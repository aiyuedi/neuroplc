#!/usr/bin/env python3
"""
NeuroPLC — Student KAN Training via VRM Knowledge Distillation
================================================================
Distill knowledge from a pre-trained Teacher CNN into a lightweight
KAN student using Virtual Relation Matching (VRM) + temperature KD.

Loss:
    L = α·KL(qT||qS) + (1-α)·CE(y, qS) + λ_rel·L_VRM + λ_feat·L_feat

where:
    KL:  Kullback-Leibler divergence between temperature-scaled logits
    CE:  standard cross-entropy with hard labels
    VRM: cosine similarity matrix MSE (Virtual Relation Matching, ICCV 2025)
    feat: L2 distance between mapped teacher/student feature vectors

Reference:
    VRM: Zhang et al., ICCV 2025 (Highlight)
    Hinton KD: Hinton et al., arXiv 2015

Usage:
    python train_student_kd.py --teacher results/teacher/teacher_best.pt
    python train_student_kd.py --teacher results/teacher/teacher_best.pt --epochs 20 --test-mode
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from models.student_kan import StudentKAN
from models.student_mlp import StudentMLP
from models.teacher_cnn import TeacherCNN
from neuroplc.utils.mlflow_tracker import ExperimentTracker


# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
TEACHER_DIR = RESULTS_DIR / "teacher"
STUDENT_DIR = RESULTS_DIR / "student"
STUDENT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Data Loading
# ============================================================================

def load_data() -> tuple:
    """Load preprocessed features (28-D) and waveform (for teacher)."""
    X_feat = np.load(PROCESSED_DIR / "features_X.npy")    # (N, 28)
    y = np.load(PROCESSED_DIR / "features_y.npy")          # (N,)
    X_wav = np.load(PROCESSED_DIR / "waveform_X.npy")      # (N, 1024)
    return (torch.from_numpy(X_feat).float(),
            torch.from_numpy(X_wav).float(),
            torch.from_numpy(y).long())


def create_dataloaders(X_feat, X_wav, y, batch_size=128,
                       test_size=0.2, val_size=0.1, seed=42):
    """Stratified split → DataLoaders for both features and waveform."""
    from sklearn.model_selection import train_test_split

    N = len(y)
    idx = np.arange(N)
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed)
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac,
        stratify=y[train_val_idx], random_state=seed)

    loaders = {}
    splits = {}
    for name, subset_idx in [("train", train_idx), ("val", val_idx),
                              ("test", test_idx)]:
        ds = TensorDataset(X_feat[subset_idx], X_wav[subset_idx], y[subset_idx])
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"))
        splits[name] = len(subset_idx)
    return loaders, splits


# ============================================================================
# VRM Loss
# ============================================================================

def cosine_similarity_matrix(z: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix for a batch of logits.

    Args:
        z: (B, C) — logits or features

    Returns:
        S: (B, B) — S[i,j] = cos_sim(z_i, z_j)
    """
    z_norm = F.normalize(z, p=2, dim=1)  # (B, C)
    S = z_norm @ z_norm.T                 # (B, B)
    return S


def vrm_loss(teacher_logits: torch.Tensor,
             student_logits: torch.Tensor) -> torch.Tensor:
    """
    Virtual Relation Matching loss.

    L_VRM = MSE(cos_sim_matrix(T), cos_sim_matrix(S))

    This enforces the student to preserve the relative structure
    between samples that the teacher has learned.
    """
    T_sim = cosine_similarity_matrix(teacher_logits)  # (B, B)
    S_sim = cosine_similarity_matrix(student_logits)   # (B, B)
    return F.mse_loss(S_sim, T_sim)


# ============================================================================
# Feature Alignment
# ============================================================================

class FeatureAdapter(nn.Module):
    """Linear projection from student feature dim to teacher feature dim."""

    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.fc = nn.Linear(student_dim, teacher_dim)

    def forward(self, x):
        return self.fc(x)


# ============================================================================
# Training
# ============================================================================

def distillation_loss(student_logits, teacher_logits, labels,
                      temperature=4.0, alpha=0.3,
                      lambda_rel=0.5, lambda_feat=0.1,
                      student_features=None, teacher_features=None,
                      feature_adapter=None):
    """
    Combined knowledge distillation loss.

    Returns:
        total_loss, loss_dict
    """
    # Temperature-scaled soft targets
    T = temperature
    soft_teacher = F.log_softmax(teacher_logits / T, dim=1)
    soft_student = F.log_softmax(student_logits / T, dim=1)

    loss_kl = F.kl_div(soft_student, soft_teacher, log_target=True,
                        reduction="batchmean") * (T ** 2)
    loss_ce = F.cross_entropy(student_logits, labels)

    loss_kd = alpha * loss_kl + (1.0 - alpha) * loss_ce
    total = loss_kd

    losses = {"KL": loss_kl.item(), "CE": loss_ce.item(), "KD": loss_kd.item()}

    # VRM
    if lambda_rel > 0:
        loss_vrm = vrm_loss(teacher_logits, student_logits)
        total = total + lambda_rel * loss_vrm
        losses["VRM"] = loss_vrm.item()

    # Feature alignment
    if lambda_feat > 0 and teacher_features is not None and \
       student_features is not None and feature_adapter is not None:
        student_mapped = feature_adapter(student_features)
        loss_feat = F.mse_loss(student_mapped, teacher_features)
        total = total + lambda_feat * loss_feat
        losses["Feat"] = loss_feat.item()

    return total, losses


def train_one_epoch(student, teacher, adapter, loader, optimizer,
                    device, temperature, alpha, lambda_rel, lambda_feat):
    """One epoch of KD training."""
    student.train()
    teacher.eval()

    total_loss, correct, n = 0.0, 0, 0
    all_losses = {}

    for x_feat, x_wav, y_batch in loader:
        x_feat = x_feat.to(device)
        x_wav = x_wav.unsqueeze(1).to(device)
        y_batch = y_batch.to(device)
        B = x_feat.size(0)

        # Teacher forward (no grad)
        with torch.no_grad():
            t_logits, t_features = teacher(x_wav, return_features=True)

        # Student forward
        s_logits = student(x_feat)  # KAN forward

        # For feature alignment: use student's intermediate output
        # KAN doesn't have a natural 'features' layer, so we use logits
        # as the feature representation (both are low-dim)
        s_features = s_logits  # (B, 4) for alignment to teacher (B, 64)

        loss, losses = distillation_loss(
            s_logits, t_logits, y_batch,
            temperature=temperature, alpha=alpha,
            lambda_rel=lambda_rel, lambda_feat=lambda_feat,
            student_features=s_features, teacher_features=t_features,
            feature_adapter=adapter,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item() * B
        correct += (s_logits.argmax(1) == y_batch).sum().item()
        n += B

        for k, v in losses.items():
            all_losses[k] = all_losses.get(k, 0.0) + v * B

    for k in all_losses:
        all_losses[k] /= max(n, 1)

    return total_loss / max(n, 1), correct / max(n, 1), all_losses


@torch.no_grad()
def evaluate_student(student, loader, device):
    """Evaluate student on features only (no teacher needed)."""
    student.eval()
    correct, n = 0, 0
    all_preds, all_labels = [], []
    for x_feat, _, y_batch in loader:
        x_feat = x_feat.to(device)
        logits = student(x_feat)
        preds = logits.argmax(1)
        correct += (preds == y_batch.to(device)).sum().item()
        n += x_feat.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
    return correct / max(n, 1), np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="NeuroPLC Student KAN via VRM-KD")
    parser.add_argument("--teacher", type=str,
                        default=str(TEACHER_DIR / "teacher_best.pt"),
                        help="Path to trained teacher checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--lambda-rel", type=float, default=0.5)
    parser.add_argument("--lambda-feat", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--student-type", choices=["kan", "mlp"], default="kan")
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument("--no-vrm", action="store_true",
                        help="Disable VRM (Hinton KD only)")
    parser.add_argument("--no-kd", action="store_true",
                        help="Disable KD entirely (student from scratch)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading data...")
    X_feat, X_wav, y = load_data()
    loaders, splits = create_dataloaders(
        X_feat, X_wav, y, batch_size=args.batch_size)
    print(f"Samples: {splits['train']} train / {splits['val']} val / "
          f"{splits['test']} test")
    print(f"Feature dim: {X_feat.shape[1]}")
    print(f"Classes: {len(torch.unique(y))}")

    # ── Teacher ──
    print(f"\nLoading teacher: {args.teacher}")
    teacher = TeacherCNN(num_classes=4).to(device)
    if not Path(args.teacher).exists():
        print(f"WARNING: Teacher checkpoint not found at {args.teacher}")
        print("Training student WITHOUT teacher supervision...")
        args.no_kd = True
    else:
        ckpt = torch.load(args.teacher, map_location=device, weights_only=True)
        teacher.load_state_dict(ckpt["model_state_dict"])
        print(f"Teacher loaded (val_acc={ckpt.get('val_acc', '?')})")

    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # ── Student ──
    if args.student_type == "kan":
        student = StudentKAN([28, 16, 4]).to(device)
        tag_prefix = "kan_kd"
    else:
        student = StudentMLP(input_dim=28, hidden_dims=[32, 16],
                             num_classes=4).to(device)
        tag_prefix = "mlp_kd"

    print(student.summary() if args.student_type == "kan" else
          f"StudentMLP — {student.parameter_count} params")

    # Feature adapter: student logits (4) → teacher features (64)
    adapter = FeatureAdapter(
        4 if args.student_type == "kan" else 4, 64).to(device)

    # ── Optimizer ──
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(adapter.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs)

    # ── Mode string ──
    mode_parts = [tag_prefix]
    if args.no_kd:
        mode_parts.append("noKD")
    elif args.no_vrm:
        mode_parts.append("hintonKD")
    else:
        mode_parts.append("vrmKD")
    if args.tag:
        mode_parts.append(args.tag)
    mode = "_".join(mode_parts)

    # ── KD params (zero out if no_kd) ──
    temp = 0.0 if args.no_kd else args.temperature
    alpha = 0.0 if args.no_kd else args.alpha
    l_rel = 0.0 if (args.no_kd or args.no_vrm) else args.lambda_rel
    l_feat = 0.0 if args.no_kd else args.lambda_feat

    print(f"\nMode: {mode}")
    print(f"KD: τ={temp}, α={alpha}, λ_vrm={l_rel}, λ_feat={l_feat}")

    # ── MLflow ──
    tracker = ExperimentTracker(
        run_name=mode,
        config={
            "student_type": args.student_type,
            "student_params": student.parameter_count,
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "temperature": temp, "alpha": alpha,
            "lambda_rel": l_rel, "lambda_feat": l_feat,
        },
        experiment_name="neuroplc",
        enabled=not args.no_mlflow,
    )

    with tracker:
        best_acc = 0.0
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc, loss_dict = train_one_epoch(
                student, teacher, adapter, loaders["train"], optimizer,
                device, temp, alpha, l_rel, l_feat)

            val_acc, _, _ = evaluate_student(
                student, loaders["val"], device)

            scheduler.step()

            # Log
            metrics = {"train_loss": train_loss, "train_acc": train_acc,
                       "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"]}
            if loss_dict:
                metrics.update({f"loss_{k}": v for k, v in loss_dict.items()})
            tracker.log_metrics_batch(metrics, step=epoch)

            if epoch % 10 == 0 or epoch == 1:
                loss_str = " ".join(f"{k}={v:.3f}" for k, v in loss_dict.items()
                                    if k != "KD")
                print(f"Epoch {epoch:3d}/{args.epochs} | "
                      f"acc={train_acc:.3f}/{val_acc:.3f} | {loss_str}")

            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch, "val_acc": val_acc,
                    "student_state_dict": student.state_dict(),
                    "adapter_state_dict": adapter.state_dict(),
                    "config": vars(args),
                }, STUDENT_DIR / f"{mode}_best.pt")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0

        # ── Test ──
        ckpt_path = STUDENT_DIR / f"{mode}_best.pt"
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        student.load_state_dict(ckpt["student_state_dict"])
        test_acc, test_preds, test_labels = evaluate_student(
            student, loaders["test"], device)

        report = classification_report(
            test_labels, test_preds,
            target_names=["Normal", "InnerRace", "Ball", "OuterRace"],
            digits=4)
        print(f"\n{report}")

        tracker.log_classification_report(
            test_labels, test_preds,
            class_names=["Normal", "InnerRace", "Ball", "OuterRace"])
        tracker.log_metric("test_acc", test_acc)
        tracker.log_model(student, f"student_{args.student_type}")

    print(f"\nDone. Best val_acc={best_acc:.4f} | test_acc={test_acc:.4f} | "
          f"{elapsed:.0f}s")
    print(f"Checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
