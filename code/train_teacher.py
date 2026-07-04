#!/usr/bin/env python3
"""
NeuroPLC — Teacher CNN Training
=================================
Train a 1D-CNN with Self-Attention on CWRU waveform data.
The teacher is a large model (~49K params) that achieves 99%+ accuracy.
It serves as the knowledge source for VRM-KD distillation into the Student KAN.

Workflow:
    1. Load waveform data: data/processed/waveform_X.npy, waveform_y.npy
    2. Standard stratified train/val/test split
    3. Train TeacherCNN (80 epochs, Adam + cosine annealing)
    4. Save checkpoint to results/teacher/teacher_best.pt

Usage:
    python train_teacher.py                     # Full training
    python train_teacher.py --epochs 20         # Quick run
    python train_teacher.py --batch-size 32     # Smaller batch for CPU
    python train_teacher.py --tag v1            # Experiment tag

Tracks: MLflow (experiment name: "neuroplc")
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
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report

from models.teacher_cnn import TeacherCNN
from neuroplc.utils.mlflow_tracker import ExperimentTracker


# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results" / "teacher"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> tuple:
    """Load preprocessed waveform data."""
    X = np.load(PROCESSED_DIR / "waveform_X.npy")      # (N, 1024)
    y = np.load(PROCESSED_DIR / "waveform_y.npy")       # (N,)
    return torch.from_numpy(X).float(), torch.from_numpy(y).long()


def create_dataloaders(X, y, batch_size=64, test_size=0.2, val_size=0.1, seed=42
                       ) -> tuple:
    """Stratified train/val/test split → DataLoaders."""
    from sklearn.model_selection import train_test_split

    N = len(y)
    idx = np.arange(N)

    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=seed)
    val_frac = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac, stratify=y[train_val_idx],
        random_state=seed)

    loaders = {}
    for name, subset_idx in [("train", train_idx), ("val", val_idx),
                              ("test", test_idx)]:
        ds = TensorDataset(X[subset_idx], y[subset_idx])
        loaders[name] = DataLoader(
            ds, batch_size=batch_size, shuffle=(name == "train"),
            drop_last=(name == "train"))

    return loaders, {"train": len(train_idx), "val": len(val_idx),
                     "test": len(test_idx)}


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Single training epoch."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x_batch, y_batch in loader:
        x_batch = x_batch.unsqueeze(1).to(device)  # (B, 1, 1024)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_batch.size(0)
        correct += (logits.argmax(1) == y_batch).sum().item()
        total += x_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation or test set."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x_batch, y_batch in loader:
        x_batch = x_batch.unsqueeze(1).to(device)
        y_batch = y_batch.to(device)
        logits = model(x_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * x_batch.size(0)
        preds = logits.argmax(1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="NeuroPLC Teacher CNN Training")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--tag", type=str, default="")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load data ──
    print("Loading data...")
    X, y = load_data()
    loaders, splits = create_dataloaders(X, y, batch_size=args.batch_size)
    print(f"Samples: {splits['train']} train / {splits['val']} val / "
          f"{splits['test']} test")

    # ── Model ──
    model = TeacherCNN(num_classes=4).to(device)
    print(model.summary())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    criterion = nn.CrossEntropyLoss()

    # ── MLflow ──
    tracker = ExperimentTracker(
        run_name=f"teacher{'_'+args.tag if args.tag else ''}",
        config={
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "model_params": model.parameter_count,
        },
        experiment_name="neuroplc",
        enabled=not args.no_mlflow,
    )

    with tracker:
        best_acc = 0.0
        patience_counter = 0
        t0 = time.time()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, loaders["train"], optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(
                model, loaders["val"], criterion, device)

            scheduler.step()

            tracker.log_metrics_batch({
                "train_loss": train_loss, "train_acc": train_acc,
                "val_loss": val_loss, "val_acc": val_acc, "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{args.epochs} | "
                      f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
                      f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

            # Early stopping
            if val_acc > best_acc + 1e-4:
                best_acc = val_acc
                patience_counter = 0
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "config": vars(args),
                }, RESULTS_DIR / "teacher_best.pt")
            else:
                patience_counter += 1

            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0

        # ── Final test evaluation ──
        print("\nLoading best checkpoint for test evaluation...")
        ckpt = torch.load(RESULTS_DIR / "teacher_best.pt", map_location=device,
                          weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        test_loss, test_acc, test_preds, test_labels = evaluate(
            model, loaders["test"], criterion, device)

        report = classification_report(
            test_labels, test_preds,
            target_names=["Normal", "InnerRace", "Ball", "OuterRace"],
            digits=4)
        print(f"\n{report}")

        tracker.log_classification_report(test_labels, test_preds,
                                          class_names=["Normal", "InnerRace",
                                                       "Ball", "OuterRace"])
        tracker.log_metric("test_acc", test_acc)
        tracker.log_model(model, "teacher_cnn")

    print(f"\nDone. Best val_acc={best_acc:.4f} | test_acc={test_acc:.4f} | "
          f"{elapsed:.0f}s")
    print(f"Checkpoint: {RESULTS_DIR / 'teacher_best.pt'}")


if __name__ == "__main__":
    main()
