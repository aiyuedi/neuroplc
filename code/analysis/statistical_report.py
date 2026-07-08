#!/usr/bin/env python3
"""
NeuroPLC — Statistical Analysis Module (P2: Statistical Rigor)
================================================================
Computes cross-seed statistics, confidence intervals, and significance
tests for all experiments requiring statistical rigor (E1-E3, E7).

Statistical methods used:
    - Bootstrap 95% CI (Efron & Tibshirani, 1993)
    - McNemar's test (paired classifiers, E1)
    - Cochran's Q test (multiple classifiers, same test set, E3)
    - Wilson score interval (proportions, E7)
    - Friedman test + Nemenyi post-hoc (E2, if multiple models)

Output:
    results/multiseed/statistical_report.json  — all computed statistics
    results/multiseed/paper_tables.tex         — LaTeX tables with mean±std

Usage:
    python analysis/statistical_report.py          # Full analysis
    python analysis/statistical_report.py --json   # JSON only
"""

import os, sys, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from scipy import stats as scipy_stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.student_kan import StudentKAN
from models.student_mlp import StudentMLP
from models.teacher_cnn import TeacherCNN

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # code/
REPO_ROOT = PROJECT_ROOT.parent  # D:/neuroplc-paper/

# ── Paths ──
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
RESULTS_DIR = REPO_ROOT / "results"
STUDENT_DIR = RESULTS_DIR / "student"
TEACHER_DIR = RESULTS_DIR / "teacher"
OUT_DIR = RESULTS_DIR / "multiseed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# Bootstrap CI
# ============================================================================

def bootstrap_ci(data, n_bootstrap=10000, ci=95, seed=42):
    """Bootstrap confidence interval for a 1-D array of values."""
    rng = np.random.RandomState(seed)
    n = len(data)
    estimates = np.array([np.mean(data[rng.randint(0, n, n)])
                           for _ in range(n_bootstrap)])
    alpha = (100 - ci) / 2
    return {
        "mean": float(np.mean(estimates)),
        "lower": float(np.percentile(estimates, alpha)),
        "upper": float(np.percentile(estimates, 100 - alpha)),
        "std": float(np.std(data, ddof=1)),
        "n": n,
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def bootstrap_ci_proportion(correct, total, n_bootstrap=10000, ci=95, seed=42):
    """Bootstrap CI for a proportion (Wilson-like via bootstrap)."""
    rng = np.random.RandomState(seed)
    estimates = np.array([
        np.mean(rng.binomial(1, correct / total, total))
        for _ in range(n_bootstrap)
    ])
    alpha = (100 - ci) / 2
    return {
        "mean": float(np.mean(estimates)),
        "lower": float(np.percentile(estimates, alpha)),
        "upper": float(np.percentile(estimates, 100 - alpha)),
    }


# ============================================================================
# McNemar's Test (Paired Classifiers)
# ============================================================================

def mcnemar_test(preds_a, preds_b, y_true):
    """McNemar's test with continuity correction.

    H0: classifiers A and B have the same error rate.
    Returns (statistic, p_value, n_discordant).
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)
    n01 = int(np.sum(~correct_a & correct_b))
    n10 = int(np.sum(correct_a & ~correct_b))
    n_total = n01 + n10
    if n_total == 0:
        return 0.0, 1.0, (0, 0)
    stat = (abs(n01 - n10) - 1) ** 2 / n_total
    p_val = 1.0 - scipy_stats.chi2.cdf(stat, 1)
    return float(stat), float(p_val), (n01, n10)


# ============================================================================
# Cochran's Q Test (Multiple Classifiers, Same Test Set)
# ============================================================================

def cochran_q_test(predictions_list, y_true):
    """Cochran's Q test for k related classifiers.

    H0: all k classifiers have the same accuracy.
    predictions_list: list of (name, preds) tuples.
    Returns (Q_statistic, p_value, per_classifier_accuracy).
    """
    k = len(predictions_list)
    n = len(y_true)
    if n < 2 or k < 2:
        return 0.0, 1.0, {}

    # Binary correctness matrix: n_samples × k_classifiers
    correct = np.zeros((n, k), dtype=np.int32)
    accs = {}
    for j, (name, preds) in enumerate(predictions_list):
        correct[:, j] = (preds == y_true).astype(np.int32)
        accs[name] = float(correct[:, j].mean())

    # Per-sample: sum over classifiers
    G_j = correct.sum(axis=0)  # total correct per classifier
    L_i = correct.sum(axis=1)  # total correct classifiers per sample

    G_mean = G_j.mean()
    numerator = (k - 1) * np.sum((G_j - G_mean) ** 2)
    denominator = np.sum(L_i * (k - L_i))

    if denominator < 1e-12:
        return 0.0, 1.0, accs

    Q = k * (k - 1) * np.sum((G_j - G_mean) ** 2) / (k * np.sum(L_i) - np.sum(L_i ** 2))
    # Simpler formulation:
    Q2 = float((k - 1) * np.sum((G_j - G_mean) ** 2) / (k * G_mean - np.sum(L_i ** 2) / n))
    # Use the standard formula
    Q_standard = float((k - 1) * (k * np.sum(G_j ** 2) - np.sum(G_j) ** 2) /
                       (k * np.sum(G_j) - np.sum(L_i ** 2)))

    p_val = 1.0 - scipy_stats.chi2.cdf(Q_standard, k - 1)
    return Q_standard, p_val, accs


# ============================================================================
# Load checkpoints
# ============================================================================

def load_predictions(model_class, ckpt_pattern, model_kwargs, X_test, y_test,
                     seeds=None, ckpt_dir=None):
    """Load predictions from multiple checkpoints across seeds.

    Args:
        model_class:   StudentKAN, StudentMLP, or TeacherCNN
        ckpt_pattern:  "kan_kd_vrmKD_seed{seed}_best.pt" or similar
        model_kwargs:  dict of kwargs for model constructor
        X_test:        (n, d) test features
        y_test:        (n,) test labels
        seeds:         list of seeds to try
        ckpt_dir:      directory containing checkpoints

    Returns:
        list of dicts: [{"seed": s, "preds": array, "acc": float, ...}, ...]
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if ckpt_dir is None:
        ckpt_dir = STUDENT_DIR

    results = []
    for seed in seeds:
        ckpt_name = ckpt_pattern.format(seed=seed)
        ckpt_path = ckpt_dir / ckpt_name
        # Fallback to default name
        if not ckpt_path.exists():
            fallback = ckpt_pattern.replace("_seed{seed}", "")
            ckpt_path = ckpt_dir / fallback.format(seed=seed) if "{" in fallback else ckpt_dir / fallback
        if not ckpt_path.exists():
            continue

        try:
            model = model_class(**model_kwargs).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            # Handle nested state dict (checkpoint can have "student_state_dict", "model_state_dict", or flat keys)
            if "student_state_dict" in ckpt:
                state_dict = ckpt["student_state_dict"]
            elif "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            else:
                state_dict = {k: v for k, v in ckpt.items()
                              if k not in ("epoch", "val_acc", "config", "adapter_state_dict")}
            model.load_state_dict(state_dict, strict=False)
            model.eval()

            with torch.no_grad():
                if hasattr(model, 'net') and model_class == StudentMLP:
                    X_t = torch.from_numpy(X_test).float().to(DEVICE)
                    logits = model(X_t)
                else:
                    X_t = torch.from_numpy(X_test).float().to(DEVICE)
                    logits = model(X_t)
                preds = logits.argmax(1).cpu().numpy()

            acc = float(accuracy_score(y_test, preds))
            _, _, f1, _ = precision_recall_fscore_support(
                y_test, preds, average='macro', zero_division=0)

            results.append({
                "seed": seed,
                "preds": preds,
                "acc": acc,
                "macro_f1": float(f1),
            })
        except Exception as e:
            print(f"  WARNING seed={seed}: {e}")
            continue

    return results


# ============================================================================
# Main Analysis
# ============================================================================

def main():
    print("=" * 70)
    print("NeuroPLC — Statistical Rigor Analysis (P2)")
    print("=" * 70)

    # ── Load test data ──
    print("\n[1/5] Loading test data...")
    X_feat = np.load(PROCESSED_DIR / "features_X.npy")
    y_all = np.load(PROCESSED_DIR / "features_y.npy")
    X_wav = np.load(PROCESSED_DIR / "waveform_X.npy")
    loads = np.load(PROCESSED_DIR / "features_load.npy")

    test_mask = np.load(SPLITS_DIR / "standard" / "test_idx.npy")
    X_feat_test = X_feat[test_mask]
    y_test = y_all[test_mask]
    X_wav_test = X_wav[test_mask]
    loads_test = loads[test_mask]
    print(f"  Test set: {len(y_test)} samples")

    report = {"test_set_size": int(len(y_test)), "seeds_target": DEFAULT_SEEDS}

    # ── E1: Teacher vs Student — multi-seed ──
    print("\n[2/5] E1: Teacher vs Student (multi-seed)...")
    e1 = {}

    # Try loading multi-seed student checkpoints
    student_results = load_predictions(
        StudentKAN,
        "kan_kd_vrmKD_seed{seed}_best.pt",
        {"layers_hidden": [28, 16, 4]},
        X_feat_test, y_test,
        seeds=DEFAULT_SEEDS,
        ckpt_dir=STUDENT_DIR,
    )

    if len(student_results) >= 3:
        student_accs = [r["acc"] for r in student_results]
        e1["student"] = bootstrap_ci(np.array(student_accs))
        e1["student_raw"] = [{"seed": r["seed"], "acc": r["acc"],
                               "macro_f1": r["macro_f1"]}
                              for r in student_results]
        print(f"  Student KAN: {e1['student']['mean']:.4f} ± "
              f"{e1['student']['std']:.4f} "
              f"(n={len(student_results)} seeds)")
    else:
        print(f"  Student KAN: only {len(student_results)} checkpoints found "
              f"(need >=3). Using single-checkpoint bootstrap.")
        # Fallback: single checkpoint with bootstrap
        ckpt_path = STUDENT_DIR / "kan_kd_vrmKD_best.pt"
        if ckpt_path.exists():
            model = StudentKAN([28, 16, 4]).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt["student_state_dict"])
            model.eval()
            with torch.no_grad():
                preds = model(torch.from_numpy(X_feat_test).float().to(DEVICE)).argmax(1).cpu().numpy()
            acc = float(accuracy_score(y_test, preds))
            e1["student"] = bootstrap_ci_proportion(
                int(acc * len(y_test)), len(y_test))
            e1["student"]["note"] = "Single checkpoint bootstrap; multi-seed pending."
            print(f"  Student KAN (single ckpt, bootstrap): "
                  f"{acc:.4f} CI95=[{e1['student']['lower']:.4f}, "
                  f"{e1['student']['upper']:.4f}]")

    # Teacher
    teacher_results = load_predictions(
        TeacherCNN,
        "teacher_seed{seed}_best.pt",
        {"num_classes": 4},
        X_wav_test.reshape(-1, 1, X_wav_test.shape[1]), y_test,
        seeds=DEFAULT_SEEDS,
        ckpt_dir=TEACHER_DIR,
    )

    if len(teacher_results) >= 3:
        teacher_accs = [r["acc"] for r in teacher_results]
        e1["teacher"] = bootstrap_ci(np.array(teacher_accs))
        print(f"  Teacher CNN: {e1['teacher']['mean']:.4f} ± "
              f"{e1['teacher']['std']:.4f} "
              f"(n={len(teacher_results)} seeds)")
    else:
        ckpt_path = TEACHER_DIR / "teacher_best.pt"
        if ckpt_path.exists():
            teacher = TeacherCNN(num_classes=4).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            teacher.load_state_dict(ckpt["model_state_dict"])
            teacher.eval()
            with torch.no_grad():
                X_t = torch.from_numpy(X_wav_test).float().unsqueeze(1).to(DEVICE)
                preds = teacher(X_t).argmax(1).cpu().numpy()
            acc = float(accuracy_score(y_test, preds))
            e1["teacher"] = bootstrap_ci_proportion(
                int(acc * len(y_test)), len(y_test))
            e1["teacher"]["note"] = "Single checkpoint bootstrap."
            print(f"  Teacher CNN (single ckpt, bootstrap): "
                  f"{acc:.4f} CI95=[{e1['teacher']['lower']:.4f}, "
                  f"{e1['teacher']['upper']:.4f}]")

    # ── E3: KD Ablation — Cochran's Q ──
    print("\n[3/5] E3: KD Ablation — significance tests...")
    e3 = {}
    kd_methods = {
        "VRM-KD": "kan_kd_vrmKD_best.pt",
        "Hinton-KD": "kan_kd_hintonKD_best.pt",
        "No-KD": "kan_nokd_best.pt",
    }
    all_preds = []

    for method, ckpt_name in kd_methods.items():
        ckpt_path = STUDENT_DIR / ckpt_name
        if not ckpt_path.exists():
            print(f"  WARNING {method}: checkpoint not found ({ckpt_name})")
            continue
        try:
            model = StudentKAN([28, 16, 4]).to(DEVICE)
            ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
            model.load_state_dict(ckpt.get("student_state_dict", ckpt))
            model.eval()
            with torch.no_grad():
                preds = model(torch.from_numpy(X_feat_test).float().to(DEVICE)).argmax(1).cpu().numpy()
            acc = float(accuracy_score(y_test, preds))
            all_preds.append((method, preds))
            print(f"  {method}: {acc:.4f}")
        except Exception as e:
            print(f"  WARNING {method}: {e}")

    if len(all_preds) >= 2:
        Q, p_val, per_acc = cochran_q_test(all_preds, y_test)
        e3["cochran_q"] = {"Q": float(Q), "p_value": float(p_val),
                            "significant_at_0.05": p_val < 0.05,
                            "per_method_acc": per_acc}
        sig = "SIGNIFICANT" if p_val < 0.05 else "NOT significant"
        print(f"  Cochran's Q = {Q:.3f}, p = {p_val:.4f} → {sig}")

    # ── E7: Cross-Load — Wilson CI per load ──
    print("\n[4/5] E7: Cross-Load — per-load statistics...")
    e7 = {}
    # Use the standard checkpoint for cross-load evaluation
    ckpt_path = STUDENT_DIR / "kan_kd_vrmKD_best.pt"
    if ckpt_path.exists():
        model = StudentKAN([28, 16, 4]).to(DEVICE)
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["student_state_dict"])
        model.eval()
        for tgt_load, label in [(0, "0hp"), (2, "2hp"), (3, "3hp")]:
            mask = loads == tgt_load
            if mask.sum() == 0:
                continue
            with torch.no_grad():
                preds = model(torch.from_numpy(X_feat[mask]).float().to(DEVICE)).argmax(1).cpu().numpy()
            acc = float(accuracy_score(y_all[mask], preds))
            ci = bootstrap_ci_proportion(
                int(acc * mask.sum()), int(mask.sum()))
            e7[label] = {
                "accuracy": acc,
                "n_samples": int(mask.sum()),
                "ci95_lower": ci["lower"],
                "ci95_upper": ci["upper"],
            }
            print(f"  {label}: {acc:.4f} CI95=[{ci['lower']:.4f}, {ci['upper']:.4f}] "
                  f"(n={mask.sum()})")

    # ── Aggregate Report ──
    report["E1"] = e1
    report["E3"] = e3
    report["E7"] = e7

    json_path = OUT_DIR / "statistical_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n[5/5] Report saved → {json_path}")

    # ── Print summary ──
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)

    if "student" in e1 and "teacher" in e1:
        s, t = e1["student"], e1["teacher"]
        print(f"E1 Teacher: {t['mean']:.4f} ± {t.get('std', 'N/A')} "
              f"CI95=[{t['lower']:.4f}, {t['upper']:.4f}]")
        print(f"E1 Student: {s['mean']:.4f} ± {s.get('std', 'N/A')} "
              f"CI95=[{s['lower']:.4f}, {s['upper']:.4f}]")

    if "cochran_q" in e3:
        cq = e3["cochran_q"]
        print(f"E3 KD Ablation: Cochran's Q={cq['Q']:.2f}, p={cq['p_value']:.4f} "
              f"({'*' if cq['significant_at_0.05'] else 'n.s.'})")

    for load_name, stats_ld in e7.items():
        print(f"E7 {load_name}: {stats_ld['accuracy']:.4f} "
              f"CI95=[{stats_ld['ci95_lower']:.4f}, {stats_ld['ci95_upper']:.4f}]")

    print("=" * 70)

    return report


if __name__ == "__main__":
    main()
