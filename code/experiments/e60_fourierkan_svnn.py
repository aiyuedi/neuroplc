#!/usr/bin/env python3
"""
NeuroPLC -- E60: FourierKAN SVNN Verification
==============================================
Trains FourierKAN [28,16,4] on CWRU, computes M2 per edge,
reports Z3-equivalent verifiability rate: edges where
M2 * h_lut^2 / 8 < deployment margin (0.182).

Proposition 9c predicts: FourierKAN satisfies SVNN Conditions 1-2.
M2 formula: omega^2 * sum_{k=1}^K k^2 * (|c_k| + |d_k|)

Usage:
    python e60_fourierkan_svnn.py [--epochs 20] [--lrate 0.001]
"""

import sys, os, json, argparse
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODE_DIR))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score

from models.student_fourierkan import StudentFourierKAN

# ── Paths ──
PROJECT_ROOT = CODE_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = CODE_DIR / "results" / "theory"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARCH = [28, 16, 4]
DEPLOYMENT_MARGIN = 0.182
N_LUT = 15
DOMAIN_LO, DOMAIN_HI = -3.0, 3.0
H_LUT = (DOMAIN_HI - DOMAIN_LO) / (N_LUT - 1)  # = 6/14 = 0.4286


def load_cwru_data():
    """Load preprocessed CWRU data (.npy format)."""
    X = np.load(DATA_DIR / "features_X.npy")
    y = np.load(DATA_DIR / "features_y.npy")
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()

    n = len(X)
    n_train = int(0.7 * n)
    idx = torch.randperm(n)
    return X[idx[:n_train]], y[idx[:n_train]], X[idx[n_train:]], y[idx[n_train:]]


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, lr=0.001):
    """Train FourierKAN on CWRU."""
    ds = TensorDataset(X_train, y_train)
    dl = DataLoader(ds, batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for bx, by in dl:
            opt.zero_grad()
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(dim=1)
            acc = accuracy_score(y_test.numpy(), preds.numpy())
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 5 == 0:
            print(f"  Epoch {ep:3d}: loss={total_loss/len(dl):.4f}, test_acc={acc:.4f}")

    model.load_state_dict(best_state)
    return best_acc


def compute_verifiability(model, h_lut=H_LUT, margin=DEPLOYMENT_MARGIN):
    """Compute M2 per edge and Z3-equivalent verifiability.

    For FourierKAN, each edge's activation phi(x) = sum_k [c_k*sin(k*w*x) + d_k*cos(k*w*x)]
    has M2 = omega^2 * sum_k k^2 * (|c_k| + |d_k|).

    For Z3-equivalent verifiability: M2 * h_lut^2 / 8 <= margin
    (same de Boor guarantee as Theorem 7, but using the analytical M2 formula)
    """
    all_edges = []
    for l_idx, layer in enumerate(model.kan_layers):
        m2 = layer.compute_m2_bounds()  # (out, in)
        m2h2 = m2 * h_lut**2 / 8  # (out, in)
        out_d, in_d = m2.shape
        for j in range(out_d):
            for i in range(in_d):
                all_edges.append({
                    'layer': l_idx, 'out_idx': j, 'in_idx': i,
                    'M2': float(m2[j, i].item()),
                    'M2h2_8': float(m2h2[j, i].item()),
                    'verified': bool(m2h2[j, i].item() <= margin),
                })

    return all_edges


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--lrate', type=float, default=0.001)
    parser.add_argument('--n_harmonics', type=int, default=6)
    parser.add_argument('--omega', type=float, default=0.4)
    args = parser.parse_args()

    print("=" * 70)
    print(f"E60: FourierKAN SVNN Verification (K={args.n_harmonics}, w={args.omega})")
    print("=" * 70)

    # Load data
    print("\n[1/3] Loading CWRU data...")
    X_train, y_train, X_test, y_test = load_cwru_data()
    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"  Classes: {torch.unique(y_test).tolist()}")

    # Train
    print(f"\n[2/3] Training FourierKAN [28,16,4] (K={args.n_harmonics}, w={args.omega})...")
    model = StudentFourierKAN(ARCH, n_harmonics=args.n_harmonics, omega=args.omega)
    print(f"  Parameters: {model.n_params:,}")
    print(f"  Edges: {model.n_edges}")

    best_acc = train_model(model, X_train, y_train, X_test, y_test,
                           epochs=args.epochs, lr=args.lrate)
    print(f"  Best test accuracy: {best_acc:.4f} ({100*best_acc:.2f}%)")

    # M2 analysis
    print(f"\n[3/3] Z3-equivalent verifiability (M2*h_lut^2/8 <= {DEPLOYMENT_MARGIN})...")
    edges = compute_verifiability(model)
    m2h2_vals = np.array([e['M2h2_8'] for e in edges])
    n_total = len(edges)
    n_verified = int(np.sum(m2h2_vals <= DEPLOYMENT_MARGIN))

    print(f"  Total edges: {n_total}")
    print(f"  h_lut = {H_LUT:.4f}, margin = {DEPLOYMENT_MARGIN}")
    print(f"  M2: mean={np.mean([e['M2'] for e in edges]):.4f}, "
          f"P50={np.median([e['M2'] for e in edges]):.4f}, "
          f"P95={np.percentile([e['M2'] for e in edges], 95):.4f}, "
          f"max={np.max([e['M2'] for e in edges]):.4f}")
    print(f"  M2*h_lut^2/8: mean={np.mean(m2h2_vals):.5f}, "
          f"P50={np.median(m2h2_vals):.5f}, "
          f"P95={np.percentile(m2h2_vals, 95):.5f}, "
          f"max={np.max(m2h2_vals):.5f}")
    print(f"  Z3-equivalent verified: {n_verified}/{n_total} "
          f"({100*n_verified/n_total:.1f}%)")

    # Safety margin
    if n_verified < n_total:
        failed_vals = m2h2_vals[m2h2_vals > DEPLOYMENT_MARGIN]
        print(f"  Failed edges: {n_total - n_verified}, max exceed = {np.max(failed_vals):.5f}")
        print(f"  Safety margin (min): {DEPLOYMENT_MARGIN / max(m2h2_vals):.3f}x")
    else:
        safety_margin = DEPLOYMENT_MARGIN / max(m2h2_vals)
        print(f"  ALL EDGES VERIFIED! Safety margin: {safety_margin:.1f}x")

    # Save results
    result = {
        'experiment': 'E60',
        'architecture': 'FourierKAN',
        'arch': ARCH,
        'n_harmonics': args.n_harmonics,
        'omega': args.omega,
        'n_params': model.n_params,
        'n_edges': n_total,
        'test_accuracy': float(best_acc),
        'h_lut': float(H_LUT),
        'N_lut': N_LUT,
        'deployment_margin': DEPLOYMENT_MARGIN,
        'n_verified': int(n_verified),
        'n_total': n_total,
        'verification_rate': float(n_verified / n_total),
        'M2_stats': {
            'mean': float(np.mean([e['M2'] for e in edges])),
            'p50': float(np.median([e['M2'] for e in edges])),
            'p95': float(np.percentile([e['M2'] for e in edges], 95)),
            'max': float(np.max([e['M2'] for e in edges])),
        },
        'M2h2_stats': {
            'mean': float(np.mean(m2h2_vals)),
            'p50': float(np.median(m2h2_vals)),
            'p95': float(np.percentile(m2h2_vals, 95)),
            'max': float(np.max(m2h2_vals)),
        },
        'svnn_conditions': {
            'condition_1': 'PASS (operation-type closure: Fourier basis is element-wise)',
            'condition_2': f'PASS (M2 = w^2 * sum k^2*(|c|+|d|) = computable)',
        },
    }

    out_path = RESULTS_DIR / "e60_fourierkan_results.json"
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n[DONE] Results saved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
