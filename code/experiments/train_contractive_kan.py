#!/usr/bin/env python3
"""
train_contractive_kan.py — contractive KAN training (Condition 3, gamma < 1).

Goal: train a KAN [28,16,4] on CWRU whose per-activation-layer amplification
gamma = [gamma_1, gamma_2] < 1 (measured numerically, E68 semantics), with
accuracy >= 99%.  Two mechanisms:
  (a) soft penalty on gamma in the loss (smooth);
  (b) hard projection every PROJ_EVERY steps (scale both base and spline
      weights of a layer so its measured amplification <= GAMMA_TARGET).

Output: results/student/kan_contractive.pt + results/theory/contractive.json
Run: python experiments/train_contractive_kan.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.student_kan import StudentKAN

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "student"
OUT_T = ROOT / "results" / "theory"

ARCH = [28, 16, 4]
XR = (-3.0, 3.0)
EPOCHS = 80
BATCH = 128
LR = 3e-3
SEED = 42
GAMMA_TARGET = 0.85          # Condition 3: per-layer amplification < 1
PROJ_EVERY = 10              # hard projection cadence (steps)
GAMMA_LAMBDA = 20.0          # soft penalty weight
N_GAMMA_SAMPLES = 600        # samples for numeric gamma measurement


def load_data():
    X = np.load(DATA / "features_X.npy").astype(np.float32)
    y = np.load(DATA / "features_y.npy").astype(np.int64)
    X = np.clip(X, XR[0], XR[1])
    idx = np.arange(len(y))
    train_val, test_idx = train_test_split(idx, test_size=0.2,
                                           stratify=y, random_state=42)
    train_idx, val_idx = train_test_split(
        train_val, test_size=0.125, stratify=y[train_val], random_state=42)
    return (X[train_idx], y[train_idx], X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def measure_gamma(model):
    """Per-activation-layer amplification, numeric (E68 semantics).

    gamma_l = max over inputs of ||h_l(x)||_inf / ||h_{l-1}(x)||_inf
    (with ||.||_inf over the activation dimension), sampled on the domain.
    """
    model.eval()
    xs = np.random.default_rng(0).uniform(XR[0], XR[1],
                                          (N_GAMMA_SAMPLES, ARCH[0]))
    gammas = []
    with torch.no_grad():
        h = torch.from_numpy(xs).float()
        for layer in model.kan_layers:
            h_next = layer(h)
            ratio = (h_next.abs().max(dim=1)[0] /
                     (h.abs().max(dim=1)[0] + 1e-8))
            gammas.append(float(ratio.max()))
            h = h_next
    return gammas


def project_contractive(model, target):
    """Numeric per-layer projection: measure gamma, scale weights iteratively.

    Uses the measured (E68-semantics) per-layer amplification; each
    layer is scaled by target/gamma_l (base and spline jointly), then
    re-measured (up to 4 iterations).  This matches the metric that
    Condition 3 is stated on.
    """
    for _ in range(4):
        g = measure_gamma(model)
        if max(g) <= target:
            return g
        with torch.no_grad():
            for l, layer in enumerate(model.kan_layers):
                if g[l] > target and g[l] > 0:
                    s = target / g[l]
                    layer.base_weight.mul_(s)
                    layer.spline_weight.mul_(s)
    return measure_gamma(model)


def main():
    ap = argparse.ArgumentParser(description="contractive KAN training (v2)")
    ap.add_argument("--gamma-target", type=float, default=GAMMA_TARGET)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--gamma-lambda", type=float, default=GAMMA_LAMBDA)
    ap.add_argument("--w-bound", type=float, default=0.0,
                    help="clip |spline_weight| <= w-bound (bounded-amplitude basis, "
                         "v2: unlocks gamma<1; 0 disables)")
    ap.add_argument("--out", type=str, default="kan_contractive_v2.pt")
    ap.add_argument("--arch", type=str, default="28,16,4",
                    help="comma-separated layer widths (default 2-layer 28,16,4)")
    args = ap.parse_args()
    gamma_target = args.gamma_target
    n_epochs = args.epochs
    gamma_lambda = args.gamma_lambda

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xt, yt, Xv, yv, Xte, yte = load_data()
    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv); yv_t = torch.from_numpy(yv)

    arch = [int(x) for x in args.arch.split(",")]
    m = StudentKAN(arch)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    rng = np.random.RandomState(SEED)
    best_val, best_sd, bad, n = -1.0, None, 0, len(Xt)

    gamma_history = []
    for ep in range(n_epochs):
        m.train()
        perm = rng.permutation(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            logits = m(Xt_t[idx])
            loss = F.cross_entropy(logits, yt_t[idx])
            # (a) soft gamma penalty
            with torch.no_grad():
                g = measure_gamma(m)
            pen = sum(max(0.0, gi - gamma_target) ** 2 for gi in g)
            loss = loss + gamma_lambda * pen
            loss.backward()
            opt.step()
            # (b) bounded-amplitude basis (v2): clip spline control points
            if args.w_bound > 0:
                with torch.no_grad():
                    for layer in m.kan_layers:
                        layer.spline_weight.data.clamp_(-args.w_bound, args.w_bound)
            # (c) hard projection cadence
            if (b // BATCH) % PROJ_EVERY == 0:
                project_contractive(m, gamma_target)
        sched.step()
        m.eval()
        with torch.no_grad():
            va = float((m(Xv_t).argmax(1) == yv_t).float().mean())
        g = measure_gamma(m)
        gamma_history.append(g)
        if va > best_val:
            best_val = va
            best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 20:
            break
        if ep % 5 == 0 or ep == n_epochs - 1:
            print(f"ep {ep}: val {va:.4f} gamma {[round(x, 3) for x in g]}",
                  flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    with torch.no_grad():
        acc = float((m(torch.from_numpy(Xte)).argmax(1) ==
                     torch.from_numpy(yte)).float().mean())
    g_final = measure_gamma(m)

    out_model = OUT / args.out
    torch.save(best_sd, out_model)
    res = {"date": "2026-08-04", "script": "train_contractive_kan.py",
           "gamma_target": gamma_target, "epochs": n_epochs,
           "gamma_lambda": gamma_lambda, "w_bound": args.w_bound,
           "gamma_measured": [round(x, 4) for x in g_final],
           "gamma_history_last": [round(x, 4) for x in gamma_history[-1]],
           "test_acc": float(acc), "best_val": float(best_val),
           "condition3_satisfied": bool(max(g_final) < 1.0),
           "verdict": "PASS" if (max(g_final) < 1.0 and acc >= 0.99)
                      else "CHECK"}
    os.makedirs(OUT_T, exist_ok=True)
    with open(OUT_T / "contractive.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    print(f"Saved: {out_model}")


if __name__ == "__main__":
    main()
