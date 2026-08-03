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
EPOCHS = 60
BATCH = 128
LR = 3e-3
SEED = 42
GAMMA_TARGET = 0.95          # Condition 3: per-layer amplification < 1
PROJ_EVERY = 10              # hard projection cadence (steps)
GAMMA_LAMBDA = 8.0           # soft penalty weight
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
        h = torch.from_numpy(xs)
        for layer in model.kan_layers:
            h_next = layer(h)
            ratio = (h_next.abs().max(dim=1)[0] /
                     (h.abs().max(dim=1)[0] + 1e-8))
            gammas.append(float(ratio.max()))
            h = h_next
    return gammas


def project_contractive(model, target):
    """Scale base+spline weights per layer so measured gamma <= target."""
    for attempt in range(3):
        g = measure_gamma(model)
        if max(g) <= target:
            return g
        with torch.no_grad():
            for l, layer in enumerate(model.kan_layers):
                # row-sum-inf norm of effective weight
                w_eff = (layer.base_weight + layer.spline_weight.mean(-1))
                nrm = w_eff.abs().sum(dim=1).max().item()
                if nrm > 1e-9:
                    s = min(1.0, (target * nrm) / nrm)  # scale to target row-sum
                    # row-wise scaling: keep relative structure, bound the max row
                    rowsum = w_eff.abs().sum(dim=1)
                    scale = torch.clamp(target / (rowsum + 1e-9), max=1.0)
                    layer.base_weight.mul_(scale[:, None])
                    layer.spline_weight.mul_(scale[:, None, None])
        g = measure_gamma(model)
        if max(g) <= target:
            return g
    return measure_gamma(model)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    Xt, yt, Xv, yv, Xte, yte = load_data()
    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv); yv_t = torch.from_numpy(yv)

    m = StudentKAN(ARCH)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    rng = np.random.RandomState(SEED)
    best_val, best_sd, bad, n = -1.0, None, 0, len(Xt)

    gamma_history = []
    for ep in range(EPOCHS):
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
            pen = sum(max(0.0, gi - GAMMA_TARGET) ** 2 for gi in g)
            loss = loss + GAMMA_LAMBDA * pen
            loss.backward()
            opt.step()
            # (b) hard projection cadence
            if (b // BATCH) % PROJ_EVERY == 0:
                project_contractive(m, GAMMA_TARGET)
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
        if bad >= 8:
            break
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"ep {ep}: val {va:.4f} gamma {[round(x, 3) for x in g]}",
                  flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    with torch.no_grad():
        acc = float((m(torch.from_numpy(Xte)).argmax(1) ==
                     torch.from_numpy(yte)).float().mean())
    g_final = measure_gamma(m)

    out_model = OUT / "kan_contractive.pt"
    torch.save(best_sd, out_model)
    res = {"date": "2026-08-04", "script": "train_contractive_kan.py",
           "gamma_target": GAMMA_TARGET,
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
