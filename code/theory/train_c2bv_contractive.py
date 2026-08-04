#!/usr/bin/env python3
"""Bounded-amplitude basis contractive training (2026-08-04 P0).

FourierKAN (|sin/cos|<=1) and WaveletKAN (Mexican-hat, bounded) are the
bounded-amplitude basis family of the C^2-BV class. Unlike B-spline KAN
(sup|B|>=1 architectural constant blocks full contractivity), these bases
can reach gamma < 1 under the same soft-gamma recipe (E68 semantics,
per-activation-layer amplification ||h_l||inf / ||h_{l-1}||inf).

Closes the "bounded-amplitude basis = future work" item: if gamma<1 is
attained with accuracy >= ~98%, the Thm-4 depth-independent certificate
branch is instantiated on a non-B-spline basis.
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
from models.student_fourierkan import StudentFourierKAN
from models.student_waveletkan import StudentWaveletKAN

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data" / "processed"
OUT = ROOT / "results" / "student"
OUT_T = ROOT / "results" / "theory"

XR = (-3.0, 3.0)
N_GAMMA_SAMPLES = 600

# weight attributes to scale under the contraction projection
ATTRS = {"fourier": ["fourier_coeffs", "base_weight"],
         "wavelet": ["wavelet_coeffs", "base_weight"]}


def load_data():
    X = np.load(DATA / "features_X.npy").astype(np.float32)
    y = np.load(DATA / "features_y.npy").astype(np.int64)
    X = np.clip(X, XR[0], XR[1])
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=0.2, stratify=y, random_state=42)
    tr, va = train_test_split(tr, test_size=0.125, stratify=y[tr],
                              random_state=42)
    return (X[tr], y[tr], X[va], y[va], X[te], y[te])


def measure_gamma(model):
    model.eval()
    xs = np.random.default_rng(0).uniform(XR[0], XR[1],
                                          (N_GAMMA_SAMPLES, 28))
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


def project_contractive(model, target, attrs):
    for _ in range(5):
        g = measure_gamma(model)
        if max(g) <= target:
            return g
        with torch.no_grad():
            for l, layer in enumerate(model.kan_layers):
                if g[l] > target and g[l] > 0:
                    s = target / g[l]
                    for a in attrs:
                        getattr(layer, a).mul_(s)
    return measure_gamma(model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["fourier", "wavelet"], required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gamma-target", type=float, default=0.85)
    ap.add_argument("--gamma-lambda", type=float, default=30.0)
    ap.add_argument("--contractive", action="store_true")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    attrs = ATTRS[args.model]
    torch.manual_seed(42)
    np.random.seed(42)

    Xt, yt, Xv, yv, Xte, yte = load_data()
    Xt_t, yt_t = torch.from_numpy(Xt), torch.from_numpy(yt)
    Xv_t, yv_t = torch.from_numpy(Xv), torch.from_numpy(yv)

    if args.model == "fourier":
        m = StudentFourierKAN([28, 16, 4], n_harmonics=6, omega=0.4)
    else:
        m = StudentWaveletKAN([28, 16, 4], n_scales=8)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    rng = np.random.RandomState(42)
    best_val, best_sd, bad, n = -1.0, None, 0, len(Xt)

    for ep in range(args.epochs):
        m.train()
        perm = rng.permutation(n)
        for b in range(0, n, 128):
            idx = perm[b:b + 128]
            opt.zero_grad()
            logits = m(Xt_t[idx])
            loss = F.cross_entropy(logits, yt_t[idx])
            if args.contractive:
                with torch.no_grad():
                    g = measure_gamma(m)
                pen = sum(max(0.0, gi - args.gamma_target) ** 2 for gi in g)
                loss = loss + args.gamma_lambda * pen
            loss.backward()
            opt.step()
            if args.contractive and (b // 128) % 10 == 0:
                project_contractive(m, args.gamma_target, attrs)
        sched.step()
        m.eval()
        with torch.no_grad():
            va = float((m(Xv_t).argmax(1) == yv_t).float().mean())
        g = measure_gamma(m)
        if va > best_val:
            best_val, best_sd, bad = va, {k: v.clone()
                                          for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 20:
            break
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"ep {ep}: val {va:.4f} gamma {[round(x,3) for x in g]}",
                  flush=True)

    m.load_state_dict(best_sd)
    m.eval()
    with torch.no_grad():
        acc = float((m(torch.from_numpy(Xte)).argmax(1) ==
                     torch.from_numpy(yte)).float().mean())
    g_final = measure_gamma(m)
    out_name = args.out or f"kan_{args.model}_contractive.pt"
    torch.save(best_sd, OUT / out_name)
    res = {"date": "2026-08-04", "model": args.model,
           "contractive": args.contractive, "gamma_target": args.gamma_target,
           "gamma_measured": [round(x, 4) for x in g_final],
           "test_acc": float(acc), "best_val": float(best_val),
           "condition3_satisfied": bool(max(g_final) < 1.0),
           "verdict": "PASS" if (max(g_final) < 1.0 and acc >= 0.98)
                      else "CHECK"}
    with open(OUT_T / f"{args.model}_contractive.json", "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2))
    print(f"Saved: {OUT / out_name}")


if __name__ == "__main__":
    main()
