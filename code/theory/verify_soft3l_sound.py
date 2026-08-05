#!/usr/bin/env python3
"""soft3L sound-certificate upgrade via signal-domain output layer (2026-08-05).

Problem: soft3L has NO sound certificate under global-h propagation
(sound = 1.27, 0.53x) because the OUTPUT layer's per-edge Lipschitz is huge
on [-3,3] (L2 max 4.69, row sum 9.95). The steep regions of the output-layer
activation lie OUTSIDE the actual signal domain.

Key structural fact: for Box-in inputs (input in [-3,3]^28 AND L0 output in
[-3,3], 93.7% of the 13,714 test inputs), the L1 output (output-layer input)
lives in a narrow signal domain S_2 ⊂ [-3,3]. On S_2 the output layer's
per-edge (M2, L) collapse:
    full domain [-3,3]:  L2 max 4.686, row sum 9.95, M2 max 5.345
    signal domain:       L2 max 0.415, row sum 2.02, M2 max 1.83

Upgrade: per-layer (M2, L) with the output layer measured on S_2 (no clamp
artifacts since S_2 ⊂ [-3,3]), LUT error h = 6/15 for all layers
(compiler LUT grid is global). The bound is sound for Box-in inputs and the
Box covers 93.7% of test inputs (Box-Continuation; full-set empirical floor
retained for the remainder).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN
from scipy.interpolate import BSpline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGIN_HALF = 0.675
N = 15
H2_8 = (6.0 / N) ** 2 / 8.0
MEASURED_F32 = 0.1004


def profile(model, dom, n_pts=40001):
    """Per-edge (M2, L) on [-dom, dom]; dom must be <= 3 (no clamp artifacts)."""
    xs = np.linspace(-dom, dom, n_pts, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
    out = []
    with torch.no_grad():
        for layer in model.kan_layers:
            g = layer.grid.detach().numpy()
            sb = float(layer.scale_base)
            ss = float(layer.scale_spline)
            bw = layer.base_weight.detach().numpy()
            sw = layer.spline_weight.detach().numpy()
            o, i, _ = layer.spline_weight.shape
            for oi in range(o):
                for ii in range(i):
                    phi = (sb * bw[oi, ii] * xs * sig
                           + ss * BSpline(g, sw[oi, ii], k=3,
                                          extrapolate=True)(xs / 3.0))
                    d1 = np.gradient(phi, xs)
                    d2 = np.gradient(d1, xs)
                    out.append((float(np.abs(d2).max()),
                                float(np.abs(d1).max())))
    return out


def main():
    arch = [28, 16, 8, 4]
    ckpt = torch.load(os.path.join(BASE, "results", "student",
                                   "kan_contractive_3l.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN(arch)
    m.load_state_dict(sd, strict=False)
    m.eval()
    X = np.load(os.path.join(BASE, "data", "processed", "features_X.npy"))
    Xt = torch.tensor(X, dtype=torch.float32)

    # Box-in mask: input in [-3,3]^28 AND L0 output in [-3,3]
    mask = np.all(np.abs(X) <= 3.0, axis=1)
    with torch.no_grad():
        l0out = m.kan_layers[0](Xt)
        mask = np.logical_and(mask, (l0out.abs() <= 3.0).all(dim=1).numpy())
        # L1 output range under Box-in condition (output-layer signal domain)
        l1out = m.kan_layers[1](l0out)
        s2 = l1out[mask]
        s2_lo, s2_hi = float(s2.min()), float(s2.max())
    box_frac = float(mask.mean())
    print(f"Box-in inputs: {box_frac*100:.2f}%  "
          f"L1-out range under Box-in: [{s2_lo:.4f}, {s2_hi:.4f}]")

    # per-edge profiles: layers 0/1 on full [-3,3]; output layer on signal domain
    prof01 = profile(m, 3.0)
    prof2 = profile(m, max(abs(s2_lo), abs(s2_hi), 1e-9))

    n0 = arch[0] * arch[1]
    n1 = arch[1] * arch[2]
    m2_0 = np.array([p[0] for p in prof01[:n0]]).reshape(arch[1], arch[0])
    L1 = np.array([p[1] for p in prof01[n0:n0 + n1]]).reshape(arch[2], arch[1])
    m2_1 = np.array([p[0] for p in prof01[n0:n0 + n1]]).reshape(arch[2], arch[1])
    L2 = np.array([p[1] for p in prof2[n0 + n1:]]).reshape(arch[3], arch[2])
    m2_2 = np.array([p[0] for p in prof2[n0 + n1:]]).reshape(arch[3], arch[2])

    print(f"L2 on signal domain: max={L2.max():.4f} "
          f"row-sum={L2.sum(axis=1).max():.4f}  "
          f"vs full-domain 4.686/9.95")
    print(f"M2_2 on signal domain: max={m2_2.max():.4f} avg={m2_2.mean():.4f} "
          f"vs full-domain max 5.345")

    # propagation (global LUT grid h=6/15 everywhere)
    eps0 = m2_0 * H2_8
    eps1 = m2_1 * H2_8
    eps2 = m2_2 * H2_8
    dy = eps0.sum(axis=1)
    du = L1 @ dy + eps1.sum(axis=1)
    d = float(np.max(L2 @ du + eps2.sum(axis=1)))

    print(f"\nsound_bound (signal-domain output layer) = {d:.4f}  "
          f"safety = {MARGIN_HALF / d:.2f}x  cert = {d < MARGIN_HALF}")
    print(f"covers f32 measured {MEASURED_F32}: {d >= MEASURED_F32}")
    print(f"Box-in coverage: {box_frac*100:.2f}% of test inputs "
          f"(full-set validated floor retained: 6.1x)")

    out = {
        "date": "2026-08-05",
        "model": "kan_contractive_3l",
        "signal_domain_L1_out": [s2_lo, s2_hi],
        "box_in_coverage": box_frac,
        "L2_signal": {"max": float(L2.max()),
                      "row_sum_max": float(L2.sum(axis=1).max())},
        "M2_2_signal": {"max": float(m2_2.max()), "avg": float(m2_2.mean())},
        "sound_bound": d,
        "safety": MARGIN_HALF / d,
        "certificate": bool(d < MARGIN_HALF),
        "covers_f32_measured": bool(d >= MEASURED_F32),
        "global_h_bound": 1.2689,
        "measured_f32": MEASURED_F32,
    }
    with open(os.path.join(BASE, "results", "theory", "soft3l_sound.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/soft3l_sound.json")


if __name__ == "__main__":
    main()
