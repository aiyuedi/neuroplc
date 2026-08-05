#!/usr/bin/env python3
"""v5 sound certificate, corrected-propagation口径 (2026-08-06).

Same methodology as verify_perfunc_v2.py (the paper's 2.34x sound,
dy = sum eps, per-edge L1), recomputed on:
  - kan_contractive.pt  (the checkpoint the paper's 0.288/2.34x reports)
  - kan_contractive_v3.pt (post-hoc output projection)
  - kan_contractive_v5.pt (pervasive contractivity, gamma=[0.95,0.95])
The v5 weights are globally scaled (L0 x0.842, L1 x0.884) plus a
constrained L1 fine-tune, so per-edge M2 and L1 shrink accordingly.
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
H2_8 = (6.0 / 14) ** 2 / 8.0
MEASURED = 0.0527     # Tier-4 f32, soft2L family


def profile_lipschitz(model, n_pts=40001):
    """Per-edge (M2, L) of the full activation phi, float64."""
    xs = np.linspace(-3.0, 3.0, n_pts, dtype=np.float64)
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


def run(name, path, arch):
    ckpt = torch.load(os.path.join(BASE, path), map_location="cpu",
                      weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN(arch)
    m.load_state_dict(sd, strict=False)
    m.eval()
    pl = profile_lipschitz(m)
    n0 = arch[0] * arch[1]
    n1 = arch[1] * arch[2]
    m2_0 = np.array([p[0] for p in pl[:n0]]).reshape(arch[1], arch[0])
    L1 = np.array([p[1] for p in pl[n0:n0 + n1]]).reshape(arch[2], arch[1])
    m2_1 = np.array([p[0] for p in pl[n0:n0 + n1]]).reshape(arch[2], arch[1])
    eps0 = m2_0 * H2_8
    eps1 = m2_1 * H2_8
    dy = eps0.sum(axis=1)
    d = float(np.max(L1 @ dy + eps1.sum(axis=1)))
    print(f"{name:6s} sound={d:.4f} (safety {MARGIN_HALF/d:.2f}x)  "
          f"L1 row-sum max={L1.sum(axis=1).max():.4f}  "
          f"M2_0 max={m2_0.max():.4f} M2_1 max={m2_1.max():.4f}  "
          f"covers f32 {MEASURED:.4f}: {d >= MEASURED}")
    return {"sound_bound": d, "safety": MARGIN_HALF / d,
            "covers_f32_measured": bool(d >= MEASURED),
            "measured": MEASURED, "certificate": bool(d < MARGIN_HALF),
            "L1_rowsum_max": float(L1.sum(axis=1).max()),
            "M2_0_max": float(m2_0.max()), "M2_1_max": float(m2_1.max())}


def main():
    out = {
        "base": run("base", "results/student/kan_contractive.pt", [28, 16, 4]),
        "v3": run("v3", "results/student/kan_contractive_v3.pt", [28, 16, 4]),
        "v5": run("v5", "results/student/kan_contractive_v5.pt", [28, 16, 4]),
    }
    out["date"] = "2026-08-06"
    out["note"] = ("corrected propagation (dy=sum eps, per-edge L1), "
                   "full-phi float64 M2; v5 = gamma=[0.95,0.95] pervasive")
    with open(os.path.join(BASE, "results", "theory", "v5_sound2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/v5_sound2.json")


if __name__ == "__main__":
    main()
