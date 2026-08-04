#!/usr/bin/env python3
"""Corrected two-layer error propagation (2026-08-04 P0, δ_fp32 pursuit).

Structural correction: the LUT error eps_{j,i} is the approximation error
of phi_{j,i} itself (weights already inside phi). Propagation:

  layer 0 (y_j = sum_i phi_{j,i}(x_i), input exact in-domain):
      dy_j  = sum_i eps0_{j,i}                    (direct sum, no |W0|)
  layer 1 (z_k = sum_j phi_{k,j}(y_j)):
      dz_k  = sum_j [ L1_{k,j} * dy_j + eps1_{k,j} ]
              (per-edge Lipschitz L1_{k,j}, not a global L_B)
  sound   = max_k dz_k,  eps = M2_full * h^2/8  (full-phi float64 M2)

This yields the correct conservative envelope; if it covers the float32
deployment measurement (Tier-4 simulator), the certificate is purely
theoretical (no empirical floor), and IEEE-754 accumulation is the
residual delta_fp32 quantified separately.
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
MEASURED = {"main": 0.5175, "soft2L": 0.0527, "soft3L": 0.1004}  # Tier-4 f32


def profile_lipschitz(model, n_pts=40001):
    """Per-edge (M2, L) of the full activation phi (base + spline), float64."""
    xs = np.linspace(-3.0, 3.0, n_pts, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
    h = xs[1] - xs[0]
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
                    out.append((float(np.abs(d2).max()),   # M2
                                float(np.abs(d1).max())))  # L
    return out


def main():
    out = {}
    for name, ck, arch in [
        ("main", "results/student/kan_kd_vrmKD_best.pt", [28, 16, 4]),
        ("soft2L", "results/student/kan_contractive.pt", [28, 16, 4]),
        ("soft3L", "results/student/kan_contractive_3l.pt", [28, 16, 8, 4]),
    ]:
        ckpt = torch.load(os.path.join(BASE, ck), map_location="cpu",
                          weights_only=False)
        sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
        m = StudentKAN(arch)
        m.load_state_dict(sd, strict=False)
        m.eval()
        pl = profile_lipschitz(m)
        n0 = arch[0] * arch[1]          # layer-0 edges
        n1 = arch[1] * arch[2]          # layer-1 edges
        m2_0 = np.array([p[0] for p in pl[:n0]]).reshape(arch[1], arch[0])
        L1 = np.array([p[1] for p in pl[n0:n0 + n1]]).reshape(arch[2], arch[1])
        m2_1 = np.array([p[0] for p in pl[n0:n0 + n1]]).reshape(arch[2], arch[1])
        eps0 = m2_0 * H2_8
        eps1 = m2_1 * H2_8
        dy = eps0.sum(axis=1)                           # (h1,) per hidden node
        if len(arch) == 4:
            n2 = arch[2] * arch[3]
            L2 = np.array([p[1] for p in pl[n0 + n1:]]).reshape(arch[3], arch[2])
            m2_2 = np.array([p[0] for p in pl[n0 + n1:]]).reshape(arch[3], arch[2])
            eps2 = m2_2 * H2_8
            du = L1 @ dy + eps1.sum(axis=1)             # (h2,)
            d = float(np.max(L2 @ du + eps2.sum(axis=1)))
        else:
            d = float(np.max(L1 @ dy + eps1.sum(axis=1)))   # (c,)
        meas = MEASURED[name]
        covers = bool(d >= meas)
        out[name] = {"sound_bound": d, "safety": MARGIN_HALF / d,
                     "covers_f32_measured": covers,
                     "measured": meas,
                     "certificate": bool(d < MARGIN_HALF)}
        print(f"{name:7s} corrected sound={d:.4f} (safety {MARGIN_HALF/d:.2f}x)  "
              f"covers f32 measured {meas:.4f}: {'YES' if covers else 'NO'}  "
              f"cert={'YES' if d < MARGIN_HALF else 'no'}")
    with open(os.path.join(BASE, "results", "theory", "perfunc_v2.json"), "w") as f:
        json.dump({"date": "2026-08-04", "note":
                   "corrected propagation (dy=sum eps, per-edge L1)",
                   "out": out}, f, indent=2)
    print("Saved: results/theory/perfunc_v2.json")


if __name__ == "__main__":
    main()
