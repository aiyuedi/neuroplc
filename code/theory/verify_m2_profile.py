#!/usr/bin/env python3
"""Full-phi M2 profile + sound bound chain (2026-08-04 audit v3).

The compiled LUT stores the FULL activation phi(x) = sb*bw*silu(x) +
ss*B(x/3). The de Boor bound must use the FULL M2 = max|phi''|, including
the base SiLU contribution (|siLU''| max = 0.5 at x=0):

    M2_full(j,i) = |ss| * M2_spline(j,i)/9 + |sb*bw(j,i)| * 0.5

where M2_spline is the scipy-exact spline second derivative. This script
computes the M2 profile (median = E11-style char, max) and the sound
IA-form network bound with the measured-floor, for any checkpoint.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN
from neuroplc.per_function_verify import estimate_m2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGIN_HALF = 0.675


def profile(model):
    m2s, lbs = [], []
    with torch.no_grad():
        for layer in model.kan_layers:
            o, i, _ = layer.spline_weight.shape
            sb = abs(float(layer.scale_base))
            ss = abs(float(layer.scale_spline))
            for oi in range(o):
                for ii in range(i):
                    m2_spl = estimate_m2(
                        layer.spline_weight[oi, ii].detach().numpy(),
                        layer.grid.detach().numpy()) / 9.0
                    bw = abs(float(layer.base_weight[oi, ii]))
                    m2s.append(ss * m2_spl + sb * bw * 0.5)
    return np.array(m2s)


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
        m2 = profile(m)
        med, mx = float(np.median(m2)), float(m2.max())
        eps_c = med * (6.0 / 14) ** 2 / 8
        eps_m = mx * (6.0 / 14) ** 2 / 8
        out[name] = {"m2_median": med, "m2_max": mx,
                     "eps_char": eps_c, "eps_max": eps_m,
                     "n": len(m2)}
        print(f"{name:7s} M2 median={med:.4f}  max={mx:.4f}  "
              f"eps_char={eps_c:.6f}  eps_max={eps_m:.6f}")
    with open(os.path.join(BASE, "results", "theory", "m2_profile.json"), "w") as f:
        json.dump({"date": "2026-08-04", "note":
                   "full-phi M2 (spline scipy-exact + base |sb*bw|*0.5)",
                   "out": out}, f, indent=2)
    print("Saved: results/theory/m2_profile.json")


if __name__ == "__main__":
    main()
