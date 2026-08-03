#!/usr/bin/env python3
"""Verify the IA/DA logit bounds with the CURRENT checkpoint constants.

2026-08-03 audit follow-up: the paper's Thm-1 instantiations
(dIA=0.172, dDA=0.079, safety 3.9x/8.5x) used stale constants
(L_B=0.65, ||W||=0.28/0.32) that no longer reproduce from
results/student/kan_kd_vrmKD_best.pt (training extended the B-spline
grids and grew scale_base). This script recomputes the bounds with the
measured constants (L_B = 2.21 layer-0 / 2.05 layer-1, folded weights)
and dumps results/theory/da_bounds_recomputed.json.
"""
import json
import os
import sys

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")
OUT = os.path.join(BASE, "results", "theory", "da_bounds_recomputed.json")

EPS_LUT_CHAR = 0.00406      # M2_char=0.177, h=6/14 @ N=15
EPS_LUT_MAX = 0.03641       # M2_max=1.586, h=6/14 @ N=15
MARGIN_HALF = 0.675         # min inter-class margin 1.35 / 2
L_B = (2.21, 2.05)          # measured max |phi'| layer-0 / layer-1 (E68)


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"]
    W0 = sd["kan_layers.0.base_weight"] * sd["kan_layers.0.scale_base"]
    S0 = (sd["kan_layers.0.spline_weight"]
          * sd["kan_layers.0.scale_spline"]).mean(-1)
    W1 = sd["kan_layers.1.base_weight"] * sd["kan_layers.1.scale_base"]
    S1 = (sd["kan_layers.1.spline_weight"]
          * sd["kan_layers.1.scale_spline"]).mean(-1)

    results = {}
    for wname, (A0, A1) in [("base-folded", (W0, W1)),
                            ("base+spline-mean", (W0 + S0, W1 + S1))]:
        # Correct propagation: the layer-0 LUT error eps0 sits at the
        # activation output and is carried to the logits by W1 followed by
        # the layer-1 activation (Lipschitz LB1); the layer-1 LUT error
        # eps1 enters directly at the logit combination.
        row0 = A0.abs().sum(1)                # (16,) per-input rowsum
        t1 = (A1.abs() @ row0).max()          # max_k sum_j |W1_kj| rowsum0_j
        t2 = A1.abs().sum(1).max()            # max_k sum_j |W1_kj|
        m_row = (A1 @ A0).abs().sum(1).max()  # max_k sum_i |(W1 W0)_ki| (DA)
        s1 = A1.sum(1).abs().max()            # max_k |sum_j W1_kj| (DA)
        for eps_name, eps in [("eps_char=0.00406", EPS_LUT_CHAR),
                              ("eps_max=0.03641", EPS_LUT_MAX),
                              ("eps_mean=0.01739", 0.01739),
                              ("eps_p95=0.03011", 0.03011)]:
            d_ia = eps * (L_B[1] * t1 + t2)
            d_da = eps * (L_B[1] * m_row + s1)
            results[f"{wname}/{eps_name}"] = {
                "t1": float(t1), "t2": float(t2),
                "m_row": float(m_row), "s1": float(s1),
                "d_IA": float(d_ia), "d_DA": float(d_da),
                "IA_safe_vs_margin_half": bool(d_ia < MARGIN_HALF),
                "DA_safe_vs_margin_half": bool(d_da < MARGIN_HALF),
                "IA_safety_factor": float(MARGIN_HALF / d_ia) if d_ia > 0 else None,
                "DA_safety_factor": float(MARGIN_HALF / d_da) if d_da > 0 else None,
            }

    headline = results["base-folded/eps_char=0.00406"]
    print(json.dumps(headline, indent=2))
    print(f"\nPaper claim was: IA 0.172 (3.9x) / DA 0.079 (8.5x).")
    print(f"Recomputed (eps_char): IA {headline['d_IA']:.3f} "
          f"(safe={headline['IA_safe_vs_margin_half']}) / "
          f"DA {headline['d_DA']:.3f} (safe={headline['DA_safe_vs_margin_half']}, "
          f"factor {headline['DA_safety_factor']:.1f}x)")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({"date": "2026-08-03", "margin_half": MARGIN_HALF,
                   "L_B_measured": list(L_B), "results": results},
                  f, indent=2)
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
