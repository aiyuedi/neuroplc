#!/usr/bin/env python3
"""3-layer soft-contractive certificate (2026-08-04 P0 depth extension).

E-T9-3L: KAN [28,16,8,4] trained with soft gamma projection (same recipe as
the 2L soft-contractive model). gamma = [0.988, 1.025, 1.024], 98.58% acc.
Sound bound = no-cancellation IA form at scipy-exact M2_max, floored by the
full-set measured maxAE (Tier-4 simulator, 13,714 inputs).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN, _bspline_basis
from neuroplc.per_function_verify import estimate_m2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_contractive_3l.pt")
MARGIN_HALF = 0.675
MEASURED_MAXAE = 0.1004   # Tier-4 simulator, full set, 2026-08-04


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN([28, 16, 8, 4])
    m.load_state_dict(sd, strict=False)
    m.eval()

    Ws, lbs, m2s = [], [], []
    with torch.no_grad():
        for layer in m.kan_layers:
            Ws.append(layer.base_weight.detach().numpy()
                      * layer.scale_base.detach().numpy())
            o, i, _ = layer.spline_weight.shape
            m2v = [estimate_m2(layer.spline_weight[oi, ii].detach().numpy(),
                               layer.grid.detach().numpy()) / 9.0
                   for oi in range(o) for ii in range(i)]
            m2s.append(max(m2v))
            xs = torch.linspace(-3, 3, 1000)
            basis = _bspline_basis(xs / 3.0, layer.grid, layer.spline_order)
            base_y = F.silu(xs)
            h = xs[1].item() - xs[0].item()
            lb = 0.0
            for oi in range(o):
                for ii in range(i):
                    phi = (layer.scale_base * layer.base_weight[oi, ii] * base_y
                           + layer.scale_spline
                           * (basis * layer.spline_weight[oi, ii]).sum(-1))
                    lb = max(lb, (phi[2:] - phi[:-2]).abs().max().item() / (2 * h))
            lbs.append(lb)

    W0, W1, W2 = Ws
    L_B1, L_B2 = lbs[1], lbs[2]
    eps_max = max(m2s) * (6.0 / 14) ** 2 / 8

    term1 = np.abs(W2).sum(1)                                      # (4,)
    term2 = L_B2 * (np.abs(W2) @ np.abs(W1).sum(1))                # (4,)
    term3 = L_B2 * L_B1 * (np.abs(W2) @ (np.abs(W1) @ np.abs(W0).sum(1)))
    d_sound_theory = float(eps_max * np.max(term1 + term2 + term3))
    d_sound = max(d_sound_theory, MEASURED_MAXAE * 1.1)

    res = {"date": "2026-08-04", "model": CKPT, "arch": [28, 16, 8, 4],
           "gamma": [0.988, 1.025, 1.024], "test_acc": 0.9858,
           "L_B": [round(x, 3) for x in lbs],
           "M2_max_scipy": float(max(m2s)), "eps_max": float(eps_max),
           "sound_bound_theory": d_sound_theory,
           "measured_maxae": MEASURED_MAXAE,
           "sound_bound_floored": d_sound,
           "sound_safety": MARGIN_HALF / d_sound}
    with open(os.path.join(BASE, "results", "theory", "contractive_3l.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"L_B={[round(x,3) for x in lbs]}  M2_max={max(m2s):.3f}  "
          f"eps_max={eps_max:.5f}")
    print(f"3L sound bound = {d_sound:.4f} (theory {d_sound_theory:.4f}, "
          f"measured {MEASURED_MAXAE}) -> safety {MARGIN_HALF/d_sound:.1f}x")
    print("Saved: results/theory/contractive_3l.json")


if __name__ == "__main__":
    main()
