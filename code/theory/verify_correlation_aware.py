#!/usr/bin/env python3
"""Recompute the correlation-aware aggregate bound (Eq. eq:correlation_aware)
on the released checkpoint (2026-08-04 audit, R1).

    d_agg = eps * (d_out * L_net + L_B * sum_i max_j |W1_k*,j W0_j,i|)

with L_net = max_k sum_j |W1_kj| * rowsum0_j  (IA network aggregation, t1),
     k*    = argmax_k sum_i |sum_j W1_kj W0_ji|.
The paper's ~6.4 used the stale L_net = 19.6 (pre-2026-08-03 constants).
"""
import json
import os

import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")

EPS_LUT = 0.00406   # M2_char=0.177, h=6/14 @ N=15
L_B = 2.05          # layer-1 B-spline Lipschitz (E68)
D_OUT = 4
MAXAE = 3.65        # empirical MaxAE (paper)


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"]
    W0 = sd["kan_layers.0.base_weight"] * sd["kan_layers.0.scale_base"]
    W1 = sd["kan_layers.1.base_weight"] * sd["kan_layers.1.scale_base"]

    l_net = (W1.abs() @ W0.abs().sum(1)).max().item()   # IA aggregation t1
    cross = (W1 @ W0).abs().sum(1)                       # (4,) per-output sum
    kstar = int(cross.argmax())
    P = (W1[kstar].abs().unsqueeze(1) * W0.abs()).max(dim=0).values.sum().item()

    d_agg = EPS_LUT * (D_OUT * l_net + L_B * P)
    print(f"L_net(IA) = {l_net:.3f}   (paper TODO said stale 19.6)")
    print(f"k* = {kstar}   correlation penalty P = {P:.3f}")
    print(f"correlation-aware aggregate bound = {d_agg:.3f}   (paper: ~6.4)")
    print(f"safety vs empirical MaxAE {MAXAE} = {d_agg / MAXAE:.2f}x  (paper: 1.75x)")

    with open(os.path.join(BASE, "results", "theory", "correlation_aware_recomputed.json"), "w") as f:
        json.dump({"date": "2026-08-04", "L_net_IA": l_net, "kstar": kstar,
                   "P": P, "d_agg": d_agg, "maxae": MAXAE,
                   "safety_factor": d_agg / MAXAE}, f, indent=2)
    print("\nSaved: results/theory/correlation_aware_recomputed.json")


if __name__ == "__main__":
    main()
