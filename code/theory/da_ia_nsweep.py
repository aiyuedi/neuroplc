#!/usr/bin/env python3
"""N-sweep of DA/IA logit bounds with measured constants (2026-08-03).

Reuses the released checkpoint and the constants/measurements from
verify_da_bounds_recomputed.py:
  L_B1 = 2.05   (layer-1 activation Lipschitz, E68)
  M2_char = 0.177
  eps_char(N) = M2_char * (6/(N-1))^2 / 8
  IA(N) = eps_char(N) * (L_B1 * t1 + t2)
  DA(N) = eps_char(N) * (L_B1 * m_row + s1)
with folded weights (base_weight*scale_base + spline-part mean).
Grid: N = [8 10 12 15 18 20] (same as the .m figure scripts).
"""
import os

import numpy as np
import torch

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")

L_B1 = 2.05      # layer-1 activation Lipschitz (E68)
M2_char = 0.177   # char-activation second derivative constant
N_GRID = [8, 10, 12, 15, 18, 20]


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"]
    W0 = sd["kan_layers.0.base_weight"] * sd["kan_layers.0.scale_base"]
    S0 = (sd["kan_layers.0.spline_weight"]
          * sd["kan_layers.0.scale_spline"]).mean(-1)
    W1 = sd["kan_layers.1.base_weight"] * sd["kan_layers.1.scale_base"]
    S1 = (sd["kan_layers.1.spline_weight"]
          * sd["kan_layers.1.scale_spline"]).mean(-1)

    # Headline values use base-folded weights (matches
    # results/theory/da_bounds_recomputed.json "base-folded" entry:
    # DA(15)=0.659, IA(15)=1.378); the spline-mean variant is printed for
    # comparison but not used for the figure arrays.
    A0 = W0         # folded (base*scale) layer-0 weights
    A1 = W1         # folded (base*scale) layer-1 weights

    row0 = A0.abs().sum(1)                 # (16,) per-input rowsum
    t1 = (A1.abs() @ row0).max().item()    # max_k sum_j |W1_kj| rowsum0_j
    t2 = A1.abs().sum(1).max().item()      # max_k sum_j |W1_kj|
    m_row = (A1 @ A0).abs().sum(1).max().item()  # max_k sum_i |(W1 W0)_ki|
    s1 = A1.sum(1).abs().max().item()      # max_k |sum_j W1_kj|

    da = []
    ia = []
    for N in N_GRID:
        eps = M2_char * (6.0 / (N - 1)) ** 2 / 8.0
        da.append(eps * (L_B1 * m_row + s1))
        ia.append(eps * (L_B1 * t1 + t2))

    print("L_B1=%.3f M2_char=%.3f" % (L_B1, M2_char))
    print("t1=%.4f  t2=%.4f  m_row=%.4f  s1=%.4f" % (t1, t2, m_row, s1))
    print("N     :", N_GRID)
    print("DA(N) :", " ".join("%.4f" % v for v in da))
    print("IA(N) :", " ".join("%.4f" % v for v in ia))
    print("IA/DA :", " ".join("%.3f" % (i / d) for i, d in zip(ia, da)))
    print("eps   :", " ".join("%.5f" % (M2_char * (6.0 / (N - 1)) ** 2 / 8.0)
                              for N in N_GRID))


if __name__ == "__main__":
    main()
