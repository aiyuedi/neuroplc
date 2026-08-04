#!/usr/bin/env python3
"""Adaptive-allocation x DA/IA bounds on the released checkpoint (2026-08-04 P0).

E15 (greedy adaptive allocation, same S7-1200 budget B=30,720 B as uniform
N=15) reduces the worst-case per-function LUT error from 0.00406 (uniform)
to 0.00115 (71.6% reduction). Propagating the adaptive worst epsilon through
the E68 DA/IA forms:

  d_DA  = eps * (L_B1 * m_row + s1)          = 0.6586  at eps=0.00406
  d_IA  = eps * (L_B1 * t1 + t2)             = 1.3784  at eps=0.00406
  -> with eps_adapt = 0.00115, both scale linearly (eps is a common factor).
"""
import json
import os

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

L_B1 = 2.05
M_ROW = 77.7102279663086   # verify_da_bounds_recomputed (base-folded)
S1 = 2.9155447483062744
T1 = 160.3999786376953
T2 = 10.693679809570312
MARGIN_HALF = 0.675

EPS_UNIFORM = 0.00406    # M2_char=0.177, N=15
EPS_ADAPT = 0.00115      # E15: greedy adaptive, B=30,720 B, worst-case (71.6% reduction)
EPS_ADAPT_S1500 = 0.00109  # E15: S7-1500 budget B=102,400 B (73.1% reduction, N=50-equiv)


def da_ia(eps):
    d_da = eps * (L_B1 * M_ROW + S1)
    d_ia = eps * (L_B1 * T1 + T2)
    return d_da, d_ia


def main():
    rows = {}
    for name, eps in [("uniform_N15", EPS_UNIFORM), ("adaptive_N15", EPS_ADAPT),
                      ("adaptive_S1500", EPS_ADAPT_S1500)]:
        d_da, d_ia = da_ia(eps)
        rows[name] = {
            "eps": eps,
            "DA": d_da,
            "DA_safety": MARGIN_HALF / d_da,
            "IA": d_ia,
            "IA_safety": MARGIN_HALF / d_ia,
            "IA_is_certificate": d_ia < MARGIN_HALF,
        }
        print(f"{name:16s} eps={eps:.5f}  DA={d_da:.4f} (safety {rows[name]['DA_safety']:.2f}x)  "
              f"IA={d_ia:.4f} (safety {rows[name]['IA_safety']:.2f}x, cert={rows[name]['IA_is_certificate']})")

    with open(os.path.join(BASE, "results", "theory", "adaptive_da_recomputed.json"), "w") as f:
        json.dump({"date": "2026-08-04", "margin_half": MARGIN_HALF,
                   "constants": {"L_B1": L_B1, "m_row": M_ROW, "s1": S1,
                                 "t1": T1, "t2": T2}, "rows": rows}, f, indent=2)
    print("\nSaved: results/theory/adaptive_da_recomputed.json")


if __name__ == "__main__":
    main()
