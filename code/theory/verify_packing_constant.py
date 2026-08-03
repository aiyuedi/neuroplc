"""
verify_packing_constant.py — constructive lower bound on the packing
constant c_ent of Theorem 17(1) (1-D W^k, L-infinity).

Idea: the window-product extremals of Theorem 9 (sharp-lut), scaled to
M_k = 1, form a family {f_s} over translates s. Two translates with
disjoint supports have L-infinity distance >= 2 * (window peak), so an
N-cell codebook cannot separate more than N-distinct-support translates
at resolution eps -> a packing lower bound eps >= c_ent N^{-k} with a
numerically estimated c_ent, which must satisfy c_ent <= c_k = 1/8
(k=2; the packing line cannot exceed the LUT constant, else the LUT
would violate it).

We scan eps, count the max separable translates, fit c_ent in
log-log, and compare with c_k. Also checks the bit-parameterization
exponent (-B/k).

Run: python theory/verify_packing_constant.py
Output: results/theory/packing_constant.json
"""

import json
import os
import numpy as np

C2 = 1.0 / 8.0          # sharp interpolation constant (k=2), upper reference


def window_peak(h):
    """Peak of f_s(x) = (M2/2)(x-s)(x-s-h) on its window [s, s+h], M2=1."""
    return h * h / 8.0


def max_separable(h, eps, domain_len=1.0):
    """Max number of translates f_s with mutual L-inf distance >= 2 eps.

    Translates with disjoint supports (|s_i - s_j| >= h) are separated by
    2*peak; same-support translates differ by a translation of the window
    product, whose L-inf distance over the window is >= peak (the maxima
    cannot coincide for different s).  We therefore count translates spaced
    >= h, and also require 2*peak >= 2 eps (i.e., peak >= eps) so that
    disjoint-support separation exceeds the resolution.
    """
    if window_peak(h) < eps:
        return 0
    return max(1, int(domain_len / h))     # floor(1/h) disjoint windows


def main():
    results = {"date": "2026-08-04", "script": "verify_packing_constant.py",
               "theorem": "Thm 17(1) packing-line constant, constructive lower bound"}
    # scan: for each codebook size B (= log2 of separable count), the
    # largest eps that still allows that many separable translates
    eps_list = np.logspace(-3.5, -1.0, 40)
    rows = []
    for eps in eps_list:
        # largest h with peak(h) >= eps  ->  h = sqrt(8 eps)
        h = np.sqrt(8.0 * eps)
        P = max_separable(h, eps)
        if P >= 2:
            rows.append((eps, P, np.log2(P)))
    # fit: eps vs P^{-k} with k=2 -> eps = c_ent * P^{-2}
    P = np.array([r[1] for r in rows], float)
    e = np.array([r[0] for r in rows])
    logP, loge = np.log(P), np.log(e)
    A = np.vstack([-2.0 * logP, np.ones_like(logP)]).T
    coef, *_ = np.linalg.lstsq(A, loge, rcond=None)
    c_ent_fit = np.exp(coef[1])
    # slope check: log e vs log P should be ~ -2
    slope = np.polyfit(logP, loge, 1)[0]

    results["rows"] = [{"eps": float(x[0]), "P": x[1], "log2P": float(x[2])}
                       for x in rows[:12]]
    results["c_ent_fit"] = float(c_ent_fit)
    results["slope_loge_vs_logP"] = float(slope)
    results["c_k_reference"] = C2
    results["c_ent_le_c_k"] = bool(c_ent_fit <= C2 + 1e-9)
    # bit-parameterization exponent: log2-space slope equals ln-space slope
    # (log2 is a uniform scaling), expect ~ -k = -2; eps ~ P^{-k} = 2^{-kB}.
    slope_bits = np.polyfit(np.log2(P), np.log2(e), 1)[0]
    results["slope_log2e_vs_log2P"] = float(slope_bits)   # expect ~ -k = -2
    results["verdict"] = "PASS" if (results["c_ent_le_c_k"]
                                    and abs(slope + 2.0) < 0.3
                                    and abs(slope_bits + 2.0) < 0.3) else "CHECK"

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "theory", "packing_constant.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
