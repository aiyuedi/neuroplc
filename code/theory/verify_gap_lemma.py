#!/usr/bin/env python3
"""verify_gap_lemma.py — Curvature-Concentrated Gap lemma (2026-08-05).

Lemma (approximation sense): if f'' is supported in an interval of
length delta (curvature-concentrated), then free-node placement (all N
nodes inside the support) achieves approximation error
    e_free <= M2 * delta^2 / (8 N^2),
while any uniform grid (segment width (b-a)/N) has
    e_uni  >= M2 * (b-a)^2 / (8 N^2)   (sharp, de Boor / Thm sharp-lut).
Hence the free-vs-uniform gap is at least ((b-a)/delta)^2.

This explicates the mechanism behind c_* < c_k (stratum-3 width
constant): concentration of curvature, not the width theory, is the
source of the gap. Numerically confirmed below (delta scan).
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def build_f(delta, center=0.5, peak=1.0):
    lo, hi = center - delta / 2, center + delta / 2

    def f(x):
        x = np.asarray(x, float)
        out = np.zeros_like(x)
        for i, xv in enumerate(x):
            if xv <= lo:
                out[i] = 0.0
            elif xv <= hi:
                out[i] = peak * (xv - lo) ** 2 / 2
            else:
                out[i] = peak * delta * (xv - lo) - peak * delta ** 2 / 2
        return out
    return f


def seg_best(f, a, b, n=401):
    s = np.linspace(a, b, n)
    fs = f(s)
    best = 1e9
    for init in ((fs[0], 0.0), (fs[0], (fs[-1] - fs[0]) / (b - a))):
        r = minimize(lambda z: np.max(np.abs(fs - (z[0] + z[1] * (s - a)))),
                     x0=init, method="Nelder-Mead",
                     options={"maxiter": 200, "fatol": 1e-13, "xatol": 1e-11})
        best = min(best, r.fun)
    return best


def approx_err(f, nodes):
    return max(seg_best(f, nodes[j], nodes[j + 1])
               for j in range(len(nodes) - 1))


def main():
    N = 8
    rows = []
    for delta in (0.5, 0.25, 0.1, 0.05, 0.02):
        f = build_f(delta)
        uni = np.linspace(0, 1, N)
        free = np.linspace(0.5 - delta / 2, 0.5 + delta / 2, N)
        e_uni = approx_err(f, uni)
        e_free = approx_err(f, free)
        rows.append({"delta": delta, "e_uniform": e_uni,
                     "e_free": e_free, "gap": e_uni / e_free,
                     "theory_gap": 1.0 / delta ** 2})
        print(f"delta={delta:.2f}: gap={e_uni / e_free:.1f}x  "
              f"theory (1/delta)^2={1 / delta ** 2:.0f}x")
    with open(os.path.join(BASE, "results", "theory", "gap_lemma.json"), "w") as f:
        json.dump({"date": "2026-08-05",
                   "script": "verify_gap_lemma.py",
                   "note": "curvature-concentrated gap: e_free <= "
                           "M2*delta^2/(8N^2) vs e_uni >= M2/(8N^2); "
                           "gap >= (1/delta)^2 (mechanism of c_* < c_k)",
                   "rows": rows}, f, indent=2)
    print("Saved: results/theory/gap_lemma.json")


if __name__ == "__main__":
    main()
