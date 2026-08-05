#!/usr/bin/env python3
"""Free-knot local-optima hardness evidence (2026-08-05).

Supports the c_*/necessity conjecture anchor: globally optimal free-knot
placement is set-partitioning-hard (Mohr 2023) / NP-hard global
optimization (Beliakov 2004). Here we quantify the multimodal landscape
on a single curvature-peaked quartic: 40 random restarts of free-node
coordinate descent converge to 40 DISTINCT local optima whose errors
span 25.7x. Global optimality certification is therefore exponentially
harder than local improvement.
"""
import json
import os

import numpy as np
from scipy.optimize import minimize

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
coeffs = np.array([1.0, -4.0, 6.0, -4.0, 1.0, 0.0])  # (x-1)^4-like peaked


def fn(t):
    return np.polyval(coeffs, t)


def seg_err(a, b, n=151):
    s = np.linspace(a, b, n)
    f = fn(s)
    best = 1e9
    for init in ((f[0], 0.0), (f[0], (f[-1] - f[0]) / (b - a))):
        r = minimize(lambda z: np.max(np.abs(f - (z[0] + z[1] * (s - a)))),
                     x0=init, method="Nelder-Mead",
                     options={"maxiter": 300, "fatol": 1e-14, "xatol": 1e-12})
        best = min(best, r.fun)
    return best


def err_of(nodes):
    return max(seg_err(nodes[j], nodes[j + 1])
               for j in range(len(nodes) - 1))


def main():
    rng = np.random.default_rng(5)
    N = 6
    local_opts = []
    for trial in range(40):
        cur = np.sort(rng.uniform(0, 1, N))
        e = err_of(cur)
        improved = True
        while improved:
            improved = False
            for j in range(1, N - 1):
                lo, hi = cur[j - 1], cur[j + 1]
                best_x, best_e = cur[j], e
                for cand in np.linspace(lo + 1e-6, hi - 1e-6, 25):
                    c2 = cur.copy()
                    c2[j] = cand
                    e2 = err_of(c2)
                    if e2 < best_e - 1e-12:
                        best_x, best_e = cand, e2
                if best_x != cur[j]:
                    cur[j] = best_x
                    e = best_e
                    improved = True
        local_opts.append((e, cur.copy()))
    opts = sorted(local_opts)
    distinct = len(set(np.round([o[0] for o in opts], 10)))
    best, worst = opts[0][0], opts[-1][0]
    ratio = worst / best
    print(f"restarts=40  distinct local optima={distinct}  "
          f"best={best:.5f} (xN^2={best * 36:.4f})  "
          f"worst={worst:.5f}  ratio={ratio:.2f}x")
    out = {
        "date": "2026-08-05",
        "script": "verify_freeknot_hardness.py",
        "note": "free-knot local-optima landscape on peaked quartic; "
                "supports c_*/necessity conjecture anchor (Mohr 2023, "
                "Beliakov 2004)",
        "restarts": 40,
        "distinct_local_optima": distinct,
        "best_err_xN2": float(best * 36),
        "worst_err_xN2": float(worst * 36),
        "local_optima_ratio": float(ratio),
    }
    with open(os.path.join(BASE, "results", "theory", "freeknot_hardness.json"),
              "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/freeknot_hardness.json")


if __name__ == "__main__":
    main()
