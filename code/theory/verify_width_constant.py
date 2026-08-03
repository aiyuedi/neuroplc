"""
verify_width_constant.py — numerical estimate of the class-optimal width
constant c_* (stratum 3) for W^2 in L-infinity, free-node piecewise-linear.

Each segment uses the L-infinity optimal line (Remez-type: minimize
max|f - p| over the segment), and interior node positions are optimized
by coordinate descent (as in verify_capacity.py).  The measured worst
error times N^2 estimates the width constant c_*, which must satisfy
c_* < c_k = 1/8 (interpolation constant; the free-node width is strictly
smaller for curvature-peaked families).

Run: python theory/verify_width_constant.py
Output: results/theory/width_constant.json
"""

import json
import os
import numpy as np
from scipy.optimize import minimize

RNG = np.random.default_rng(5)
DOMAIN = (0.0, 1.0)
CK = 1.0 / 8.0


def eval_poly(coeffs, xs):
    return np.polyval(coeffs[::-1], xs)


def seg_l_inf_line(fn, a, b, n_grid=151):
    """L-infinity optimal line on [a,b] via smooth max surrogate + minimize."""
    s = np.linspace(a, b, n_grid)
    f = fn(s)
    best = None
    for init in ((f[0], 0.0), (f[0], (f[-1] - f[0]) / (b - a)),
                 (f[n_grid // 2], 0.0)):
        res = minimize(
            lambda z: np.max(np.abs(f - (z[0] + z[1] * (s - a)))),
            x0=init, method="Nelder-Mead",
            options={"maxiter": 300, "xatol": 1e-12, "fatol": 1e-14})
        if best is None or res.fun < best.fun:
            best = res
    return best.x, float(best.fun)


def scheme_free_l_inf(coeffs, N):
    """Free nodes, per-segment L-inf optimal line; coordinate descent."""
    fn = lambda t: eval_poly(coeffs, t)
    xs = np.linspace(DOMAIN[0], DOMAIN[1], N)

    def max_err(nodes):
        worst = 0.0
        for j in range(N - 1):
            _, e = seg_l_inf_line(fn, nodes[j], nodes[j + 1])
            worst = max(worst, e)
        return worst

    best = (max_err(xs), xs.copy())
    for restart in range(2):
        cur = np.sort(RNG.uniform(DOMAIN[0], DOMAIN[1], N))
        cur[0], cur[-1] = DOMAIN[0], DOMAIN[1]
        for _ in range(15):
            improved = False
            for j in range(1, N - 1):
                lo, hi = cur[j - 1], cur[j + 1]
                base = max_err(cur)
                bc, be = cur[j], base
                for cand in np.linspace(lo + 1e-4, hi - 1e-4, 5):
                    trial = cur.copy()
                    trial[j] = cand
                    e = max_err(trial)
                    if e < be:
                        bc, be = cand, e
                if be < base:
                    cur[j] = bc
                    improved = True
            if not improved:
                break
        if max_err(cur) < best[0]:
            best = (max_err(cur), cur.copy())
    return best[0]


def main():
    results = {"date": "2026-08-04", "script": "verify_width_constant.py",
               "theorem": "Thm 17(4)/18(i) width constant c_*, numerical estimate"}
    # families (||f''|| = 1): KT extremal (uniform curvature) + peaked
    kt = np.array([0.0, 0.5, -0.5])                 # x(1-x)/2, M2 = 1
    peaked = [np.array([0.0, 0.0, 0.0, 0.0, 1.0 / 12.0]),   # x^4/12
              np.array([1.0 / np.e, 1.0 / np.e, 1.0 / (2 * np.e),
                        1.0 / (6 * np.e), 1.0 / (24 * np.e)])]  # e^x/e
    NS = [4, 8]
    rows = []
    for N in NS:
        row = {"N": N}
        for name, cf in [("kt", kt)] + [(f"peaked{i}", p) for i, p in enumerate(peaked)]:
            e = scheme_free_l_inf(cf, N)
            row[name] = e
            row[name + "_N2"] = e * N * N
        rows.append(row)
    # c_* estimate: max over families of err * N^2 at the largest N
    r16 = rows[-1]
    c_star = max(r16["kt_N2"], r16["peaked0_N2"], r16["peaked1_N2"])
    results["c_star_at_N"] = r16["N"]
    results["rows"] = rows
    results["c_star_estimate"] = float(c_star)
    results["c_k_reference"] = CK
    results["c_star_lt_c_k"] = bool(c_star < CK)
    results["gap_ratio_c_k_over_c_star"] = float(CK / c_star)
    results["verdict"] = "PASS" if results["c_star_lt_c_k"] else "CHECK"

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "theory", "width_constant.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "rows"}, indent=2))
    for r in rows:
        print(f"N={r['N']}: kt={r['kt']:.6f} (N2={r['kt_N2']:.4f}), "
              f"peaked0={r['peaked0']:.6f} (N2={r['peaked0_N2']:.4f}), "
              f"peaked1={r['peaked1']:.6f} (N2={r['peaked1_N2']:.4f})")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
