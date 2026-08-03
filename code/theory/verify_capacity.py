"""
verify_capacity.py v2 — Packing-line constant matching for Certifiable Capacity Theory (v4).

Verifies on synthetic W^k classes (k=2) the empirical facts behind
Theorem 2 (stratification) and Theorem 3 (constant purchase):

  (a) Uniform-LUT (linear interpolation) error matches the sharp constant
      c_2 = Q_2/2! = 1/8 on the KT extremal (exact), slope ~ -2 in log-log;
  (b) Constant ordering on the *worst-case* (curvature-peaked) family:
      c_free <= c_fixed <= c_k  (free nodes help exactly where curvature is
      concentrated; on uniformly-curved random families free ~= fixed, an
      empirical echo of Chaskalovic: uniform grid is (near-)minimax there);
  (c) Slope fit: min-max deployment error decays as N^{-k} (packing rate).

Honest scope: corroboration only, no proofs. The free-node optimizer is a
cheap coordinate-descent (upper bound on the true free-node optimum), so
c_free estimates are conservative.

Run:  python theory/verify_capacity.py
Output: results/theory/capacity_packing.json
"""

import json
import os
import numpy as np

RNG = np.random.default_rng(42)
DOMAIN = (0.0, 1.0)


# ---------------------------------------------------------------------------
# Test families, ||f''||_inf = 1
# ---------------------------------------------------------------------------

def _max_kth_deriv(coeffs, k):
    der = np.polyder(coeffs[::-1], k)
    if der.size == 0:
        return 0.0
    xs = np.linspace(0.0, 1.0, 4001)
    return float(np.max(np.abs(np.polyval(der, xs))))


def random_poly_family(n_funcs=30, degree=6, seed_offset=0):
    """Random degree-d polynomials with ||f''|| = 1 (uniformly curved)."""
    rng = np.random.default_rng(seed_offset)
    fs = []
    for _ in range(n_funcs):
        c = rng.normal(0.0, 1.0, degree + 1)
        m = _max_kth_deriv(c, 2)
        if m < 1e-9:
            continue
        fs.append(c / m)
    return fs


def kt_extremal_k2():
    """KT-type window extremal: f(x) = x(1-x)/2 on [0,1]; ||f''||=1.

    Interpolation error of the linear LUT attains exactly M2 h^2/8
    (the sharp constant c_2), pointwise worst case at window midpoints.
    """
    return np.array([0.0, 0.5, -0.5])   # x/2 - x^2/2 = x(1-x)/2 (low->high)


def peaked_family():
    """Curvature-peaked (worst-case) family: ||f''|| = 1, curvature concentrated.

    - x^4/12  : M2 = max|12x^2|/12 = 1 at x=1 (curvature at right endpoint)
    - sin(pi x)/pi^2 : M2 = 1 at both endpoints
    - e^x / e : M2 = 1 at x=1 (exponentially concentrated)
    """
    return [
        np.array([0.0, 0.0, 0.0, 0.0, 1.0 / 12.0]),      # x^4/12
        np.array([0.0, 1.0 / np.pi, 0.0, -np.pi / 6.0, 0.0, np.pi ** 3 / 120.0,
                  0.0, -np.pi ** 5 / 5040.0]),           # sin(pi x)/pi^2
        np.array([1.0 / np.e, 1.0 / np.e, 1.0 / (2 * np.e), 1.0 / (6 * np.e),
                  1.0 / (24 * np.e), 1.0 / (120 * np.e)]),  # e^x/e
    ]


def eval_poly(coeffs, xs):
    return np.polyval(coeffs[::-1], xs)


# ---------------------------------------------------------------------------
# Schemes (piecewise linear, k=2)
# ---------------------------------------------------------------------------

def scheme_lut_interp(coeffs, N):
    """Uniform-node linear interpolation through node values (the DA/LUT)."""
    xs = np.linspace(DOMAIN[0], DOMAIN[1], N)
    ys = eval_poly(coeffs, xs)

    def val(t):
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(xs, t) - 1, 0, N - 2)
        return ys[i] + (ys[i + 1] - ys[i]) * (t - xs[i]) / (xs[i + 1] - xs[i])
    return val


def scheme_proj_fixed(coeffs, N):
    """Uniform nodes, per-segment least-squares linear projection."""
    xs = np.linspace(DOMAIN[0], DOMAIN[1], N)
    segs = []
    for j in range(N - 1):
        s = np.linspace(xs[j], xs[j + 1], 101)
        f = eval_poly(coeffs, s)
        A = np.vstack([np.ones_like(s), (s - xs[j])]).T
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        segs.append((xs[j], xs[j + 1], coef))

    def val(t):
        t = np.asarray(t, dtype=float)
        flat = t.ravel()
        i = np.clip(np.searchsorted(xs, flat) - 1, 0, N - 2)
        out = np.empty_like(flat)
        for j in range(N - 1):
            m = i == j
            if m.any():
                x0, _, coef = segs[j]
                out[m] = coef[0] + coef[1] * (flat[m] - x0)
        return out.reshape(t.shape)
    return val


def scheme_proj_free(coeffs, N):
    """Free nodes: coordinate descent over interior node positions.

    Initialized by the T-I curvature-aware density rho ~ sqrt(M2(x))
    (inverse-CDF sampling), then refined by coordinate descent.  The
    evaluation grid per segment is 1001 points for a smooth objective.
    Conservative upper bound on the true free-node optimum (10 restarts).
    """
    # curvature-aware init: density proportional to sqrt(M2) via inverse CDF
    xs_init = np.linspace(DOMAIN[0], DOMAIN[1], N)
    fpp = np.polyder(coeffs[::-1], 2)
    if fpp.size:
        gg = np.linspace(DOMAIN[0], DOMAIN[1], 2001)
        w = np.abs(np.polyval(fpp, gg)) ** 0.5 + 1e-6
        cdf = np.cumsum(w)
        cdf /= cdf[-1]
        xs_init = np.interp(np.linspace(0, 1, N), cdf, gg)
        xs_init[0], xs_init[-1] = DOMAIN[0], DOMAIN[1]

    def max_err(nodes):
        worst = 0.0
        for j in range(N - 1):
            s = np.linspace(nodes[j], nodes[j + 1], 1001)
            f = eval_poly(coeffs, s)
            A = np.vstack([np.ones_like(s), (s - nodes[j])]).T
            coef, *_ = np.linalg.lstsq(A, f, rcond=None)
            e = np.abs(f - (coef[0] + coef[1] * (s - nodes[j])))
            worst = max(worst, float(e.max()))
        return worst

    best = (max_err(xs_init), xs_init.copy())
    for restart in range(10):
        cur = np.sort(RNG.uniform(DOMAIN[0], DOMAIN[1], N))
        cur[0], cur[-1] = DOMAIN[0], DOMAIN[1]
        for _ in range(60):
            improved = False
            for j in range(1, N - 1):
                lo, hi = cur[j - 1], cur[j + 1]
                base = max_err(cur)
                best_cand, best_e = cur[j], base
                for cand in np.linspace(lo + 1e-4, hi - 1e-4, 9):
                    trial = cur.copy()
                    trial[j] = cand
                    e = max_err(trial)
                    if e < best_e:
                        best_cand, best_e = cand, e
                if best_e < base:
                    cur[j] = best_cand
                    improved = True
            if not improved:
                break
        if max_err(cur) < best[0]:
            best = (max_err(cur), cur.copy())

    best_xs = best[1]
    seg_coefs = []
    for j in range(N - 1):
        s = np.linspace(best_xs[j], best_xs[j + 1], 3)
        f = eval_poly(coeffs, s)
        A = np.vstack([np.ones_like(s), (s - best_xs[j])]).T
        coef, *_ = np.linalg.lstsq(A, f, rcond=None)
        seg_coefs.append((best_xs[j], best_xs[j + 1], coef))

    def val(t):
        t = np.asarray(t, dtype=float)
        flat = t.ravel()
        i = np.clip(np.searchsorted(best_xs, flat) - 1, 0, N - 2)
        out = np.empty_like(flat)
        for j in range(N - 1):
            m = i == j
            if m.any():
                x0, _, coef = seg_coefs[j]
                out[m] = coef[0] + coef[1] * (flat[m] - x0)
        return out.reshape(t.shape)
    return val, best_xs


def deployment_error(f_orig, scheme_val, n_grid=4001):
    t = np.linspace(DOMAIN[0], DOMAIN[1], n_grid)
    return float(np.max(np.abs(f_orig(t) - scheme_val(t))))


def fit_slope(logN, logE):
    A = np.vstack([logN, np.ones_like(logN)]).T
    coef, *_ = np.linalg.lstsq(A, logE, rcond=None)
    return float(coef[0])


def scan_family(coeffs_list, Ns, with_free=True):
    """Return per-N worst-case (over family) errors for the three schemes."""
    rows = []
    for N in Ns:
        lut = proj = free = 0.0
        for cf in coeffs_list:
            f = lambda t, cf=cf: eval_poly(cf, t)
            lut = max(lut, deployment_error(f, scheme_lut_interp(cf, N)))
            proj = max(proj, deployment_error(f, scheme_proj_fixed(cf, N)))
            if with_free and N <= 16:
                v, _ = scheme_proj_free(cf, N)
                free = max(free, deployment_error(f, v))
        rows.append({"N": N, "lut": lut, "fixed": proj,
                     "free": free if (with_free and N <= 16) else None})
    return rows


def main():
    results = {"date": "2026-08-04", "script": "verify_capacity.py",
               "theorem": "capacity packing-line constant matching (v4)", "k": 2}
    c2_theory = 1.0 / 8.0
    NS = [8, 16, 32, 64]

    # --- (a) KT extremal: LUT attains c_2 exactly at every N ---
    kt = kt_extremal_k2()
    kt_ratios = []
    for N in NS:
        e = deployment_error(lambda t: eval_poly(kt, t),
                             scheme_lut_interp(kt, N))
        kt_ratios.append(e / (c2_theory / (N - 1) ** 2))

    # --- (b) random family (uniform curvature): free ~= fixed (Chaskalovic echo) ---
    rand = random_poly_family(n_funcs=20, seed_offset=7)
    rand_rows = scan_family(rand, NS, with_free=True)

    # --- (c) peaked family (worst case): free <= fixed <= lut ---
    peak = peaked_family()
    peak_rows = scan_family(peak, NS, with_free=True)

    # ordering on peaked family at N=8,16 (where free is measured)
    ord_ok = True
    ord_detail = []
    for r in peak_rows:
        if r["free"] is None:
            continue
        ok = r["free"] <= r["fixed"] + 1e-14 <= r["lut"] + 1e-14
        ord_ok = ord_ok and ok
        ord_detail.append({"N": r["N"], "ok": ok, "free": r["free"],
                           "fixed": r["fixed"], "lut": r["lut"]})

    # slopes (N=8..64, lut & fixed)
    slope_lut = fit_slope(np.log(NS), np.log([r["lut"] for r in peak_rows]))
    slope_fix = fit_slope(np.log(NS), np.log([r["fixed"] for r in peak_rows]))

    # constants at N=64 (peaked family worst case)
    c_lut64 = peak_rows[-1]["lut"] * 64 ** 2
    c_fix64 = peak_rows[-1]["fixed"] * 64 ** 2

    results["kt_extremal"] = {"c2_theory": c2_theory,
                              "lut_ratio_at_N": list(zip(NS, [round(x, 6) for x in kt_ratios]))}
    results["random_family"] = {"note": "uniform curvature: free ~= fixed (Chaskalovic echo)",
                                "rows": rand_rows}
    results["peaked_family"] = {"note": "worst-case curvature: expected free <= fixed <= lut",
                                "rows": peak_rows,
                                "ordering_ok": ord_ok,
                                "ordering_detail": ord_detail,
                                "slope_lut": slope_lut, "slope_fixed": slope_fix,
                                "c_lut_N64": c_lut64, "c_fixed_N64": c_fix64}
    results["verdict"] = "PASS" if (
        all(0.995 < r < 1.005 for r in kt_ratios[:2]) and
        ord_ok and abs(slope_lut + 2.0) < 0.2 and abs(slope_fix + 2.0) < 0.2
    ) else "CHECK"

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "theory", "capacity_packing.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
