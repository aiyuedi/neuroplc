#!/usr/bin/env python3
"""T-I: Curvature-aware optimal LUT allocation and grid (2026-08-03).

Two-level optimality theorem for LUT compilation of a B-spline KAN:
  (a) CROSS-FUNCTION allocation: with total budget N over F functions,
      the minimax-optimal per-function node count is n_i ~ N * M2_i^{1/2}
      / sum_j M2_j^{1/2}, giving
          e* = (b-a)^2 / (8 (sum_i M2_i^{-1/2})^2)
      vs uniform e = max_i M2_i (b-a)^2 / (8 N^2).
  (b) WITHIN-FUNCTION grid: minimax segment error M2(seg) Delta^2/8 is
      balanced by node density rho(x) ~ sqrt(M2(x)), i.e. sqrt(|phi''(x)|).
      The previous curvature density |y''|/(1+y'^2)^{3/2} is measurably
      worse than uniform on real activations.

Reference: Micchelli-Rivlin-Winograd (1976) optimal recovery; the k=2
piecewise-linear LUT case with per-segment M2. Runs on the released
checkpoint; dumps results/theory/ti_optimal_lut.json.
"""
import json
import os

import numpy as np
import torch
from scipy.interpolate import BSpline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")
OUT = os.path.join(BASE, "results", "theory", "ti_optimal_lut.json")
X0, X1, G, K = -3.0, 3.0, 8, 3
NS = 2001


def knots():
    return np.concatenate([np.full(K, X0), np.linspace(X0, X1, G + 1),
                           np.full(K, X1)])


def per_function_m2(sd):
    """M2_i = max |phi''| per B-spline activation (folded scale)."""
    xs = np.linspace(X0, X1, NS)
    t = knots()
    out = []
    for L in ("0", "1"):
        st = sd[f"kan_layers.{L}.spline_weight"].numpy()
        ss = float(sd[f"kan_layers.{L}.scale_spline"])
        for o in range(st.shape[0]):
            for i in range(st.shape[1]):
                sp = BSpline(t, st[o, i] * ss, K)
                out.append(float(np.abs(sp.derivative(2)(xs)).max()))
    return np.array(out)


def seg_minimax(grid, d2, xs):
    errs = []
    for kk in range(len(grid) - 1):
        seg = (xs >= grid[kk]) & (xs < grid[kk + 1])
        if seg.sum() < 2:
            continue
        errs.append(d2[seg].max() * (grid[kk + 1] - grid[kk]) ** 2 / 8)
    return float(max(errs))


def worst_function(sd):
    st0 = sd["kan_layers.0.spline_weight"].numpy()
    ss0 = float(sd["kan_layers.0.scale_spline"])
    t0 = knots()
    xs = np.linspace(X0, X1, NS)
    wm, wd, wf = 0.0, None, None
    for o in range(st0.shape[0]):
        for i in range(st0.shape[1]):
            sp = BSpline(t0, st0[o, i] * ss0, K)
            d2 = np.abs(sp.derivative(2)(xs))
            mm = d2.max()
            if mm > wm:
                wm, wd, wf = mm, d2, sp(xs)
    return wm, wd, wf, xs


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"]
    m2 = per_function_m2(sd)
    N = len(m2)
    b_a = X1 - X0

    # (a) cross-function allocation
    inv_sqrt = float(np.sum(1.0 / np.sqrt(np.maximum(m2, 1e-9))))
    e_star = (b_a ** 2) / (8 * inv_sqrt ** 2)
    e_uniform = float(m2.max()) * (b_a ** 2) / (8 * N ** 2)

    # (b) within-function grid on the worst activation
    wm, wd, wf, xs = worst_function(sd)
    dx = xs[1] - xs[0]
    npts = 15
    g_u = np.linspace(X0, X1, npts)
    w = np.sqrt(wd + 1e-10)
    cdf = np.cumsum(w)
    cdf /= cdf[-1]
    g_sqrt = np.interp(np.linspace(0, 1, npts), cdf, xs)
    dy = np.gradient(wf, dx)
    w2 = wd / (1.0 + dy ** 2) ** 1.5 + 1e-10
    cdf2 = np.cumsum(w2)
    cdf2 /= cdf2[-1]
    g_kap = np.interp(np.linspace(0, 1, npts), cdf2, xs)
    e_u = seg_minimax(g_u, wd, xs)
    e_sqrt = seg_minimax(g_sqrt, wd, xs)
    e_kap = seg_minimax(g_kap, wd, xs)

    result = {
        "date": "2026-08-03",
        "theorem": "T-I curvature-aware optimal LUT",
        "m2": {"max": float(m2.max()), "mean": float(m2.mean()),
               "p90": float(np.percentile(m2, 90))},
        "cross_function": {
            "e_star_optimal": e_star, "e_uniform": e_uniform,
            "improvement_x": float(e_uniform / e_star)},
        "within_function_worst": {
            "uniform": e_u, "sqrt_M2_optimal": e_sqrt,
            "kappa_old_impl": e_kap,
            "sqrt_vs_uniform_x": float(e_u / e_sqrt),
            "kappa_vs_uniform_x": float(e_kap / e_u)},
        "verdict": "PASS" if (e_uniform / e_star > 1.5
                              and e_sqrt < e_u < e_kap) else "CHECK",
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
