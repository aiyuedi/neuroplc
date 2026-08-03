#!/usr/bin/env python3
"""
Theorem 10' (Verifiability Trichotomy) verification: LUT-compiled NN error regimes.

Claim: as N (LUT points per function) increases, interpolation error rates are:

  R1  C^2 activations (tanh, sin, cubic B-spline):  max error ~ N^{-2}  (slope -2)
  R2  Lipschitz-only activations (ReLU):            max error ~ N^{-1}  (slope -1)
  R3  multivariate gates (product / softmax):
      per-dimension error rate still ~ N^{-2}, BUT tensor-product LUT
      storage cost is N^m per m-ary gate (exponential in arity).

Checks performed:
  1. Rate measurement: max LUT interpolation error of N-point piecewise-linear
     LUTs on [-3,3] for f in {tanh, ReLU, sin, cubic B-spline}, N in
     {8,16,32,64,128,256}; log-log slope fit + R^2.
  2. Regime-3 cost: 2-ary tensor LUT storage = N^2, 4-ary storage = N^4;
     per-dimension error rate ~ N^{-2} verified on generic (non-exactly-
     reproducible) C^2 multivariate gates.  Note: pure product w*x*y*z is
     EXACTLY reproduced by multilinear tensor LUTs (error ~ machine eps),
     so the honest N^{-2}-rate statement is checked on generic gates.
  3. Sharp constant: for tanh on [-3,3], measured max error vs M_2*h^2/8 with
     M_2 = max|tanh''| = 4*sqrt(3)/9 ~ 0.7698 (verified numerically);
     ratio within 2x.

Honesty policy: measured slopes are reported as measured (fit can give -1.9
or -2.1); PASS requires the qualitative rate separation (slope -1 vs -2)
with R^2 > 0.9.

Output: results/theory/thm10p_trichotomy.json with PASS/FAIL assertions.
"""

import json
import math
import os
import sys
import datetime

import numpy as np
from scipy.interpolate import BSpline, RegularGridInterpolator

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
LO, HI = -3.0, 3.0
N_VALUES = [8, 16, 32, 64, 128, 256]
SEED = 42
N_EVAL_1D = 400_001          # dense evaluation grid per 1D measurement
OUT_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                 "results", "theory"))
OUT_JSON = os.path.join(OUT_DIR, "thm10p_trichotomy.json")

# qualitative thresholds (loose: theorem is about rate separation, not exact -2)
C2_SLOPE_BAND = (-2.30, -1.70)
R2_SLOPE_BAND = (-1.30, -0.70)
R2_MIN = 0.90


def h_of(N: int) -> float:
    """Uniform grid spacing on [LO, HI] with N points (endpoints included)."""
    return (HI - LO) / (N - 1)


# ---------------------------------------------------------------------------
# 1D LUT machinery
# ---------------------------------------------------------------------------
def lut_max_error_1d(f, N: int) -> float:
    """Max |f - LUT_N(f)| over a dense grid (x=0 added so ReLU's kink is seen)."""
    xs = np.linspace(LO, HI, N)
    ys = np.asarray(f(xs), dtype=float)
    xe = np.unique(np.concatenate([np.linspace(LO, HI, N_EVAL_1D), [0.0]]))
    ye = np.interp(xe, xs, ys)
    return float(np.max(np.abs(np.asarray(f(xe), dtype=float) - ye)))


def fit_loglog(Ns, errs):
    """Least-squares slope of log10(err) vs log10(N) plus R^2."""
    x = np.log10(np.asarray(Ns, dtype=float))
    y = np.log10(np.asarray(errs, dtype=float))
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    yhat = A @ np.array([slope, intercept])
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r2)


# ---------------------------------------------------------------------------
# activation functions
# ---------------------------------------------------------------------------
def make_cubic_bspline(seed: int = SEED):
    """Cubic (k=3) B-spline with random control points; C^2 continuous.

    Knots in [-6,6] with valid domain [t[3], t[16]] ~ [-4.1, 4.1] covering
    [-3,3].  Interpolation error of a C^2 function is O(h^2) -> slope -2.
    """
    rng = np.random.default_rng(seed)
    n_cp = 16
    ctrl = rng.uniform(-1.0, 1.0, n_cp)
    knots = np.linspace(-6.0, 6.0, n_cp + 4)
    return BSpline(knots, ctrl, k=3)


FUNCS = {
    "tanh": lambda x: np.tanh(x),
    "relu": lambda x: np.maximum(0.0, x),
    "sin": lambda x: np.sin(x),
    "bspline_cubic": make_cubic_bspline(),
}


# ---------------------------------------------------------------------------
# multivariate (tensor-product LUT) machinery
# ---------------------------------------------------------------------------
def tensor_lut_stats(f, N: int, dims: int, n_eval: int):
    """Max error of an N^dims multilinear tensor LUT of f on [LO,HI]^dims.

    Returns (max_err, storage_entries) where storage_entries = N^dims.
    """
    axes = [np.linspace(LO, HI, N)] * dims
    mesh = np.meshgrid(*axes, indexing="ij")
    vals = np.asarray(f(*mesh), dtype=float)                 # shape (N,)*dims
    lut = RegularGridInterpolator(axes, vals, method="linear")
    eg = [np.linspace(LO, HI, n_eval)] * dims
    emesh = np.meshgrid(*eg, indexing="ij")
    pts = np.stack([m.ravel() for m in emesh], axis=-1)
    approx = lut(pts)
    exact = np.asarray(f(*emesh), dtype=float).ravel()
    err = float(np.max(np.abs(exact - approx)))
    return err, int(vals.size)


# ---------------------------------------------------------------------------
# regime 3 gates
# ---------------------------------------------------------------------------
def f_xy(x, y):
    return x * y


def f_x2y(x, y):
    return x * x * y


def f_wxyz(w, x, y, z):
    return w * x * y * z


def f_gen4(w, x, y, z):
    return np.sin(w) * np.sin(x) * np.sin(y) * np.sin(z)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> int:
    results = {
        "theorem": "Thm10p_verifiability_trichotomy",
        "claim": (
            "R1: C^2 activations (tanh/sin/B-spline) LUT error ~ N^-2; "
            "R2: Lipschitz-only (ReLU) LUT error ~ N^-1; "
            "R3: multivariate gates keep ~N^-2 per dimension but cost N^m "
            "per m-ary tensor-product LUT."
        ),
        "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": __import__("scipy").__version__,
        "method": {
            "domain": [LO, HI],
            "N_values": N_VALUES,
            "interpolation": "piecewise-linear LUT (uniform grid, endpoints included)",
            "eval_grid_1d": N_EVAL_1D,
            "rng_seed": SEED,
        },
    }

    # ---------------- 1. rate measurement ----------------
    rates = {}
    for name, f in FUNCS.items():
        errs = [lut_max_error_1d(f, N) for N in N_VALUES]
        slope, r2 = fit_loglog(N_VALUES, errs)
        rates[name] = {
            "N": N_VALUES,
            "err": errs,
            "slope": slope,
            "r2": r2,
        }
    results["rates"] = rates

    # ---------------- 2. sharp constant (tanh) ----------------
    def tanh_d2(x):
        t = np.tanh(x)
        return -2.0 * t * (1.0 - t * t)

    M2_numeric = float(np.max(np.abs(tanh_d2(np.linspace(LO, HI, 1_000_001)))))
    M2_analytic = 4.0 * math.sqrt(3.0) / 9.0
    tanh_errs = rates["tanh"]["err"]
    tanh_ratios = []
    for N, err in zip(N_VALUES, tanh_errs):
        h = h_of(N)
        pred = M2_numeric * h * h / 8.0
        tanh_ratios.append(err / pred)
    results["sharp_constant"] = {
        "M2_numeric_max_abs_tanh_pp": M2_numeric,
        "M2_analytic_4sqrt3_over_9": M2_analytic,
        "prediction": "M2 * h^2 / 8 with h = 6/(N-1)",
        "h_per_N": [h_of(N) for N in N_VALUES],
        "predicted_err": [M2_numeric * h_of(N) ** 2 / 8.0 for N in N_VALUES],
        "measured_err": tanh_errs,
        "ratio_measured_over_predicted": tanh_ratios,
        "mean_ratio": float(np.mean(tanh_ratios)),
    }

    # ---------------- 3. regime 3: cost explosion + per-dim rate ----------------
    # 3a. 2-ary: storage N^2; pure product exact; generic C^2 gate rate -2
    n2 = [8, 16, 32, 64]
    xy_errs = {}
    xy_storage = {}
    for N in n2:
        err, storage = tensor_lut_stats(f_xy, N, 2, n_eval=401)
        xy_errs[str(N)] = err
        xy_storage[str(N)] = storage
    x2y_errs, x2y_storage = {}, {}
    for N in n2:
        err, storage = tensor_lut_stats(f_x2y, N, 2, n_eval=401)
        x2y_errs[str(N)] = err
        x2y_storage[str(N)] = storage
    x2y_slope, x2y_r2 = fit_loglog(n2, [x2y_errs[str(N)] for N in n2])

    # 3b. 4-ary: storage N^4; generic C^2 gate per-dim rate -2
    n4 = [4, 6, 8, 12]
    wxyz_storage = {}
    for N in [4, 6]:
        _, storage = tensor_lut_stats(f_wxyz, N, 4, n_eval=11)
        wxyz_storage[str(N)] = storage
    gen4_errs, gen4_storage = {}, {}
    for N in n4:
        err, storage = tensor_lut_stats(f_gen4, N, 4, n_eval=21)
        gen4_errs[str(N)] = err
        gen4_storage[str(N)] = storage
    gen4_slope, gen4_r2 = fit_loglog(n4, [gen4_errs[str(N)] for N in n4])

    results["regime3"] = {
        "tensor_product_lut": {
            "storage_formula": "N^m entries per m-ary gate (multilinear grid)",
        },
        "gate_2ary_xy": {
            "gate": "f(x,y) = x*y",
            "storage_per_N": xy_storage,
            "expected_storage": "N^2",
            "max_err_per_N": xy_errs,
            "note": (
                "bilinear tensor LUT reproduces xy EXACTLY (multilinear span "
                "contains the monomial xy); measured error is floating-point "
                "noise (~1e-15), so no slope is fitted -- pure-product gates "
                "are exact, cost N^2 is the binding constraint"
            ),
        },
        "gate_2ary_generic": {
            "gate": "f(x,y) = x^2*y (generic C^2, not bilinear-exact)",
            "N": n2,
            "storage_per_N": x2y_storage,
            "expected_storage": "N^2",
            "max_err_per_N": x2y_errs,
            "slope_per_dim": x2y_slope,
            "r2": x2y_r2,
        },
        "gate_4ary_product": {
            "gate": "f(w,x,y,z) = w*x*y*z",
            "storage_at_N6": wxyz_storage.get("6"),
            "expected_storage": "N^4 = 6^4 = 1296",
            "note": "storage measured as size of the stored values array",
        },
        "gate_4ary_generic": {
            "gate": "f(w,x,y,z) = sin(w)sin(x)sin(y)sin(z)",
            "N": n4,
            "storage_per_N": gen4_storage,
            "expected_storage": "N^4",
            "max_err_per_N": gen4_errs,
            "slope_per_dim": gen4_slope,
            "r2": gen4_r2,
        },
        "cost_scaling_demo": {
            "N": 64,
            "storage_N_m": {
                "m=1 (univariate)": 64,
                "m=2 (2-ary)": 64 ** 2,
                "m=3 (3-ary)": 64 ** 3,
                "m=4 (4-ary)": 64 ** 4,
            },
            "note": "exponential-in-arity cost: N^m (vs N for univariate LUTs)",
        },
    }

    # ---------------- 4. assertions ----------------
    assertions = []
    for name in ("tanh", "sin", "bspline_cubic"):
        s, r2 = rates[name]["slope"], rates[name]["r2"]
        assertions.append({
            "name": f"R1 rate for {name}",
            "pass": C2_SLOPE_BAND[0] <= s <= C2_SLOPE_BAND[1] and r2 > R2_MIN,
            "measured_slope": s,
            "measured_r2": r2,
            "threshold": f"slope in {C2_SLOPE_BAND} and R^2 > {R2_MIN} (expect ~-2)",
        })
    s_r, r2_r = rates["relu"]["slope"], rates["relu"]["r2"]
    assertions.append({
        "name": "R2 rate for ReLU",
        "pass": R2_SLOPE_BAND[0] <= s_r <= R2_SLOPE_BAND[1] and r2_r > R2_MIN,
        "measured_slope": s_r,
        "measured_r2": r2_r,
        "threshold": f"slope in {R2_SLOPE_BAND} and R^2 > {R2_MIN} (expect ~-1)",
    })
    # qualitative regime separation: C^2 vs Lipschitz
    c2_slopes = [rates[n]["slope"] for n in ("tanh", "sin", "bspline_cubic")]
    sep = abs(float(np.mean(c2_slopes)) - s_r)
    assertions.append({
        "name": "regime separation C^2 vs ReLU",
        "pass": sep >= 0.5,
        "measured_separation": sep,
        "threshold": "|mean(C^2 slope) - ReLU slope| >= 0.5 (-2 vs -1)",
    })
    # regime 3: storage exactness
    sto_ok2 = all(xy_storage[str(N)] == N ** 2 for N in n2)
    assertions.append({
        "name": "R3 2-ary storage = N^2",
        "pass": sto_ok2,
        "measured_storage_per_N": xy_storage,
        "threshold": "storage == N^2 for N in {8,16,32,64}",
    })
    assertions.append({
        "name": "R3 4-ary storage = N^4",
        "pass": wxyz_storage.get("6") == 6 ** 4,
        "measured_storage_at_N6": wxyz_storage.get("6"),
        "threshold": "storage == 6^4 = 1296",
    })
    assertions.append({
        "name": "R3 2-ary per-dimension rate ~ N^-2",
        "pass": C2_SLOPE_BAND[0] <= x2y_slope <= C2_SLOPE_BAND[1] and x2y_r2 > R2_MIN,
        "measured_slope": x2y_slope,
        "measured_r2": x2y_r2,
        "threshold": f"slope in {C2_SLOPE_BAND} and R^2 > {R2_MIN}",
    })
    assertions.append({
        "name": "R3 4-ary per-dimension rate ~ N^-2",
        "pass": -2.50 <= gen4_slope <= -1.50 and gen4_r2 > 0.85,
        "measured_slope": gen4_slope,
        "measured_r2": gen4_r2,
        "threshold": "slope in [-2.5,-1.5] and R^2 > 0.85 (4 N-points)",
    })
    # sharp constant: measured / (M2 h^2 / 8) within 2x
    mean_ratio = float(np.mean(tanh_ratios))
    assertions.append({
        "name": "tanh sharp constant M2*h^2/8 within 2x",
        "pass": 0.5 <= mean_ratio <= 2.0,
        "measured_mean_ratio": mean_ratio,
        "threshold": "0.5 <= mean(measured/(M2 h^2/8)) <= 2.0",
    })

    results["assertions"] = assertions
    results["pass"] = bool(all(a["pass"] for a in assertions))

    # regime classification verdict
    results["regime_classification"] = {
        "R1 (C^2 activations: tanh, sin, B-spline)": {
            "predicted_rate": "N^-2",
            "measured_slopes": {
                n: rates[n]["slope"] for n in ("tanh", "sin", "bspline_cubic")
            },
            "pass": all(any(
                a["name"] == f"R1 rate for {n}" and a["pass"] for a in assertions
            ) for n in ("tanh", "sin", "bspline_cubic")),
        },
        "R2 (Lipschitz-only: ReLU)": {
            "predicted_rate": "N^-1",
            "measured_slope": s_r,
            "pass": assertions[3]["pass"],
        },
        "R3 (multivariate gates)": {
            "predicted_rate": "N^-2 per dim, cost N^m per gate",
            "pass": all(assertions[i]["pass"] for i in range(4, 9)),
        },
    }

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # ---------------- console summary ----------------
    print("=" * 72)
    print("Verifiability Trichotomy (Thm10') -- LUT interpolation error rates")
    print("=" * 72)
    print(f"{'function':<16}{'slope':>9}{'R^2':>8}   regime (expect)")
    for name in ("tanh", "sin", "bspline_cubic"):
        s, r2 = rates[name]["slope"], rates[name]["r2"]
        print(f"{name:<16}{s:>9.3f}{r2:>8.4f}   R1: C^2  (expect ~-2)")
    print(f"{'relu':<16}{s_r:>9.3f}{r2_r:>8.4f}   R2: Lipschitz (expect ~-1)")
    print("-" * 72)
    print(f"2-ary generic gate x^2*y: slope {x2y_slope:.3f} (R^2={x2y_r2:.4f})  [expect -2 per dim]")
    print(f"4-ary generic gate sin*sin*sin*sin: slope {gen4_slope:.3f} (R^2={gen4_r2:.4f})  [expect -2 per dim]")
    print(f"2-ary storage: {xy_storage} = N^2  |  4-ary storage @N=6: "
          f"{wxyz_storage.get('6')} = 6^4  |  cost demo @N=64: "
          f"{results['regime3']['cost_scaling_demo']['storage_N_m']}")
    print(f"2-ary pure product x*y: error {max(xy_errs.values()):.2e} "
          "(exact reproduction by bilinear LUT, floating-point noise)")
    print("-" * 72)
    print(f"tanh sharp constant: M2 = {M2_numeric:.6f} (analytic 4sqrt(3)/9 = "
          f"{M2_analytic:.6f})")
    print(f"  mean measured/(M2 h^2/8) = {mean_ratio:.3f}  [threshold 0.5..2]")
    print("-" * 72)
    for a in assertions:
        print(f"[{'PASS' if a['pass'] else 'FAIL':<4}] {a['name']}")
    print("=" * 72)
    print(f"VERDICT: {('PASS' if results['pass'] else 'FAIL')}  ->  {OUT_JSON}")
    return 0 if results["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
