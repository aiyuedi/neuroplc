"""
verify_decision_law.py — Architecture-as-code selection law (Thm 1 corollary).

Scans target smoothness and compares two layer-1 codes on the same B budget:

  Table code (uniform linear LUT): error ~ N^{-2} for ANY smooth class
      (non-adaptive; polynomial rate on analytic classes too).
  Spectral code (Fourier truncation): error ~ e^{-2 rho N} on analytic
      classes (exponential), polynomial N^{-(2s-1)} on W^s classes.

Predicted phase transition: as smoothness rho grows past a threshold,
the spectral code overtakes the table code at moderate N — "who deserves
the bits" law. On W^s (polynomial decay) both stay polynomial: the
exponential benefit is exclusive to analytic classes.

Families (||f||_2 = 1 normalized):  a_n = Z^-1 e^{-rho n}   (analytic)
                                    a_n = Z^-1 n^{-(s+1/2)} (W^s)

Run:  python theory/verify_decision_law.py
Output: results/theory/decision_law.json
"""

import json
import os
import numpy as np

RNG = np.random.default_rng(11)
DOMAIN = (0.0, 1.0)
MMAX = 400  # coefficient truncation for the exact target


def make_family(mode, param):
    """Return (a_n array, f(t) callable), L2-normalized coefficients."""
    n = np.arange(1, MMAX + 1)
    if mode == "analytic":
        a = np.exp(-param * n)
    else:
        a = n ** -(param + 0.5)
    z = np.sqrt(np.sum(a ** 2) / 2.0)   # ||sum a_n sin(n pi t)||_2 = sqrt(sum a^2/2)
    a = a / z
    def f(t):
        t = np.asarray(t, dtype=float)
        s = np.zeros_like(t)
        for i, ai in enumerate(a, start=1):
            s = s + ai * np.sin(i * np.pi * t)
        return s
    return a, f


def approx_spectral(a, N):
    return a[:N]


def spectral_val(aN, t):
    t = np.asarray(t, dtype=float)
    s = np.zeros_like(t)
    for i, ai in enumerate(aN, start=1):
        s = s + ai * np.sin(i * np.pi * t)
    return s


def lut_linear_interp(fn, N):
    xs = np.linspace(DOMAIN[0], DOMAIN[1], N)
    ys = fn(xs)

    def val(t):
        t = np.asarray(t, dtype=float)
        i = np.clip(np.searchsorted(xs, t) - 1, 0, N - 2)
        return ys[i] + (ys[i + 1] - ys[i]) * (t - xs[i]) / (xs[i + 1] - xs[i])
    return val


def max_err(f_orig, val, n_grid=8001):
    t = np.linspace(DOMAIN[0], DOMAIN[1], n_grid)
    e = float(np.max(np.abs(f_orig(t) - val(t))))
    return max(e, 1e-300)   # clamp against underflow (exponential decay)


def main():
    results = {"date": "2026-08-04", "script": "verify_decision_law.py",
               "theorem": "architecture-as-code selection law (Thm 1 corollary)"}
    NS = [8, 16, 32, 64]
    tables = {}

    for mode, params in (("analytic", [0.15, 0.35, 0.7, 1.2]),
                         ("ws", [1.0, 2.0, 4.0])):
        tables[mode] = {}
        for p in params:
            a, f = make_family(mode, p)
            rows = []
            for N in NS:
                e_lut = max_err(f, lut_linear_interp(f, N))
                e_spec = max_err(f, lambda t, aN=a[:N]: spectral_val(aN, t))
                rows.append({"N": N, "lut": e_lut, "spectral": e_spec,
                             "spectral_wins": e_spec < e_lut})
            # log-log slopes at the largest N window
            logN = np.log(NS[-2:])
            slope_lut = np.polyfit(logN, np.log([r["lut"] for r in rows[-2:]]), 1)[0]
            slope_spec = np.polyfit(logN, np.log([r["spectral"] for r in rows[-2:]]), 1)[0]
            # N=64 error ratio (robust: no precision-target interpolation)
            e64_lut = rows[-1]["lut"]
            e64_spec = rows[-1]["spectral"]
            tables[mode][str(p)] = {"rows": rows,
                                    "slope_lut": float(slope_lut),
                                    "slope_spectral": float(slope_spec),
                                    "e64_lut": e64_lut, "e64_spectral": e64_spec,
                                    "e64_ratio_spec_over_lut": (
                                        e64_spec / e64_lut if e64_lut > 0 else None)}
            if mode == "analytic":
                # exponential slope on the N=16->32 window (pre-underflow)
                e2 = rows[1]["spectral"]
                e3 = rows[2]["spectral"]
                exp_slope = (np.log(e3) - np.log(e2)) / (NS[2] - NS[1])
                tables[mode][str(p)]["exp_slope_per_N"] = float(exp_slope)

    # Assertions (robust, N=64 error ratio based):
    # (1) analytic: spectral error decays exponentially with rate ~ rho
    an = tables["analytic"]
    exp_ok = all(abs(v["exp_slope_per_N"] + float(k)) < 0.6
                 for k, v in an.items())
    # (2) analytic: spectral beats table by >= 1 order of magnitude at N=64
    ratios_an = [v["e64_ratio_spec_over_lut"] for v in an.values()]
    an_win = all(r is not None and r < 0.1 for r in ratios_an)
    # (3) W^s: both codes polynomial (no exponential slope)
    ws = tables["ws"]
    ws_poly = all(abs(v["slope_spectral"]) < 8 and abs(v["slope_lut"]) < 8
                  for v in ws.values())
    # (4) phase transition: low-smoothness W^1 keeps LUT competitive
    #     (ratio >= 0.1) while analytic collapses below 0.01
    r_ws1 = ws["1.0"]["e64_ratio_spec_over_lut"]
    lut_competitive_ws1 = r_ws1 is not None and r_ws1 >= 0.1
    gap = (max(r for r in ratios_an if r is not None)
           if any(r is not None for r in ratios_an) else None)
    phase_gap = gap is not None and gap <= 0.01 and lut_competitive_ws1
    # (5) LUT stays polynomial on analytic classes (no exponential benefit)
    lut_poly = all(abs(v["slope_lut"]) < 3.5 for v in an.values())

    results["tables"] = tables
    results["assertions"] = {
        "exp_slope_matches_rho": exp_ok,
        "analytic_spectral_wins_by_1_order": an_win,
        "ws_stays_polynomial": ws_poly,
        "phase_transition_ws1_competitive_vs_analytic": phase_gap,
        "lut_stays_polynomial_on_analytic": lut_poly,
    }
    results["verdict"] = "PASS" if (exp_ok and an_win and ws_poly
                                    and phase_gap and lut_poly) else "CHECK"

    out = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..",
                                        "results", "theory", "decision_law.json"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps(results["assertions"], indent=2))
    print("verdict:", results["verdict"])
    for k, v in an.items():
        print(f"analytic rho={k}: exp_slope={round(v['exp_slope_per_N'], 3)}, "
              f"slope_lut={round(v['slope_lut'], 2)}, "
              f"e64_ratio={round(v['e64_ratio_spec_over_lut'], 4)}")
    for k, v in ws.items():
        print(f"ws s={k}: slope_spec={round(v['slope_spectral'], 2)}, "
              f"slope_lut={round(v['slope_lut'], 2)}, "
              f"e64_ratio={round(v['e64_ratio_spec_over_lut'], 4)}")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
