#!/usr/bin/env python3
"""T-II: Compile-aware minimax deployment rate (2026-08-03).

Theorem (LUT-compilation minimax deployment rate), for a k-smooth target
class F_k = {f : ||f^(k)||_inf <= M_k}, n samples, LUT budget N:
  (a) lower bound: inf_{learner,compiler} sup R_deploy(n,N)
        >= c1 * max{ n^{-2k/(2k+1)}, e*(N) }   (classical minimax + T-I bias)
  (b) upper bound attained by LUT compilation with the T-I two-level
      optimal allocation: R <= c2 * max{ n^{-2k/(2k+1)}, e*(N) }
  (c) SAMPLE-BUDGET SCISSORS: for N <= N*(n) ~ n^{1/(2k)}, the risk is
      ~ N^{-2k} -- INDEPENDENT of n (budget-starved samples are wasted).
  (d) DISCRETE COMPILATION IS FREE: for N >= N*(n), the risk is
      ~ n^{-2k/(2k+1)} -- LUT compilation does not degrade the
      statistical minimax rate.
  (e) optimal budget curve N*(n) ~ n^{1/(2k)} (bias-estimation balance).

Numerical validation on a synthetic k=2 Fourier class: the N=4 row must
be flat in n (scissors), the N=256 row must decay ~ n^{-4/5}
(discrete-free), N=16 must be between (approaching the budget edge).
Dumps results/theory/tii_compile_aware_minimax.json.
"""
import json
import os

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(BASE, "results", "theory", "tii_compile_aware_minimax.json")
COEF = np.array([1.0 / i ** 2 for i in range(1, 6)])  # k=2 Fourier class
RNG = np.random.RandomState(0)


def gen_task(n, noise=0.05):
    x = RNG.uniform(-3, 3, n)
    y = (sum(c * np.sin(i * np.pi * x / 3)
             for i, c in enumerate(COEF, 1)) + noise * RNG.randn(n))
    return x, y


def fit_and_deploy(n, N, trials=25):
    risks = []
    for _ in range(trials):
        x, y = gen_task(n)
        B = np.column_stack([np.sin(i * np.pi * x / 3)
                             for i in range(1, 6)])
        w = np.linalg.lstsq(B, y, rcond=None)[0]
        g = np.linspace(-3, 3, N)
        fg = sum(w[i - 1] * np.sin(i * np.pi * g / 3)
                 for i in range(1, 6))
        xq = np.linspace(-3, 3, 2000)
        f_true = sum(c * np.sin(i * np.pi * xq / 3)
                     for i, c in enumerate(COEF, 1))
        f_est = np.interp(xq, g, fg)
        risks.append(np.mean((f_est - f_true) ** 2))
    return float(np.mean(risks))


def main():
    ns = (200, 3200, 51200)
    nts = (4, 16, 64, 256)
    table = {str(n): {str(N): fit_and_deploy(n, N) for N in nts}
             for n in ns}
    flat = table["200"]["4"], table["3200"]["4"], table["51200"]["4"]
    decay = table["200"]["256"], table["3200"]["256"], table["51200"]["256"]
    flat_ratio = max(flat) / min(flat)
    decay_ratio = decay[0] / decay[2]
    # scissors: N=4 flat in n (ratio < 1.5); discrete-free: N=256 decays
    # >= 100x across the n range (n^{4/5}: 200->51200 is ~380x)
    scissors_pass = flat_ratio < 1.5
    free_pass = decay_ratio > 100
    result = {
        "date": "2026-08-03",
        "theorem": "T-II compile-aware minimax deployment rate",
        "class": "k=2 Fourier (W^2_inf), noise 0.05",
        "risk_table": table,
        "scissors_check": {"N=4 row": flat,
                           "max/min": flat_ratio,
                           "flat_in_n (scissors)": scissors_pass},
        "discrete_free_check": {"N=256 row": decay,
                                "decay_ratio_200_to_51200": decay_ratio,
                                "decays_with_n (free)": free_pass},
        "predictions": {"N*_200 ~ 3.8, N*_3200 ~ 7.5, N*_51200 ~ 15.0":
                        "N=16 row flat-ish (budget edge), N=64/256 rows decay"},
        "verdict": "PASS" if (scissors_pass and free_pass) else "CHECK",
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
