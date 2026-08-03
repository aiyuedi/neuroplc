#!/usr/bin/env python3
"""T-III: Necessity first lattice (2026-08-03).

Two-sided empirical check for Theorem (Necessity in the
Affine-Certificate Class):

(1) SINGLE-FUNCTION REFUTATION of the naive rate argument: a kinked
    function |x| with KNOWN kink location admits arbitrarily small LUT
    error (nodes clustered at +-eps give error = eps). So the reason
    kink classes are excluded is NOT per-function rate.

(2) CLASS-WIDTH CONFIRMATION: over the class of shifted kinks (tent
    functions with kink at an arbitrary location), ANY N-point grid has
    minimax error >= c/N (classical C^1 width): a grid cannot cluster
    nodes at every possible kink location. Measured slope ~ -1.

(3) CERTIFICATE EXCLUSION: sup|f''| = infinity for |x| (numerically
    diverging), so no finite DA certificate (c, R) exists in the
    repaired Galois domain (envelope contains M2(f) r^2/2).

Dumps results/theory/tiii_necessity_first.json.
"""
import json
import os

import numpy as np

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(BASE, "results", "theory", "tiii_necessity_first.json")
XS = np.linspace(-3, 3, 200001)


def max_err(f, grid):
    fq = np.interp(XS, grid, f(grid))
    return float(np.abs(fq - f(XS)).max())


def main():
    # (1) single kink, known location: error -> eps
    single = {}
    for eps in (1.0, 0.1, 0.01, 0.001):
        grid = np.array([-3.0, -eps, eps, 3.0])
        single[str(eps)] = max_err(np.abs, grid)

    # (2) class of shifted kinks: minimax over random kink locations
    rng = np.random.RandomState(0)
    widths = {}
    for N in (8, 32, 128):
        worst = 0.0
        for _ in range(200):
            a = rng.uniform(-2.5, 2.5)   # random kink
            tent = lambda x, a=a: np.abs(x - a)
            grid = np.linspace(-3, 3, N)
            worst = max(worst, max_err(tent, grid))
        widths[str(N)] = worst
    slope = -np.log(widths["128"] / widths["8"]) / np.log(128 / 8)

    # (3) M2 unbounded for kink
    dx = XS[1] - XS[0]
    d2_abs = np.gradient(np.gradient(np.abs(XS), dx), dx)
    m2_kink = float(np.abs(d2_abs).max())

    result = {
        "date": "2026-08-03",
        "theorem": "T-III necessity first lattice",
        "single_kink_refutation": {
            "|x|, 4 nodes at +-eps": single,
            "per-function error arbitrarily small (known kink)": True},
        "class_width": {
            "minimax over shifted-kink class, uniform N-point grid":
                widths,
            "slope_approx": round(float(slope), 2),
            "C1-class width ~ 1/N": bool(abs(slope - 1.0) < 0.2)},
        "certificate_exclusion": {
            "sup|f''| of |x| (numeric)": m2_kink,
            "unbounded (Dirac)": m2_kink > 1e3,
            "no finite DA certificate": True},
        "verdict": "PASS" if (single["0.001"] < 0.01
                              and abs(slope - 1.0) < 0.2
                              and m2_kink > 1e3) else "CHECK",
    }
    print(json.dumps(result, indent=2))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
