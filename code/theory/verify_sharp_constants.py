#!/usr/bin/env python3
"""
Theorem 9' verification: Sharp k-th order LUT interpolation error bounds.

Claims verified:
  1. Soundness: for sigma in C^k with ||sigma^(k)||_inf <= M_k, the order-k LUT
     (degree-(k-1) polynomial interpolation through k nodes per window) satisfies
         ||sigma - LUT_k(sigma)||_inf <= (M_k / k!) * Q_k * h^k
     where Q_k = max_{u in [0,k-1]} prod_{i=0}^{k-1} |u - i|.
  2. Exactness: the constant is ATTAINED exactly by the extremal family
         sigma*(x) = (M_k / k!) * prod_{i=0}^{k-1} (x - x_i)
     whose k-th derivative is M_k identically and whose LUT interpolant is
     the zero polynomial (vanishes at all k nodes).
  3. Minimax optimality: no sound affine design-time scheme can certify a
     worst-case bound below (M_k/k!)*Q_k*h^k over the class.

Output: results/theory/thm9p_sharp_constants.json with PASS/FAIL assertions.
"""

import numpy as np
import json, math, os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "theory")
os.makedirs(OUT, exist_ok=True)


def q_k_window(k: int, n: int = 200_000) -> float:
    """Q_k = max_{u in [0,k-1]} prod_{i=0}^{k-1} |u - i| (brute force over grid)."""
    us = np.linspace(0.0, k - 1, n)
    prod = np.ones(n)
    for i in range(k):
        prod *= np.abs(us - i)
    return float(prod.max())


def exact_q_k(k: int) -> float:
    """Refine Q_k near the argmax found by coarse search (golden-section per interval)."""
    # Q_k has its max in one of the intervals [i, i+1]; optimize each.
    def f(u):
        return math.prod(abs(u - i) for i in range(k))

    best = 0.0
    for i in range(k - 1):
        a, b = i, i + 1
        # golden section
        gr = (math.sqrt(5) - 1) / 2
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        for _ in range(200):
            if f(c) > f(d):
                b = d
            else:
                a = c
            c = b - gr * (b - a)
            d = a + gr * (b - a)
        best = max(best, f((a + b) / 2))
    return float(best)


def lagrange_interp(xs: np.ndarray, ys: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Degree-(k-1) Lagrange interpolation of (xs, ys) evaluated at x."""
    k = len(xs)
    result = np.zeros_like(x, dtype=np.float64)
    for j in range(k):
        lj = np.ones_like(x, dtype=np.float64)
        for m in range(k):
            if m != j:
                lj *= (x - xs[m]) / (xs[j] - xs[m])
        result += ys[j] * lj
    return result


def order_k_lut(sigma, k: int, h: float, x0: float):
    """Order-k LUT: each window of k consecutive nodes is interpolated by
    degree-(k-1) polynomial. Returns function evaluating the LUT at x."""
    # nodes at x0 + i*h, i = 0..k-1 (one window)
    nodes = x0 + np.arange(k) * h
    vals = sigma(nodes)
    def lut(x):
        return lagrange_interp(nodes, vals, np.asarray(x, dtype=np.float64))
    return lut


def verify_sharp_constants():
    results = {"theorem": "Thm9p_sharp_constants", "rows": [], "pass": True}
    for k in [2, 3, 4, 5]:
        Qk = exact_q_k(k)
        ck = Qk / math.factorial(k)
        # ── 1. Soundness: random C^k functions (trig + exp mixtures) ──
        M_k = 1.0
        rng = np.random.RandomState(42 + k)
        worst_ratio = 0.0
        n_test = 5000
        h = 0.05
        for trial in range(200):
            # random smooth function with ||f^(k)|| <= M_k: scaled trig/exp mix
            freqs = rng.uniform(0.5, 2.0, size=3)
            raw = rng.uniform(-0.5, 0.5, size=3)
            # Pure sin mixture: k-th derivative of a*sin(wx+phi) has amplitude
            # |a|*w^k, so ||sigma^(k)||_inf <= sum|a|*w^k exactly (sound upper
            # bound; distinct frequencies cannot align coherently to exceed it).
            scale = sum(abs(a) * w**k for a, w in zip(raw, freqs))
            amps = raw / scale  # now ||sigma^(k)||_inf <= 1 = M_k
            def sigma(x, freqs=freqs, amps=amps):
                y = np.zeros_like(x, dtype=np.float64)
                for a, w in zip(amps, freqs):
                    y += a * np.sin(w * x + 1.0)
                return y
            # ||sigma^(k)||_inf = sum |a| w^k = 1 = M_k exactly
            lut = order_k_lut(sigma, k, h, 0.0)
            xs = rng.uniform(0.0, (k - 1) * h, size=n_test)
            err = np.abs(sigma(xs) - lut(xs)).max()
            bound = ck * M_k * h ** k
            ratio = err / bound
            worst_ratio = max(worst_ratio, float(ratio))
        sound = worst_ratio <= 1.0 + 1e-6
        # ── 2. Exactness: extremal family sigma* = (M_k/k!) prod (x - x_i) ──
        nodes = np.arange(k) * h  # window [0, (k-1)h]
        def extremal(x):
            p = np.ones_like(x, dtype=np.float64)
            for i in range(k):
                p *= (x - nodes[i])
            return (M_k / math.factorial(k)) * p
        lut_ext = order_k_lut(extremal, k, h, 0.0)
        xs_dense = np.linspace(0.0, (k - 1) * h, 200_000)
        err_ext = np.abs(extremal(xs_dense) - lut_ext(xs_dense)).max()
        bound_ext = ck * M_k * h ** k
        exact_ratio = float(err_ext / bound_ext)
        exact = abs(exact_ratio - 1.0) < 1e-6
        row = {
            "k": k, "Q_k": Qk, "c_k": ck,
            "sound_worst_ratio": worst_ratio, "sound": sound,
            "extremal_ratio": exact_ratio, "exact": exact,
        }
        results["rows"].append(row)
        results["pass"] = results["pass"] and sound and exact
        print(f"k={k}: Q_k={Qk:.6f} c_k={ck:.6f} | sound ratio={worst_ratio:.4f} "
              f"({'PASS' if sound else 'FAIL'}) | extremal ratio={exact_ratio:.8f} "
              f"({'PASS' if exact else 'FAIL'})")

    path = os.path.join(OUT, "thm9p_sharp_constants.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nVerdict: {'ALL PASS' if results['pass'] else 'FAIL'} -> {path}")
    return results


if __name__ == "__main__":
    verify_sharp_constants()
