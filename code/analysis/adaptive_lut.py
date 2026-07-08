#!/usr/bin/env python3
"""
Adaptive Mixed-Precision LUT Density Allocation for KAN B-Spline Functions
===========================================================================
Greedy resource allocation: given a total LUT storage budget B (bytes),
assign different numbers of LUT points N_{o,i} to each of the 576 B-spline
activation functions, minimizing the worst-case per-function error.

Problem formulation:
    min   max_{o,i} ε(φ_{o,i}, N_{o,i})
    s.t.  Σ_{o,i} N_{o,i} · 4 ≤ B              (4 bytes per REAL)
          N_{o,i} ≥ 2, integer

    where ε(φ, N) = M₂(φ) · (b-a)² / [8 · (N-1)²]   (de Boor bound)

Algorithm: Greedy incremental allocation
    1. Start: ∀(o,i), N_{o,i} = 3
    2. Compute ε(o,i) = M₂(o,i) · 4.5 / (N-1)²
    3. While Σ N_{o,i} · 4 < B:
       a. (o*, i*) = argmax ε(o,i)
       b. N_{o*,i*} += 1
       c. Recompute ε(o*,i*)
    4. Return N allocation

Properties:
    - Monotone: ε(N) strictly decreases with N
    - Convex marginal benefit: ε(N)-ε(N+1) = M₂·4.5·(1/(N-1)² - 1/N²)
    - Greedy achieves optimal worst-case ε for convex descending cost functions
    - Near-optimal for general case (within factor of 2)

Expected result at budget = 34,560 bytes (uniform N=15):
    - Worst-case ε: reduced by 40-60%
    - Mean ε: reduced by 35-50%
    - Storage distribution: 5-40 points per function (vs uniform 15)
    - "Flat" functions drop to 3-5 points, "wiggly" ones get 25-40

Compilation integration:
    Functions are grouped by N value; each group shares one grid array.
    Groups with same N can be compiled using the same binary-search grid.
    Groups with different N need separate grid arrays (small overhead).

Usage:
    python adaptive_lut.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import torch
import json
import heapq
from typing import Optional
from dataclasses import dataclass, field

from models.student_kan import StudentKAN

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Per-function M₂ estimation
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_function_m2(model) -> np.ndarray:
    """
    Compute global M₂ = max|φ''(x)| for each of the 576 activation functions.

    Uses dense sampling (500 pts) + central differences for accurate estimation.
    This is a one-time offline computation; the result is stored for allocation.

    Returns:
        m2_values: (n_functions,) — per-function M₂ values
        metadata: list of dicts with (layer_idx, out_idx, in_idx) per function
    """
    from models.student_kan import _bspline_basis

    xs_dense = np.linspace(-3.0, 3.0, 500, dtype=np.float64)
    xs_bsp = xs_dense / 3.0  # map to B-spline domain
    xs_t = torch.from_numpy(xs_bsp).float()

    m2_values = []
    metadata = []

    for l_idx, layer in enumerate(model.kan_layers):
        spline_w = layer.spline_weight.detach().cpu().numpy()  # (out, in, n_bases)
        grid = layer.grid.detach().cpu().float()
        out_d, in_d, n_bases = spline_w.shape

        # Precompute basis (same for all (o,i) in this layer at this x-grid)
        basis_3 = _bspline_basis(xs_t, grid, k=3).double().numpy()  # (500, n_bases)

        for o in range(out_d):
            for i in range(in_d):
                coeffs = spline_w[o, i, :].astype(np.float64)
                phi = basis_3 @ coeffs  # (500,)

                # Second derivative via gradient
                dx = xs_dense[1] - xs_dense[0]
                d1 = np.gradient(phi, dx)
                d2 = np.gradient(d1, dx)
                m2 = float(np.max(np.abs(d2)))

                m2_values.append(m2)
                metadata.append({"layer": l_idx, "out": o, "in": i})

    return np.array(m2_values), metadata


# ─────────────────────────────────────────────────────────────────────────────
# Greedy allocation algorithm
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AllocationResult:
    """Result of adaptive LUT density allocation."""

    budget_bytes: int = 0
    n_functions: int = 0

    # Per-function allocation
    n_allocated: np.ndarray = field(default_factory=lambda: np.array([]))
    m2_values: np.ndarray = field(default_factory=lambda: np.array([]))

    # Error statistics
    worst_eps_uniform: float = 0.0      # worst ε with uniform N
    worst_eps_adaptive: float = 0.0     # worst ε with adaptive allocation
    mean_eps_uniform: float = 0.0
    mean_eps_adaptive: float = 0.0
    eps_reduction_worst: float = 0.0    # % reduction in worst-case ε
    eps_reduction_mean: float = 0.0

    # Storage statistics
    storage_uniform: int = 0
    storage_adaptive: int = 0
    storage_saving_pct: float = 0.0

    # Distribution
    n_distribution: dict = field(default_factory=dict)  # {N: count}


def greedy_allocate(
    m2_values: np.ndarray,
    budget_bytes: int,
    n_min: int = 3,
    n_max: int = 100,
    domain_width: float = 6.0,
) -> AllocationResult:
    """
    Greedy incremental allocation minimizing worst-case per-function LUT error.

    ε(φ, N) = M₂(φ) · (domain_width)² / [8 · (N-1)²]

    At each step, increment N for the function with currently largest ε.
    Uses a max-heap for O(log K) per increment.

    Args:
        m2_values:    (K,) — per-function M₂ = max|φ''|
        budget_bytes: total storage budget in bytes
        n_min:        minimum LUT points per function
        n_max:        maximum LUT points per function
        domain_width: b-a = 6 for [-3,3]

    Returns:
        AllocationResult
    """
    K = len(m2_values)
    result = AllocationResult()
    result.budget_bytes = budget_bytes
    result.n_functions = K
    result.m2_values = m2_values

    # Scale factor: ε(N) = M₂ · W² / [8 · (N-1)²]
    W = domain_width
    scale = W * W / 8.0  # = 36/8 = 4.5 for [-3,3]

    # Initialize: N = n_min for all functions
    n_current = np.full(K, n_min, dtype=np.int32)

    # Compute initial ε values
    # ε = M₂ · scale / (N-1)²
    eps_current = m2_values * scale / ((n_current - 1) ** 2)

    # Storage used
    storage_used = K * n_min * 4  # 4 bytes per REAL

    # Max-heap: store (-ε, idx) for max extraction
    heap = [(-float(eps_current[i]), i) for i in range(K)]
    heapq.heapify(heap)

    # ── Uniform benchmark (all same N) ──
    n_uniform = budget_bytes // (K * 4)  # floor division
    n_uniform = max(n_min, min(n_uniform, n_max))
    eps_uniform = m2_values * scale / ((n_uniform - 1) ** 2)
    result.worst_eps_uniform = float(eps_uniform.max())
    result.mean_eps_uniform = float(eps_uniform.mean())
    result.storage_uniform = K * n_uniform * 4

    # ── Greedy allocation ──
    step = 0
    while storage_used + 4 <= budget_bytes:
        # Extract function with max ε
        if not heap:
            break
        neg_eps, idx = heapq.heappop(heap)

        # Skip stale entries (ε may have been updated)
        current_eps = float(eps_current[idx])
        if abs(-neg_eps - current_eps) > 1e-12:
            heapq.heappush(heap, (-current_eps, idx))
            continue

        # Increment N for this function
        if n_current[idx] >= n_max:
            continue  # at max, can't increase

        n_current[idx] += 1
        storage_used += 4
        step += 1

        # Recompute ε
        new_eps = m2_values[idx] * scale / ((n_current[idx] - 1) ** 2)
        eps_current[idx] = new_eps
        heapq.heappush(heap, (-new_eps, idx))

    result.n_allocated = n_current
    result.worst_eps_adaptive = float(eps_current.max())
    result.mean_eps_adaptive = float(eps_current.mean())
    result.storage_adaptive = storage_used

    # Reduction statistics
    result.eps_reduction_worst = (
        (1.0 - result.worst_eps_adaptive / max(result.worst_eps_uniform, 1e-15)) * 100)
    result.eps_reduction_mean = (
        (1.0 - result.mean_eps_adaptive / max(result.mean_eps_uniform, 1e-15)) * 100)

    # Storage saving: what storage would be needed for same worst ε with uniform?
    # uniform ε = worst_M₂ · scale / (N-1)² ⇒ N = 1 + sqrt(worst_M₂ · scale / ε)
    if result.worst_eps_adaptive > 0:
        n_needed = 1 + int(np.sqrt(m2_values.max() * scale / result.worst_eps_adaptive))
        storage_equiv_uniform = K * n_needed * 4
        result.storage_saving_pct = (
            (1.0 - storage_used / max(storage_equiv_uniform, 1)) * 100)

    # Distribution
    unique, counts = np.unique(n_current, return_counts=True)
    result.n_distribution = {int(u): int(c) for u, c in zip(unique, counts)}

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  Adaptive Mixed-Precision LUT Density Allocation")
    print("=" * 70)

    BASE = os.path.join(os.path.dirname(__file__), "..")

    # Load trained KAN
    kan = StudentKAN([28, 16, 4])
    ckpt = torch.load(
        os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt"),
        map_location="cpu", weights_only=True)
    kan.load_state_dict(ckpt["student_state_dict"])
    kan.eval()
    print("\nKAN [28,16,4] loaded OK\n")

    # Compute per-function M₂
    print("Computing per-function M2 values...")
    m2_values, metadata = compute_per_function_m2(kan)

    n_total = len(m2_values)
    print(f"  Functions: {n_total}")
    print(f"  M2 statistics: min={m2_values.min():.4f}, median={np.median(m2_values):.4f}, "
          f"max={m2_values.max():.4f}, mean={m2_values.mean():.4f}")
    print(f"  M2 percentiles: P10={np.percentile(m2_values, 10):.4f}, "
          f"P25={np.percentile(m2_values, 25):.4f}, "
          f"P75={np.percentile(m2_values, 75):.4f}, "
          f"P90={np.percentile(m2_values, 90):.4f}")

    # ── Run allocation at multiple budgets ──
    budgets = [
        ("S7-1200 tight",  n_total * 10 * 4),   # N=10 uniform → 23,040 bytes
        ("S7-1200 default", n_total * 15 * 4),   # N=15 uniform → 34,560 bytes
        ("S7-1200 generous", n_total * 20 * 4),  # N=20 uniform → 46,080 bytes
        ("S7-1500",         n_total * 50 * 4),   # N=50 uniform → 115,200 bytes
    ]

    all_results = {}
    for label, budget in budgets:
        print(f"\n─── Budget: {label} ({budget:,} bytes, {n_total} fns × "
              f"{budget // (n_total * 4)} pts uniform) ───")

        result = greedy_allocate(m2_values, budget)
        all_results[label] = result

        print(f"  Uniform N={budget // (n_total * 4)} pts:")
        print(f"    Worst ε:     {result.worst_eps_uniform:.6f}")
        print(f"    Mean ε:      {result.mean_eps_uniform:.6f}")
        print(f"    Storage:     {result.storage_uniform:,} bytes")

        print(f"  Adaptive allocation:")
        print(f"    Worst ε:     {result.worst_eps_adaptive:.6f}")
        print(f"    Mean ε:      {result.mean_eps_adaptive:.6f}")
        print(f"    Reduction:   worst {result.eps_reduction_worst:.1f}%, "
              f"mean {result.eps_reduction_mean:.1f}%")
        print(f"    Storage:     {result.storage_adaptive:,} bytes "
              f"(saving {result.storage_saving_pct:.1f}% vs equiv uniform)")

        # N distribution
        n_min_used = result.n_allocated.min()
        n_max_used = result.n_allocated.max()
        n_mean = result.n_allocated.mean()
        print(f"    N range:     [{n_min_used}, {n_max_used}], mean={n_mean:.1f}")
        print(f"    N quartiles: P25={np.percentile(result.n_allocated, 25):.0f}, "
              f"P50={np.percentile(result.n_allocated, 50):.0f}, "
              f"P75={np.percentile(result.n_allocated, 75):.0f}")

        # Top-5 distribution buckets
        dist = result.n_distribution
        top_buckets = sorted(dist.items(), key=lambda x: x[1], reverse=True)[:8]
        buckets_str = ", ".join(f"N={n}:{c}" for n, c in top_buckets)
        print(f"    Distribution: {buckets_str}")

    # ── Find "quality parity" budget ──
    print(f"\n--- Storage-Precision Trade-off ---")
    # How much storage does adaptive need to match uniform N=15's worst ε?
    uniform_15 = all_results["S7-1200 default"]
    target_eps = uniform_15.worst_eps_uniform

    # Binary search on budget
    lo, hi = n_total * 5 * 4, n_total * 15 * 4
    for _ in range(20):
        mid = (lo + hi) // 2
        r = greedy_allocate(m2_values, mid)
        if r.worst_eps_adaptive <= target_eps:
            hi = mid
        else:
            lo = mid

    quality_parity_budget = hi
    quality_parity_n_mean = quality_parity_budget / (n_total * 4)
    saving_pct = (1.0 - quality_parity_budget / uniform_15.storage_uniform) * 100
    print(f"  To match uniform N=15 worst eps ({target_eps:.6f}):")
    print(f"    Adaptive needs: {quality_parity_budget:,} bytes "
          f"(avg {quality_parity_n_mean:.1f} pts/fn)")
    print(f"    Storage saving: {saving_pct:.1f}% vs uniform N=15")

    # ── Save results ──
    output = {
        "m2_statistics": {
            "min": float(m2_values.min()),
            "max": float(m2_values.max()),
            "mean": float(m2_values.mean()),
            "median": float(np.median(m2_values)),
            "p10": float(np.percentile(m2_values, 10)),
            "p25": float(np.percentile(m2_values, 25)),
            "p75": float(np.percentile(m2_values, 75)),
            "p90": float(np.percentile(m2_values, 90)),
        },
        "num_functions": n_total,
        "allocations": {},
        "quality_parity": {
            "target_eps_uniform_n15": float(target_eps),
            "adaptive_budget_bytes": quality_parity_budget,
            "adaptive_mean_pts": float(quality_parity_n_mean),
            "storage_saving_pct": float(saving_pct),
        },
    }

    for label, r in all_results.items():
        output["allocations"][label] = {
            "budget_bytes": r.budget_bytes,
            "n_uniform": r.storage_uniform // (r.n_functions * 4),
            "uniform_worst_eps": float(r.worst_eps_uniform),
            "uniform_mean_eps": float(r.mean_eps_uniform),
            "adaptive_worst_eps": float(r.worst_eps_adaptive),
            "adaptive_mean_eps": float(r.mean_eps_adaptive),
            "eps_reduction_worst_pct": float(r.eps_reduction_worst),
            "eps_reduction_mean_pct": float(r.eps_reduction_mean),
            "storage_saving_pct": float(r.storage_saving_pct),
            "n_range": [int(r.n_allocated.min()), int(r.n_allocated.max())],
            "n_mean": float(r.n_allocated.mean()),
            "n_distribution": {str(k): v for k, v in r.n_distribution.items()},
        }

    json_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "adaptive_lut.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n[OK] Saved: {json_path}")

    print(f"\n[OK] Adaptive LUT allocation analysis complete")
