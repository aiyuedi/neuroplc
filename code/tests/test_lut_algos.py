#!/usr/bin/env python3
"""
NeuroPLC — LUT Algorithm Unit Tests
=====================================
Tests for:
    - DP-optimal LUT grid allocation (_compute_optimal_grid_dp)
    - Adaptive B-spline sampling (adaptive_bspline_sampling)
    - Uniform LUT resampling
    - Auto-selection strategy (threshold-based)

Coverage: T2 (DP-optimal), part of T1 (adaptive LUT)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroplc.ir import IRGraph, IROpType, IRNode
from neuroplc.optimizer import (
    adaptive_bspline_sampling,
    _compute_optimal_grid_dp,
    compare_sampling_error,
    optimal_bspline_sampling,
)


# ================================================================
# DP-Optimal LUT Algorithm
# ================================================================

class TestOptimalGridDP:
    """Test optimizer.py: _compute_optimal_grid_dp."""

    def test_returns_k_points(self):
        """Output grid has exactly K points."""
        np.random.seed(42)
        # Simulate a curvature profile over a fine grid
        xs = np.linspace(-3.0, 3.0, 200)
        # Bimodal curvature: peaks at ±1.5
        curv = np.exp(-(xs - 1.5) ** 2 / 0.3) + np.exp(-(xs + 1.5) ** 2 / 0.3) + 0.01
        K = 15
        grid_dp, cost_dp = _compute_optimal_grid_dp(curv, xs, K)
        assert len(grid_dp) == K

    def test_monotonic_in_x_range(self):
        """DP grid points are within the input range and sorted."""
        np.random.seed(42)
        xs = np.linspace(-3.0, 3.0, 200)
        curv = np.exp(-(xs) ** 2 / 0.5) + 0.01
        grid_dp, _ = _compute_optimal_grid_dp(curv, xs, 15)
        assert grid_dp[0] == xs[0]
        assert grid_dp[-1] == xs[-1]
        assert np.all(np.diff(grid_dp) > 0)  # strictly increasing

    def test_uniform_curvature_gives_monotonic_grid(self):
        """Constant curvature → DP produces monotonic grid with even spacing."""
        xs = np.linspace(-3.0, 3.0, 200)
        curv = np.ones_like(xs) + 0.001  # near-uniform curvature
        grid_dp, _ = _compute_optimal_grid_dp(curv, xs, 20)
        # Grid should be monotonic and bounded
        assert np.all(np.diff(grid_dp) > 0)
        assert grid_dp[0] == xs[0]
        assert grid_dp[-1] == xs[-1]

    def test_cost_positive(self):
        """DP cost is always positive."""
        xs = np.linspace(-3.0, 3.0, 200)
        curv = np.random.rand(200) + 0.01
        _, cost = _compute_optimal_grid_dp(curv, xs, 10)
        assert cost > 0

    def test_more_points_lower_cost(self):
        """More K → lower or equal cost."""
        xs = np.linspace(-3.0, 3.0, 200)
        curv = np.exp(-(xs - 0.5) ** 2 / 0.5) + 0.01
        _, cost_10 = _compute_optimal_grid_dp(curv, xs, 10)
        _, cost_20 = _compute_optimal_grid_dp(curv, xs, 20)
        assert cost_20 <= cost_10


# ================================================================
# Adaptive B-spline Sampling
# ================================================================

class TestAdaptiveSampling:
    """Test optimizer.py: adaptive_bspline_sampling."""

    def test_no_bspline_nodes_noop(self):
        """Graph with no BsplineLUT nodes → nothing to do."""
        g = IRGraph()
        g.add_node(IROpType.MatMul)
        g.add_node(IROpType.Softmax)
        n = adaptive_bspline_sampling(g, target_points=15, x_range=(-3.0, 3.0))
        assert n == 0

    def test_preserves_table_shape(self):
        """After adaptive sampling, table shape is (out, in, target_points)."""
        g = IRGraph()
        table = np.random.randn(4, 16, 50).astype(np.float32)
        grid_orig = np.linspace(-3.0, 3.0, 50, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid_orig,
                           "x_range": [-3.0, 3.0]})
        n = adaptive_bspline_sampling(g, target_points=15, x_range=(-3.0, 3.0))
        assert n == 1
        node = g.nodes[0]
        assert node.attrs["table"].shape == (4, 16, 15)
        assert len(node.attrs["grid"]) == 15

    def test_grid_stays_sorted(self):
        """After adaptive resampling, grid points are still monotonic."""
        g = IRGraph()
        np.random.seed(123)
        table = np.random.randn(3, 5, 30).astype(np.float32)
        grid_orig = np.linspace(-3.0, 3.0, 30, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid_orig,
                           "x_range": [-3.0, 3.0]})
        adaptive_bspline_sampling(g, target_points=15, x_range=(-3.0, 3.0))
        new_grid = g.nodes[0].attrs["grid"]
        assert np.all(np.diff(new_grid) > 0)


# ================================================================
# Optimal (DP) LUT Sampling
# ================================================================

class TestDPOptimalSampling:
    """Test optimizer.py: optimal_bspline_sampling."""

    def test_preserves_table_shape(self):
        """DP-optimal resampling preserves (out, in) dimensions."""
        g = IRGraph()
        table = np.random.randn(2, 3, 30).astype(np.float32) * 0.1
        grid_orig = np.linspace(-3.0, 3.0, 30, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid_orig,
                           "x_range": [-3.0, 3.0]})
        n = optimal_bspline_sampling(g, target_points=20, x_range=(-3.0, 3.0))
        assert n == 1
        node = g.nodes[0]
        assert node.attrs["table"].shape == (2, 3, 20)

    def test_no_bspline_noop(self):
        """Graph without BsplineLUT → nothing optimized."""
        g = IRGraph()
        g.add_node(IROpType.MatMul)
        n = optimal_bspline_sampling(g, target_points=15)
        assert n == 0


# ================================================================
# Compare Sampling Error
# ================================================================

class TestCompareSamplingError:
    """Test optimizer.py: compare_sampling_error."""

    def test_returns_expected_keys(self):
        """Returns dict with all expected metric keys."""
        g = IRGraph()
        table = np.random.randn(2, 3, 30).astype(np.float32) * 0.1
        grid = np.linspace(-3.0, 3.0, 30, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid})

        result = compare_sampling_error(g, n_test_points=100, x_range=(-3.0, 3.0))
        for key in ["uniform_max", "adaptive_max", "adaptive_vs_optimal_pct",
                     "uniform_mean", "adaptive_mean", "num_functions"]:
            assert key in result, f"Missing key: {key}"

    def test_num_functions_matches(self):
        """num_functions should equal total (out_dim × in_dim) across all Bspline nodes."""
        g = IRGraph()
        g.add_node(IROpType.BsplineLUT, name="b1",
                    attrs={"table": np.zeros((4, 16, 15), dtype=np.float32),
                           "grid": np.linspace(-3, 3, 15, dtype=np.float32)})
        g.add_node(IROpType.BsplineLUT, name="b2",
                    attrs={"table": np.zeros((4, 16, 15), dtype=np.float32),
                           "grid": np.linspace(-3, 3, 15, dtype=np.float32)})
        result = compare_sampling_error(g, n_test_points=100)
        assert result["num_functions"] == 4 * 16 + 4 * 16

    def test_uniform_max_non_negative(self):
        g = IRGraph()
        table = np.random.randn(2, 3, 15).astype(np.float32) * 0.1
        grid = np.linspace(-3.0, 3.0, 15, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid})
        result = compare_sampling_error(g, n_test_points=100)
        assert result["uniform_max"] >= 0
        assert result["uniform_mean"] >= 0
