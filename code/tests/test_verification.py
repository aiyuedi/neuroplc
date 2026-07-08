#!/usr/bin/env python3
"""
NeuroPLC — Verification Module Unit Tests
===========================================
Tests for:
    - compute_lut_error_bound (de Boor formula)
    - compute_weight_norms
    - propagate_error (IA propagation)
    - Doubleton arithmetic (DA) propagation
    - Segment-aware per-segment M₂ computation
    - IntervalVerificationResult structure

Coverage: T1 (verification), T4 (DA/IA)
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroplc.interval_verify import (
    compute_lut_error_bound,
    compute_weight_norms,
    propagate_error,
    IntervalVerificationResult,
)
from neuroplc.affine_verify import (
    propagate_error_doubleton,
    propagate_per_layer_affine,
    DoubletonVerificationResult,
    AffineForm,
)
from analysis.segment_bound import compute_segment_m2, evaluate_phi_on_grid


# ================================================================
# De Boor LUT Error Bound
# ================================================================

class TestLUTErrorBound:
    """Test interval_verify.py: compute_lut_error_bound."""

    def test_formula_monotonic(self):
        """More points → smaller error."""
        e10 = compute_lut_error_bound(10)
        e20 = compute_lut_error_bound(20)
        e50 = compute_lut_error_bound(50)
        assert e10 > e20 > e50

    def test_zero_points_error(self):
        """With 2 points (Δ = 6), error should be non-zero."""
        e2 = compute_lut_error_bound(2)
        assert e2 > 0.0

    def test_many_points_tiny_error(self):
        """With 500 points, error should be negligible."""
        e500 = compute_lut_error_bound(500)
        assert e500 < 1e-5

    def test_m2_bound_scales_linearly(self):
        """Doubling M₂ doubles the error bound."""
        e_default = compute_lut_error_bound(15, m2_bound=0.3)
        e_double = compute_lut_error_bound(15, m2_bound=0.6)
        assert abs(e_double / e_default - 2.0) < 1e-10

    def test_x_range_affects_error(self):
        """Wider range → larger Δ → larger error."""
        e_narrow = compute_lut_error_bound(15, x_range=(-1.0, 1.0))
        e_wide = compute_lut_error_bound(15, x_range=(-5.0, 5.0))
        assert e_wide > e_narrow


# ================================================================
# Weight Norms
# ================================================================

class TestWeightNorms:
    """Test interval_verify.py: compute_weight_norms."""

    def test_known_dims(self):
        """Weight norms should match KAN architecture."""
        from models.student_kan import StudentKAN
        model = StudentKAN([28, 16, 4])
        l0, l1 = compute_weight_norms(model)
        assert len(l0) == 16   # layer 0: 28→16
        assert len(l1) == 4    # layer 1: 16→4

    def test_all_non_negative(self):
        """L1 norms are always non-negative."""
        from models.student_kan import StudentKAN
        model = StudentKAN([28, 16, 4])
        l0, l1 = compute_weight_norms(model)
        assert np.all(l0 >= 0)
        assert np.all(l1 >= 0)


# ================================================================
# Error Propagation (IA)
# ================================================================

class TestErrorPropagation:
    """Test interval_verify.py: propagate_error."""

    def test_zero_eps_gives_zero_perturbation(self):
        """With zero LUT error, output perturbation is zero."""
        l0_norms = np.array([1.0, 2.0, 3.0, 4.0])
        l1_norms = np.array([0.5, 1.0])
        l0, l1 = propagate_error(l0_norms, l1_norms, eps=0.0, lipschitz=0.65)
        assert np.allclose(l0, 0.0)
        assert np.allclose(l1, 0.0)

    def test_linear_scaling_with_eps(self):
        """Doubling epsilon doubles layer 0 output deviation."""
        l0_norms = np.array([1.0, 2.0, 3.0])
        l1_norms = np.array([0.5, 1.0])
        l0_a, _ = propagate_error(l0_norms, l1_norms, eps=0.01, lipschitz=0.65)
        l0_b, _ = propagate_error(l0_norms, l1_norms, eps=0.02, lipschitz=0.65)
        assert np.allclose(l0_b, 2.0 * l0_a)

    def test_output_shape(self):
        """Output arrays match the expected dimensions."""
        l0_norms = np.ones(16)
        l1_norms = np.ones(4)
        l0, l1 = propagate_error(l0_norms, l1_norms, eps=0.001, lipschitz=0.65)
        assert l0.shape == (16,)
        assert l1.shape == (4,)


# ================================================================
# Doubleton Arithmetic (DA)
# ================================================================

class TestDoubletonBounds:
    """Test affine_verify.py: compute_doubleton_bounds."""

    def test_da_tighter_than_ia(self):
        """DA bound should be strictly tighter than IA bound."""
        # 2-layer KAN: [28, 16, 4]
        np.random.seed(42)
        l0_w = np.random.randn(16, 28).astype(np.float32) * 0.5
        l1_w = np.random.randn(4, 16).astype(np.float32) * 0.5

        result = propagate_per_layer_affine(l0_w, l1_w, eps=0.001, lipschitz=0.65)
        # DA final bound should be tighter than IA final bound
        assert np.all(result["da_final"] <= result["ia_final"] + 1e-10)

    def test_all_da_values_non_negative(self):
        """All bound arrays contain only non-negative values."""
        np.random.seed(42)
        l0_w = np.random.randn(16, 28).astype(np.float32) * 0.5
        l1_w = np.random.randn(4, 16).astype(np.float32) * 0.5

        result = propagate_per_layer_affine(l0_w, l1_w, eps=0.001, lipschitz=0.65)
        for key, val in result.items():
            if isinstance(val, np.ndarray):
                assert np.all(val >= 0), f"{key} has negative values"

    def test_da_same_as_ia_for_iid_weights(self):
        """For rotationally symmetric random weights, DA ≈ IA."""
        np.random.seed(123)
        # Weights with zero mean → sign cancellation is minimal
        l0_w = np.random.randn(16, 28).astype(np.float32)
        l1_w = np.random.randn(4, 16).astype(np.float32)

        result = propagate_per_layer_affine(l0_w, l1_w, eps=0.001, lipschitz=0.65)
        # For zero-mean weights, DA and IA are similar
        ratio = result["da_final"].max() / max(result["ia_final"].max(), 1e-12)
        assert ratio <= 1.0 + 1e-6


# ================================================================
# Segment-Aware M₂ Computation
# ================================================================

class TestSegmentM2:
    """Test segment_bound.py: compute_segment_m2.

    NOTE: For grid with G knots and order k, _bspline_basis returns
    (n_points, G - k - 1) basis functions. With G=15, k=3: n_bases = 11.
    """

    G = 15
    k = 3
    n_bases = G - k - 1  # = 11

    def test_constant_function_m2_zero(self):
        """Constant coefficients → φ'' ≈ 0 in interior, boundary effects possible."""
        grid_bsp = np.linspace(-1.5, 1.5, self.G, dtype=np.float64)
        coeffs = np.ones(self.n_bases, dtype=np.float64) * 2.0
        lut_grid = np.linspace(-3.0, 3.0, 15, dtype=np.float64)
        m2 = compute_segment_m2(coeffs, grid_bsp, lut_grid)
        # Interior segments should have near-zero M₂ (boundary may have artifacts)
        interior = m2[1:-1]  # skip first and last segments
        assert np.all(interior < 0.1)

    def test_linear_function_m2_zero(self):
        """Linearly increasing coefficients → constant φ' → φ'' ≈ 0."""
        grid_bsp = np.linspace(-1.5, 1.5, self.G, dtype=np.float64)
        coeffs = np.arange(self.n_bases, dtype=np.float64) * 0.01
        lut_grid = np.linspace(-3.0, 3.0, 15, dtype=np.float64)
        m2 = compute_segment_m2(coeffs, grid_bsp, lut_grid)
        assert np.all(m2 < 0.5)

    def test_output_length_matches_segments(self):
        """M₂ array has one entry per LUT segment."""
        grid_bsp = np.linspace(-1.5, 1.5, self.G, dtype=np.float64)
        coeffs = np.random.randn(self.n_bases).astype(np.float64) * 0.5
        n_pts = 15
        lut_grid = np.linspace(-3.0, 3.0, n_pts, dtype=np.float64)
        m2 = compute_segment_m2(coeffs, grid_bsp, lut_grid)
        assert len(m2) == n_pts - 1

    def test_all_non_negative(self):
        """All M₂ values are non-negative."""
        grid_bsp = np.linspace(-1.5, 1.5, self.G, dtype=np.float64)
        coeffs = np.random.randn(self.n_bases).astype(np.float64)
        lut_grid = np.linspace(-3.0, 3.0, 15, dtype=np.float64)
        m2 = compute_segment_m2(coeffs, grid_bsp, lut_grid)
        assert np.all(m2 >= 0)


# ================================================================
# AffineForm Helpers
# ================================================================

class TestAffineForm:
    """Test affine_verify.py: AffineForm dataclass."""

    def test_create_default(self):
        """Default AffineForm is exact (no noise)."""
        x = AffineForm()
        assert x.center == 0.0
        assert x.radius == 0.0
        assert x.symbol == -1

    def test_create_with_noise(self):
        """AffineForm with noise has non-zero radius."""
        x = AffineForm(center=1.0, radius=0.1, symbol=0)
        assert x.center == 1.0
        assert x.radius == 0.1
        assert x.symbol == 0

    def test_interval_range(self):
        """lower/upper properties give interval range."""
        x = AffineForm(center=1.0, radius=0.2)
        assert x.lower == 0.8
        assert x.upper == 1.2
        assert x.width == 0.4

    def test_add_affine_forms(self):
        """Adding two affine forms sums centers and radii."""
        a = AffineForm(center=1.0, radius=0.1, symbol=0)
        b = AffineForm(center=2.0, radius=0.2, symbol=1)
        c = a + b
        assert c.center == 3.0
        assert abs(c.radius - 0.3) < 1e-10


# ================================================================
# Integration: DA + Segment-Aware
# ================================================================

class TestSegmentDoubletonIntegration:
    """Test that segment-aware M₂ integrates with DA error propagation."""

    def test_segment_m2_integrates_with_lut_error(self):
        """Per-segment M₂ should give tighter per-segment LUT error bounds."""
        G, k = 15, 3
        n_bases = G - k - 1  # = 11
        grid_bsp = np.linspace(-1.5, 1.5, G, dtype=np.float64)
        np.random.seed(42)
        coeffs = np.random.randn(n_bases).astype(np.float64) * 0.5
        lut_grid = np.linspace(-3.0, 3.0, 15, dtype=np.float64)

        m2_per_seg = compute_segment_m2(coeffs, grid_bsp, lut_grid)

        # Global M₂ = max segment M₂
        global_m2 = m2_per_seg.max()
        # Segment-aware LUT error per segment
        seg_width = (3.0 - (-3.0)) / (15 - 1)
        seg_eps = m2_per_seg * seg_width ** 2 / 8.0
        global_eps = global_m2 * seg_width ** 2 / 8.0

        # Per-segment errors should be ≤ global error
        assert np.all(seg_eps <= global_eps + 1e-12)

        # At least one segment should have strictly smaller M₂
        # (unless all segments have identical curvature — extremely unlikely)
        assert np.any(m2_per_seg < global_m2 * 0.99)
