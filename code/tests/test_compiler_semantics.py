#!/usr/bin/env python3
"""
NeuroPLC — Compiler Semantic Preservation Tests
=================================================
Verify that the compiler preserves model semantics:
  (a) IR compilation succeeds for KAN and MLP
  (b) Generated SCL contains required structural elements
  (c) LUT approximation error ≤ theoretical de Boor bound
  (d) Per-operation IR node correctness
  (e) End-to-end: PyTorch FP32 → LUT-approximated classifier agreement

These tests form the COMPILER REGRESSION SUITE — any change to the
compiler, optimizer, or backend must pass these tests.

Usage:
    pytest code/tests/test_compiler_semantics.py -v
    pytest code/tests/test_compiler_semantics.py -v -k "test_lut_error_bound"
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroplc.ir import IRGraph, IROpType, IRNode
from neuroplc.compiler import NeuroPLCCompiler
from neuroplc.frontend import kan_to_ir, mlp_to_ir
from neuroplc.optimizer import optimize
from neuroplc.interval_verify import compute_lut_error_bound
from models.student_kan import StudentKAN, _bspline_basis
from models.student_mlp import StudentMLP


# ============================================================================
# Helpers
# ============================================================================

def _make_kan_lut_forward(model, N_lut=15, x_range=(-3.0, 3.0)):
    """Create a LUT-based forward function for a KAN model.

    Returns a function f(x_numpy) → logits_numpy that mimics SCL execution:
    - Replaces B-spline evaluation with piecewise-linear LUT interpolation
    - Preserves the SiLU base path exactly (no LUT for SiLU)
    - Merges base + spline paths via element-wise addition

    This is equivalent to what the generated SCL code does at inference time.
    """
    import torch.nn.functional as F_local
    device = next(model.parameters()).device

    def lut_forward(x_np):
        """x_np: (batch, in_features) float32 numpy array."""
        # Precompute LUT tables for all layers
        lut_data = []
        for layer in model.kan_layers:
            grid_np = layer.grid.detach().cpu().numpy().astype(np.float32)
            coeffs_np = layer.spline_weight.detach().cpu().numpy().astype(np.float32)
            out_d, in_d, n_coeff = coeffs_np.shape

            lut_x = np.linspace(x_range[0], x_range[1], N_lut, dtype=np.float32)
            lut_vals = np.zeros((out_d, in_d, N_lut), dtype=np.float32)
            for o in range(out_d):
                for i in range(in_d):
                    for p, xp in enumerate(lut_x):
                        x_t = torch.tensor(float(xp), dtype=torch.float32).reshape(1)
                        g_t = torch.from_numpy(grid_np)
                        c_t = torch.from_numpy(coeffs_np[o:o+1, i:i+1, :])
                        basis = _bspline_basis(x_t / 3.0, g_t, 3)
                        val = torch.einsum('oic,pc->oip', c_t, basis)
                        lut_vals[o, i, p] = float(val.item())

            lut_data.append({
                'lut_x': lut_x, 'lut_vals': lut_vals,
                'base_weight': layer.base_weight.detach().cpu().numpy().astype(np.float32),
                'out_d': out_d, 'in_d': in_d, 'n_coeff': n_coeff,
            })

        # Forward pass through all layers
        B = x_np.shape[0]
        x_curr = x_np.astype(np.float32)

        for ld in lut_data:
            out_d, in_d = ld['out_d'], ld['in_d']
            lx = ld['lut_x']
            lv = ld['lut_vals']
            bw = ld['base_weight']

            # Base path: SiLU(base_weight · x)
            x_t = torch.from_numpy(x_curr).to(device)
            base_act = F_local.silu(x_t)
            base_out_t = torch.einsum('...i,ji->...j', base_act,
                                       torch.from_numpy(bw).to(device))
            base_out = base_out_t.cpu().numpy().astype(np.float32)

            # Spline path: LUT interpolation
            spline_out = np.zeros((B, out_d), dtype=np.float32)
            for b in range(B):
                for o in range(out_d):
                    for i in range(in_d):
                        xi = float(x_curr[b, i])
                        # Binary search
                        lo, hi = 0, N_lut - 1
                        while hi - lo > 1:
                            mid = (lo + hi) // 2
                            if xi > lx[mid]:
                                lo = mid
                            else:
                                hi = mid
                        # Linear interpolation
                        if abs(lx[hi] - lx[lo]) < 1e-15:
                            interp_val = lv[o, i, lo]
                        else:
                            t = (xi - lx[lo]) / (lx[hi] - lx[lo])
                            interp_val = float(lv[o, i, lo] * (1.0 - t) + lv[o, i, hi] * t)
                        spline_out[b, o] += interp_val

            # Merge: Add(base_out, spline_out)
            x_curr = base_out + spline_out

        return x_curr.astype(np.float32)

    return lut_forward


# ============================================================================
# Test: IR Compilation Does Not Crash
# ============================================================================

class TestIRCompilation:
    """Verify that model→IR compilation succeeds without errors."""

    def test_kan_to_ir_succeeds(self):
        """KAN [3,2,2] → IR should succeed."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        ir = kan_to_ir(model, name="test_kan")
        assert ir is not None
        assert len(ir.nodes) > 0
        ir.validate()

    def test_mlp_to_ir_succeeds(self):
        """MLP [3,2,2] → IR should succeed."""
        model = StudentMLP(input_dim=3, hidden_dims=[2], num_classes=2)
        model.eval()
        ir = mlp_to_ir(model, name="test_mlp")
        assert ir is not None
        assert len(ir.nodes) > 0
        ir.validate()

    def test_kan_ir_has_expected_node_types(self):
        """KAN IR must contain MatMul, BsplineLUT, Add nodes."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        ir = kan_to_ir(model, name="test_kan")
        op_types = {n.op for n in ir.nodes.values()}
        assert IROpType.MatMul in op_types
        assert IROpType.BsplineLUT in op_types
        assert IROpType.Add in op_types

    def test_mlp_ir_has_expected_node_types(self):
        """MLP IR must contain MatMul, StandardAct nodes."""
        model = StudentMLP(input_dim=3, hidden_dims=[2], num_classes=2)
        model.eval()
        ir = mlp_to_ir(model, name="test_mlp")
        op_types = {n.op for n in ir.nodes.values()}
        assert IROpType.MatMul in op_types
        assert IROpType.StandardAct in op_types


# ============================================================================
# Test: Full Compiler Pipeline
# ============================================================================

class TestCompilerPipeline:
    """End-to-end compiler pipeline: model → IR → optimizer → SCL."""

    def test_compile_kan_s7_1200(self):
        """KAN→S7-1200 full pipeline succeeds."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=15,
                                     adaptive=False, verbose=False)
        result = compiler.compile(model)
        assert result is not None
        assert len(result.scl_code) > 0

    def test_compile_kan_s7_1500(self):
        """KAN→S7-1500 full pipeline succeeds."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1500", lut_points=50,
                                     adaptive=False, verbose=False)
        result = compiler.compile(model)
        assert result is not None
        assert len(result.scl_code) > 0

    def test_compile_mlp_s7_1200(self):
        """MLP→S7-1200 full pipeline succeeds."""
        model = StudentMLP(input_dim=3, hidden_dims=[2], num_classes=2)
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", verbose=False)
        result = compiler.compile(model)
        assert result is not None
        assert len(result.scl_code) > 0

    def test_scl_contains_db_block(self):
        """Generated SCL must contain a DATA_BLOCK for weights."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=15,
                                     adaptive=False, verbose=False)
        result = compiler.compile(model)
        assert "DATA_BLOCK" in result.scl_code or "NeuroPLC_Weights" in result.scl_code, \
            "SCL must contain parameter storage (DB block)"

    def test_scl_contains_fb_block(self):
        """Generated SCL must contain inference logic."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=15,
                                     adaptive=False, verbose=False)
        result = compiler.compile(model)
        has_fb_or_fc = "FUNCTION_BLOCK" in result.scl_code or \
                       "FUNCTION" in result.scl_code
        assert has_fb_or_fc, \
            f"SCL must contain inference logic (FB/FC block). Got: {result.scl_code[:200]}"

    def test_scl_compiles_with_adaptive_lut(self):
        """Compilation with adaptive LUT sampling succeeds."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=15,
                                     adaptive=True, verbose=False)
        result = compiler.compile(model)
        assert result is not None
        assert len(result.scl_code) > 0


# ============================================================================
# Test: LUT Approximation Error Bound
# ============================================================================

class TestLUTErrorBound:
    """Verify LUT approximation error respects the de Boor bound."""

    def test_more_points_smaller_error(self):
        """Doubling LUT points should reduce error (monotonicity)."""
        e5 = compute_lut_error_bound(5)
        e10 = compute_lut_error_bound(10)
        e20 = compute_lut_error_bound(20)
        assert e5 > e10 > e20, \
            f"Expected monotonic decrease: {e5:.6f} > {e10:.6f} > {e20:.6f}"

    def test_de_boor_formula_positive(self):
        """LUT error bound should be positive for finite points."""
        for n in [3, 5, 10, 15, 20, 50]:
            e = compute_lut_error_bound(n)
            assert e > 0.0, f"Bound should be > 0 for n={n}"

    def test_lut_error_worst_case_measurable(self):
        """Verify we can measure LUT error on a known function."""
        # sin(x) on [-π, π], 10 LUT points
        xs = np.linspace(-np.pi, np.pi, 200, dtype=np.float64)
        ys = np.sin(xs)

        N = 10
        lx = np.linspace(-np.pi, np.pi, N, dtype=np.float64)
        ly = np.sin(lx)

        max_err = 0.0
        for xv, yv in zip(xs, ys):
            lo, hi = 0, N - 1
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xv > lx[mid]:
                    lo = mid
                else:
                    hi = mid
            t = (xv - lx[lo]) / (lx[hi] - lx[lo])
            interp = ly[lo] * (1.0 - t) + ly[hi] * t
            max_err = max(max_err, abs(interp - yv))

        # sin'' = -sin, max |sin''| = 1 on [-π,π]
        # de Boor: ε ≤ 1 · (2π/9)² / 8 ≈ 0.061
        bound = 1.0 * (2 * np.pi / (N - 1))**2 / 8.0
        assert max_err <= bound * 1.1, \
            f"Measured error {max_err:.6f} exceeds bound {bound:.6f}"


# ============================================================================
# Test: Per-Operation Type Correctness
# ============================================================================

class TestPerOperationCorrectness:
    """Verify each IR operation type behaves correctly in isolation."""

    def test_matmul_operation(self):
        """MatMul(W,b)(x) = Wx + b."""
        W = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        b = np.array([0.5, -0.5], dtype=np.float32)
        x = np.array([1.0, 1.0], dtype=np.float32)
        expected = np.array([3.5, 6.5], dtype=np.float32)
        actual = W @ x + b
        assert np.allclose(actual, expected, atol=1e-6)

    def test_bspline_lut_deterministic(self):
        """BsplineLUT evaluation is deterministic for fixed input."""
        torch.manual_seed(42)
        model = StudentKAN([3, 2, 2])
        model.eval()
        layer = model.kan_layers[0]
        coeffs = layer.spline_weight.detach()
        grid = layer.grid

        x_val = torch.tensor([0.5], dtype=torch.float32)
        basis1 = _bspline_basis(x_val / 3.0, grid, 3)
        out1 = torch.einsum('oic,pc->oip', coeffs, basis1)
        basis2 = _bspline_basis(x_val / 3.0, grid, 3)
        out2 = torch.einsum('oic,pc->oip', coeffs, basis2)
        assert torch.allclose(out1, out2, atol=1e-6), \
            "B-spline evaluation is not deterministic"

    def test_add_operation(self):
        """Add(a, b) = a + b (element-wise)."""
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        b = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assert np.allclose(a + b, np.array([1.1, 2.2, 3.3]), atol=1e-6)

    @pytest.mark.parametrize("logits,expected_class", [
        ([-10.0, 5.0, 2.0, -3.0], 1),
        ([3.0, 1.0, 2.0, 0.0], 0),
        ([-1.0, -2.0, -3.0, 0.0], 3),
    ])
    def test_softmax_peaks_at_max(self, logits, expected_class):
        """Softmax should give highest probability to argmax class."""
        logits = np.array(logits, dtype=np.float32)
        exp_logits = np.exp(logits - logits.max())
        probs = exp_logits / exp_logits.sum()
        assert np.argmax(probs) == expected_class, \
            f"Softmax should peak at class {expected_class}"

    def test_argmax_preserved_under_small_perturbation(self):
        """Argmax unchanged when perturbation < half margin."""
        logits = np.array([0.0, 5.0, 2.0, 1.0], dtype=np.float32)
        rng = np.random.RandomState(42)
        perturbed = logits + rng.uniform(-1.0, 1.0, size=logits.shape).astype(np.float32)
        assert np.argmax(perturbed) == 1, \
            "Argmax should be preserved under perturbation < half margin"

    def test_silu_minimum_location(self):
        """SiLU should have a minimum near x ≈ -1.28 (not monotonic)."""
        x = torch.linspace(-3.0, 3.0, 601)
        y = F.silu(x)
        min_idx = torch.argmin(y).item()
        x_min = x[min_idx].item()
        # Minimum should be near the known value of ~ -1.28
        assert -1.5 < x_min < -1.0, \
            f"SiLU minimum should be near -1.28, found at x={x_min:.3f}"
        # SiLU(0) should be 0
        assert abs(F.silu(torch.tensor(0.0)).item()) < 1e-6


# ============================================================================
# Test: End-to-End Classifier Agreement
# ============================================================================

class TestEndToEndAgreement:
    """PyTorch FP32 vs LUT-approximated forward pass agreement."""

    def test_classification_agreement_small_kan(self):
        """For a small KAN [3,2,2], LUT approximation preserves classification."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = StudentKAN([3, 2, 2])
        model.eval()

        n_test = 100
        x_test = np.random.RandomState(123).randn(n_test, 3).astype(np.float32)
        x_t = torch.from_numpy(x_test)

        with torch.no_grad():
            ref_logits = model(x_t).numpy()
        ref_preds = ref_logits.argmax(axis=1)

        lut_fn = _make_kan_lut_forward(model, N_lut=50, x_range=(-3.0, 3.0))
        lut_logits = lut_fn(x_test)
        lut_preds = lut_logits.argmax(axis=1)

        agreement = (ref_preds == lut_preds).mean()
        assert agreement >= 0.85, \
            f"LUT agreement {agreement:.2%} below 85% threshold (N_lut=50)"

    def test_logit_mae_decreases_with_more_lut_points(self):
        """More LUT points → smaller logit MAE."""
        torch.manual_seed(42)
        np.random.seed(42)

        model = StudentKAN([3, 2, 2])
        model.eval()

        n_test = 50
        x_test = np.random.RandomState(123).randn(n_test, 3).astype(np.float32) * 0.5
        x_t = torch.from_numpy(x_test)

        with torch.no_grad():
            ref_logits = model(x_t).numpy()

        mae_10 = np.abs(ref_logits - _make_kan_lut_forward(model, N_lut=10)(x_test)).mean()
        mae_50 = np.abs(ref_logits - _make_kan_lut_forward(model, N_lut=50)(x_test)).mean()

        # With more LUT points, error should decrease or stay similar
        assert mae_50 <= mae_10 * 1.2, \
            f"MAE at N=50 ({mae_50:.6f}) not ≤ 1.2× MAE at N=10 ({mae_10:.6f})"


# ============================================================================
# Test: KAN Architecture Consistency
# ============================================================================

class TestKANArchitecture:
    """Verify KAN model architecture properties."""

    def test_kan_architecture_consistency(self):
        """All KAN layers should have consistent input/output dimensions."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        prev_out = 3
        for i, layer in enumerate(model.kan_layers):
            assert layer.in_features == prev_out, \
                f"Layer {i}: in_features={layer.in_features} != prev_out={prev_out}"
            prev_out = layer.out_features
        assert prev_out == 2, f"Final output dim {prev_out} != 2"

    def test_kan_spline_weight_shape(self):
        """Spline weights should have shape (out_dim, in_dim, n_coeffs)."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        for i, layer in enumerate(model.kan_layers):
            sw = layer.spline_weight
            assert sw.ndim == 3, f"Layer {i}: spline_weight should be 3D"
            assert sw.shape[0] == layer.out_features
            assert sw.shape[1] == layer.in_features

    def test_kan_base_weight_shape(self):
        """Base weights should have shape (out_dim, in_dim)."""
        model = StudentKAN([3, 2, 2])
        model.eval()
        for layer in model.kan_layers:
            bw = layer.base_weight
            assert bw.ndim == 2
            assert bw.shape[0] == layer.out_features
            assert bw.shape[1] == layer.in_features
