#!/usr/bin/env python3
"""
NeuroPLC — Backend Variants Unit Tests
=======================================
Tests for ALL backend variants:
    - FB-Only backend (backend_s7_fbonly.py)
    - DB+FB split backend (backend_s7_db.py)
    - S7-1200 and S7-1500 subclasses of each

Coverage: T2 (DB+FB backend), T3 (FB-Only backend), T10 (SCL structure verification)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroplc.ir import IRGraph, IROpType
from neuroplc.frontend import kan_to_ir, mlp_to_ir

# ── Backend imports ──
from neuroplc.backend_s7_fbonly import S71200FBOnlyBackend, S71500FBOnlyBackend
from neuroplc.backend_s7_db import S71200DBBackend, S71500DBBackend


# ================================================================
# Fixtures: reusable IR graphs for testing backends
# ================================================================


@pytest.fixture(scope="module")
def kan_ir():
    """Small KAN IR graph for backend testing."""
    g = IRGraph(name="test_kan")
    table = np.random.randn(4, 8, 15).astype(np.float32) * 0.1
    grid_pts = np.linspace(-3.0, 3.0, 15, dtype=np.float32)
    W0 = np.random.randn(8, 8).astype(np.float32) * 0.1
    W1 = np.random.randn(4, 8).astype(np.float32) * 0.1

    # Virtual input
    inp = g.add_node(IROpType.MatMul, name="input",
                     attrs={"W": np.eye(8, dtype=np.float32),
                            "b": np.zeros(8, dtype=np.float32),
                            "_virtual_input": True},
                     shape_in=(8,), shape_out=(8,))

    # SiLU
    silu = g.add_node(IROpType.StandardAct, name="silu", attrs={"type": "silu"},
                      shape_in=(8,), shape_out=(8,))
    g.add_edge(inp, silu)

    # Base MatMul
    base = g.add_node(IROpType.MatMul, name="base",
                      attrs={"W": W0, "b": np.zeros(8, dtype=np.float32)},
                      shape_in=(8,), shape_out=(8,))
    g.add_edge(silu, base)

    # BsplineLUT
    bspline = g.add_node(IROpType.BsplineLUT, name="bspline",
                          attrs={"table": table, "grid": grid_pts,
                                 "x_range": [-3.0, 3.0]},
                          shape_in=(8,), shape_out=(8, 4))
    g.add_edge(inp, bspline)

    # Add merge
    merge = g.add_node(IROpType.Add, name="merge",
                       shape_in=(8,), shape_out=(8,))
    g.add_edge(base, merge, port=0)
    g.add_edge(bspline, merge, port=1)

    # Second layer
    W1_node = g.add_node(IROpType.MatMul, name="layer1",
                         attrs={"W": W1, "b": np.zeros(4, dtype=np.float32)},
                         shape_in=(8,), shape_out=(4,))
    g.add_edge(merge, W1_node)

    # Softmax
    sm = g.add_node(IROpType.Softmax, name="softmax",
                    shape_in=(4,), shape_out=(4,))
    g.add_edge(W1_node, sm)

    # Argmax
    am = g.add_node(IROpType.Argmax, name="argmax",
                    shape_in=(4,), shape_out=(1,))
    g.add_edge(sm, am)

    return g


@pytest.fixture(scope="module")
def mlp_ir():
    """Small MLP IR graph for backend testing."""
    g = IRGraph(name="test_mlp")
    W0 = np.random.randn(16, 8).astype(np.float32) * 0.1
    W1 = np.random.randn(4, 16).astype(np.float32) * 0.1

    inp = g.add_node(IROpType.MatMul, name="input",
                     attrs={"W": np.eye(8, dtype=np.float32),
                            "b": np.zeros(8, dtype=np.float32),
                            "_virtual_input": True},
                     shape_in=(8,), shape_out=(8,))

    fc0 = g.add_node(IROpType.MatMul, name="fc0",
                     attrs={"W": W0, "b": np.zeros(16, dtype=np.float32)},
                     shape_in=(8,), shape_out=(16,))
    g.add_edge(inp, fc0)

    relu = g.add_node(IROpType.StandardAct, name="relu", attrs={"type": "relu"},
                      shape_in=(16,), shape_out=(16,))
    g.add_edge(fc0, relu)

    fc1 = g.add_node(IROpType.MatMul, name="fc1",
                     attrs={"W": W1, "b": np.zeros(4, dtype=np.float32)},
                     shape_in=(16,), shape_out=(4,))
    g.add_edge(relu, fc1)

    sm = g.add_node(IROpType.Softmax, name="softmax", shape_in=(4,), shape_out=(4,))
    g.add_edge(fc1, sm)

    am = g.add_node(IROpType.Argmax, name="argmax", shape_in=(4,), shape_out=(1,))
    g.add_edge(sm, am)

    return g


# ================================================================
# FB-Only Backend Tests
# ================================================================

class TestFBOnlyBackend:
    """Test backend_s7_fbonly.py: S71200FBOnlyBackend + S71500FBOnlyBackend."""

    def test_s71200_generates_code(self, kan_ir):
        backend = S71200FBOnlyBackend(lut_pts=15)
        scl = backend.generate(kan_ir)
        assert len(scl) > 1000, "Should generate substantial SCL code"
        assert "FUNCTION_BLOCK" in scl
        assert "END_FUNCTION_BLOCK" in scl

    def test_s71500_generates_code(self, kan_ir):
        backend = S71500FBOnlyBackend(lut_pts=50)
        scl = backend.generate(kan_ir)
        assert len(scl) > 1000
        assert "FUNCTION_BLOCK" in scl

    def test_no_data_block_in_fbonly(self, kan_ir):
        """FB-Only mode should NOT generate a separate DATA_BLOCK."""
        backend = S71200FBOnlyBackend(lut_pts=15)
        scl = backend.generate(kan_ir)
        assert "DATA_BLOCK" not in scl

    def test_has_var_declarations(self, kan_ir):
        """FB-Only SCL should contain VAR sections with array declarations."""
        backend = S71200FBOnlyBackend(lut_pts=15)
        scl = backend.generate(kan_ir)
        assert "VAR" in scl
        assert "ARRAY" in scl  # uppercase after B9b fix

    def test_mlp_generates_code(self, mlp_ir):
        backend = S71200FBOnlyBackend(lut_pts=15)
        scl = backend.generate(mlp_ir)
        assert len(scl) > 500
        assert "FUNCTION_BLOCK" in scl
        # MLP should not have BsplineLUT references
        assert "BsplineLUT" not in scl
        assert "BsplineEval" not in scl

    def test_uppercase_keywords(self, kan_ir):
        """After B9b fix, SCL uses uppercase IEC keywords."""
        backend = S71200FBOnlyBackend(lut_pts=15)
        scl = backend.generate(kan_ir)
        assert "ARRAY" in scl
        assert "OF REAL" in scl or "OF INT" in scl
        assert ": INT;" in scl
        assert ": REAL;" in scl
        # Should NOT have lowercase variants
        assert "Array[" not in scl
        assert " of Real" not in scl


# ================================================================
# DB+FB Backend Tests
# ================================================================

class TestDBFBBackend:
    """Test backend_s7_db.py: S71200DBBackend + S71500DBBackend."""

    def test_s71200_generates_db_and_fb(self, kan_ir):
        backend = S71200DBBackend(lut_pts=15)
        db_scl, fb_scl = backend.generate(kan_ir)
        assert len(db_scl) > 1000
        assert len(fb_scl) > 500
        assert "DATA_BLOCK" in db_scl
        assert "FUNCTION_BLOCK" in fb_scl

    def test_s71500_optimized_db(self, kan_ir):
        """S7-1500 uses optimized DB access to handle larger DBs."""
        backend = S71500DBBackend(lut_pts=50)
        db_scl, fb_scl = backend.generate(kan_ir)
        assert len(db_scl) > 1000
        assert len(fb_scl) > 500
        # S7-1500 should use optimized access
        assert "S7_Optimized_Access := 'TRUE'" in db_scl

    def test_s71200_non_optimized_db(self, kan_ir):
        """S7-1200 uses non-optimized DB for compatibility."""
        backend = S71200DBBackend(lut_pts=15)
        db_scl, fb_scl = backend.generate(kan_ir)
        assert "S7_Optimized_Access := 'FALSE'" in db_scl

    def test_db_has_struct_wrapper(self, kan_ir):
        """After B9a fix, DB has STRUCT/END_STRUCT wrapper."""
        backend = S71200DBBackend(lut_pts=15)
        db_scl, _ = backend.generate(kan_ir)
        assert "STRUCT" in db_scl  # uppercase
        assert "END_STRUCT;" in db_scl
        assert "BEGIN" in db_scl
        assert "END_DATA_BLOCK" in db_scl

    def test_fb_references_db_name(self, kan_ir):
        """FB inference code should reference DB arrays by qualified name."""
        backend = S71200DBBackend(lut_pts=15, db_name="NeuroPLC_Weights")
        db_scl, fb_scl = backend.generate(kan_ir)
        # FB should reference the DB name for array access
        assert '"NeuroPLC_Weights"' in fb_scl

    def test_db_contains_array_declarations(self, kan_ir):
        """DB should contain ARRAY type declarations for weights/LUT."""
        backend = S71200DBBackend(lut_pts=15)
        db_scl, _ = backend.generate(kan_ir)
        assert "ARRAY" in db_scl  # uppercase

    def test_fb_uppercase_keywords(self, kan_ir):
        """After B9b fix, FB uses uppercase keywords."""
        backend = S71200DBBackend(lut_pts=15)
        _, fb_scl = backend.generate(kan_ir)
        assert "ARRAY" in fb_scl
        assert "OF REAL" in fb_scl
        assert ": INT;" in fb_scl
        assert ": REAL;" in fb_scl

    def test_mlp_generates_db_and_fb(self, mlp_ir):
        backend = S71200DBBackend(lut_pts=15)
        db_scl, fb_scl = backend.generate(mlp_ir)
        assert len(db_scl) > 500
        assert len(fb_scl) > 300
        assert "DATA_BLOCK" in db_scl
        assert "FUNCTION_BLOCK" in fb_scl
        # MLP: no BsplineLUT references
        assert "BsplineLUT" not in fb_scl


# ================================================================
# Cross-Backend Comparison
# ================================================================

class TestBackendComparison:
    """Cross-backend consistency tests."""

    def test_all_backends_output_different_scl(self, kan_ir):
        """Each backend should produce structurally different SCL."""
        fb1200 = S71200FBOnlyBackend(lut_pts=15)
        fb1500 = S71500FBOnlyBackend(lut_pts=50)
        db1200 = S71200DBBackend(lut_pts=15)
        db1500 = S71500DBBackend(lut_pts=50)

        scl_fb1200 = fb1200.generate(kan_ir)
        db_scl_1200, fb_scl_1200 = db1200.generate(kan_ir)
        scl_fb1500 = fb1500.generate(kan_ir)
        db_scl_1500, fb_scl_1500 = db1500.generate(kan_ir)

        # All outputs should be different (different structures, different line counts)
        outputs = [scl_fb1200, db_scl_1200 + fb_scl_1200,
                   scl_fb1500, db_scl_1500 + fb_scl_1500]
        for i in range(len(outputs)):
            for j in range(i + 1, len(outputs)):
                assert outputs[i] != outputs[j], \
                    f"Backends {i} and {j} produce identical output"

    def test_s71500_larger_than_s71200(self, kan_ir):
        """S7-1500 backends produce larger code (more LUT points, unrolled)."""
        scl_1200 = S71200FBOnlyBackend(lut_pts=15).generate(kan_ir)
        scl_1500 = S71500FBOnlyBackend(lut_pts=50).generate(kan_ir)
        # 50-point LUT → more init assignments (3.3× more). Code may differ based on data.
        assert len(scl_1500) > len(scl_1200), \
            f"S7-1500 ({len(scl_1500)}) should be larger than S7-1200 ({len(scl_1200)})"
