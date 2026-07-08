#!/usr/bin/env python3
"""
NeuroPLC — Compiler Pipeline Integration Tests
=================================================
End-to-end tests for the IR → Frontend → Optimizer → Backend → Validator pipeline.

Tests cover:
    - IR graph construction, validation, serialization
    - KAN + MLP → IR conversion (frontend)
    - Adaptive B-spline sampling (optimizer)
    - SCL code generation (backend)
    - Compiler orchestrator (end-to-end)
"""

import sys
import json
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.student_kan import StudentKAN, KANLinear
from models.student_mlp import StudentMLP
from neuroplc.ir import IRGraph, IROpType, IRNode, build_mlp_ir


# ================================================================
# IR Graph Tests
# ================================================================

class TestIRGraph:
    """Test ir.py: IRGraph construction and validation."""

    def test_create_empty_graph(self):
        g = IRGraph(name="test")
        assert g.node_count == 0
        assert len(g.input_nodes) == 0
        assert len(g.output_nodes) == 0

    def test_add_nodes(self):
        g = IRGraph()
        n1 = g.add_node(IROpType.MatMul, name="fc1")
        n2 = g.add_node(IROpType.StandardAct, name="relu1")
        assert g.node_count == 2
        assert n1.id == 0
        assert n2.id == 1
        assert n1.op == IROpType.MatMul

    def test_add_edge(self):
        g = IRGraph()
        n1 = g.add_node(IROpType.MatMul)
        n2 = g.add_node(IROpType.StandardAct)
        g.add_edge(n1, n2)
        assert n2.id in n1.outputs
        assert n1.id in n2.inputs

    def test_topological_order_linear(self):
        g = IRGraph()
        a = g.add_node(IROpType.MatMul)
        b = g.add_node(IROpType.StandardAct)
        c = g.add_node(IROpType.Softmax)
        g.add_edge(a, b)
        g.add_edge(b, c)
        order = g.topological_order()
        assert order == [0, 1, 2]

    def test_topological_order_merge(self):
        """Two parallel paths merging into Add."""
        g = IRGraph()
        a = g.add_node(IROpType.MatMul, name="base")
        b = g.add_node(IROpType.BsplineLUT, name="spline")
        c = g.add_node(IROpType.Add, name="merge")
        g.add_edge(a, c, port=0)
        g.add_edge(b, c, port=1)
        order = g.topological_order()
        # a and b are independent, but both must come before c
        assert order.index(0) < order.index(2)
        assert order.index(1) < order.index(2)

    def test_cycle_detection(self):
        g = IRGraph()
        a = g.add_node(IROpType.MatMul)
        b = g.add_node(IROpType.Add)
        g.add_edge(a, b)
        g.add_edge(b, a)  # cycle!
        with pytest.raises(ValueError, match="cycle"):
            g.topological_order()

    def test_validation_passes(self):
        g = IRGraph()
        a = g.add_node(IROpType.MatMul, attrs={
            "W": np.eye(4, dtype=np.float32),
            "b": np.zeros(4, dtype=np.float32)})
        b = g.add_node(IROpType.StandardAct)
        c = g.add_node(IROpType.Softmax)
        g.add_edge(a, b)
        g.add_edge(b, c)
        warnings = g.validate()
        assert len(warnings) == 0
        assert g.is_valid

    def test_validation_missing_attrs(self):
        g = IRGraph()
        a = g.add_node(IROpType.MatMul)  # no W, b
        b = g.add_node(IROpType.BsplineLUT)  # no grid, table
        g.add_edge(a, b)
        warnings = g.validate()
        assert len(warnings) >= 2  # missing W, missing grid/table

    def test_serialization_roundtrip(self):
        g = IRGraph(name="test_serialize")
        a = g.add_node(IROpType.MatMul, name="fc")
        b = g.add_node(IROpType.Softmax, name="sm")
        g.add_edge(a, b)
        json_str = g.to_json()
        assert "test_serialize" in json_str
        assert "fc" in json_str
        assert "sm" in json_str
        # Parse back
        d = json.loads(json_str)
        assert d["name"] == "test_serialize"
        assert d["node_count"] == 2

    def test_build_mlp_ir(self):
        g = build_mlp_ir([28, 32, 16, 4])
        assert g.node_count == 7
        assert "softmax" in [n.op.value for n in g.nodes.values()]

    def test_edge_port_ordering(self):
        g = IRGraph()
        a = g.add_node(IROpType.MatMul)  # base path
        b = g.add_node(IROpType.BsplineLUT)  # spline path
        c = g.add_node(IROpType.Add)
        g.add_edge(a, c, port=0)
        g.add_edge(b, c, port=1)
        assert c.inputs[0] == a.id  # port 0 = primary
        assert c.inputs[1] == b.id  # port 1 = secondary


# ================================================================
# Frontend Tests
# ================================================================

class TestFrontend:
    """Test frontend.py: PyTorch model → IR conversion."""

    @pytest.fixture
    def kan_model(self):
        model = StudentKAN([28, 16, 4])
        model.eval()
        return model

    @pytest.fixture
    def mlp_model(self):
        model = StudentMLP()
        model.eval()
        return model

    def test_kan_to_ir_structure(self, kan_model):
        from neuroplc.frontend import kan_to_ir
        ir = kan_to_ir(kan_model, lut_points=10)
        assert ir.node_count > 5
        assert ir.is_valid
        # KAN should have BsplineLUT + Add nodes
        ops = set(n.op.value for n in ir.nodes.values())
        assert "bspline_lut" in ops
        assert "add" in ops
        assert "softmax" in ops

    def test_mlp_to_ir_structure(self, mlp_model):
        from neuroplc.frontend import mlp_to_ir
        ir = mlp_to_ir(mlp_model)
        assert ir.node_count > 3
        assert ir.is_valid
        ops = set(n.op.value for n in ir.nodes.values())
        assert "matmul" in ops
        assert "softmax" in ops
        # MLP should NOT have BsplineLUT
        assert "bspline_lut" not in ops

    def test_lut_table_shape(self, kan_model):
        from neuroplc.frontend import kan_to_ir
        ir = kan_to_ir(kan_model, lut_points=15)
        for n in ir.nodes.values():
            if n.op == IROpType.BsplineLUT:
                tbl = n.attrs["table"]
                gd = n.attrs["grid"]
                # table: (out, in, n_points)
                assert tbl.ndim == 3
                assert tbl.shape[2] == 15
                # grid: (n_points,)
                assert len(gd) == 15

    def test_kan_ir_produces_output(self, kan_model):
        """IR graph should have exactly one output node (Argmax)."""
        from neuroplc.frontend import kan_to_ir
        ir = kan_to_ir(kan_model)
        outputs = ir.output_nodes
        assert len(outputs) == 1
        assert outputs[0].op == IROpType.Argmax


# ================================================================
# Optimizer Tests
# ================================================================

class TestOptimizer:
    """Test optimizer.py: adaptive sampling + cleanup passes."""

    def test_adaptive_sampling_reduces_points(self):
        from neuroplc.optimizer import adaptive_bspline_sampling, optimize

        g = IRGraph(name="opt_test")
        # Create BsplineLUT with 100 points
        xs = np.linspace(-3, 3, 100, dtype=np.float32)
        ys = np.sin(xs * 2) * 0.5
        tbl = np.tile(ys.reshape(1, 1, -1), (4, 4, 1)).astype(np.float32)

        in_node = g.add_node(IROpType.MatMul, attrs={
            "W": np.eye(4, dtype=np.float32), "b": np.zeros(4, dtype=np.float32)})
        lut_node = g.add_node(IROpType.BsplineLUT,
                               attrs={"table": tbl, "grid": xs})
        out = g.add_node(IROpType.Softmax)
        g.add_edge(in_node, lut_node)
        g.add_edge(lut_node, out)

        before = lut_node.attrs["table"].shape[2]
        optimize(g, passes=["adaptive_bspline"], target_points=20)
        after = lut_node.attrs["table"].shape[2]

        assert after == 20
        assert after < before
        assert lut_node.attrs.get("_adaptive_sampled")

    def test_dead_node_elimination(self):
        from neuroplc.optimizer import dead_node_elimination

        g = IRGraph()
        a = g.add_node(IROpType.MatMul, attrs={
            "W": np.eye(4, dtype=np.float32), "b": np.zeros(4, dtype=np.float32)})
        # Create a truly orphan node: no inputs from any reachable source
        orphan = g.add_node(IROpType.StandardAct, name="orphan")
        # Add edge from orphan to itself (dead cycle, nothing feeds it)
        # orphan has no inputs and no connection to the main graph

        b = g.add_node(IROpType.StandardAct)
        c = g.add_node(IROpType.Softmax)
        g.add_edge(a, b)
        g.add_edge(b, c)
        n_before = g.node_count
        dead_node_elimination(g)
        assert g.node_count == n_before - 1  # orphan removed

    def test_compare_sampling_error(self):
        from neuroplc.optimizer import compare_sampling_error, optimize

        g = IRGraph(name="err_test")
        xs = np.linspace(-3, 3, 100, dtype=np.float32)
        ys = np.sin(xs * 2) * 0.5 + np.cos(xs * 1.5) * 0.3
        tbl = np.tile(ys.reshape(1, 1, -1), (2, 2, 1)).astype(np.float32)

        in_node = g.add_node(IROpType.MatMul, attrs={
            "W": np.eye(2, dtype=np.float32), "b": np.zeros(2, dtype=np.float32)})
        lut_node = g.add_node(IROpType.BsplineLUT,
                               attrs={"table": tbl, "grid": xs})
        out = g.add_node(IROpType.Softmax)
        g.add_edge(in_node, lut_node)
        g.add_edge(lut_node, out)

        optimize(g, passes=["adaptive_bspline"], target_points=20)
        errs = compare_sampling_error(g)
        assert errs["num_functions"] > 0
        # Three-way comparison: uniform, curvature-adaptive, DP-optimal
        assert "uniform_max" in errs
        assert "adaptive_max" in errs
        assert "optimal_max" in errs
        assert errs["optimal_max"] <= errs["uniform_max"]  # DP-optimal is provably best


# ================================================================
# Backend Tests
# ================================================================

class TestBackend:
    """Test backend_s7.py: SCL code generation."""

    @pytest.fixture
    def kan_ir(self):
        from neuroplc.frontend import kan_to_ir
        model = StudentKAN([28, 16, 4])
        model.eval()
        return kan_to_ir(model, lut_points=10)

    @pytest.fixture
    def mlp_ir(self):
        from neuroplc.frontend import mlp_to_ir
        model = StudentMLP()
        model.eval()
        return mlp_to_ir(model)

    def test_s71200_backend_generates_code(self, kan_ir):
        from neuroplc.backend_s7 import S71200Backend
        b = S71200Backend(lut_pts=10)
        scl = b.generate(kan_ir)
        assert len(scl) > 1000
        assert "DATA_BLOCK" in scl
        assert "FUNCTION_BLOCK" in scl
        assert "END_FUNCTION_BLOCK" in scl

    def test_s71500_backend_generates_code(self, mlp_ir):
        from neuroplc.backend_s7 import S71500Backend
        b = S71500Backend(lut_pts=20)
        scl = b.generate(mlp_ir)
        assert len(scl) > 1000
        assert "END_FUNCTION_BLOCK" in scl

    def test_kan_scl_has_bspline_fc(self, kan_ir):
        from neuroplc.backend_s7 import S71200Backend
        b = S71200Backend(lut_pts=10)
        scl = b.generate(kan_ir)
        assert "BsplineEval" in scl
        assert "Binary search" in scl

    def test_mlp_scl_no_bspline_fc(self, mlp_ir):
        from neuroplc.backend_s7 import S71200Backend
        b = S71200Backend(lut_pts=10)
        scl = b.generate(mlp_ir)
        assert "BsplineEval" not in scl

    def test_db_has_arrays(self, kan_ir):
        from neuroplc.backend_s7 import S71200Backend
        b = S71200Backend(lut_pts=10)
        scl = b.generate(kan_ir)
        assert "ARRAY[" in scl  # B9b: uppercase keywords required by TIA Portal V21


# ================================================================
# Compiler Integration Tests
# ================================================================

class TestCompilerIntegration:
    """End-to-end compiler pipeline tests."""

    def test_kan_full_compile(self):
        from neuroplc.compiler import NeuroPLCCompiler

        model = StudentKAN([28, 16, 4])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=10, verbose=False)
        result = compiler.compile(model)

        assert result.scl_code
        assert result.ir_graph.is_valid
        assert result.analyzer_report["fits_budget"]
        assert len(result.scl_code) > 5000

    def test_mlp_full_compile(self):
        from neuroplc.compiler import NeuroPLCCompiler

        model = StudentMLP()
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1500", lut_points=20, verbose=False)
        result = compiler.compile(model)

        assert result.scl_code
        assert result.ir_graph.is_valid
        assert result.analyzer_report["fits_budget"]

    def test_compiler_auto_detect(self):
        from neuroplc.compiler import NeuroPLCCompiler

        kan = StudentKAN([28, 16, 4])
        kan.eval()
        mlp = StudentMLP()
        mlp.eval()

        c = NeuroPLCCompiler(target="s7-1500", verbose=False)
        assert c._detect_type(kan) == "kan"
        assert c._detect_type(mlp) == "mlp"

    def test_analyzer_report_values(self):
        from neuroplc.compiler import NeuroPLCCompiler

        model = StudentKAN([28, 16, 4])
        model.eval()
        compiler = NeuroPLCCompiler(target="s7-1200", lut_points=10, verbose=False)
        result = compiler.compile(model)

        r = result.analyzer_report
        assert "memory" in r
        assert "flops" in r
        assert "budget_utilization_pct" in r
        assert r["memory"]["total_kb"] > 0
        assert r["flops"]["total_per_inference"] > 0

    def test_validator_cross_validate(self):
        from neuroplc.validator import Validator

        py_out = np.random.randn(100, 4).astype(np.float32)
        scl_out = py_out + np.random.normal(0, 1e-5, py_out.shape).astype(np.float32)

        val = Validator(tolerance=1e-4)
        result = val.compare(py_out, scl_out)

        assert result["passes"]
        assert result["classification_agreement"] > 0.95
        assert result["max_absolute_error"] < 1e-4

    def test_validator_rejects_bad_output(self):
        from neuroplc.validator import Validator

        py_out = np.random.randn(100, 4).astype(np.float32)
        scl_out = py_out + 0.1  # large error

        val = Validator(tolerance=1e-4)
        result = val.compare(py_out, scl_out)

        assert not result["passes"]
        assert result["max_absolute_error"] > 0.01


# ================================================================
# Memory Analyzer Tests
# ================================================================

class TestAnalyzer:
    """Test analyzer.py."""

    def test_memory_analyzer_kan(self):
        from neuroplc.analyzer import MemoryAnalyzer
        from neuroplc.frontend import kan_to_ir

        kan = StudentKAN([28, 16, 4])
        kan.eval()
        ir = kan_to_ir(kan, lut_points=10)

        analyzer = MemoryAnalyzer(ir, target_work_memory_kb=75)
        report = analyzer.analyze()

        assert report["memory"]["total_kb"] > 10
        assert report["memory"]["lut_kb"] > 0  # KAN has LUTs
        assert report["fits_budget"]

    def test_memory_analyzer_mlp(self):
        from neuroplc.analyzer import MemoryAnalyzer
        from neuroplc.frontend import mlp_to_ir

        mlp = StudentMLP()
        mlp.eval()
        ir = mlp_to_ir(mlp)

        analyzer = MemoryAnalyzer(ir, target_work_memory_kb=75)
        report = analyzer.analyze()

        assert report["memory"]["total_kb"] > 5
        assert report["memory"]["lut_kb"] == 0  # MLP has no LUTs
        assert report["fits_budget"]
