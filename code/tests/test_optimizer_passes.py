#!/usr/bin/env python3
"""
NeuroPLC — Optimizer Passes Unit Tests
=======================================
Tests for each optimization pass independently:
    - fuse_matmul_add: operator fusion
    - lutize_exp: EXP → LUT strength reduction
    - dead_node_elimination: reachability + dead node removal
    - constant_folding: virtual input handling (B9f fix)
    - optimal_bspline_sampling: DP-based LUT placement

Coverage: T5 (cross-validation), T7 (training), T9 (real weight distributions)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neuroplc.ir import IRGraph, IROpType, IRNode
from neuroplc.optimizer import (
    fuse_matmul_add,
    lutize_exp,
    dead_node_elimination,
    constant_folding,
    optimal_bspline_sampling,
)


# ================================================================
# Fixtures
# ================================================================


@pytest.fixture
def kan_style_graph():
    """A minimal 2-layer KAN-style IR graph."""
    g = IRGraph(name="test_kan")

    W0 = np.random.randn(8, 4).astype(np.float32) * 0.1
    table = np.random.randn(8, 4, 15).astype(np.float32) * 0.1
    grid = np.linspace(-3.0, 3.0, 15, dtype=np.float32)

    inp = g.add_node(IROpType.MatMul, name="input",
                     attrs={"W": np.eye(4, dtype=np.float32),
                            "b": np.zeros(4, dtype=np.float32),
                            "_virtual_input": True},
                     shape_in=(4,), shape_out=(4,))

    silu = g.add_node(IROpType.StandardAct, name="silu", attrs={"type": "silu"},
                      shape_in=(4,), shape_out=(4,))
    g.add_edge(inp, silu)

    base = g.add_node(IROpType.MatMul, name="base",
                      attrs={"W": W0, "b": np.zeros(8, dtype=np.float32)},
                      shape_in=(4,), shape_out=(8,))
    g.add_edge(silu, base)

    bspline = g.add_node(IROpType.BsplineLUT, name="bspline",
                          attrs={"table": table, "grid": grid,
                                 "x_range": [-3.0, 3.0]},
                          shape_in=(4,), shape_out=(4, 8))
    g.add_edge(inp, bspline)

    merge = g.add_node(IROpType.Add, name="merge",
                       shape_in=(8,), shape_out=(8,))
    g.add_edge(base, merge, port=0)
    g.add_edge(bspline, merge, port=1)

    sm = g.add_node(IROpType.Softmax, name="softmax",
                    shape_in=(8,), shape_out=(8,))
    g.add_edge(merge, sm)

    am = g.add_node(IROpType.Argmax, name="argmax",
                    shape_in=(8,), shape_out=(1,))
    g.add_edge(sm, am)

    return g


# ================================================================
# Fuse MatMul + Add
# ================================================================

class TestFuseMatMulAdd:
    """Test optimizer.py: fuse_matmul_add pass."""

    def test_detects_fusion_pattern(self, kan_style_graph):
        """KAN graph has MatMul → ... → Add(MatMul, BsplineLUT) pattern."""
        n = fuse_matmul_add(kan_style_graph)
        assert n > 0, "Should detect at least one fusion pattern"

    def test_marks_attributes(self, kan_style_graph):
        """Fused nodes get _fused_matmul_add and port markers."""
        fuse_matmul_add(kan_style_graph)
        merge_node = None
        for node in kan_style_graph.nodes.values():
            if node.op == IROpType.Add:
                merge_node = node
                break
        assert merge_node is not None
        assert merge_node.attrs.get("_fused_matmul_add") is True
        assert "_mm_input" in merge_node.attrs
        assert "_bs_input" in merge_node.attrs

    def test_no_fusion_without_bspline(self):
        """MLP-style graph (no BsplineLUT) should have zero fusions."""
        g = IRGraph()
        a = g.add_node(IROpType.MatMul, name="fc")
        b = g.add_node(IROpType.StandardAct, name="relu", attrs={"type": "relu"})
        g.add_edge(a, b)
        n = fuse_matmul_add(g)
        assert n == 0


# ================================================================
# LUTize EXP
# ================================================================

class TestLutizeExp:
    """Test optimizer.py: lutize_exp pass."""

    def test_marks_silu_nodes(self, kan_style_graph):
        """SiLU activation nodes should get LUT metadata."""
        n = lutize_exp(kan_style_graph)
        assert n > 0, "Should mark SiLU nodes"

        silu_nodes = [n for n in kan_style_graph.nodes.values()
                      if n.op == IROpType.StandardAct and n.attrs.get("type") == "silu"]
        for node in silu_nodes:
            assert "_lut_silu" in node.attrs
            assert node.attrs.get("_lut_silu") is True
            assert "_lut_silu_n" in node.attrs

    def test_marks_softmax_nodes(self, kan_style_graph):
        """Softmax nodes should get EXP LUT metadata."""
        n = lutize_exp(kan_style_graph)
        sm_nodes = [n for n in kan_style_graph.nodes.values()
                     if n.op == IROpType.Softmax]
        for node in sm_nodes:
            assert "_lut_exp" in node.attrs
            assert node.attrs.get("_lut_exp") is True

    def test_no_false_positives(self):
        """Graph without SiLU should not mark unrelated nodes."""
        g = IRGraph()
        g.add_node(IROpType.StandardAct, name="relu", attrs={"type": "relu"})
        n = lutize_exp(g)
        assert n == 0


# ================================================================
# Dead Node Elimination
# ================================================================

class TestDeadNodeElimination:
    """Test optimizer.py: dead_node_elimination pass."""

    def test_removes_unused_node(self):
        """A node with no path to output should be removed."""
        g = IRGraph()
        a = g.add_node(IROpType.MatMul, name="a")
        dead = g.add_node(IROpType.MatMul, name="dead")
        b = g.add_node(IROpType.Softmax, name="b")
        g.add_edge(a, b)
        # dead node has no edges at all

        n = dead_node_elimination(g)
        assert n == 1
        assert dead.id not in g.nodes

    def test_preserves_connected_nodes(self, kan_style_graph):
        """Connected KAN graph should have no dead nodes."""
        node_count_before = kan_style_graph.node_count
        n = dead_node_elimination(kan_style_graph)
        assert n == 0, "All nodes are connected to output"
        assert kan_style_graph.node_count == node_count_before

    def test_handles_empty_graph(self):
        """Empty graph should not crash."""
        g = IRGraph()
        n = dead_node_elimination(g)
        assert n == 0


# ================================================================
# Constant Folding (B9f fix — virtual input preserved)
# ================================================================

class TestConstantFolding:
    """Test optimizer.py: constant_folding pass."""

    def test_preserves_virtual_input_connectivity(self, kan_style_graph):
        """After B9f fix, constant_folding does NOT destroy graph connectivity."""
        node_count_before = kan_style_graph.node_count

        # Verify graph is valid before folding
        assert kan_style_graph.is_valid

        n = constant_folding(kan_style_graph)

        # After B9f fix: virtual input is preserved, so node count unchanged
        assert kan_style_graph.node_count == node_count_before
        assert n == 0

        # Graph must still be topologically sortable
        order = kan_style_graph.topological_order()
        assert len(order) == kan_style_graph.node_count

    def test_graph_remains_valid(self, kan_style_graph):
        """After constant_folding, graph should still be valid and sortable."""
        constant_folding(kan_style_graph)
        assert kan_style_graph.is_valid
        order = kan_style_graph.topological_order()
        assert len(order) == kan_style_graph.node_count


# ================================================================
# DP-Optimal B-spline Sampling
# ================================================================

class TestOptimalBsplineSampling:
    """Test optimizer.py: optimal_bspline_sampling pass."""

    def test_preserves_table_shape(self):
        """DP-optimal resampling preserves (out, in) dimensions."""
        g = IRGraph()
        table = np.random.randn(3, 5, 30).astype(np.float32) * 0.1
        grid = np.linspace(-3.0, 3.0, 30, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid,
                           "x_range": [-3.0, 3.0]})

        n = optimal_bspline_sampling(g, target_points=15, x_range=(-3.0, 3.0))
        assert n == 1
        node = g.nodes[0]
        new_table = node.attrs["table"]
        assert new_table.shape == (3, 5, 15)

    def test_grid_monotonic(self):
        """DP grid must be strictly increasing."""
        g = IRGraph()
        table = np.random.randn(2, 3, 30).astype(np.float32) * 0.1
        grid = np.linspace(-3.0, 3.0, 30, dtype=np.float32)
        g.add_node(IROpType.BsplineLUT, name="bspline",
                    attrs={"table": table, "grid": grid})

        optimal_bspline_sampling(g, target_points=10, x_range=(-3.0, 3.0))
        new_grid = g.nodes[0].attrs["grid"]
        assert np.all(np.diff(new_grid) > 0)

    def test_no_bspline_noop(self):
        """Graph without BsplineLUT → nothing optimized."""
        g = IRGraph()
        g.add_node(IROpType.MatMul)
        n = optimal_bspline_sampling(g, target_points=15)
        assert n == 0


# ================================================================
# Pass Ordering: Fusion before backend
# ================================================================

class TestPassCombinations:
    """Test that passes work together in correct order."""

    def test_fuse_then_dead_elim(self, kan_style_graph):
        """Fusion marks nodes; dead_elim should not remove fused nodes."""
        fuse_matmul_add(kan_style_graph)
        n_removed = dead_node_elimination(kan_style_graph)
        # All fused nodes are still connected to output
        assert n_removed == 0

    def test_full_pipeline_no_crash(self, kan_style_graph):
        """Full optimization pipeline should not crash or break graph."""
        # Passes with positional args only
        optimal_bspline_sampling(kan_style_graph, 15, (-3.0, 3.0))
        fuse_matmul_add(kan_style_graph)
        lutize_exp(kan_style_graph)
        dead_node_elimination(kan_style_graph)
        constant_folding(kan_style_graph)

        # Graph should still be valid
        assert kan_style_graph.is_valid
        # Graph should be sortable
        order = kan_style_graph.topological_order()
        assert len(order) == kan_style_graph.node_count
