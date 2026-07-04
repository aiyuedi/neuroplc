#!/usr/bin/env python3
"""
NeuroPLC — Static Analyzer
=============================
Analyzes an IR graph for memory usage and computational cost.

Reports:
    Memory budget:     DB weights + LUTs + code estimation + variables
    FLOPs:             multiply-add operations per inference
    Per-operation:     breakdown by node type
    PLC budget check:  utilization percentage for target PLC

Usage:
    from neuroplc.analyzer import MemoryAnalyzer

    analyzer = MemoryAnalyzer(graph, target_kb=75)
    report = analyzer.analyze()
    print(report.summary())
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .ir import IRGraph, IROpType


class MemoryAnalyzer:
    """Compute memory usage for an IR graph on a specific PLC."""

    def __init__(self, graph: IRGraph, target_work_memory_kb: int = 75,
                 array_overhead_bytes: int = 8):
        """
        Args:
            graph:                    compiled IR graph
            target_work_memory_kb:    PLC work memory limit
            array_overhead_bytes:     per-array metadata overhead (estimated)
        """
        self.graph = graph
        self.target_kb = target_work_memory_kb
        self.overhead = array_overhead_bytes

    def analyze(self) -> dict:
        """Run all analyses. Returns a dict suitable for JSON export."""
        mem = self._memory()
        flops = self._flops()
        return {
            "graph_name": self.graph.name,
            "target_plc_kb": self.target_kb,
            "memory": mem,
            "flops": flops,
            "budget_utilization_pct": round(
                mem["total_kb"] / self.target_kb * 100, 1),
            "fits_budget": mem["total_kb"] <= self.target_kb,
        }

    def _memory(self) -> dict:
        """Compute memory breakdown."""
        weights_kb = 0.0
        lut_kb = 0.0
        array_blocks = 0

        for node in self.graph.nodes.values():
            if node.op == IROpType.MatMul:
                W = node.attrs.get("W")
                b = node.attrs.get("b")
                if W is not None:
                    weights_kb += (W.nbytes + (
                        b.nbytes if b is not None else 0)) / 1024.0
                    array_blocks += 1
            elif node.op == IROpType.BsplineLUT:
                table = node.attrs.get("table")
                grid = node.attrs.get("grid")
                if table is not None:
                    lut_kb += table.nbytes / 1024.0
                    array_blocks += 1
                if grid is not None:
                    lut_kb += grid.nbytes / 1024.0
                    array_blocks += 1

        # Code estimate: ~200B per IR node + overhead
        code_kb = (self.graph.node_count * 200 + 2000) / 1024.0

        # Variable allocation: ~4B per scalar × node outputs
        var_kb = 0.0
        for node in self.graph.nodes.values():
            if node.shape_out:
                d = node.shape_out[0]
                var_kb += d * 4 / 1024.0

        total_kb = weights_kb + lut_kb + code_kb + var_kb + (
            array_blocks * self.overhead / 1024.0)

        return {
            "weights_kb": round(weights_kb, 1),
            "lut_kb": round(lut_kb, 1),
            "code_kb": round(code_kb, 1),
            "variables_kb": round(var_kb, 1),
            "array_blocks": array_blocks,
            "total_kb": round(total_kb, 1),
        }

    def _flops(self) -> dict:
        """Count floating-point operations per inference."""
        matmul_flops = 0
        bspline_flops = 0
        act_flops = 0
        softmax_flops = 0

        for node in self.graph.nodes.values():
            if node.op == IROpType.MatMul:
                W = node.attrs.get("W")
                if W is not None:
                    out_d, in_d = W.shape
                    # Multiply-add: out_d × in_d multiplies + out_d adds
                    matmul_flops += out_d * in_d * 2
            elif node.op == IROpType.BsplineLUT:
                table = node.attrs.get("table")
                if table is not None:
                    out_d, in_d, n_pts = table.shape
                    # Binary search: ~log2(n_pts) comparisons
                    # Linear interp: 2 subs + 2 muls + 1 add
                    bspline_flops += out_d * in_d * (
                        int(np.ceil(np.log2(n_pts))) + 5)
            elif node.op == IROpType.StandardAct:
                d = node.shape_in[0] if node.shape_in else 28
                at = node.attrs.get("type", "relu")
                if at in ("relu",):
                    act_flops += d  # one comparison
                elif at in ("silu", "sigmoid"):
                    act_flops += d * 4  # exp + add + div + mul
            elif node.op == IROpType.Softmax:
                d = node.shape_in[0] if node.shape_in else 4
                softmax_flops += d * 3  # exp + sum + div

        total = matmul_flops + bspline_flops + act_flops + softmax_flops
        return {
            "matmul": matmul_flops,
            "bspline_lut": bspline_flops,
            "activation": act_flops,
            "softmax": softmax_flops,
            "total_per_inference": total,
        }

    def summary(self) -> str:
        """Human-readable summary string."""
        r = self.analyze()
        m, f = r["memory"], r["flops"]
        fit = "FITS" if r["fits_budget"] else "EXCEEDS"
        return (
            f"Memory: {m['total_kb']:.1f}KB / {self.target_kb}KB ({r['budget_utilization_pct']}%) {fit}\n"
            f"  Weights: {m['weights_kb']:.1f}KB | LUT: {m['lut_kb']:.1f}KB | "
            f"Code: {m['code_kb']:.1f}KB | Vars: {m['variables_kb']:.1f}KB\n"
            f"FLOPs: {f['total_per_inference']} per inference\n"
            f"  MatMul: {f['matmul']} | Bspline: {f['bspline_lut']} | "
            f"Act: {f['activation']} | Softmax: {f['softmax']}"
        )


class FLOPsAnalyzer:
    """Lightweight FLOPs counter (subset of MemoryAnalyzer)."""

    def __init__(self, graph: IRGraph):
        self.graph = graph

    def count(self) -> dict:
        return MemoryAnalyzer(self.graph)._flops()
