#!/usr/bin/env python3
"""
NeuroPLC — Compiler Orchestrator
==================================
Ties together Frontend → Optimizer → Backend → Analyzer.

Usage:
    from neuroplc.compiler import NeuroPLCCompiler
    from models.student_kan import StudentKAN

    model = StudentKAN([28, 16, 4])
    model.eval()

    compiler = NeuroPLCCompiler(target="s7-1200")
    result = compiler.compile(model, output="results/scl_output/kan.scl")
    print(result.summary)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .ir import IRGraph, IROpType
from .frontend import kan_to_ir, mlp_to_ir
from .optimizer import optimize, compare_sampling_error
from .backend_s7 import S71200Backend, S71500Backend
from .analyzer import MemoryAnalyzer


class NeuroPLCCompiler:
    """End-to-end NeuroPLC compiler.

    Pipeline:
        1. Frontend:    PyTorch model → IR graph
        2. Optimizer:   Adaptive B-spline sampling + cleanup
        3. Analyzer:    Memory budget + FLOPs report
        4. Backend:     IR graph → IEC 61131-3 SCL code
        5. Export:      Write SCL + IR JSON + analyzer report

    Args:
        target:          "s7-1200" or "s7-1500"
        lut_points:      B-spline LUT sampling density (15 for S7-1200, 50 for S7-1500)
        adaptive:        use curvature-aware non-uniform sampling
        optimize_passes: IR optimization passes to run
        x_range:         B-spline input domain
        verbose:         print per-stage progress
    """

    def __init__(
        self,
        target: str = "s7-1200",
        lut_points: Optional[int] = None,
        adaptive: bool = True,
        optimize_passes: Optional[list[str]] = None,
        x_range: tuple = (-3.0, 3.0),
        verbose: bool = False,
    ):
        self.target = target.lower()
        self.adaptive = adaptive
        self.optimize_passes = optimize_passes or [
            "adaptive_bspline", "dead_node_elim", "constant_folding"]
        if not adaptive:
            self.optimize_passes = [
                p for p in self.optimize_passes if p != "adaptive_bspline"]
        self.x_range = x_range
        self.verbose = verbose

        # Backend selection
        if self.target == "s7-1200":
            self.lut_points = lut_points or 15
            self.backend = S71200Backend(lut_pts=self.lut_points)
        elif self.target == "s7-1500":
            self.lut_points = lut_points or 50
            self.backend = S71500Backend(lut_pts=self.lut_points)
        else:
            raise ValueError(f"Unknown target: '{target}'")

        # Results storage
        self.ir_graph: Optional[IRGraph] = None
        self.scl_code: str = ""
        self.analyzer_report: dict = {}
        self.optimizer_stats: dict = {}
        self.sampling_error: dict = {}

    def compile(self, model, output: Optional[str] = None,
                model_type: Optional[str] = None) -> "CompileResult":
        """Compile a PyTorch model to SCL.

        Args:
            model:      PyTorch model (StudentKAN or StudentMLP)
            output:     optional SCL output path
            model_type: "kan" or "mlp" (auto-detected if None)

        Returns:
            CompileResult with .scl_code, .ir_graph, .analyzer_report
        """
        # ── Auto-detect model type ──
        if model_type is None:
            model_type = self._detect_type(model)

        # ── Stage 1: Frontend ──
        if self.verbose:
            print(f"[1/5] Frontend: PyTorch {model_type.upper()} → IR")

        model.eval()
        if model_type == "kan":
            self.ir_graph = kan_to_ir(
                model, lut_points=self.lut_points, x_range=self.x_range,
                adaptive=self.adaptive)
        elif model_type == "mlp":
            self.ir_graph = mlp_to_ir(model)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if not self.ir_graph.is_valid:
            warnings = self.ir_graph.validate()
            raise ValueError(
                f"IR graph is not valid: {len(warnings)} warnings\n"
                + "\n".join(warnings))

        if self.verbose:
            print(f"       IR: {self.ir_graph.node_count} nodes, "
                  f"{self.ir_graph.op_counts}")

        # ── Stage 2: Optimizer ──
        if self.verbose:
            print(f"[2/5] Optimizer: {self.optimize_passes}")

        self.optimizer_stats = optimize(
            self.ir_graph, passes=self.optimize_passes,
            target_points=self.lut_points, x_range=self.x_range,
            verbose=self.verbose)

        self.sampling_error = compare_sampling_error(self.ir_graph)

        if self.verbose:
            n_opt = self.optimizer_stats.get("adaptive_bspline", 0)
            if n_opt > 0:
                print(f"       Adaptive sampling: {n_opt} LUTs, "
                      f"max error reduction: ~{self._error_reduction():.1f}%")

        # ── Stage 3: Analyzer ──
        if self.verbose:
            print(f"[3/5] Analyzer: memory + FLOPs")

        target_kb = 75 if self.target == "s7-1200" else 1500
        analyzer = MemoryAnalyzer(self.ir_graph, target_kb)
        self.analyzer_report = analyzer.analyze()

        if self.verbose:
            print(f"       {analyzer.summary()}")

        # ── Stage 4: Backend ──
        if self.verbose:
            print(f"[4/5] Backend: IR → SCL ({self.target.upper()})")

        self.scl_code = self.backend.generate(self.ir_graph)

        if self.verbose:
            print(f"       SCL: {len(self.scl_code)} chars, "
                  f"{self.scl_code.count(chr(10))} lines")

        # ── Stage 5: Export ──
        if output:
            if self.verbose:
                print(f"[5/5] Export: {output}")
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                f.write(self.scl_code)

            # Also write IR graph JSON
            ir_json = str(Path(output).with_suffix(".ir.json"))
            self.ir_graph.to_json(ir_json)

            # Write analyzer report
            report_json = str(Path(output).with_suffix(".report.json"))
            with open(report_json, "w", encoding="utf-8") as f:
                json.dump(self.analyzer_report, f, indent=2)

        if self.verbose:
            print(f"       Done.")

        return CompileResult(
            scl_code=self.scl_code,
            ir_graph=self.ir_graph,
            analyzer_report=self.analyzer_report,
            optimizer_stats=self.optimizer_stats,
            sampling_error=self.sampling_error,
        )

    def _detect_type(self, model) -> str:
        """Detect KAN vs MLP from model class name."""
        cls_name = type(model).__name__.lower()
        if "kan" in cls_name:
            return "kan"
        if "mlp" in cls_name:
            return "mlp"
        # Fallback: check for kan_layers attribute
        if hasattr(model, "kan_layers"):
            return "kan"
        if hasattr(model, "net"):
            return "mlp"
        raise ValueError(
            f"Cannot detect model type from {type(model).__name__}. "
            f"Pass model_type='kan' or 'mlp' explicitly.")

    def _error_reduction(self) -> float:
        """Estimate % error reduction from adaptive sampling."""
        u = self.sampling_error.get("uniform_max", 0)
        a = self.sampling_error.get("adaptive_max", 0)
        if u < 1e-10:
            return 0.0
        return (1.0 - a / u) * 100.0


class CompileResult:
    """Result of a compilation."""

    def __init__(self, scl_code: str, ir_graph: IRGraph,
                 analyzer_report: dict, optimizer_stats: dict,
                 sampling_error: dict):
        self.scl_code = scl_code
        self.ir_graph = ir_graph
        self.analyzer_report = analyzer_report
        self.optimizer_stats = optimizer_stats
        self.sampling_error = sampling_error

    @property
    def summary(self) -> str:
        """Multi-line summary."""
        m = self.analyzer_report.get("memory", {})
        f = self.analyzer_report.get("flops", {})
        return (
            f"CompileResult:\n"
            f"  SCL: {len(self.scl_code)} chars\n"
            f"  IR nodes: {self.ir_graph.node_count}\n"
            f"  Memory: {m.get('total_kb', '?')}KB / "
            f"{self.analyzer_report.get('target_plc_kb', '?')}KB "
            f"({self.analyzer_report.get('budget_utilization_pct', '?')}%)\n"
            f"  FLOPs: {f.get('total_per_inference', '?')} per inference\n"
            f"  Fits budget: {self.analyzer_report.get('fits_budget', '?')}"
        )
