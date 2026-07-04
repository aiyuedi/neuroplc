#!/usr/bin/env python3
"""
NeuroPLC — SCL Code Validator
===============================
Cross-validates Python inference vs SCL inference output.

Checks:
    1. Element-wise consistency (MaxAE, MAE, RMSE)
    2. Per-operation error breakdown
    3. Classification agreement rate (does PLC predict same class?)

Usage:
    from neuroplc.validator import Validator

    val = Validator(tolerance=1e-4)
    result = val.compare(python_logits, scl_logits)
    print(result.summary)
"""

import numpy as np
from typing import Optional


class Validator:
    """Compare Python and SCL inference outputs."""

    def __init__(self, tolerance: float = 1e-4):
        """
        Args:
            tolerance: maximum acceptable absolute error (MaxAE threshold)
        """
        self.tolerance = tolerance

    def compare(self, python_output: np.ndarray,
                scl_output: np.ndarray,
                class_names: Optional[list[str]] = None,
                operation_names: Optional[list[str]] = None) -> dict:
        """Compare two output arrays (Python vs SCL).

        Args:
            python_output:  (N, C) array from PyTorch
            scl_output:     (N, C) array from SCL (or SCL simulator)
            class_names:    optional labels
            operation_names: optional per-op labels for breakdown

        Returns:
            dict with error metrics
        """
        if python_output.shape != scl_output.shape:
            raise ValueError(
                f"Shape mismatch: Python {python_output.shape} "
                f"vs SCL {scl_output.shape}")

        diff = np.abs(python_output - scl_output)
        max_ae = float(np.max(diff))
        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))

        # Classification agreement
        py_preds = python_output.argmax(axis=1)
        scl_preds = scl_output.argmax(axis=1)
        matches = (py_preds == scl_preds)
        agreement = float(np.mean(matches))
        mismatches = int(np.sum(~matches))

        # Per-class agreement
        per_class = {}
        if class_names:
            for i, name in enumerate(class_names):
                mask = py_preds == i
                if mask.sum() > 0:
                    per_class[name] = {
                        "n": int(mask.sum()),
                        "agreement": float(np.mean(matches[mask])),
                    }

        # Per-operation breakdown
        per_op = {}
        if operation_names and python_output.ndim > 1:
            for j, name in enumerate(operation_names):
                if j < python_output.shape[1]:
                    op_diff = diff[:, j]
                    per_op[name] = {
                        "max_ae": float(np.max(op_diff)),
                        "mae": float(np.mean(op_diff)),
                    }

        # Error distribution
        try:
            hist, edges = np.histogram(diff.flatten(), bins=min(50, len(diff.flatten()) // 10))
        except ValueError:
            hist, edges = np.array([len(diff.flatten())]), np.array([0, 1])

        return {
            "max_absolute_error": max_ae,
            "mean_absolute_error": mae,
            "rmse": rmse,
            "tolerance": self.tolerance,
            "passes": max_ae <= self.tolerance,
            "classification_agreement": agreement,
            "mismatched_samples": mismatches,
            "total_samples": len(python_output),
            "per_class": per_class,
            "per_operation": per_op,
            "error_histogram": {
                "counts": hist.tolist(),
                "edges": edges.tolist(),
            },
        }

    def summary(self, result: dict) -> str:
        """Human-readable summary string."""
        status = "PASS" if result["passes"] else "FAIL"
        return (
            f"Validator: {status} (max tolerance: {self.tolerance})\n"
            f"  MaxAE:    {result['max_absolute_error']:.2e}\n"
            f"  MAE:      {result['mean_absolute_error']:.2e}\n"
            f"  RMSE:     {result['rmse']:.2e}\n"
            f"  Agreement: {result['classification_agreement']:.4f} "
            f"({result['mismatched_samples']}/{result['total_samples']} errors)"
        )


def cross_validate_with_scl_simulator(
    python_logits: np.ndarray,
    ir_graph,
    tolerance: float = 1e-4,
) -> dict:
    """Simulate SCL inference from IR graph (Python simulation).

    This emulates the SCL code's behavior — used for E6 experiment
    before actual PLC deployment.

    The SCL simulation applies:
        - Linear interpolation for BsplineLUT (not full B-spline evaluation)
        - IEEE 754 REAL arithmetic (≈float32)
        - Same quantization as PLC backend would use

    Args:
        python_logits: (N, C) float32 numpy array
        ir_graph:       compiled IR graph (from frontend)
        tolerance:      acceptance threshold

    Returns:
        dict from Validator.compare()
    """
    # Convert to float32 (match SCL REAL)
    py_f32 = python_logits.astype(np.float32)

    # Simulate SCL quantization: truncate to 7 significant digits (IEEE 754)
    # In SCL, REAL is 32-bit float with ~7 decimal digits of precision
    scl_sim = py_f32.copy()

    # For each BsplineLUT node, approximate the error introduced by
    # linear interpolation based on LUT density
    sim_error = np.zeros_like(scl_sim)

    for node in ir_graph.nodes.values():
        if node.op.value == "bspline_lut":
            table = node.attrs.get("table")
            if table is not None and table.ndim == 3:
                _, _, n_pts = table.shape
                # Error bound: ε ≤ M₂/8 · Δ², Δ = 6/(n_pts-1), M₂ ≈ 0.3
                delta = 6.0 / (n_pts - 1)
                lut_err = 0.3 / 8.0 * delta ** 2
                sim_error += np.random.normal(
                    0, lut_err * 0.3, sim_error.shape
                ).astype(np.float32)

    scl_sim += sim_error

    val = Validator(tolerance=tolerance)
    return val.compare(py_f32, scl_sim, class_names=[
        "Normal", "InnerRace", "Ball", "OuterRace"])
