#!/usr/bin/env python3
"""
NeuroPLC — Frontend: PyTorch Model → IR Graph
================================================
Converts trained PyTorch models into IRGraph for compilation.

Supports:
    - KAN (StudentKAN)   → IR Graph with MatMul + BsplineLUT + Add
    - MLP (StudentMLP)   → IR Graph with MatMul + StandardAct
    - TeacherCNN         → NOT supported (too large for PLC)

KAN layer decomposition:
    Each KANLinear layer produces TWO IR paths:
      1. Base (SiLU) path:      input → MatMul(base_weight) → SiLU → merge
      2. B-spline path:         input → BsplineLUT → merge
    Then:                       merge = Add(base_out, spline_out)

    This decomposition is what makes KAN compilable:
    - MatMul handles the linear combination of inputs
    - SiLU is a StandardAct (simple formula in SCL)
    - BsplineLUT is a lookup table (the core innovation)
    - Add merges the two paths

Usage:
    from neuroplc.frontend import kan_to_ir, mlp_to_ir
    from models.student_kan import StudentKAN

    kan = StudentKAN([28, 16, 4])
    ir = kan_to_ir(kan, name="kan_28_16_4")
    print(ir.summary())
"""

import numpy as np
import torch
from typing import Optional

from .ir import IRGraph, IROpType, IRNode


# ============================================================================
# Weight extraction
# ============================================================================

def extract_kan_weights(model) -> dict:
    """
    Extract all trainable parameters from a StudentKAN model.

    Returns:
        {
            "layers": [
                {
                    "spline_weight": np.array (out, in, n_bases),
                    "base_weight":   np.array (out, in),
                    "scale_base":    float,
                    "scale_spline":  float,
                    "grid":          np.array (n_grid_points,),
                    "in_dim":  int,
                    "out_dim": int,
                },
                ...
            ],
            "num_classes": int,
        }
    """
    data = {"layers": [], "num_classes": model.num_classes}

    for layer in model.kan_layers:
        ld = {
            "spline_weight": layer.spline_weight.detach().cpu().numpy().copy(),
            "base_weight": layer.base_weight.detach().cpu().numpy().copy(),
            "scale_base": layer.scale_base.detach().cpu().item(),
            "scale_spline": layer.scale_spline.detach().cpu().item(),
            "grid": layer.grid.detach().cpu().numpy().copy(),
            "in_dim": layer.in_features,
            "out_dim": layer.out_features,
            "grid_size": layer.grid_size,
            "spline_order": layer.spline_order,
        }
        data["layers"].append(ld)

    return data


def extract_mlp_weights(model) -> dict:
    """
    Extract all Linear layer weights from a StudentMLP model.

    Returns:
        {
            "layers": [
                {"W": np.array (out, in), "b": np.array (out,),
                 "activation": "relu"},
                ...
            ],
            "num_classes": int,
        }
    """
    data = {"layers": [], "num_classes": model.num_classes}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            ld = {
                "W": module.weight.detach().cpu().numpy().copy(),
                "b": module.bias.detach().cpu().numpy().copy(),
                "in_dim": module.in_features,
                "out_dim": module.out_features,
                "activation": "relu",  # StudentMLP uses ReLU
            }
            data["layers"].append(ld)

    return data


# ============================================================================
# KAN → IR
# ============================================================================

def _build_bspline_lut(layer_data: dict, n_points: int = 20,
                        x_range: tuple = (-3.0, 3.0),
                        adaptive: bool = False) -> np.ndarray:
    """
    Build a B-spline lookup table for one KAN layer.

    For each (input, output) pair, sample the B-spline at N points
    and store as a (out, in, n_points) array.

    In the compiled SCL code, linear interpolation between neighboring
    table entries approximates the full B-spline evaluation.

    Args:
        layer_data: dict with spline_weight, base_weight, grid, etc.
        n_points:   number of sample points per activation function
        x_range:    input range for sampling
        adaptive:   if True, use curvature-aware non-uniform sampling

    Returns:
        table: (out_dim, in_dim, n_points) — sampled activation values
        grid_points: (n_points,) — x-coordinates of the table
    """
    import torch.nn.functional as F
    from models.student_kan import _bspline_basis

    spline_w = torch.from_numpy(layer_data["spline_weight"])  # (out, in, n_bases)
    grid = torch.from_numpy(layer_data["grid"])                # (G,)
    k = layer_data["spline_order"]
    out_dim, in_dim, _ = spline_w.shape

    if adaptive:
        # Adaptive sampling is handled as an optimizer pass
        # (adaptive_bspline_sampling in optimizer.py).
        # The frontend produces uniform sampling as a baseline;
        # the optimizer then refines knot placement based on curvature.
        # To apply adaptive sampling at IR construction time, call
        # optimizer.adaptive_bspline_sampling() as a post-frontend pass.
        xs = np.linspace(x_range[0], x_range[1], n_points)
    else:
        xs = np.linspace(x_range[0], x_range[1], n_points)

    xs_t = torch.from_numpy(xs).float()
    xs_scaled = xs_t / 3.0  # map [-3,3] → [-1,1] where grid lives

    # Compute B-spline basis at all sample points
    basis = _bspline_basis(xs_scaled, grid, k)  # (n_points, n_bases)

    # Evaluate all activation functions at once
    # spline_w: (out, in, n_bases), basis: (n_points, n_bases)
    # → table: (out, in, n_points)
    table = torch.einsum("o i b, p b -> o i p", spline_w, basis)

    return table.numpy(), xs


def kan_to_ir(model, name: str = "kan",
              lut_points: int = 20,
              x_range: tuple = (-3.0, 3.0),
              adaptive: bool = False) -> IRGraph:
    """
    Convert a trained StudentKAN model to IRGraph.

    For each KAN layer:
        1. Extract weights (spline_weight, base_weight, grid, scales)
        2. Pre-compute B-spline LUT (out_dim, in_dim, lut_points)
        3. Create IR nodes: MatMul(base) + SiLU + BsplineLUT + Add(merge)

    Args:
        model:      StudentKAN instance
        name:       graph name
        lut_points: B-spline LUT sampling density
        x_range:    input range for LUT
        adaptive:   use curvature-aware sampling (reserved for optimizer)

    Returns:
        IRGraph ready for optimization and code generation
    """
    weights = extract_kan_weights(model)
    g = IRGraph(name=name)

    # ── Per-layer decomposition ──
    prev_out_dim = weights["layers"][0]["in_dim"]

    # Input marker node (virtual — represents 28-D feature input)
    input_node = g.add_node(
        IROpType.MatMul,
        name="input_features",
        attrs={
            "W": np.eye(prev_out_dim, dtype=np.float32),
            "b": np.zeros(prev_out_dim, dtype=np.float32),
            "_virtual_input": True  # marker: this is the graph entry
        },
        shape_in=(prev_out_dim,),
        shape_out=(prev_out_dim,),
    )
    last_out = input_node

    for l_idx, ld in enumerate(weights["layers"]):
        in_d, out_d = ld["in_dim"], ld["out_dim"]

        # ── Path A: Base activation (SiLU via base_weight) ──
        # In KAN: y_base_j = Σ_i base_weight[j,i] · SiLU(x_i)
        # We decompose this into: SiLU(x) → MatMul(base_weight)
        # This is more SCL-friendly: apply SiLU first, then matrix multiply

        # SiLU activation (element-wise on the previous output)
        silu_node = g.add_node(
            IROpType.StandardAct,
            name=f"l{l_idx}_silu",
            attrs={"type": "silu"},
            shape_in=(prev_out_dim,),
            shape_out=(prev_out_dim,),
        )
        g.add_edge(last_out, silu_node)

        # Base linear transform
        base_matmul = g.add_node(
            IROpType.MatMul,
            name=f"l{l_idx}_base",
            attrs={
                "W": ld["base_weight"],         # (out, in)
                "b": np.zeros(out_d, dtype=np.float32),
                "_scale": ld["scale_base"],
            },
            shape_in=(in_d,),
            shape_out=(out_d,),
        )
        g.add_edge(silu_node, base_matmul)

        # ── Path B: B-spline activation ──
        table, grid_pts = _build_bspline_lut(
            ld, n_points=lut_points, x_range=x_range, adaptive=adaptive)

        bspline_node = g.add_node(
            IROpType.BsplineLUT,
            name=f"l{l_idx}_bspline",
            attrs={
                "table": table,                 # (out, in, lut_points)
                "grid": grid_pts,                # (lut_points,)
                "x_range": list(x_range),
                "_scale": ld["scale_spline"],
                "_spline_order": ld["spline_order"],
                "_grid_size": ld["grid_size"],
            },
            shape_in=(in_d,),
            shape_out=(in_d, out_d),  # intermediate: per-input, per-output
        )
        g.add_edge(last_out, bspline_node)

        # ── Merge: Add(base_out, spline_sum) ──
        # KAN: y_j = scale_base · Σ_i base_weight[j,i]·SiLU(x_i)
        #          + scale_spline · Σ_i Σ_c c[j,i,c]·B_c(x_i)
        # In IR: Add(base_matmul_output, sum_over_inputs(bspline_output))
        merge_node = g.add_node(
            IROpType.Add,
            name=f"l{l_idx}_merge",
            attrs={
                "scale_base": ld["scale_base"],
                "scale_spline": ld["scale_spline"],
            },
            shape_in=(out_d,),
            shape_out=(out_d,),
        )
        g.add_edge(base_matmul, merge_node, port=0)
        g.add_edge(bspline_node, merge_node, port=1)

        last_out = merge_node
        prev_out_dim = out_d

    # ── Output: Softmax + Argmax ──
    softmax_node = g.add_node(
        IROpType.Softmax, name="softmax",
        shape_in=(weights["num_classes"],),
        shape_out=(weights["num_classes"],))
    g.add_edge(last_out, softmax_node)

    argmax_node = g.add_node(
        IROpType.Argmax, name="argmax",
        shape_in=(weights["num_classes"],),
        shape_out=(1,))
    g.add_edge(softmax_node, argmax_node)

    return g


# ============================================================================
# MLP → IR
# ============================================================================

def mlp_to_ir(model, name: str = "mlp") -> IRGraph:
    """
    Convert a trained StudentMLP model to IRGraph.

    Each hidden layer: MatMul → StandardAct(ReLU)
    Output layer:     MatMul → Softmax → Argmax

    Args:
        model: StudentMLP instance
        name:  graph name

    Returns:
        IRGraph ready for code generation
    """
    weights = extract_mlp_weights(model)
    g = IRGraph(name=name)

    # Input
    in_dim = weights["layers"][0]["in_dim"]
    input_node = g.add_node(
        IROpType.MatMul, name="input_features",
        attrs={
            "W": np.eye(in_dim, dtype=np.float32),
            "b": np.zeros(in_dim, dtype=np.float32),
            "_virtual_input": True,
        },
        shape_in=(in_dim,), shape_out=(in_dim,),
    )
    last_out = input_node

    for l_idx, ld in enumerate(weights["layers"]):
        is_last = (l_idx == len(weights["layers"]) - 1)

        # MatMul
        fc = g.add_node(
            IROpType.MatMul, name=f"fc{l_idx}",
            attrs={"W": ld["W"], "b": ld["b"]},
            shape_in=(ld["in_dim"],), shape_out=(ld["out_dim"],),
        )
        g.add_edge(last_out, fc)

        if is_last:
            # Output
            sm = g.add_node(
                IROpType.Softmax, name="softmax",
                shape_in=(ld["out_dim"],), shape_out=(ld["out_dim"],))
            g.add_edge(fc, sm)
            am = g.add_node(
                IROpType.Argmax, name="argmax",
                shape_in=(ld["out_dim"],), shape_out=(1,))
            g.add_edge(sm, am)
            last_out = am
        else:
            # Hidden activation
            act = g.add_node(
                IROpType.StandardAct, name=f"relu{l_idx}",
                attrs={"type": ld["activation"]},
                shape_in=(ld["out_dim"],), shape_out=(ld["out_dim"],))
            g.add_edge(fc, act)
            last_out = act

    return g
