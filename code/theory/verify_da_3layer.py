#!/usr/bin/env python3
"""Recompute the 3-layer DA bound with E68-measured constants (2026-08-04 R4).

Uses the E56 manual 3-layer sign-structural propagation (same form) with the
audit constants: eps = 0.00406 (M2_char=0.177, h=6/14, N=15) and per-layer
L_B measured on the 3L checkpoint itself (1,000-pt grid, E68 method).

2L reference (same eps/L_B family): DA = 0.66, from
verify_da_bounds_recomputed.py on the released main student.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN, _bspline_basis

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "kan_3layer", "kan_28x16x8x4.pt")
EPS = 0.00406   # M2_char=0.177, h=6/14 @ N=15
DA_2L = 0.6586  # released main student (verify_da_bounds_recomputed.py)


def layer_lipschitz(layer, n=1000):
    """Max |phi'| over all activation functions of a KANLinear layer (E68 method)."""
    xs = torch.linspace(-3.0, 3.0, n, dtype=torch.float32)
    xs_scaled = xs / 3.0
    basis = _bspline_basis(xs_scaled, layer.grid, layer.spline_order)  # (n, G+k)
    base_y = torch.nn.functional.silu(xs)
    h = xs[1].item() - xs[0].item()
    lb = 0.0
    o, i, _ = layer.spline_weight.shape
    with torch.no_grad():
        for oi in range(o):
            for ii in range(i):
                phi = (layer.scale_base * layer.base_weight[oi, ii] * base_y
                       + layer.scale_spline
                       * (basis * layer.spline_weight[oi, ii]).sum(-1))
                dphi = (phi[2:] - phi[:-2]) / (2 * h)
                lb = max(lb, dphi.abs().max().item())
    return lb


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    model = StudentKAN([28, 16, 8, 4])
    model.load_state_dict(sd, strict=False)
    model.eval()

    layers = model.kan_layers
    # base-folded weights (matches verify_da_bounds_recomputed.py)
    ws = [l.base_weight.detach().numpy() * l.scale_base.detach().numpy()
          for l in layers]
    w0, w1, w2 = ws
    d0, d1, d2 = w0.shape[1], w1.shape[1], w2.shape[1]

    lbs = [layer_lipschitz(l) for l in layers]
    print(f"per-layer L_B (3L checkpoint): [{lbs[0]:.3f}, {lbs[1]:.3f}, {lbs[2]:.3f}]")

    # E56 manual 3-layer propagation, audit constants (eps, L_B per layer)
    err_l0 = EPS * d0                                # layer-0 output LUT error
    da_l1 = np.abs(w1).sum(axis=1) * err_l0          # (8,) through W1
    da_l1_max = float(da_l1.max())
    fresh_l2 = EPS * d1 * lbs[1]                     # fresh layer-1 LUT error
    da_l2 = np.abs(w2).sum(axis=1) * da_l1_max + fresh_l2  # (4,)
    da_3l = float(da_l2.max())

    print(f"d0={d0} d1={d1} d2={d2}")
    print(f"3L DA = {da_3l:.4f}   (E56 2026-08-04 rerun said 1.899 with stale lb=0.65)")
    print(f"2L DA = {DA_2L:.4f}   (verify_da_bounds_recomputed, same eps family)")
    print(f"depth ratio 3L/2L = {da_3l / DA_2L:.2f}x   (paper said 15.3x on stale 0.064 base)")
    print(f"worst-case envelope gamma^(L-1) = 5.3 -> sub-exponential holds: {da_3l / DA_2L < 5.3}")


if __name__ == "__main__":
    main()
