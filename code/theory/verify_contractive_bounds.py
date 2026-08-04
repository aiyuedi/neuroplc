#!/usr/bin/env python3
"""Full E68 bound chain for the contractive models (2026-08-04 P0 breakthrough).

The soft-contractive model (E-T9, gamma=[1.03,1.38], 98.5% acc) confines
layer-1 signals inside [-3,3] (Box-Continuation coverage 99.9% vs 0% for
the main student). This script recomputes the full certificate chain on
that checkpoint: per-layer L_B, DA/IA amplification constants, M2 profile,
DA/IA bounds and safety factors (verify_da_bounds_recomputed semantics).
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN, _bspline_basis

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_contractive.pt")
MARGIN_HALF = 0.675


def _m2_median(model):
    from neuroplc.per_function_verify import estimate_m2
    vals = []
    for layer in model.kan_layers:
        o, i, _ = layer.spline_weight.shape
        with torch.no_grad():
            for oi in range(o):
                for ii in range(i):
                    c = layer.spline_weight[oi, ii].numpy()
                    g = layer.grid.numpy()
                    vals.append(estimate_m2(c, g) / 9.0)   # input-domain M2
    return float(np.median(vals))


def layer_lipschitz(layer, n=1000):
    """L_B (numeric) and M2_max (scipy-exact, estimate_m2 — the numeric
    finite-difference M2 under-samples narrow segments; audit 2026-08-04)."""
    from neuroplc.per_function_verify import estimate_m2
    xs = torch.linspace(-3.0, 3.0, n, dtype=torch.float32)
    xs_scaled = xs / 3.0
    basis = _bspline_basis(xs_scaled, layer.grid, layer.spline_order)
    base_y = F.silu(xs)
    h = xs[1].item() - xs[0].item()
    lb, m2 = 0.0, 0.0
    o, i, _ = layer.spline_weight.shape
    with torch.no_grad():
        for oi in range(o):
            for ii in range(i):
                phi = (layer.scale_base * layer.base_weight[oi, ii] * base_y
                       + layer.scale_spline
                       * (basis * layer.spline_weight[oi, ii]).sum(-1))
                d1 = (phi[2:] - phi[:-2]) / (2 * h)
                lb = max(lb, float(d1.abs().max()))
                m2 = max(m2,
                         estimate_m2(layer.spline_weight[oi, ii].numpy(),
                                     layer.grid.numpy()) / 9.0)
    return lb, m2


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    model = StudentKAN([28, 16, 4])
    model.load_state_dict(sd, strict=False)
    model.eval()

    W0 = sd["kan_layers.0.base_weight"].numpy() * sd["kan_layers.0.scale_base"].numpy()
    W1 = sd["kan_layers.1.base_weight"].numpy() * sd["kan_layers.1.scale_base"].numpy()

    lb0, m2_max0 = layer_lipschitz(model.kan_layers[0])
    lb1, m2_max1 = layer_lipschitz(model.kan_layers[1])
    m2_max = max(m2_max0, m2_max1)
    # E11 median calibration scaled by the soft-contractive/main median ratio
    # (same methodology on both checkpoints; main median = 0.193 measured
    # with estimate_m2). Soft-contractive M2 distribution is far lighter.
    m2_median_soft = _m2_median(model)
    m2_char = 0.177 * m2_median_soft / 0.193

    eps_char = m2_char * (6.0 / 14) ** 2 / 8
    eps_max = m2_max * (6.0 / 14) ** 2 / 8

    t1 = float((np.abs(W1) @ np.abs(W0).sum(1)).max())
    t2 = float(np.abs(W1).sum(1).max())
    m_row = float(np.abs(W1 @ W0).sum(1).max())
    s1 = float(np.abs(W1.sum(1)).max())

    res = {"model": CKPT, "date": "2026-08-04",
           "L_B": [round(lb0, 3), round(lb1, 3)],
           "M2_max_measured": float(m2_max),
           "eps_char": eps_char, "eps_max": eps_max,
           "t1": t1, "t2": t2, "m_row": m_row, "s1": s1}
    for name, eps in [("M2_char", eps_char), ("M2_max", eps_max)]:
        d_da = eps * (lb1 * m_row + s1)
        d_ia = eps * (lb1 * t1 + t2)
        res[f"DA_{name}"] = d_da
        res[f"DA_safety_{name}"] = MARGIN_HALF / d_da
        res[f"IA_{name}"] = d_ia
        res[f"IA_safety_{name}"] = MARGIN_HALF / d_ia
        print(f"{name:6s}: DA={d_da:.4f} (safety {MARGIN_HALF/d_da:.2f}x)  "
              f"IA={d_ia:.4f} (safety {MARGIN_HALF/d_ia:.2f}x)")

    # Sound worst-function certificate: IA form (no cancellation) at M2_max,
    # floored by the full-set measured maxAE (Tier-4 simulator, 13,714
    # inputs, 2026-08-04). The DA sign-cancellation form under-bounds the
    # fresh-term path on this checkpoint (measured 0.053 vs DA-max 0.030),
    # so the sound row uses the no-cancellation envelope.
    MEASURED_MAXAE = 0.0527
    d_sound_ia = eps_max * (lb1 * t1 + t2)
    d_sound = max(d_sound_ia, MEASURED_MAXAE * 1.1)
    res["sound_bound_M2max_IA"] = d_sound_ia
    res["sound_bound_floored"] = d_sound
    res["sound_safety"] = MARGIN_HALF / d_sound
    res["measured_maxae_fullset"] = MEASURED_MAXAE
    print(f"SOUND (M2_max, IA form, floored): bound={d_sound:.4f} "
          f"(safety {MARGIN_HALF/d_sound:.1f}x; measured {MEASURED_MAXAE})")

    with open(os.path.join(BASE, "results", "theory", "contractive_bounds.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"\nSaved: results/theory/contractive_bounds.json")


if __name__ == "__main__":
    main()
