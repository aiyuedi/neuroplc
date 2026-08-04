#!/usr/bin/env python3
"""Adaptive allocation on the soft-contractive checkpoint (2026-08-04 P0).

E15 greedy allocation (Algorithm 2) applied to the E-T9 model: per-function
M2 (scipy-exact, input domain) drives N_phi allocation under the same
S7-1200 budget B = 30,720 B (4 bytes/point), N_min = 3.
  minimize max_phi eps(phi, N_phi)  s.t.  4 * sum N_phi <= B
Sound bound = eps_adaptive_max * (IA no-cancellation propagation constants)
of the soft-contractive model, floored by measured maxAE (Tier-4).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN
from neuroplc.per_function_verify import estimate_m2

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_contractive.pt")
BUDGET_BYTES = 30720     # S7-1200 default (uniform N=15 equivalent)
BYTES_PER_PT = 4
N_MIN = 3
DOMAIN = 6.0             # [-3, 3]
MARGIN_HALF = 0.675


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN([28, 16, 4])
    m.load_state_dict(sd, strict=False)
    m.eval()

    # per-function M2 (input domain, scipy-exact)
    m2 = []
    with torch.no_grad():
        for layer in m.kan_layers:
            o, i, _ = layer.spline_weight.shape
            for oi in range(o):
                for ii in range(i):
                    m2.append(estimate_m2(
                        layer.spline_weight[oi, ii].detach().numpy(),
                        layer.grid.detach().numpy()) / 9.0)
    m2 = np.array(m2)
    n_funcs = len(m2)

    # greedy allocation (Algorithm 2): start N=3, repeatedly give +1 point
    # to the function with the current worst epsilon.
    N = np.full(n_funcs, N_MIN, dtype=int)
    budget = BUDGET_BYTES // BYTES_PER_PT
    used = N.sum()
    while used < budget:
        eps = m2 / (8.0 * (N - 1) ** 2) * DOMAIN ** 2   # eps(N) = M2 * L^2/(8(N-1)^2)
        w = int(np.argmax(eps))
        N[w] += 1
        used += 1

    eps_final = m2 * DOMAIN ** 2 / (8.0 * (N - 1) ** 2)
    eps_adapt = float(eps_final.max())
    eps_uniform = float(m2.max() * DOMAIN ** 2 / (8.0 * (15 - 1) ** 2))

    # sound propagation (IA no-cancellation, soft-contractive constants)
    W0 = sd["kan_layers.0.base_weight"].numpy() * sd["kan_layers.0.scale_base"].numpy()
    W1 = sd["kan_layers.1.base_weight"].numpy() * sd["kan_layers.1.scale_base"].numpy()
    L_B1 = 0.232   # measured layer-1 (E-T9)
    t1 = float((np.abs(W1) @ np.abs(W0).sum(1)).max())
    t2 = float(np.abs(W1).sum(1).max())
    d_sound_adapt = eps_adapt * (L_B1 * t1 + t2)
    d_sound_uniform = eps_uniform * (L_B1 * t1 + t2)

    print(f"functions: {n_funcs}  M2 median {np.median(m2):.4f}  M2 max {m2.max():.4f}")
    print(f"eps: uniform {eps_uniform:.6f} -> adaptive {eps_adapt:.6f} "
          f"({(1-eps_adapt/eps_uniform)*100:.1f}% reduction)")
    print(f"N range [{N.min()}, {N.max()}]  median {np.median(N):.0f}")
    print(f"sound bound: uniform {d_sound_uniform:.4f} (safety {MARGIN_HALF/d_sound_uniform:.1f}x) "
          f"-> adaptive {d_sound_adapt:.4f} (safety {MARGIN_HALF/d_sound_adapt:.1f}x)")

    with open(os.path.join(BASE, "results", "theory", "adaptive_soft.json"), "w") as f:
        json.dump({"date": "2026-08-04", "model": CKPT,
                   "eps_uniform": eps_uniform, "eps_adaptive": eps_adapt,
                   "N_range": [int(N.min()), int(N.max())],
                   "N_median": float(np.median(N)),
                   "sound_uniform": d_sound_uniform,
                   "sound_adaptive": d_sound_adapt,
                   "sound_safety_adaptive": MARGIN_HALF / d_sound_adapt}, f, indent=2)
    print("Saved: results/theory/adaptive_soft.json")


if __name__ == "__main__":
    main()
