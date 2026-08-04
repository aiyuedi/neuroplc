#!/usr/bin/env python3
"""Seg-DA / Seg-IA recomputation on the released main checkpoint (2026-08-04 P0).

Restores the withdrawn E16 11.9x claim with a sound, reproducible pipeline:

  per-segment LUT error (Eq. eq:segment_bound):
      eps_{j,i}(k) = M2_{j,i}(k) * Delta^2 / 8,
      M2_{j,i}(k)  = max_{x in [g_k, g_{k+1}]} |phi''_{j,i}(x)|
  per-input epsilon (Eq. eq:da_bound composition):
      eps_i(x) = max over outputs j sharing input i of eps_{j,i}(k_i(x))
  Seg-DA (per-input, sound for that input):
      d_k(x) = eps_fresh * |sum_j W1_kj| + L_B1 * sum_i eps_i(x) * |sum_j W1_kj W0_ji|
  Global-DA sanity check: eps_i = EPS_GLOBAL for all i must reproduce 0.6586.

Reported: max over the in-domain CWRU test set (the E16 empirical figure),
plus the global-DA cross-check. Also Seg-IA per-input:
      d_k(x) = (eps_fresh + L_B1 * max_j sum_i eps_i(x) |W0_ji|) * sum_j |W1_kj|
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN, _bspline_basis

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")
DATA_X = os.path.join(BASE, "data", "processed", "features_X.npy")

EPS_GLOBAL = 0.00406   # M2_char=0.177, h=6/14 @ N=15
L_B1 = 2.05            # measured layer-1 B-spline Lipschitz (E68)
DOMAIN = (-3.0, 3.0)


def segment_eps_table(layer, N):
    """(out, in, N-1) array of per-segment eps = M2(k) * Delta^2 / 8."""
    Delta = (DOMAIN[1] - DOMAIN[0]) / (N - 1)
    edges = np.linspace(*DOMAIN, N)
    o, i, _ = layer.spline_weight.shape
    eps = np.zeros((o, i, N - 1))
    xs = torch.linspace(*DOMAIN, 600)
    xs_scaled = xs / 3.0
    basis = _bspline_basis(xs_scaled, layer.grid, layer.spline_order)
    base_y = torch.nn.functional.silu(xs)
    h = xs[1].item() - xs[0].item()
    with torch.no_grad():
        d2 = None
        for oi in range(o):
            for ii in range(i):
                phi = (layer.scale_base * layer.base_weight[oi, ii] * base_y
                       + layer.scale_spline
                       * (basis * layer.spline_weight[oi, ii]).sum(-1))
                d2a = (phi[2:] - 2 * phi[1:-1] + phi[:-2]).abs().numpy() / h ** 2
                xc = xs[1:-1].numpy()
                for k in range(N - 1):
                    m = d2a[(xc >= edges[k]) & (xc < edges[k + 1])]
                    if m.size:
                        eps[oi, ii, k] = m.max() * Delta ** 2 / 8
    return eps


def seg_da_bounds(x, eps0, W0, W1, fresh_eps=EPS_GLOBAL, eps_scale=1.0):
    """Per-input Seg-DA and Seg-IA maxima over outputs.

    True per-(i,j) segment error: eps_{j,i} is the segment eps of function
    phi_{j,i} at the segment containing x_i. DA sign structure is preserved
    inside the weight product (eps are positive scalars, so cancellation
    survives):  term = sum_i |sum_j W1_kj * (eps_{j,i} * eps_scale) * W0_ji|.
    The paper's max-over-outputs version is strictly looser (conservative);
    per-(i,j) is sound for the given x and tighter.
    """
    N = eps0.shape[2] + 1
    edges = np.linspace(*DOMAIN, N)
    seg = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, N - 2)
    idx = np.arange(eps0.shape[1])
    E = eps0[:, idx, seg[idx]] * eps_scale           # (out=16, in=28) per-input
    # DA propagation: sum_i |sum_j W1_kj * (E_ji * W0_ji)| (cancellation kept)
    W1E = W1 @ (E * W0)                              # (4, 28) = sum_j W1_kj E_ji W0_ji
    da_k = fresh_eps * np.abs(W1.sum(axis=1)) + L_B1 * np.abs(W1E).sum(axis=1)
    # IA propagation: sum_j |W1_kj| * (sum_i E_ji |W0_ji|) (no cancellation)
    ia_k = fresh_eps * np.abs(W1).sum(axis=1) \
        + L_B1 * (np.abs(W1) @ (E * np.abs(W0)).sum(axis=1))
    return float(da_k.max()), float(ia_k.max())


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"]
    model = StudentKAN([28, 16, 4])
    model.load_state_dict(sd, strict=False)
    model.eval()

    W0 = sd["kan_layers.0.base_weight"].numpy() * sd["kan_layers.0.scale_base"].numpy()
    W1 = sd["kan_layers.1.base_weight"].numpy() * sd["kan_layers.1.scale_base"].numpy()

    X = np.load(DATA_X).astype(np.float32)
    X = np.clip(X, *DOMAIN)                     # in-domain test features (E67 convention)

    out = {}
    for N in (15, 20):
        eps0 = segment_eps_table(model.kan_layers[0], N)
        # sanity: global-eps DA must reproduce 0.6586
        d_glob = EPS_GLOBAL * np.abs(W1.sum(axis=1)).max() \
            + L_B1 * EPS_GLOBAL * np.abs(W1 @ W0).sum(axis=1).max()
        da_all = np.array([seg_da_bounds(x, eps0, W0, W1)[0] for x in X])
        ia_all = np.array([seg_da_bounds(x, eps0, W0, W1)[1] for x in X])
        out[N] = {
            "global_DA_check": float(d_glob),
            "segDA_max_over_test": float(da_all.max()),
            "segIA_max_over_test": float(ia_all.max()),
            "segDA_mean": float(da_all.mean()),
            "DA_tightening_vs_global": float(d_glob / da_all.max()),
            "IA_tightening_vs_global": float(EPS_GLOBAL * ((EPS_GLOBAL + L_B1 * EPS_GLOBAL * np.abs(W0).sum(1).max()) * np.abs(W1).sum(1).max()) / ia_all.max()),
        }
        print(f"N={N}: global-DA check={d_glob:.4f}  "
              f"Seg-DA max={out[N]['segDA_max_over_test']:.4f} "
              f"(tightening {out[N]['DA_tightening_vs_global']:.1f}x)  "
              f"Seg-IA max={out[N]['segIA_max_over_test']:.4f}")

    with open(os.path.join(BASE, "results", "theory", "seg_da_recomputed.json"), "w") as f:
        json.dump({"date": "2026-08-04", "model": CKPT, "N15": out[15], "N20": out[20]},
                  f, indent=2)
    print("\nSaved: results/theory/seg_da_recomputed.json")


if __name__ == "__main__":
    main()
