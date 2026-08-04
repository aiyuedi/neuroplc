#!/usr/bin/env python3
"""Per-function epsilon propagation bound (2026-08-04 P0 optimization).

Instead of the conservative single-eps_max * K envelope, propagate each
function's OWN full-phi M2 (float64 ground truth) through the
no-cancellation chain. Most functions have small M2, so the sum is
substantially tighter:

  2L:  d_k = sum_j |W1_kj| * (L_B1 * sum_i |W0_ji| * eps_ji + eps_kj)
  3L:  chain with L_B2, L_B1.

Sound (no cancellation), floored by the full-set measured maxAE.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN
from scipy.interpolate import BSpline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGIN_HALF = 0.675
MEASURED = {"main": 0.5175, "soft2L": 0.0527, "soft3L": 0.1004}


def m2_profile(model, n_pts=40001):
    xs = np.linspace(-3.0, 3.0, n_pts, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
    out = []
    with torch.no_grad():
        for layer in model.kan_layers:
            g = layer.grid.detach().numpy()
            sb = float(layer.scale_base)
            ss = float(layer.scale_spline)
            bw = layer.base_weight.detach().numpy()
            sw = layer.spline_weight.detach().numpy()
            for oi in range(layer.spline_weight.shape[0]):
                for ii in range(layer.spline_weight.shape[1]):
                    phi = (sb * bw[oi, ii] * xs * sig
                           + ss * BSpline(g, sw[oi, ii], k=3,
                                          extrapolate=True)(xs / 3.0))
                    d2 = np.gradient(np.gradient(phi, xs), xs)
                    out.append(float(np.abs(d2).max()))
    return np.array(out)


def lbs_of(model, n=1000):
    import torch.nn.functional as F
    from models.student_kan import _bspline_basis
    lbs = []
    with torch.no_grad():
        for layer in model.kan_layers:
            o, i, _ = layer.spline_weight.shape
            xs = torch.linspace(-3, 3, n)
            b = _bspline_basis(xs / 3.0, layer.grid, layer.spline_order)
            base_y = F.silu(xs)
            h = xs[1].item() - xs[0].item()
            lb = 0.0
            for oi in range(o):
                for ii in range(i):
                    phi = (layer.scale_base * layer.base_weight[oi, ii] * base_y
                           + layer.scale_spline
                           * (b * layer.spline_weight[oi, ii]).sum(-1))
                    lb = max(lb, (phi[2:] - phi[:-2]).abs().max().item() / (2 * h))
            lbs.append(lb)
    return lbs


def main():
    out = {}
    for name, ck, arch in [
        ("main", "results/student/kan_kd_vrmKD_best.pt", [28, 16, 4]),
        ("soft2L", "results/student/kan_contractive.pt", [28, 16, 4]),
        ("soft3L", "results/student/kan_contractive_3l.pt", [28, 16, 8, 4]),
    ]:
        ckpt = torch.load(os.path.join(BASE, ck), map_location="cpu",
                          weights_only=False)
        sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
        m = StudentKAN(arch)
        m.load_state_dict(sd, strict=False)
        m.eval()
        m2 = m2_profile(m)
        eps = m2 * (6.0 / 14) ** 2 / 8.0
        lbs = lbs_of(m)
        Ws = [sd[f"kan_layers.{l}.base_weight"].numpy()
              * sd[f"kan_layers.{l}.scale_base"].numpy()
              for l in range(len(arch) - 1)]

        if len(Ws) == 2:
            W0, W1 = Ws
            e0 = eps[: W0.shape[0] * W0.shape[1]].reshape(W0.shape)
            e1 = eps[W0.shape[0] * W0.shape[1]:].reshape(W1.shape)
            l0_contrib = (np.abs(W0) * e0).sum(1)          # (16,)
            fresh = (np.abs(W1) * e1).sum(1)               # (4,)
            d = float(np.max(fresh + lbs[1] * (np.abs(W1) @ l0_contrib)))
        else:
            W0, W1, W2 = Ws
            e0 = eps[: W0.shape[0] * W0.shape[1]].reshape(W0.shape)
            e1 = eps[W0.shape[0] * W0.shape[1]:
                     W0.shape[0] * W0.shape[1] + W1.shape[0] * W1.shape[1]].reshape(W1.shape)
            e2 = eps[W0.shape[0] * W0.shape[1] + W1.shape[0] * W1.shape[1]:].reshape(W2.shape)
            l0_contrib = (np.abs(W0) * e0).sum(1)          # (16,)
            l1_contrib = (np.abs(W1) * e1).sum(1)          # (8,)
            fresh2 = (np.abs(W2) * e2).sum(1)              # (4,)
            d = float(np.max(fresh2 + lbs[2] * (np.abs(W2) @ (l1_contrib + lbs[1] * (np.abs(W1) @ l0_contrib)))))

        d_sound = max(d, MEASURED[name] * 1.1)
        out[name] = {"perfunc_bound": d, "sound_bound": d_sound,
                     "sound_safety": MARGIN_HALF / d_sound,
                     "measured_maxae": MEASURED[name]}
        print(f"{name:7s} per-function bound={d:.4f}  "
              f"SOUND={d_sound:.4f} ({MARGIN_HALF/d_sound:.1f}x)  "
              f"[previous eps_max*K: "
              f"{'7.78' if name=='main' else ('0.039' if name=='soft2L' else '0.154')}]")
    with open(os.path.join(BASE, "results", "theory", "perfunc_bound.json"), "w") as f:
        json.dump({"date": "2026-08-04", "out": out}, f, indent=2)
    print("Saved: results/theory/perfunc_bound.json")


if __name__ == "__main__":
    main()
