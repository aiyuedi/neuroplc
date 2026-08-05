#!/usr/bin/env python3
"""v5 (pervasive-contractivity) sound-certificate recomputation (2026-08-06).

The per-edge-L recipe (verify_bspline_peredge.py) produced
kan_contractive_v5.pt with measured gamma = [0.95, 0.95] (E68 semantics)
at 98.49% accuracy.  Every weight was globally scaled (L0 x0.842,
L1 x0.884) plus a constrained L1 fine-tune, so the first-principles
sound bound (full-phi M2 x LUT error x IA propagation constant) should
shrink accordingly.  Same methodology as verify_sound_chain.py (audit
v3, 2026-08-04): sound = max(worst-tier IA bound, measured x 1.1).
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
MEASURED_V3 = 0.0527          # Tier-4 full set, v3 checkpoint


def profile(model, n_pts=80001):
    """Full-phi M2 via float64 numeric second derivative (ground truth)."""
    from scipy.interpolate import BSpline
    xs = np.linspace(-3.0, 3.0, n_pts, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
    m2s = []
    with torch.no_grad():
        for layer in model.kan_layers:
            g = layer.grid.detach().numpy()
            sb = float(layer.scale_base)
            ss = float(layer.scale_spline)
            o, i, _ = layer.spline_weight.shape
            bw = layer.base_weight.detach().numpy()
            sw = layer.spline_weight.detach().numpy()
            for oi in range(o):
                for ii in range(i):
                    phi = (sb * bw[oi, ii] * xs * sig
                           + ss * BSpline(g, sw[oi, ii], k=3,
                                          extrapolate=True)(xs / 3.0))
                    d2 = np.gradient(np.gradient(phi, xs), xs)
                    m2s.append(float(np.abs(d2).max()))
    return np.array(m2s)


def ia_prop_constants(Ws, lbs):
    """IA no-cancellation propagation constant (2-layer)."""
    W0, W1 = Ws
    t1 = float((np.abs(W1) @ np.abs(W0).sum(1)).max())
    t2 = float(np.abs(W1).sum(1).max())
    return lbs[1] * t1 + t2


def run(name, path, arch):
    ckpt = torch.load(os.path.join(BASE, path), map_location="cpu",
                      weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN(arch)
    m.load_state_dict(sd, strict=False)
    m.eval()
    m2 = profile(m)
    Ws = [sd[f"kan_layers.{l}.base_weight"].numpy()
          * sd[f"kan_layers.{l}.scale_base"].numpy()
          for l in range(len(arch) - 1)]
    import torch.nn.functional as F
    from models.student_kan import _bspline_basis
    lbs = []
    with torch.no_grad():
        for layer in m.kan_layers:
            o, i, _ = layer.spline_weight.shape
            xs = torch.linspace(-3, 3, 1000)
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
    med, mx = float(np.median(m2)), float(m2.max())
    eps_c = med * (6.0 / 14) ** 2 / 8
    eps_m = mx * (6.0 / 14) ** 2 / 8
    K = ia_prop_constants(Ws, lbs)
    d_char = eps_c * K
    d_max = eps_m * K
    d_sound = max(d_max, MEASURED_V3 * 1.1)
    print(f"{name:6s} M2 med={med:.4f} max={mx:.4f} K={K:.3f}  "
          f"expected {d_char:.4f} ({MARGIN_HALF/d_char:.1f}x)  "
          f"worst {d_max:.4f} ({MARGIN_HALF/d_max:.1f}x)  "
          f"SOUND {d_sound:.4f} ({MARGIN_HALF/d_sound:.1f}x)")
    return {"model": name, "m2_median": med, "m2_max": mx, "K": K,
            "expected_tier_bound": d_char, "expected_safety": MARGIN_HALF / d_char,
            "worst_tier_bound": d_max, "worst_safety": MARGIN_HALF / d_max,
            "sound_bound": d_sound, "sound_safety": MARGIN_HALF / d_sound}


def main():
    out = {
        "v3": run("v3", "results/student/kan_contractive_v3.pt", [28, 16, 4]),
        "v5": run("v5", "results/student/kan_contractive_v5.pt", [28, 16, 4]),
    }
    out["date"] = "2026-08-06"
    out["note"] = ("sound = max(worst-tier IA, measured_v3*1.1); "
                   "v5 = pervasive-contractivity checkpoint (gamma=[0.95,0.95])")
    with open(os.path.join(BASE, "results", "theory", "v5_sound.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/v5_sound.json")


if __name__ == "__main__":
    main()
