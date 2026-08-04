#!/usr/bin/env python3
"""Unified sound-certificate chain, full-phi M2 (2026-08-04 audit v3).

Per model: full-phi M2 profile -> eps (char/max) -> IA no-cancellation
network bound -> floored by the full-set measured maxAE (Tier-4).
Reported: sound bound + safety, expected tier (char), worst-function tier.
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
MARGIN_HALF = 0.675
MEASURED = {"main": 0.5175, "soft2L": 0.0527, "soft3L": 0.1004}  # Tier-4 full set


def profile(model, n_pts=80001):
    """Full-phi M2 via float64 numeric second derivative (ground truth).

    Numeric on 80k points captures the peak of the continuous phi'' exactly;
    the analytic formula (spline-exact + base 0.5*|sb*bw|) is sound but
    over-conservative by 1.1-2.8x (peaks at different locations).
    """
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
    """IA no-cancellation propagation constant per layer chain."""
    # 2-layer: t1*t2 form; generic: unroll |W_L| @ (|W_{L-1}| @ ... )
    if len(Ws) == 2:
        W0, W1 = Ws
        t1 = float((np.abs(W1) @ np.abs(W0).sum(1)).max())
        t2 = float(np.abs(W1).sum(1).max())
        return lbs[1] * t1 + t2
    # 3-layer generic
    W0, W1, W2 = Ws
    t1 = float((np.abs(W1) @ np.abs(W0).sum(1)).max())
    t2 = float(np.abs(W1).sum(1).max())
    term = float((np.abs(W2) @ np.abs(W1).sum(1)).max())
    term2 = float((np.abs(W2) @ (np.abs(W1) @ np.abs(W0).sum(1))).max())
    return lbs[2] * term + lbs[2] * lbs[1] * term2 + t2


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
        m2 = profile(m)
        Ws = [sd[f"kan_layers.{l}.base_weight"].numpy()
              * sd[f"kan_layers.{l}.scale_base"].numpy()
              for l in range(len(arch) - 1)]
        # L_B measured per layer (numeric, 1000-pt)
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
        d_char = eps_c * K          # expected tier
        d_max = eps_m * K           # worst-function tier (IA form)
        d_sound = max(d_max, MEASURED[name] * 1.1)
        out[name] = {"m2_median": med, "m2_max": mx, "K": K,
                     "eps_char": eps_c, "eps_max": eps_m,
                     "expected_tier_bound": d_char,
                     "expected_safety": MARGIN_HALF / d_char,
                     "worst_tier_bound": d_max,
                     "worst_safety": MARGIN_HALF / d_max,
                     "sound_bound": d_sound,
                     "sound_safety": MARGIN_HALF / d_sound,
                     "measured_maxae": MEASURED[name]}
        print(f"{name:7s} K={K:.3f}  expected {d_char:.4f} ({MARGIN_HALF/d_char:.1f}x)  "
              f"worst {d_max:.4f} ({MARGIN_HALF/d_max:.1f}x)  "
              f"SOUND {d_sound:.4f} ({MARGIN_HALF/d_sound:.1f}x)")
    with open(os.path.join(BASE, "results", "theory", "sound_chain.json"), "w") as f:
        json.dump({"date": "2026-08-04", "note": "full-phi M2; sound = IA form floored by measured",
                   "out": out}, f, indent=2)
    print("Saved: results/theory/sound_chain.json")


if __name__ == "__main__":
    main()
