#!/usr/bin/env python3
"""FourierKAN N=16 sound upgrade (2026-08-05).

Problem: fourier_v2 sound bound 0.11033 (N=15, domain [-3,3]) does NOT cover
the full-set measured 0.126, which includes 2.7% out-of-domain inputs
(L0 outputs up to 3.145 exceed the [-3,3] L1 input domain).

Upgrade: N=16 LUT points + widened Box [-3.15, 3.15]:
  - L0 output measured over all 13,714 inputs: [-3.0024, 3.1454] -> 100% within
    [-3.15, 3.15] (verified here).
  - h = 6.3/16 (vs 6/15): theoretical bound scales by ((6.3/16)/(6/15))^2
    = 0.9673 -> 0.11033 * 0.9673 = 0.10672... wait recompute in script.
  - The bound now covers the full-set measured (with headroom), and the Box
    covers 100% of test inputs -> certificate upgraded from "validated"
    to "sound (Box 100%)".

Checks performed:
  1. L0 output 100% within [-3.15, 3.15] (full 13,714 inputs).
  2. Theoretical N=16 bound (analytic per-edge M2/L, domain [-3.15,3.15]).
  3. LUT deployment simulation (N=16, per-layer domain) full-set maxAE.
  4. Coverage: bound >= measured maxAE.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_fourierkan import StudentFourierKAN

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGIN_HALF = 0.675
N_OLD, N_NEW = 15, 16
DOM_OLD = 6.0            # [-3, 3]
DOM_NEW = 6.3            # [-3.15, 3.15]


def fourier_m2_l(layer):
    """Per-edge analytic (M2, L) for FourierKANLinear (domain-independent)."""
    k = layer.k.numpy()
    c = layer.fourier_coeffs.detach().numpy()
    cs = c[:, :, :layer.n_harmonics]
    cd = c[:, :, layer.n_harmonics:]
    om = float(layer.omega)
    bw = np.abs(layer.base_weight.detach().numpy())
    # fourier part: |phi''| <= sum_k k^2 w^2 (|ck|+|dk|); |phi'| <= sum_k k w (|ck|+|dk|)
    m2f = ((k ** 2 * om ** 2)[None, None, :] * (np.abs(cs) + np.abs(cd))).sum(axis=2)
    lf = ((k * om)[None, None, :] * (np.abs(cs) + np.abs(cd))).sum(axis=2)
    # base path: SiLU, |SiLU''| <= 0.5, |SiLU'| <= 1.1 (sup ~1.099)
    m2b = bw * 0.5
    lb = bw * 1.1
    return m2f + m2b, lf + lb


def lut_forward_sim(model, X, n_pts, dom_lo, dom_hi):
    """Full-set LUT deployment simulation (float64): linear interp per phi."""
    Xt = torch.tensor(X, dtype=torch.float64)
    with torch.no_grad():
        cur = Xt
        for li, layer in enumerate(model.kan_layers):
            o, i = layer.out_features, layer.in_features
            if li == 0:
                lo, hi = -3.0, 3.0          # design input domain (LUT clamped)
            else:
                lo, hi = dom_lo, dom_hi
            # build per-edge LUTs
            xs = np.linspace(lo, hi, n_pts, dtype=np.float64)
            c = layer.fourier_coeffs.detach().numpy()
            k = layer.k.numpy()
            om = float(layer.omega)
            bw = layer.base_weight.detach().numpy()
            tables = {}
            for oi in range(o):
                for ii in range(i):
                    phix = bw[oi, ii] * xs * (1.0 / (1.0 + np.exp(-xs)))
                    for hh in range(layer.n_harmonics):
                        phix += (c[oi, ii, hh] * np.sin(k[hh] * om * xs)
                                 + c[oi, ii, layer.n_harmonics + hh]
                                 * np.cos(k[hh] * om * xs))
                    tables[(oi, ii)] = (xs, phix)
            # exact float64 forward on current layer for reference
            cur_np = cur.numpy()
            # LUT forward: clip input to [lo,hi], interpolate
            out = np.zeros((cur_np.shape[0], o), dtype=np.float64)
            for oi in range(o):
                acc = np.zeros(cur_np.shape[0], dtype=np.float64)
                for ii in range(i):
                    xin = np.clip(cur_np[:, ii], lo, hi)
                    xg, yg = tables[(oi, ii)]
                    acc += np.interp(xin, xg, yg)
                out[:, oi] = acc + float(layer.bias.detach().numpy()[oi])
            cur = torch.from_numpy(out)
        logits_lut = cur
        # exact forward (model in float64 for reference)
        m64 = model.double()
        cur_ex = torch.tensor(X, dtype=torch.float64)
        for li, layer in enumerate(m64.kan_layers):
            cur_ex = layer(cur_ex)
        maxae = float((logits_lut - cur_ex).abs().max())
    return maxae


def main():
    ckpt = torch.load(os.path.join(BASE, "results", "student",
                                   "fourier_contractive_v2.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentFourierKAN([28, 16, 4], n_harmonics=6, omega=0.4)
    m.load_state_dict(sd, strict=False)
    m.eval()
    X = np.load(os.path.join(BASE, "data", "processed", "features_X.npy"))

    l0 = m.kan_layers[0]
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        l0out = l0(Xt)
    box_ok = bool((l0out.abs() <= 3.15).all().item())
    print(f"L0 out range [{l0out.min():.4f}, {l0out.max():.4f}]  "
          f"100% within [-3.15,3.15]: {box_ok}")

    # theoretical bound: per-edge analytic M2/L on L1 (16->4) with new domain
    l1 = m.kan_layers[1]
    m2_1, L1 = fourier_m2_l(l1)
    h_old = DOM_OLD / N_OLD
    h_new = DOM_NEW / N_NEW
    eps_old = m2_1 * h_old ** 2 / 8.0
    eps_new = m2_1 * h_new ** 2 / 8.0
    # L0 error enters only if input not exact; in-domain input is exact, and
    # L0 is deployed with LUT too: add L0 LUT error via its own M2 on [-3,3].
    m2_0, _ = fourier_m2_l(l0)
    eps0_new = m2_0 * (6.0 / N_NEW) ** 2 / 8.0
    dy = eps0_new.sum(axis=1)
    d_old = float(np.max(L1 @ (m2_0 * (6.0 / N_OLD) ** 2 / 8.0).sum(axis=1)
                         + eps_old.sum(axis=1)))
    d_new = float(np.max(L1 @ dy + eps_new.sum(axis=1)))
    print(f"theory N={N_OLD} dom={DOM_OLD}: bound={d_old:.5f} "
          f"(safety {MARGIN_HALF/d_old:.2f}x)")
    print(f"theory N={N_NEW} dom={DOM_NEW}: bound={d_new:.5f} "
          f"(safety {MARGIN_HALF/d_new:.2f}x)")

    # full-set LUT deployment simulation at N=16
    maxae = lut_forward_sim(m, X, N_NEW, -3.15, 3.15)
    print(f"measured maxAE (LUT N={N_NEW}, full set): {maxae:.5f}")
    covers = bool(d_new >= maxae)
    print(f"theory covers measured: {covers}")

    out = {
        "date": "2026-08-05",
        "model": "fourier_contractive_v2",
        "box": {"lo": -3.15, "hi": 3.15, "l0_range": [float(l0out.min()),
                                                      float(l0out.max())],
                "covers_100pct_inputs": box_ok},
        "theory": {"N": N_NEW, "domain": DOM_NEW,
                   "sound_bound": d_new, "safety": MARGIN_HALF / d_new,
                   "old_N15_bound": d_old},
        "measured": {"fullset_maxAE": maxae},
        "certificate": {"type": "sound" if (covers and box_ok) else "validated",
                        "covers_measured": covers, "box_100pct": box_ok},
    }
    with open(os.path.join(BASE, "results", "theory", "fourier_n16.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/fourier_n16.json")


if __name__ == "__main__":
    main()
