#!/usr/bin/env python3
"""FourierKAN L0-direct sound upgrade (2026-08-05).

Deployment configuration: L0 (28->16) evaluated analytically (sin/cos
instructions, supported by S7-1200; zero LUT storage and zero LUT error),
L1 (16->4) LUT-compiled with N=16 points on domain [-3.15, 3.15].

  - L0 direct: no LUT error -> dy = 0, sound bound = L1 LUT error only.
  - L1 input domain: L0 output over all 13,714 inputs is [-3.002, 3.145],
    100% inside [-3.15, 3.15]; h = 6.3/16.
  - eps1 = M2_1 * h^2/8 (per-edge analytic, domain-independent for Fourier)
  - sound = max_k sum_j eps1_{k,j}  (sum over 16 input edges)
  - Full-set LUT deployment simulation verifies measured maxAE <= sound.

This upgrades FourierKAN from "validated 4.9x (theory 0.110 does not
cover full-set measured 0.126)" to a SOUND certificate at the
L0-direct deployment configuration.
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
N_L1 = 16
DOM_L1 = 6.3          # [-3.15, 3.15]


def fourier_m2_l(layer):
    k = layer.k.numpy()
    c = layer.fourier_coeffs.detach().numpy()
    cs = c[:, :, :layer.n_harmonics]
    cd = c[:, :, layer.n_harmonics:]
    om = float(layer.omega)
    bw = np.abs(layer.base_weight.detach().numpy())
    m2f = ((k ** 2 * om ** 2)[None, None, :] * (np.abs(cs) + np.abs(cd))).sum(axis=2)
    m2b = bw * 0.5
    return m2f + m2b


def lut_forward_l1(model, X, n_pts, lo, hi):
    """L0 direct (float64 analytic) + L1 LUT deployment simulation."""
    l0, l1 = model.kan_layers[0], model.kan_layers[1]
    m64 = model.double()
    # exact L0 output
    with torch.no_grad():
        l0out = m64.kan_layers[0](torch.tensor(X, dtype=torch.float64))
    # L1 LUT
    xs = np.linspace(lo, hi, n_pts, dtype=np.float64)
    c = l1.fourier_coeffs.detach().numpy()
    kk = l1.k.numpy()
    om = float(l1.omega)
    bw = l1.base_weight.detach().numpy()
    o, i = l1.out_features, l1.in_features
    out = np.zeros((X.shape[0], o), dtype=np.float64)
    with torch.no_grad():
        for oi in range(o):
            acc = np.zeros(X.shape[0], dtype=np.float64)
            for ii in range(i):
                phix = bw[oi, ii] * xs * (1.0 / (1.0 + np.exp(-xs)))
                for hh in range(l1.n_harmonics):
                    phix += (c[oi, ii, hh] * np.sin(kk[hh] * om * xs)
                             + c[oi, ii, l1.n_harmonics + hh]
                             * np.cos(kk[hh] * om * xs))
                xin = np.clip(l0out[:, ii].numpy(), lo, hi)
                acc += np.interp(xin, xs, phix)
            out[:, oi] = acc + float(l1.bias.detach().numpy()[oi])
    # exact reference
    with torch.no_grad():
        logits_exact = m64.kan_layers[1](l0out)
    return float(np.abs(out - logits_exact.numpy()).max())


def main():
    ckpt = torch.load(os.path.join(BASE, "results", "student",
                                   "fourier_contractive_v2.pt"),
                      map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentFourierKAN([28, 16, 4], n_harmonics=6, omega=0.4)
    m.load_state_dict(sd, strict=False)
    m.eval()
    X = np.load(os.path.join(BASE, "data", "processed", "features_X.npy"))

    # L0 output 100% within [-3.15,3.15]
    Xt = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        l0out = m.kan_layers[0](Xt)
    box_ok = bool((l0out.abs() <= 3.15).all().item())
    print(f"L0 out range [{l0out.min():.4f}, {l0out.max():.4f}]  "
          f"100% within [-3.15,3.15]: {box_ok}")

    # sound bound = L1 LUT error only (L0 direct)
    l1 = m.kan_layers[1]
    m2_1 = fourier_m2_l(l1)
    h = DOM_L1 / N_L1
    eps1 = m2_1 * h ** 2 / 8.0
    d = float(eps1.sum(axis=1).max())
    print(f"L1 M2 max={m2_1.max():.4f} avg={m2_1.mean():.4f}")
    print(f"sound (L0 direct, L1 LUT N={N_L1} dom={DOM_L1}): {d:.4f}  "
          f"safety={MARGIN_HALF / d:.2f}x  cert={d < MARGIN_HALF}")

    # full-set deployment simulation
    maxae = lut_forward_l1(m, X, N_L1, -3.15, 3.15)
    print(f"measured maxAE (L0 direct + L1 LUT N={N_L1}): {maxae:.5f}")
    print(f"sound covers measured: {d >= maxae}")

    out = {
        "date": "2026-08-05",
        "model": "fourier_contractive_v2",
        "config": {"L0": "direct analytic (SIN/COS, no LUT)",
                   "L1": {"N": N_L1, "domain": DOM_L1}},
        "box": {"L0_out_range": [float(l0out.min()), float(l0out.max())],
                "covers_100pct": box_ok},
        "sound_bound": d,
        "safety": MARGIN_HALF / d,
        "certificate": bool(d < MARGIN_HALF),
        "measured_maxAE": maxae,
        "covers_measured": bool(d >= maxae),
    }
    with open(os.path.join(BASE, "results", "theory", "fourier_l0direct.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/fourier_l0direct.json")


if __name__ == "__main__":
    main()
