#!/usr/bin/env python3
"""Sound bound chain for bounded-amplitude bases (2026-08-04 P0).

FourierKAN/WaveletKAN layers are per-edge function sums (y_j = sum_i
phi_{j,i}(x_i)) with NO weight-matrix multiplication: the propagation
constant is the Lipschitz sum (sum_i L_{j,i}), not a matrix norm. With
bounded-amplitude bases this yields tiny propagation envelopes.

  layer 0:  dy_j  <= sum_i L0_{j,i} * eps_in + eps0_j     (eps_in=0 in-domain)
  layer 1:  dz_k  <= sum_j L1_{k,j} * dy_j + eps1_k
  sound    = max_k [ sum_j L1_{k,j} * eps0_j + eps1_k ]   (per-edge eps)
  eps      = M2_full * h^2/8,  h = 6/14 @ N=15
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_fourierkan import StudentFourierKAN
from models.student_waveletkan import StudentWaveletKAN

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MARGIN_HALF = 0.675
H2_8 = (6.0 / 14) ** 2 / 8.0


def fourier_m2_l(layer):
    """Per-edge M2 (analytic) and Lipschitz L for FourierKANLinear."""
    k = layer.k.numpy()                       # (K,)
    c = layer.fourier_coeffs.detach().numpy()  # (out, in, 2K)
    cs = c[:, :, :layer.n_harmonics]
    cd = c[:, :, layer.n_harmonics:]
    om = float(layer.omega)
    m2 = om ** 2 * (k[None, None, :] ** 2 * (np.abs(cs) + np.abs(cd))).sum(-1)
    l_f = om * (k[None, None, :] * (np.abs(cs) + np.abs(cd))).sum(-1)
    bw = np.abs(layer.base_weight.detach().numpy())
    # |siLU'| max = 1.099 (x=1.28); base contributes bw * 1.099 to L
    l = l_f + bw * 1.099
    # base M2: |siLU''| max = 0.5
    m2 = m2 + bw * 0.5
    return m2, l


def wavelet_m2_l(layer, n=20001):
    """Per-edge M2/L for WaveletKANLinear (numeric, mexican-hat + LINEAR base).

    psi(t) = NORM*(1-t^2)e^{-t^2/2}, NORM = 2/(sqrt(3)*pi^{1/4}) ~ 0.867;
    base path is LINEAR (F.linear(x, base_weight)) — M2_base = 0, L_base = |bw|.
    """
    xs = np.linspace(-3.0, 3.0, n, dtype=np.float64)
    NORM = 2.0 / (np.sqrt(3.0) * np.pi ** 0.25)
    wc = layer.wavelet_coeffs.detach().numpy()   # (out, in, S)
    bw = np.abs(layer.base_weight.detach().numpy())
    scales = layer.scales.numpy()
    shifts = layer.shifts.numpy()
    o, i, s = wc.shape
    m2 = np.zeros((o, i))
    l = np.zeros((o, i))
    for oi in range(o):
        for ii in range(i):
            d1 = np.zeros_like(xs)
            d2 = np.zeros_like(xs)
            for si in range(s):
                a, mu = scales[si], shifts[si]
                t = (xs - mu) / a
                psi = NORM * (1 - t ** 2) * np.exp(-t ** 2 / 2)
                psi1 = NORM * (t ** 3 - 3 * t) * np.exp(-t ** 2 / 2) / a
                psi2 = NORM * ((t ** 4 - 6 * t ** 2 + 3)
                               * np.exp(-t ** 2 / 2)) / a ** 2
                d1 += wc[oi, ii, si] * psi1
                d2 += wc[oi, ii, si] * psi2
            m2[oi, ii] = np.abs(d2).max()          # linear base: M2 = 0
            l[oi, ii] = np.abs(d1).max() + bw[oi, ii]   # linear base: L = |bw|
    return m2, l


def main():
    out = {}
    for name, ck, arch, kind in [
        ("fourier_v1", "results/student/fourier_contractive.pt",
         [28, 16, 4], "fourier"),
        ("fourier_v2", "results/student/fourier_contractive_v2.pt",
         [28, 16, 4], "fourier"),
        ("fourier_base", "results/student/fourier_base.pt",
         [28, 16, 4], "fourier"),
        ("wavelet_v1", "results/student/wavelet_contractive.pt",
         [28, 16, 4], "wavelet"),
        ("wavelet_base", "results/student/wavelet_base.pt",
         [28, 16, 4], "wavelet"),
    ]:
        ckpt = torch.load(os.path.join(BASE, ck), map_location="cpu",
                          weights_only=False)
        sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
        if kind == "fourier":
            m = StudentFourierKAN(arch, n_harmonics=6, omega=0.4)
        else:
            m = StudentWaveletKAN(arch, n_scales=8)
        m.load_state_dict(sd, strict=False)
        m.eval()
        l0 = m.kan_layers[0]
        l1 = m.kan_layers[1]
        if kind == "fourier":
            m20, L0 = fourier_m2_l(l0)
            m21, L1 = fourier_m2_l(l1)
        else:
            m20, L0 = wavelet_m2_l(l0)
            m21, L1 = wavelet_m2_l(l1)
        eps0 = m20 * H2_8
        eps1 = m21 * H2_8
        # sound: dz_k = sum_j L1_{k,j} * (eps0_j) + eps1_k
        d = float(np.max((L1 @ eps0.max(axis=1)) + eps1.max(axis=1)))
        out[name] = {"sound_bound": d, "safety": MARGIN_HALF / d,
                     "certificate": bool(d < MARGIN_HALF),
                     "m2_max": float(max(m20.max(), m21.max()))}
        print(f"{name:14s} sound bound={d:.5f}  safety={MARGIN_HALF/d:.1f}x  "
              f"cert={'YES' if d < MARGIN_HALF else 'no'}")
    with open(os.path.join(BASE, "results", "theory", "c2bv_bounds.json"), "w") as f:
        json.dump({"date": "2026-08-04", "out": out}, f, indent=2)
    print("Saved: results/theory/c2bv_bounds.json")


if __name__ == "__main__":
    main()
