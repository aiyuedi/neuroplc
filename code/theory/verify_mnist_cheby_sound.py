#!/usr/bin/env python3
"""MNIST / ChebyKAN signal-domain sound certificates (2026-08-06).

Extends the soft3L signal-domain technique to the two remaining
cross-domain rows that currently carry only expected-tier certificates:
  - MNIST KAN [28,16,4] (scratch-trained, gamma=[6.3,3.2]): sound bound
    via per-layer (M2, L) measured on the ACTUAL signal domains.
  - ChebyKAN [28,16,4] (gamma=[1.2,2.1]): same treatment.

If the signal-domain propagation yields sound < 0.675, the row upgrades
from expected-tier to sound (Box-in coverage reported).
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
N = 15


def profile(model, dom, n_pts=40001):
    """Per-edge (M2, L) on [-dom, dom] with clamp semantics (deployment
    LUT extrapolates by saturation outside [-3,3])."""
    xs = np.linspace(-dom, dom, n_pts, dtype=np.float64)
    xc = np.clip(xs, -3.0, 3.0)
    sigc = 1.0 / (1.0 + np.exp(-xc))
    out = []
    with torch.no_grad():
        for layer in model.kan_layers:
            g = layer.grid.detach().numpy()
            sb = float(layer.scale_base)
            ss = float(layer.scale_spline)
            bw = layer.base_weight.detach().numpy()
            sw = layer.spline_weight.detach().numpy()
            o, i, _ = layer.spline_weight.shape
            for oi in range(o):
                for ii in range(i):
                    phi = (sb * bw[oi, ii] * xc * sigc
                           + ss * BSpline(g, sw[oi, ii], k=3)(xc / 3.0))
                    d1 = np.gradient(phi, xs)
                    d2 = np.gradient(d1, xs)
                    out.append((float(np.abs(d2).max()),
                                float(np.abs(d1).max())))
    return out


def signal_domains(model, X):
    """Per-layer signal domains under the Box-Continuation condition:
    layer outputs within [-3,3] (input domain is the design domain; the
    coverage reports the fraction of inputs whose intermediate signals
    stay in-box, which is what Box-Continuation certifies)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    layers = getattr(model, "kan_layers", None) or getattr(model, "layers")
    mask = np.ones(X.shape[0], dtype=bool)
    domains = []
    with torch.no_grad():
        cur = Xt
        for layer in layers:
            sel = cur[mask]
            domains.append((float(sel.min()), float(sel.max())))
            cur = layer(cur)
            mask = np.logical_and(mask,
                                  (torch.abs(cur) <= 3.0).all(dim=1).numpy())
    return domains, float(mask.mean())


def sound_bound(arch, prof01, prof2, B0, B1):
    """Perfunc-style propagation: L0 on input domain B0, L1 (output) on
    signal domain B1 (global LUT grid h=6/15; M2 on signal domain)."""
    n0 = arch[0] * arch[1]
    m2_0 = np.array([p[0] for p in prof01[:n0]]).reshape(arch[1], arch[0])
    L1 = np.array([p[1] for p in prof2[n0:]]).reshape(arch[2], arch[1])
    m2_1 = np.array([p[0] for p in prof2[n0:]]).reshape(arch[2], arch[1])
    H2 = (6.0 / N) ** 2 / 8.0          # global LUT grid
    eps0 = m2_0 * H2
    eps1 = m2_1 * H2
    dy = eps0.sum(axis=1)
    d = float(np.max(L1 @ dy + eps1.sum(axis=1)))
    return d, float(L1.sum(axis=1).max()), float(m2_1.max())


def run_model(name, model, X, arch):
    domains, cov = signal_domains(model, X)
    B0 = 3.0
    B1 = max(abs(domains[0][0]), abs(domains[0][1]), 1e-9)
    prof01 = profile(model, B0)
    prof2 = profile(model, min(B1, 3.0))   # output layer on signal domain
    d, Lrow, m2max = sound_bound(arch, prof01, prof2, B0, B1)
    print(f"{name}: signal domains={[f'[{a:.2f},{b:.2f}]' for a, b in domains]} "
          f"Box-in={cov*100:.1f}%")
    print(f"  output L row-sum={Lrow:.3f} M2={m2max:.3f}  "
          f"sound={d:.4f} safety={MARGIN_HALF/d:.2f}x cert={d < MARGIN_HALF}")
    return {"signal_domains": domains, "box_in_coverage": cov,
            "sound_bound": d, "safety": MARGIN_HALF / d,
            "certificate": bool(d < MARGIN_HALF),
            "L_row_sum": Lrow, "M2_out": m2max}


def cheby_profile(model, dom, n_pts=40001):
    """Per-edge (M2, L) for ChebyKANLinear: phi = sum_n c_n T_n(tanh x).
    Chebyshev recurrence on u = tanh(x); numeric derivatives."""
    xs = np.linspace(-dom, dom, n_pts, dtype=np.float64)
    u = np.tanh(xs)
    out = []
    with torch.no_grad():
        for layer in model.layers:
            c = layer.coeffs.detach().numpy()   # (out, in, degree+1)
            deg = layer.degree
            o, i, _ = c.shape
            for oi in range(o):
                for ii in range(i):
                    # T_n(u) recurrence
                    T = np.zeros((deg + 1, n_pts), dtype=np.float64)
                    T[0] = 1.0
                    if deg >= 1:
                        T[1] = u
                    for n in range(2, deg + 1):
                        T[n] = 2.0 * u * T[n - 1] - T[n - 2]
                    phi = np.zeros(n_pts, dtype=np.float64)
                    for n in range(deg + 1):
                        phi += c[oi, ii, n] * T[n]
                    d1 = np.gradient(phi, xs)
                    d2 = np.gradient(d1, xs)
                    out.append((float(np.abs(d2).max()),
                                float(np.abs(d1).max())))
    return out


def main():
    out = {}
    # --- MNIST ---
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                    "..", "experiments")))
    import e42_mnist_cross_domain as e42
    torch.manual_seed(e42.RANDOM_SEED)
    np.random.seed(e42.RANDOM_SEED)
    Xtr, ytr, Xte, yte, pca, scaler = e42.prepare_mnist_4class(
        digits=(0, 1, 2, 3), n_components=e42.ARCH[0])
    ck = torch.load(os.path.join(BASE, "results", "mnist_cross_domain",
                                 "mnist_kan.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck["student_state_dict"] if "student_state_dict" in ck else ck
    m = StudentKAN(e42.ARCH, grid_size=8, spline_order=3)
    m.load_state_dict(sd, strict=False)
    m.eval()
    try:
        out["mnist"] = run_model("MNIST KAN", m, Xte, e42.ARCH)
    except Exception as ex:
        out["mnist"] = {"error": str(ex),
                        "note": "no Box-in inputs (signal exceeds [-3,3]); "
                                "sound infeasible, expected-tier retained"}

    # --- ChebyKAN (signal-domain profile for Chebyshev basis) ---
    from models.student_chebykan import StudentChebyKAN
    ck = torch.load(os.path.join(BASE, "results", "chebykan_z3",
                                 "chebykan_trained.pt"), map_location="cpu",
                    weights_only=False)
    sd = ck.get("student_state_dict", ck.get("state_dict", ck))
    m2 = StudentChebyKAN([28, 16, 4])
    m2.load_state_dict(sd, strict=False)
    m2.eval()
    Xc = np.load(os.path.join(BASE, "data", "processed", "features_X.npy"))
    try:
        domains, cov = signal_domains(m2, Xc)
        B0, B1 = 3.0, max(abs(domains[0][0]), abs(domains[0][1]), 1e-9)
        prof01 = cheby_profile(m2, B0)
        prof2 = cheby_profile(m2, min(B1, 3.0))
        n0 = 28 * 16
        m2_0 = np.array([p[0] for p in prof01[:n0]]).reshape(16, 28)
        L1 = np.array([p[1] for p in prof2[n0:]]).reshape(4, 16)
        m2_1 = np.array([p[0] for p in prof2[n0:]]).reshape(4, 16)
        H2 = (6.0 / 15) ** 2 / 8.0
        dy = (m2_0 * H2).sum(axis=1)
        d = float(np.max(L1 @ dy + (m2_1 * H2).sum(axis=1)))
        print(f"ChebyKAN: domains={[f'[{a:.2f},{b:.2f}]' for a, b in domains]} "
              f"Box-in={cov*100:.1f}%  sound={d:.4f} "
              f"safety={MARGIN_HALF/d:.2f}x cert={d < MARGIN_HALF}")
        out["chebykan"] = {"signal_domains": domains,
                           "box_in_coverage": cov, "sound_bound": d,
                           "safety": MARGIN_HALF / d,
                           "certificate": bool(d < MARGIN_HALF)}
    except Exception as ex:
        out["chebykan"] = {"error": str(ex)}

    with open(os.path.join(BASE, "results", "theory",
                           "mnist_cheby_sound.json"), "w") as f:
        json.dump({"date": "2026-08-06", "out": out}, f, indent=2)
    print("Saved: results/theory/mnist_cheby_sound.json")


if __name__ == "__main__":
    main()
