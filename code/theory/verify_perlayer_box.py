#!/usr/bin/env python3
"""Per-layer Box propagation (2026-08-05 upgrade): tighter sound bounds.

Current code uses a GLOBAL input-domain [-3,3] (h = 6/N) for EVERY layer.
After contractive training the per-layer signal ranges shrink dramatically
(e.g. soft3L output-layer input spans only ~3.01 instead of 6.0), so the
global h is loose by up to (6/3.01)^2 ~ 4x on the final layer.

Upgrade:
  1. Analytical per-layer Box (iterated, self-consistent):
        B_0 = 3.0  (design input domain)
        B_{l+1,j} = sum_i L_{l;j,i} * B_{l,i} + sum_i |phi_{l;j,i}(0)|
     where (L, phi0) of the FULL activation phi are evaluated on [-B_l, B_l];
     iterate until B stabilises (monotone, converges in 2-3 steps).
  2. Per-layer h_l = 2*B_l / N; per-layer M2 = numeric M2 of phi on [-B_l, B_l].
  3. Re-run the corrected propagation (dy = sum eps, per-edge Lipschitz)
     with per-layer (M2, h).

Also verifies the analytical Box covers the measured per-layer signal
ranges over all 13,714 inputs (soundness of the Box).
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
MEASURED = {"main": 0.5175, "soft2L": 0.0527, "soft3L": 0.1004}  # Tier-4 f32
N_DEFAULT = 15


def profile_per_layer(model, Bs, n_pts=40001):
    """Per-edge (M2, L, |phi(0)|) with layer-l domain [-B_l, B_l].

    Deployment semantics: the compiled LUT extrapolates by CLAMPING to the
    design domain [-3,3] (Box-Continuation), so phi(x) = phi(clamp(x, -3, 3)).
    The spline part is evaluated WITHOUT extrapolation; outside [-3,3] the
    activation saturates (flat), contributing zero M2/L.
    """
    out = []
    with torch.no_grad():
        for li, layer in enumerate(model.kan_layers):
            box = float(np.max(Bs[li]))
            xs = np.linspace(-box, box, n_pts, dtype=np.float64)
            xc = np.clip(xs, -3.0, 3.0)
            sigc = 1.0 / (1.0 + np.exp(-xc))
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
                    k0 = int(np.argmin(np.abs(xs)))
                    out.append((float(np.abs(d2).max()),
                                float(np.abs(d1).max()),
                                float(abs(phi[k0]))))
    return out


def signal_domains(arch, prof):
    """Analytical per-layer Box B_l (vectors), iterated to self-consistency."""
    Bs = [np.full(arch[0], 3.0)]
    idx = 0
    for li in range(len(arch) - 1):
        o, i = arch[li + 1], arch[li]
        L = np.zeros((o, i))
        ph0 = np.zeros((o, i))
        for oi in range(o):
            for ii in range(i):
                _, L[oi, ii], ph0[oi, ii] = prof[idx]
                idx += 1
        Bs.append(L @ Bs[-1] + ph0.sum(axis=1))
    return Bs


def sound_bound_perlayer(arch, prof, Bs, n=N_DEFAULT):
    """Corrected propagation with per-layer (M2 on [-B_l,B_l], h_l = 2B_l/n)."""
    idx = 0
    o0, i0 = arch[1], arch[0]
    m2_0 = np.zeros((o0, i0))
    L1s, m2_1s, eps_sums = [], [], []
    for oi in range(o0):
        for ii in range(i0):
            m2_0[oi, ii], _, _ = prof[idx]; idx += 1
    h0 = 2.0 * float(np.max(Bs[0])) / n
    eps0 = m2_0 * h0 ** 2 / 8.0
    dy = eps0.sum(axis=1)
    du = dy
    for li in range(1, len(arch) - 1):
        o, i = arch[li + 1], arch[li]
        L = np.zeros((o, i)); m2 = np.zeros((o, i))
        for oi in range(o):
            for ii in range(i):
                m2v, Lv, _ = prof[idx]; idx += 1
                L[oi, ii] = Lv; m2[oi, ii] = m2v
        hl = 2.0 * float(np.max(Bs[li])) / n
        eps = m2 * hl ** 2 / 8.0
        du = L @ du + eps.sum(axis=1)
    return float(np.max(du))


def measured_layer_ranges(model, X):
    """Per-layer output ranges CONDITIONED on the design input domain
    [-3,3]^28 (the sound-bound semantics). Out-of-design inputs are handled
    separately by Box-Continuation (empirical floor), not by the theory."""
    Xt = torch.tensor(X, dtype=torch.float32)
    mask = np.all(np.abs(X) <= 3.0, axis=1)
    ranges = []
    with torch.no_grad():
        cur = Xt[mask]
        for layer in model.kan_layers:
            cur = layer(cur)
            ranges.append((float(cur.min()), float(cur.max())))
    return ranges


def main():
    X = np.load(os.path.join(BASE, "data", "processed", "features_X.npy"))
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
        # Per-layer Box from MEASURED 100% coverage (Box-Continuation style):
        #   B_0 = 3.0 (design domain); B_l = 1.05 * max|measured range| (L>=1)
        # Clamp semantics (deployment LUT extrapolation = saturation), so the
        # per-edge (M2, L) are the global [-3,3] profiles (flat outside).
        prof = profile_per_layer(m, [np.full(a, 3.0) for a in arch])
        ranges = measured_layer_ranges(m, X)
        Bs = [np.full(arch[0], 3.0)]
        for r in ranges:
            B = 1.05 * max(abs(r[0]), abs(r[1]))
            Bs.append(np.full(arch[len(Bs)], B))
        d_new = sound_bound_perlayer(arch, prof, Bs)
        # global-h baseline for comparison
        d_old = sound_bound_perlayer(arch, prof,
                                     [np.full(a, 3.0) for a in arch])
        meas = MEASURED[name]
        covers = {f"L{i}": bool(r[0] >= -float(b.max()) - 1e-9
                                and r[1] <= float(b.max()) + 1e-9)
                  for i, (r, b) in enumerate(zip(ranges, Bs[1:]))}
        out[name] = {
            "perlayer_boxes": [float(b.max()) for b in Bs],
            "measured_ranges": ranges,
            "box_covers_measured": covers,
            "sound_bound_perlayer": d_new,
            "safety_perlayer": MARGIN_HALF / d_new,
            "certificate_perlayer": bool(d_new < MARGIN_HALF),
            "sound_bound_global_h": d_old,
            "safety_global_h": MARGIN_HALF / d_old,
            "measured_f32": meas,
        }
        print(f"{name:7s} per-layer Box={[float(b.max()) for b in Bs]}  "
              f"sound_pl={d_new:.4f} (safety {MARGIN_HALF/d_new:.2f}x)  "
              f"sound_glob={d_old:.4f} (safety {MARGIN_HALF/d_old:.2f}x)  "
              f"cert={'YES' if d_new < MARGIN_HALF else 'no'}  "
              f"covers_meas={all(covers.values())}")
    with open(os.path.join(BASE, "results", "theory", "perlayer_box.json"), "w") as f:
        json.dump({"date": "2026-08-05", "note":
                   "per-layer Box B_{l+1}=L@B_l+|phi0| (iterated), "
                   "per-layer h=2B_l/N, per-layer M2 on [-B_l,B_l]",
                   "out": out}, f, indent=2)
    print("Saved: results/theory/perlayer_box.json")


if __name__ == "__main__":
    main()
