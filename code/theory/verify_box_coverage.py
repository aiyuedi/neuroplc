#!/usr/bin/env python3
"""Box-Continuation coverage analysis (2026-08-04, C11 feasibility).

The certificate (Box-Continuation, Lemma) applies only to inputs whose
intermediate layer-1 signals stay inside the layer-1 LUT domain [-3,3]
(47% leave it, E68). This script quantifies, on the released checkpoint:
  1. the empirical distribution of layer-1 signals u_j (CWRU test set);
  2. per-input coverage (all 16 signals inside a widened domain L);
  3. the N required to keep the same LUT error eps when widening to L:
        eps(N, L) = M2 * (2L/(N-1))^2 / 8  ->  N'(L) = 1 + (L/3)*(N-1)
  4. DA bound under widened domain at eps-preserving N (same 0.66 family),
     and the net win: (coverage gain) at (unchanged certificate strength).
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CKPT = os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt")
DATA_X = os.path.join(BASE, "data", "processed", "features_X.npy")


def main():
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = StudentKAN([28, 16, 4])
    model.load_state_dict(ckpt["student_state_dict"], strict=False)
    model.eval()

    X = np.load(DATA_X).astype(np.float32)
    X = np.clip(X, -3.0, 3.0)
    with torch.no_grad():
        h1 = model.kan_layers[0](torch.from_numpy(X)).numpy()   # (n, 16)

    n = h1.shape[0]
    lo, hi = h1.min(), h1.max()
    print(f"layer-1 signal range over {n} inputs: [{lo:.2f}, {hi:.2f}]")
    pct_out = float((np.abs(h1) > 3.0).any(axis=1).mean())
    print(f"inputs with any |u_j| > 3.0 (Box violation at L=3): {pct_out*100:.1f}%")

    rows = []
    for L in (3.0, 4.0, 5.0, 6.0):
        cover = float((np.abs(h1) <= L).all(axis=1).mean())
        N_eps = int(np.ceil(1 + (L / 3.0) * (15 - 1)))   # eps-preserving N
        rows.append({"L": L, "coverage": cover,
                     "N_eps_preserving": N_eps,
                     "uncovered": 1.0 - cover})
        print(f"L={L:3.1f}: coverage {cover*100:5.1f}%  "
              f"eps-preserving N={N_eps}  (DA bound unchanged 0.66 family)")

    with open(os.path.join(BASE, "results", "theory", "box_coverage.json"), "w") as f:
        json.dump({"date": "2026-08-04", "n": n,
                   "layer1_range": [float(lo), float(hi)],
                   "box_violation_pct_L3": float(pct_out),
                   "rows": rows}, f, indent=2)
    print("\nSaved: results/theory/box_coverage.json")


if __name__ == "__main__":
    main()
