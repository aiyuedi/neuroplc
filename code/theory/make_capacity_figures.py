"""
make_capacity_figures.py — (V_B, epsilon) plane figure for the capacity section.

Two panels, log-log:
  (a) deployment error vs storage N: LUT (stratum 1, slope -2), free-node
      projection (stratum 3 surrogate), kinked tent (stratum 2, slope -1),
      packing-line reference (c_k M N^{-2} with M=1);
  (b) verification cost vs storage: stratum-1 closed-form check (flat,
      measured wall time), stratum-3 node-search surrogate (super-linear).

Data: results/theory/capacity_packing.json, stratification.json.
Output: paper/figures/final/fig17_capacity_plane.pdf (+ .png).
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

def load(name):
    with open(os.path.join(ROOT, "results", "theory", name)) as fh:
        return json.load(fh)

def main():
    cap = load("capacity_packing.json")
    strat = load("stratification.json")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    # ---- (a) error vs N ----
    ax = axes[0]
    peak_rows = cap["peaked_family"]["rows"]
    NS = [r["N"] for r in peak_rows]
    lut = [r["lut"] for r in peak_rows]
    free = [r["free"] for r in peak_rows if r["free"] is not None]
    NSf = [r["N"] for r in peak_rows if r["free"] is not None]
    tent = strat["layer2_kink"]["tent_errors"]
    # packing reference: c_k M N^{-2}, M=1, c_k=1/8
    Nref = np.logspace(np.log10(8), np.log10(64), 50)
    ref = (1.0 / 8.0) * Nref ** -2

    ax.loglog(NS, lut, "o-", color="#1f77b4", lw=1.6, ms=4,
              label=r"Stratum 1: LUT (slope $-2$)")
    ax.loglog(NSf, free, "s--", color="#2ca02c", lw=1.4, ms=4,
              label="Stratum 3: free nodes (surrogate)")
    ax.loglog(NS, tent, "^:", color="#d62728", lw=1.4, ms=4,
              label=r"Stratum 2: kinked (slope $-1$)")
    ax.loglog(Nref, ref, "-", color="gray", lw=1.0, alpha=0.8,
              label=r"Packing line $c_k M_k N^{-2}$")
    ax.set_xlabel("Storage $N$ (cells)")
    ax.set_ylabel("Deployment error $\\varepsilon$ (worst case)")
    ax.set_title("(a) Error vs. storage budget")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    # ---- (b) verification cost vs N ----
    ax = axes[1]
    l1 = strat["verification_cost"]["layer1_closed_form"]
    l3 = strat["verification_cost"]["layer3_node_search"]
    Ns = [8, 16, 32, 64]
    t1 = [r["wall_s"] for r in l1]
    ops3 = [r["model_ops"] for r in l3]
    ax.semilogy(Ns, t1, "o-", color="#1f77b4", lw=1.6, ms=4,
                label="Stratum 1: closed-form check (flat in $N$)")
    ax.semilogy(Ns, ops3, "s--", color="#2ca02c", lw=1.4, ms=4,
                label="Stratum 3: node-search surrogate (super-linear)")
    ax.set_xlabel("Storage $N$ (cells)")
    ax.set_ylabel("Verification cost $V$ (log scale)")
    ax.set_title("(b) Verification cost vs. storage")
    ax.legend(fontsize=6.5, frameon=False)
    ax.grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    outdir = os.path.join(ROOT, "paper", "figures", "final")
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"fig17_capacity_plane.{ext}"), dpi=300)
    print("Saved: paper/figures/final/fig17_capacity_plane.{pdf,png}")


if __name__ == "__main__":
    main()
