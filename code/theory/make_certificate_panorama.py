#!/usr/bin/env python3
"""Certificate panorama figure (2026-08-04 P0 delivery).

Panel A: certificate bound tiers per checkpoint (log scale, half-margin
         0.675 reference). Tiers = expected (M2_char) / per-function
         envelope / float64 sound / float32 sound; tier shade deepens
         with strength. Model hue follows the entity.
Panel B: accuracy vs sound safety factor trade-off (capacity trade-off,
         log x). main carries no design-time certificate (open marker).
Panel C: full-phi M2 distribution (float64 ground truth) per checkpoint
         -- contractive training flattens the curvature profile.

Palette: dataviz reference slots 1-3 (blue/orange/aqua); shades from the
sequential ramps; surfaces/ink per palette.md. IEEE-style (serif, thin
grid, muted axes).
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN
from scipy.interpolate import BSpline

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(BASE, "paper", "figures", "final")

# ── palette (dataviz reference; light surface #fcfcfb) ──
SERIES = {  # categorical slots 1-3
    "main": "#2a78d6",
    "soft2L": "#eb6834",
    "soft3L": "#1baf7a",
}
# sequential shade steps per tier (same hue, deepening strength)
SHADES = {  # expected -> per-func -> f64 sound -> f32 sound
    "main": ["#9ec5f4", "#5598e7", "#256abf", "#104281"],
    "soft2L": ["#f8c9a3", "#f29463", "#e0692f", "#b94c1a"],
    "soft3L": ["#a5e3c9", "#5cc99b", "#239a6d", "#116b49"],
}
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
HALF_MARGIN = 0.675

# ── data (from verify_sound_chain / verify_perfunc_bound, 2026-08-04) ──
MODELS = [
    # name, acc, expected, perfunc, f64_sound, f32_sound, measured, k
    ("main",  99.93, 1.70, 3.38, None,     None, 0.517, 340),
    ("soft2L", 98.5, 0.0014, 0.0078, 0.026, 0.058, 0.053, 1.2),
    ("soft3L", 98.6, 0.0015, 0.019, None,  0.110, 0.100, 1.26),
]
TIERS = ["expected", "per-func", "float64 sound", "float32 sound"]


def m2_arrays():
    """full-phi M2 (float64) per checkpoint for Panel C."""
    xs = np.linspace(-3.0, 3.0, 20001, dtype=np.float64)
    sig = 1.0 / (1.0 + np.exp(-xs))
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
        vals = []
        with torch.no_grad():
            for layer in m.kan_layers:
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
                        vals.append(float(np.abs(d2).max()))
        out[name] = np.array(vals)
    return out


def main():
    m2 = m2_arrays()
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": MUTED,
    })

    fig = plt.figure(figsize=(10.2, 3.1), dpi=200)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 0.85, 0.9],
                          wspace=0.42, left=0.075, right=0.985,
                          top=0.84, bottom=0.16)

    # ── Panel A: certificate tiers (log bound) ──
    ax = fig.add_subplot(gs[0, 0])
    xpos = np.arange(len(MODELS))
    width = 0.185
    for mi, (name, acc, exp, pf, f64, f32, meas, k) in enumerate(MODELS):
        vals = [exp, pf, f64, f32]
        for ti, v in enumerate(vals):
            if v is None:
                continue
            x = xpos[mi] + (ti - 1.5) * width
            ax.bar(x, v, width, color=SHADES[name][ti],
                   edgecolor="none", linewidth=0)
            ax.text(x, v * 1.25, _fmt(v), ha="center", va="bottom",
                    fontsize=6.2, color=SECONDARY)
    ax.axhline(HALF_MARGIN, color="#e34948", linewidth=0.9, linestyle="--",
               zorder=0)
    ax.text(2.62, HALF_MARGIN * 1.06, "half-margin 0.675",
            fontsize=6.5, color="#e34948", ha="right")
    ax.set_yscale("log")
    ax.set_ylim(5e-4, 12)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"released\n99.93%" if n == "main" else
                        f"soft-2L\n98.5%" if n == "soft2L" else
                        f"soft-3L\n98.6%" for n, *_ in MODELS], fontsize=7.5)
    ax.set_ylabel("error bound $\\Delta$ (log)", fontsize=8)
    ax.set_title("(a) Certificate tiers", fontsize=8.5, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.4, which="both")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    # tier legend (shade ramp of soft2L color)
    for ti, lab in enumerate(TIERS):
        ax.bar(0, 0, color=SHADES["soft2L"][ti], label=lab)
    leg = ax.legend(loc="upper right", fontsize=6.2, frameon=False,
                    ncol=1, bbox_to_anchor=(1.0, 1.02))
    for t in leg.get_texts():
        t.set_color(SECONDARY)

    # ── Panel B: accuracy vs sound safety (capacity trade-off) ──
    ax = fig.add_subplot(gs[0, 1])
    safety = {"main": 0.2, "soft2L": 11.6, "soft3L": 6.1}
    acc = {"main": 99.93, "soft2L": 98.5, "soft3L": 98.6}
    marks = {"main": ("o", "none"), "soft2L": ("o", SERIES["soft2L"]),
             "soft3L": ("o", SERIES["soft3L"])}
    for name in ["main", "soft2L", "soft3L"]:
        mkr, fc = marks[name]
        ax.scatter(safety[name], acc[name], s=34, marker=mkr,
                   facecolor=fc if fc != "none" else "none",
                   edgecolor=SERIES[name], linewidth=1.3, zorder=3)
        dx = 0 if name != "main" else 0
        ax.annotate("released\n(no sound cert.)" if name == "main" else
                    "soft-2L" if name == "soft2L" else "soft-3L",
                    (safety[name], acc[name]),
                    textcoords="offset points", xytext=(dx, 7),
                    fontsize=7, color=INK, ha="center")
    ax.axvline(1.0, color=MUTED, linewidth=0.6, linestyle=":")
    ax.text(1.06, 99.0, "safety = 1", fontsize=6.3, color=MUTED)
    ax.set_xscale("log")
    ax.set_xlim(0.12, 25)
    ax.set_ylim(98.2, 100.1)
    ax.set_xlabel("sound safety factor (log)", fontsize=8)
    ax.set_ylabel("test accuracy (%)", fontsize=8)
    ax.set_title("(b) Accuracy–certificate trade-off", fontsize=8.5, pad=4)
    ax.grid(color=GRID, linewidth=0.4, which="both")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    # ── Panel C: M2 distribution (curvature flattening) ──
    ax = fig.add_subplot(gs[0, 2])
    bp = ax.boxplot([np.log10(m2["main"]), np.log10(m2["soft2L"]),
                     np.log10(m2["soft3L"])],
                    positions=[0, 1, 2], widths=0.5, showfliers=False,
                    patch_artist=True,
                    medianprops=dict(color="#0b0b0b", linewidth=1.0),
                    whiskerprops=dict(color=MUTED, linewidth=0.7),
                    capprops=dict(color=MUTED, linewidth=0.7),
                    boxprops=dict(linewidth=0.7))
    for patch, name in zip(bp["boxes"], ["main", "soft2L", "soft3L"]):
        patch.set_facecolor(SERIES[name])
        patch.set_edgecolor(MUTED)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["released", "soft-2L", "soft-3L"], fontsize=7.5)
    ax.set_ylabel("$\\log_{10} M_2^{\\max}(\\varphi)$", fontsize=8)
    ax.set_title("(c) Curvature profile", fontsize=8.5, pad=4)
    ax.grid(axis="y", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    os.makedirs(OUT, exist_ok=True)
    for ext in ["pdf", "png", "eps"]:
        fig.savefig(os.path.join(OUT, f"fig_cert_panorama.{ext}"),
                    dpi=300 if ext != "pdf" else 200)
    print("Saved: paper/figures/final/fig_cert_panorama.{pdf,png,eps}")


def _fmt(v):
    if v >= 1:
        return f"{v:.1f}"
    if v >= 0.01:
        return f"{v:.3f}"
    return f"{v:.2e}"


if __name__ == "__main__":
    main()
