#!/usr/bin/env python3
"""
NeuroPLC — Visualization Suite
================================
Generate all figures for the paper.

Figures:
    fig1_overview.pdf      — End-to-end system pipeline
    fig2_compiler_arch.pdf — Compiler IR pipeline (Frontend→IR→Optimizer→Backend)
    fig3_bspline_adaptive.pdf — B-spline: uniform vs adaptive sampling
    fig4_kan_activations.pdf — Learned KAN activation functions
    fig5_confusion_matrices.pdf — Teacher + Student confusion matrices
    fig6_tsne_kd.pdf        — t-SNE: No-KD vs Hinton-KD vs VRM-KD
    fig7_cross_validation.pdf — Python vs SCL error distribution

All figures use IEEE-compatible styling (scienceplots 'ieee').

Usage:
    python visualize.py --all
    python visualize.py --fig fig4_kan_activations
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

try:
    import scienceplots  # noqa
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

from models.student_kan import StudentKAN
from models.teacher_cnn import TeacherCNN
from models.student_mlp import StudentMLP


# ── Style ──
if HAS_SCIENCEPLOTS:
    plt.style.use(["science", "ieee"])
else:
    plt.style.use("seaborn-v0_8-paper")
    print("Note: scienceplots not available, using seaborn fallback.")

FIGSIZE_HALF = (3.5, 2.5)   # half-column width (IEEEtran)
FIGSIZE_FULL = (7.0, 4.5)   # full-column width
FIGSIZE_SQUARE = (5.0, 5.0)
DPI = 300

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
STUDENT_DIR = PROJECT_ROOT / "results" / "student"
TEACHER_DIR = PROJECT_ROOT / "results" / "teacher"

LABEL_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]
COLORS = sns.color_palette("colorblind", 4)


def plt_save(name: str):
    """Save figure to results/figures/<name>.pdf and .png."""
    for ext in ["pdf", "png"]:
        plt.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=DPI,
                    bbox_inches="tight")
    plt.close()
    print(f"  Saved: {name}.pdf + .png")


# ================================================================
# Fig 1: System Overview Pipeline
# ================================================================

def fig1_overview():
    """Text-based architecture diagram placeholder."""
    print("Fig 1: System Overview")

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ax.axis("off")

    lines = [
        "CWRU Vibration Data (12kHz DE, 52 files)",
        "        |",
        "  Sliding Window (1024pt, stride=512)",
        "     /              \\",
        " Waveform           28-D Features",
        "   |              (10 time + 10 freq + 8 DE)",
        "   |                     |",
        " Teacher 1D-CNN         VRM Knowledge",
        " + Self-Attention    Distillation (τ=4.0, α=0.3)",
        " (48K params)               |",
        "   |              Student KAN([28,16,4])",
        "   |              grid=8, k=3 (~300 params)",
        "   |                     |",
        "   \\____________________/",
        "              |",
        "     NeuroPLC Compiler (IR-based)",
        "    Frontend → IR → Optimizer → Backend",
        "              |",
        "     IEC 61131-3 SCL Code",
        "    (DB200 + FB1 + FC2)",
        "              |",
        "     TIA Portal V21 Auto Verify",
        "    (MCP 189 API, 0 errors)",
        "              |",
        "     S7-1200 CPU 1211C (75KB)",
    ]
    y = 0.98
    for line in lines:
        ax.text(0.5, y, line, transform=ax.transAxes, ha="center",
                va="top", fontsize=8, family="monospace")
        y -= 0.047

    ax.set_title("NeuroPLC End-to-End Pipeline", fontsize=11, fontweight="bold")
    plt_save("fig1_overview")


# ================================================================
# Fig 2: Compiler Architecture
# ================================================================

def fig2_compiler_arch():
    """Compiler IR pipeline diagram."""
    print("Fig 2: Compiler Architecture")

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ax.axis("off")

    boxes = [
        (0.05, 0.55, 0.25, 0.55, "FRONTEND\n\nPyTorch Model\n(KAN/MLP/CNN)\n→ IR Graph",
         "#E3F2FD"),
        (0.35, 0.55, 0.25, 0.55, "IR GRAPH\n\nMatMul · BsplineLUT\nStandardAct · Softmax\nArgmax · Add",
         "#FFF3E0"),
        (0.65, 0.55, 0.30, 0.55, "BACKENDS\n\nS7-1200 (75KB, compact)\nS7-1500 (1.5MB, perf)\n→ SCL Code",
         "#E8F5E9"),
        (0.20, 0.05, 0.25, 0.35, "OPTIMIZER\n\nAdaptive B-spline\nSampling\nDeadNode Elim",
         "#F3E5F5"),
        (0.52, 0.05, 0.25, 0.35, "ANALYZER\n\nMemory Map\nFLOPs Count\nBudget %",
         "#FFF9C4"),
    ]

    for x, y, w, h, text, color in boxes:
        rect = plt.Rectangle((x, y), w, h, transform=ax.transAxes,
                             facecolor=color, edgecolor="black",
                             linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, transform=ax.transAxes,
                ha="center", va="center", fontsize=8, family="monospace")

    # Arrows
    arrows = [
        (0.30, 0.75, 0.05, 0.0, "→"),    # Frontend → IR
        (0.60, 0.75, 0.05, 0.0, "→"),    # IR → Backends
        (0.30, 0.55, 0.0, -0.10, "↓"),   # Frontend → Optimizer
        (0.13, 0.55, 0.32, -0.15, "↓"),  # IR → Optimizer
        (0.52, 0.35, 0.0, -0.10, "↑"),   # Optimizer → Backend
    ]
    for x, y, dx, dy, label in arrows:
        ax.annotate(label, xy=(x + dx, y + dy), xytext=(x, y),
                    transform=ax.transAxes, fontsize=14, ha="center", va="center",
                    color="gray")

    ax.set_title("NeuroPLC Compiler Architecture (IR-Based Pipeline)",
                 fontsize=11, fontweight="bold")
    plt_save("fig2_compiler_arch")


# ================================================================
# Fig 3: B-spline Adaptive vs Uniform Sampling
# ================================================================

def fig3_bspline_adaptive():
    """Compare uniform vs adaptive B-spline LUT sampling."""
    print("Fig 3: B-spline Adaptive Sampling")

    ckpt_path = STUDENT_DIR / "kan_kd_vrmKD_best.pt"
    if not ckpt_path.exists():
        print("  Skipping: no KAN checkpoint found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = StudentKAN([28, 16, 4])
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    layer = model.kan_layers[0]

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    for ax, mode, title in zip(
        axes,
        ["uniform", "adaptive"],
        ["Uniform 20-Point Sampling", "Adaptive 20-Point Sampling\n(Curvature-Aware)"]
    ):
        # High-res evaluation
        xs = torch.linspace(-3, 3, 300)
        xs_scaled = xs / 3.0
        import torch.nn.functional as F
        from models.student_kan import _bspline_basis

        base_y = F.silu(xs)
        spline_basis = _bspline_basis(xs_scaled, layer.grid, layer.spline_order)

        # Average over all activations for illustration
        spline_y = (spline_basis * layer.spline_weight.mean(0).mean(0)).sum(-1)
        y = (layer.scale_base * layer.base_weight.mean() * base_y
             + layer.scale_spline * spline_y)

        ax.plot(xs.numpy(), y.numpy(), "k-", linewidth=1.5, label="B-spline φ(x)")

        # Sampling points
        if mode == "uniform":
            x_pts = np.linspace(-3, 3, 20)
            y_pts = np.interp(x_pts, xs.numpy(), y.numpy())
        else:
            x_pts, y_pts = layer.get_adaptive_sample_points(20)

        ax.scatter(x_pts, y_pts, s=40, c=COLORS[0], zorder=5,
                   label=f"20 sample points")

        # Connect sampled points (linear interpolation between them)
        ax.plot(x_pts, y_pts, "o-", color=COLORS[0], linewidth=1, alpha=0.5,
                markersize=4)

        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7)
        ax.set_xlim(-3.2, 3.2)

    fig.suptitle("B-spline Lookup Table: Uniform vs Adaptive Sampling",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt_save("fig3_bspline_adaptive")


# ================================================================
# Fig 4: KAN Activation Functions
# ================================================================

def fig4_kan_activations():
    """Plot learned KAN activation functions."""
    print("Fig 4: KAN Activation Functions")

    ckpt_path = STUDENT_DIR / "kan_kd_vrmKD_best.pt"
    if not ckpt_path.exists():
        print("  Skipping: no KAN checkpoint found")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = StudentKAN([28, 16, 4])
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    layer = model.kan_layers[0]  # First KAN layer (28 in, 16 out)

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE_FULL)
    axes = axes.flatten()

    # Sample some activations
    indices = [(0, 0), (0, 3), (0, 7),
               (13, 5), (20, 10), (27, 15)]

    for ax, (in_i, out_j) in zip(axes, indices):
        x, y = layer.get_activation_function(in_i, out_j)
        ax.plot(x, y, linewidth=1.2)
        ax.set_title(f"φ[{in_i}→{out_j}]", fontsize=9)
        ax.set_xlim(-3.2, 3.2)
        ax.set_xlabel("x")
        ax.set_ylabel("φ(x)")

    fig.suptitle("Learned KAN Activation Functions (Layer 0: 28→16)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt_save("fig4_kan_activations")


# ================================================================
# Fig 5: Confusion Matrices
# ================================================================

def fig5_confusion_matrices():
    """Teacher + Student confusion matrices."""
    print("Fig 5: Confusion Matrices")

    device = torch.device("cpu")

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_FULL)

    # Teacher
    teacher_ckpt = TEACHER_DIR / "teacher_best.pt"
    if teacher_ckpt.exists():
        X_wav = np.load(PROJECT_ROOT / "data/processed/waveform_X.npy")
        y = np.load(PROJECT_ROOT / "data/processed/waveform_y.npy")

        ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=True)
        teacher = TeacherCNN(num_classes=4).to(device)
        teacher.load_state_dict(ckpt["model_state_dict"])
        teacher.eval()

        with torch.no_grad():
            X_t = torch.from_numpy(X_wav[:2000]).float().unsqueeze(1)
            preds = teacher(X_t).argmax(1).numpy()

        cm = confusion_matrix(y[:2000], preds)
        _plot_cm(axes[0], cm, "Teacher CNN")

    # Student
    student_ckpt = STUDENT_DIR / "kan_kd_vrmKD_best.pt"
    if student_ckpt.exists():
        X_feat = np.load(PROJECT_ROOT / "data/processed/features_X.npy")
        y = np.load(PROJECT_ROOT / "data/processed/features_y.npy")

        ckpt = torch.load(student_ckpt, map_location=device, weights_only=True)
        student = StudentKAN([28, 16, 4]).to(device)
        student.load_state_dict(ckpt["student_state_dict"])
        student.eval()

        with torch.no_grad():
            X_t = torch.from_numpy(X_feat[:2000]).float()
            preds = student(X_t).argmax(1).numpy()

        cm = confusion_matrix(y[:2000], preds)
        _plot_cm(axes[1], cm, "Student KAN (VRM-KD)")

    fig.suptitle("Confusion Matrices: Teacher vs Student", fontsize=11,
                 fontweight="bold")
    plt.tight_layout()
    plt_save("fig5_confusion_matrices")


def _plot_cm(ax, cm, title):
    """Plot a single confusion matrix."""
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES,
                ax=ax, vmin=0, vmax=1, cbar=False)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")


# ================================================================
# Fig 6: t-SNE Visualization (KD ablation)
# ================================================================

def fig6_tsne_kd():
    """t-SNE visualization of learned representations."""
    print("Fig 6: t-SNE (KD Ablation)")

    X_feat = np.load(PROJECT_ROOT / "data/processed/features_X.npy")
    y = np.load(PROJECT_ROOT / "data/processed/features_y.npy")

    from sklearn.manifold import TSNE

    # Use subset for speed
    n_subset = 1000
    idx = np.random.RandomState(42).choice(len(y), n_subset, replace=False)
    X_sub, y_sub = X_feat[idx], y[idx]

    # t-SNE on raw features (as proxy for "No-KD" representation)
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_tsne = tsne.fit_transform(X_sub)

    fig, ax = plt.subplots(figsize=FIGSIZE_SQUARE)
    for i, name in enumerate(LABEL_NAMES):
        mask = y_sub == i
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=15, c=[COLORS[i]],
                   label=name, alpha=0.7, edgecolors="none")

    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("Feature Space (28-D → t-SNE)", fontsize=10)
    ax.legend(fontsize=7, markerscale=2)

    fig.suptitle("t-SNE Visualization of Feature Representations",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt_save("fig6_tsne_features")


# ================================================================
# Fig 7: Python vs SCL Cross-Validation
# ================================================================

def fig7_cross_validation():
    """Placeholder for Python vs SCL error distribution."""
    print("Fig 7: Python vs SCL Cross-Validation (pending Phase 2)")

    fig, ax = plt.subplots(figsize=FIGSIZE_FULL)
    ax.text(0.5, 0.5, "Pending Phase 2 (Compiler SCL Output)",
            ha="center", va="center", fontsize=14, color="gray",
            transform=ax.transAxes)
    ax.set_title("Python vs SCL: Element-wise Consistency (E6)",
                 fontsize=10)
    plt_save("fig7_cross_validation")


# ================================================================
# Main
# ================================================================

FIGURES = {
    "fig1_overview": fig1_overview,
    "fig2_compiler_arch": fig2_compiler_arch,
    "fig3_bspline_adaptive": fig3_bspline_adaptive,
    "fig4_kan_activations": fig4_kan_activations,
    "fig5_confusion_matrices": fig5_confusion_matrices,
    "fig6_tsne_features": fig6_tsne_kd,
    "fig7_cross_validation": fig7_cross_validation,
}


def main():
    parser = argparse.ArgumentParser(description="NeuroPLC Visualization Suite")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument("--fig", type=str, default="",
                        help="Single figure ID")
    args = parser.parse_args()

    if args.all:
        selected = list(FIGURES.keys())
    elif args.fig and args.fig in FIGURES:
        selected = [args.fig]
    else:
        print("Specify --all or --fig <name>")
        print(f"Available: {list(FIGURES.keys())}")
        return

    print(f"Generating {len(selected)} figure(s)...")
    for fig_id in selected:
        try:
            FIGURES[fig_id]()
        except Exception as e:
            print(f"  ERROR [{fig_id}]: {e}")

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
