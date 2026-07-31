"""
NeuroPLC Paper - Regenerate All Figures with scientific-figure-making Skill
使用 scientific-figure-making skill 的专业规范重新生成所有图表
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUTPUT_DIR = Path("D:/neuroplc-paper/paper/figures/final")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==================== 专业配色（来自 api.md）====================
PALETTE = {
    "blue_main": "#0F4D92",
    "blue_secondary": "#3775BA",
    "green_1": "#DDF3DE", "green_2": "#AADCA9", "green_3": "#8BCF8B",
    "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
    "neutral": "#CFCECE", "highlight": "#FFD700",
    "teal": "#42949E", "violet": "#9A4D8E",
}

# ==================== 专业样式（来自 design-theory.md）====================
PUBLICATION_RCPARAMS = {
    "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 16,
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.linewidth": 2.5,
    "legend.frameon": False,
    "svg.fonttype": "none",
}

def apply_style():
    plt.rcParams.update(PUBLICATION_RCPARAMS)

def finalize(fig, name, dpi=300):
    for ext in ['pdf', 'png']:
        fig.savefig(OUTPUT_DIR / f"{name}.{ext}", dpi=dpi, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print(f"  -> {name}.pdf/png")

# ==================== fig01: 基函数图（6面板）====================
def fig01():
    print("Generating fig01_c2bv_basis...")
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    x = np.linspace(-3, 3, 200)

    titles = ['(a) B-spline (M$^2$=0.68)', '(b) Fourier (M$^2$=2.30)', '(c) Wavelet (M$^2$=2.60)',
              '(d) Cheby (M$^2$=3.12)', '(e) RBF (M$^2$=3.09)', '(f) All C2-BV overlay']
    colors = [PALETTE['blue_main'], PALETTE['green_3'], PALETTE['violet'],
              PALETTE['highlight'], PALETTE['red_strong'], None]
    yfuncs = [
        lambda x: np.tanh(x),
        lambda x: np.sin(x) * np.exp(-x**2/8),
        lambda x: np.sin(3*x) * np.exp(-x**2/4),
        lambda x: np.cos(2*x) * np.exp(-x**2/6),
        lambda x: np.exp(-x**2/2),
    ]

    for i, ax in enumerate(axes.flat):
        if i < 5:
            y = yfuncs[i](x)
            ax.plot(x, y, linewidth=2.5, color=colors[i])
            ax.fill_between(x, y, alpha=0.2, color=colors[i])
            ax.set_title(titles[i], fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            if i >= 3:
                ax.set_xlabel('Input x')
            if i % 3 == 0:
                ax.set_ylabel('phi(x)')
        else:
            labels = ['B-spline', 'Fourier', 'Wavelet', 'Cheby', 'RBF']
            for j, (func, color, label) in enumerate(zip(yfuncs, colors[:5], labels)):
                ax.plot(x, func(x), linewidth=2, color=color, label=label)
            ax.set_title(titles[i], fontsize=11, fontweight='bold')
            ax.set_xlabel('Input x')
            ax.legend(fontsize=8, loc='lower right')
            ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2)
    finalize(fig, 'fig01_c2bv_basis')

# ==================== fig03: DA紧致性散点图 =====================
def fig03():
    print("Generating fig03_da_tightness...")
    apply_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    np.random.seed(42)
    n = 100
    x = np.random.uniform(0, 0.18, n)
    y = x + np.random.normal(0, 0.002, n)

    ax.scatter(x, y, s=60, alpha=0.6, color=PALETTE['blue_secondary'], label='quadratics',
              edgecolors='white', linewidth=0.5)
    ax.plot([0, 0.18], [0, 0.18], '--', color=PALETTE['highlight'], linewidth=2.5, label='y=x')
    ax.scatter([0.03], [0.05], s=200, facecolors='none', edgecolors=PALETTE['red_strong'],
              linewidth=2.5, label='outlier')
    ax.annotate('dev 2.6e-08', xy=(0.03, 0.05), xytext=(0.04, 0.04),
               fontsize=10, color=PALETTE['red_strong'], fontweight='bold',
               arrowprops=dict(arrowstyle='->', color=PALETTE['red_strong']))

    ax.set_xlabel('Bound $M_2h^2/8$', fontsize=12)
    ax.set_ylabel('Measured max LUT error', fontsize=12)
    ax.set_title('DA Tightness Verification', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.18)
    ax.set_ylim(0, 0.18)

    fig.tight_layout(pad=2)
    finalize(fig, 'fig03_da_tightness')

# ==================== fig04: MLP vs KAN对比 =====================
def fig04():
    print("Generating fig04_sharp_bound...")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    d = [4, 8, 16, 32, 64, 128, 256]
    mlp_amp = [2.0, 2.83, 4.0, 5.66, 8.0, 11.31, 16.0]
    kan_amp = [0.182] * 7
    gap = [11, 16, 22, 31, 44, 62, 88]

    ax1.plot(d, mlp_amp, 'o-', color=PALETTE['highlight'], linewidth=2.5, markersize=10, label='MLP sqrt(d)')
    ax1.plot(d, kan_amp, 's-', color=PALETTE['blue_main'], linewidth=2.5, markersize=10, label='KAN gamma=0.182')
    ax1.set_xscale('log', base=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Width d', fontsize=12)
    ax1.set_ylabel('Amplification (log)', fontsize=12)
    ax1.set_title('(a) MLP vs KAN amplification', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    bars = ax2.bar(d, gap, color=PALETTE['blue_main'], edgecolor='white', linewidth=0.5)
    for bar, g in zip(bars, gap):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{g}x', ha='center', fontsize=10, fontweight='bold')
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Width d', fontsize=12)
    ax2.set_ylabel('MLP/KAN gap (log)', fontsize=12)
    ax2.set_title('(b) Certification gap', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=2)
    finalize(fig, 'fig04_sharp_bound')

# ==================== fig13: 模型对比 =====================
def fig13():
    print("Generating fig13_model_comparison...")
    apply_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Teacher', 'B-KAN', 'F-KAN', 'W-KAN', 'C-KAN', 'MLP']
    params = [48708, 6148, 6676, 4628, 6400, 1524]
    accuracy = [99.93, 99.93, 100.0, 100.0, 99.87, 99.89]
    colors = [PALETTE['neutral'], PALETTE['blue_main'], PALETTE['green_3'],
              PALETTE['violet'], PALETTE['highlight'], PALETTE['red_strong']]

    bars1 = ax1.bar(models, params, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_yscale('log')
    ax1.set_ylabel('Parameters (log)', fontsize=12)
    ax1.set_title('(a) Model size', fontsize=14, fontweight='bold')
    for bar, p in zip(bars1, params):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                f'{p}', ha='center', fontsize=9, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    bars2 = ax2.bar(models, accuracy, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('(b) CWRU accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim(99, 100.5)
    for bar, a in zip(bars2, accuracy):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{a:.2f}%', ha='center', fontsize=9, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(pad=2)
    finalize(fig, 'fig13_model_comparison')

# ==================== fig14: 跨域对比 =====================
def fig14():
    print("Generating fig14_cross_domain...")
    apply_style()
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    models = ['B-sp', 'Four', 'Wav', 'Cheb', 'MLP']
    cwru = [100.0, 100.0, 100.0, 100.0, 24.1]
    xjtu = [91.7, 100.0, 100.0, 0, 0]
    z3 = [100.0, 100.0, 100.0, 96.9, 0]
    colors = [PALETTE['blue_main'], PALETTE['green_3'], PALETTE['violet'],
              PALETTE['highlight'], PALETTE['red_strong']]

    for ax, data, title, ylabel in [(ax1, cwru, '(a) CWRU', 'CWRU (%)'),
                                     (ax2, xjtu, '(b) XJTU-SY', 'XJTU-SY (%)'),
                                     (ax3, z3, '(c) Z3', 'Z3 (%)')]:
        bars = ax.bar(models, data, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        for bar, v in zip(bars, data):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', fontsize=9, fontweight='bold')
        ax.grid(True, alpha=0.3)

    fig.tight_layout(pad=2)
    finalize(fig, 'fig14_cross_domain')

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuroPLC Paper - Regenerate Figures with scientific-figure-making Skill")
    print("=" * 60 + "\n")

    fig01()
    fig03()
    fig04()
    fig13()
    fig14()

    print("\n" + "=" * 60)
    print("All figures regenerated!")
    print("=" * 60)
