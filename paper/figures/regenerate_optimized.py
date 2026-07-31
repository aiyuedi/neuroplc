"""
NeuroPLC Paper - Regenerate Optimized Figures
用 paper-plot-skills 风格重新生成图表
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== 风格配置 ====================
# line_confidence_band 风格（来源：Self-Distillation 论文）
COLORS = {
    'blue': '#0F4D92',
    'red': '#D00000',
    'green': '#3A8B3A',
    'gray': '#999999',
    'teal': '#42949E',
}

# ==================== 图1: DA vs IA ====================
def fig05_da_vs_ia():
    """
    DA vs IA 界紧致性对比（line_confidence_band 风格）
    """
    print("Generating Fig05: DA vs IA...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    })

    # 数据
    lut_points = [8, 10, 12, 15, 18, 20]
    da_bound = [0.419, 0.305, 0.212, 0.079, 0.055, 0.044]
    ia_bound = [0.922, 0.671, 0.466, 0.172, 0.121, 0.097]

    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制折线
    ax.plot(lut_points, ia_bound, 'o-', color=COLORS['red'], linewidth=1.8,
            label='IA Bound', markersize=6)
    ax.plot(lut_points, da_bound, 's-', color=COLORS['blue'], linewidth=1.8,
            label='DA Bound', markersize=6)

    # 添加紧致比标注
    for i, (da, ia) in enumerate(zip(da_bound, ia_bound)):
        ratio = ia / da
        ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (ia + da) / 2),
                   xytext=(10, 0), textcoords='offset points',
                   fontsize=9, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontweight='bold')
    ax.set_ylabel('Bound Value', fontweight='bold')
    ax.set_title('DA vs IA Bound Tightness', fontweight='bold')

    # 只保留左/下 spine
    for side, sp in ax.spines.items():
        sp.set_visible(side in ('left', 'bottom'))

    ax.legend(framealpha=0, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    output_dir = Path("D:/neuroplc-paper/paper/figures/final")
    plt.tight_layout()
    plt.savefig(output_dir / "fig05_da_vs_ia.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig05_da_vs_ia.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_dir / 'fig05_da_vs_ia.pdf'}")

# ==================== 图2: Adaptive LUT ====================
def fig06_adaptive_lut():
    """
    自适应 LUT 性能（line_confidence_band 风格）
    """
    print("Generating Fig06: Adaptive LUT...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    })

    # 数据
    lut_points = [10, 15, 20, 25, 30, 35, 40, 45, 50]
    uniform_error = [0.00982, 0.00406, 0.0022, 0.00145, 0.00102, 0.00076, 0.00059, 0.00047, 0.00038]
    adaptive_error = [0.00294, 0.00115, 0.00061, 0.0004, 0.00028, 0.00021, 0.00016, 0.00013, 0.0001]

    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制折线
    ax.plot(lut_points, uniform_error, 'o-', color=COLORS['red'], linewidth=1.8,
            label='Uniform LUT', markersize=6)
    ax.plot(lut_points, adaptive_error, 's-', color=COLORS['blue'], linewidth=1.8,
            label='Adaptive LUT', markersize=6)

    # 添加误差带
    ax.fill_between(lut_points, adaptive_error, uniform_error,
                    alpha=0.15, color=COLORS['green'])

    # 添加紧致比标注
    for i, (u, a) in enumerate(zip(uniform_error, adaptive_error)):
        ratio = u / a
        if i % 2 == 0:  # 只标注偶数点，避免重叠
            ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (u + a) / 2),
                       xytext=(10, 0), textcoords='offset points',
                       fontsize=9, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points', fontweight='bold')
    ax.set_ylabel('Error', fontweight='bold')
    ax.set_title('Uniform vs Adaptive LUT', fontweight='bold')
    ax.set_yscale('log')

    # 只保留左/下 spine
    for side, sp in ax.spines.items():
        sp.set_visible(side in ('left', 'bottom'))

    ax.legend(framealpha=0, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    output_dir = Path("D:/neuroplc-paper/paper/figures/final")
    plt.tight_layout()
    plt.savefig(output_dir / "fig06_adaptive_lut.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig06_adaptive_lut.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_dir / 'fig06_adaptive_lut.pdf'}")

# ==================== 图3: DA Scaling ====================
def fig07_da_scaling():
    """
    DA 缩放定律（line_confidence_band 风格）
    """
    print("Generating Fig07: DA Scaling...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'lines.linewidth': 1.8,
        'lines.markersize': 6,
    })

    # 数据
    hidden_dim = [4, 8, 12, 16, 20, 24, 32]
    sqrt_d = [2.0, 2.828, 3.464, 4.0, 4.472, 4.899, 5.657]
    ratio_mean = [2.17, 2.7, 3.39, 4.22, 4.3, 4.92, 5.22]
    ratio_std = [0.4, 0.44, 0.4, 0.55, 0.54, 0.76, 0.52]

    fig, ax = plt.subplots(figsize=(6, 4))

    # 绘制带误差棒的折线
    ax.errorbar(sqrt_d, ratio_mean, yerr=ratio_std, fmt='o-',
                color=COLORS['blue'], linewidth=1.8, markersize=6,
                capsize=4, capthick=1.5, label='Measured DA/IA Ratio')

    # 添加理论 √d 线
    ax.plot(sqrt_d, sqrt_d, '--', color=COLORS['red'], linewidth=1.5,
            alpha=0.7, label='Theoretical √d')

    ax.set_xlabel('√d (Hidden Dimension)', fontweight='bold')
    ax.set_ylabel('DA/IA Tightness Ratio', fontweight='bold')
    ax.set_title('DA Scaling Law', fontweight='bold')

    # 只保留左/下 spine
    for side, sp in ax.spines.items():
        sp.set_visible(side in ('left', 'bottom'))

    ax.legend(framealpha=0, edgecolor='none')
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    output_dir = Path("D:/neuroplc-paper/paper/figures/final")
    plt.tight_layout()
    plt.savefig(output_dir / "fig07_da_scaling.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig07_da_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_dir / 'fig07_da_scaling.pdf'}")

# ==================== 图4: Segment Bounds ====================
def fig08_segment_bounds():
    """
    段边界对比（bar_grouped_hatch 风格）
    """
    print("Generating Fig08: Segment Bounds...")

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 0.9,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
        'hatch.color': 'white',
        'hatch.linewidth': 1.0,
    })

    # 数据
    lut_points = [10, 15, 20, 50]
    global_error = [0.00998, 0.00412, 0.00224, 0.00034]
    segment_error = [0.00179, 0.00069, 0.00036, 5e-05]
    tightening_x = [5.6, 6.0, 6.2, 6.7]

    fig, ax = plt.subplots(figsize=(6, 4))

    x = np.arange(len(lut_points))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, global_error, width, color=COLORS['red'],
                   label='Global Bound', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, segment_error, width, color=COLORS['blue'],
                   label='Segment Bound', edgecolor='white', linewidth=0.5,
                   hatch='//')

    # 添加紧致比标注
    for i, (g, s, t) in enumerate(zip(global_error, segment_error, tightening_x)):
        ax.text(i, max(g, s) * 1.2, f'{t:.1f}×',
                ha='center', fontsize=10, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points', fontweight='bold')
    ax.set_ylabel('Error', fontweight='bold')
    ax.set_title('Global vs Segment Bound', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(lut_points)
    ax.set_yscale('log')

    # 只保留左/下 spine
    for side, sp in ax.spines.items():
        sp.set_visible(side in ('left', 'bottom'))

    ax.legend(framealpha=0, edgecolor='none')
    ax.grid(axis='y', alpha=0.2, linestyle='--')

    # 保存
    output_dir = Path("D:/neuroplc-paper/paper/figures/final")
    plt.tight_layout()
    plt.savefig(output_dir / "fig08_segment_bounds.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / "fig08_segment_bounds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_dir / 'fig08_segment_bounds.pdf'}")

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuroPLC Paper - Regenerate Optimized Figures")
    print("=" * 60 + "\n")

    fig05_da_vs_ia()
    fig06_adaptive_lut()
    fig07_da_scaling()
    fig08_segment_bounds()

    print("\n" + "=" * 60)
    print("All optimized figures generated!")
    print("=" * 60)
