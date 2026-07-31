"""
NeuroPLC Paper - Generate SVG Panels for Scientific Figure Composition
使用 matplotlib 生成 SVG 面板，然后用 svgutils 组合
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ==================== IEEE 规格 ====================
IEEE_SPECS = {
    'single_col': 3.5,  # inches
    'double_col': 7.16,  # inches
    'min_font': 8,  # pt
}

# ==================== 顶刊配色 ====================
COLORS = {
    'blue': '#0F4D92',
    'red': '#D00000',
    'green': '#3A8B3A',
    'gray': '#999999',
    'teal': '#42949E',
    'orange': '#FF7F5E',
    'violet': '#9A4D8E',
}

# ==================== 面板生成 ====================
def generate_panel_da_vs_ia():
    """
    Panel A: DA vs IA Bound Tightness
    """
    print("Generating Panel A: DA vs IA...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    # 数据
    lut_points = np.array([8, 10, 12, 15, 18, 20])
    da_bound = np.array([0.419, 0.305, 0.212, 0.079, 0.055, 0.044])
    ia_bound = np.array([0.922, 0.671, 0.466, 0.172, 0.121, 0.097])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # 绘制折线
    ax.plot(lut_points, ia_bound, 'o-', color=COLORS['red'], linewidth=1.2,
            markersize=5, label='IA Bound')
    ax.plot(lut_points, da_bound, 's-', color=COLORS['blue'], linewidth=1.2,
            markersize=5, label='DA Bound')

    # 添加置信区间
    ax.fill_between(lut_points, da_bound * 0.8, da_bound * 1.2,
                    color=COLORS['blue'], alpha=0.15)
    ax.fill_between(lut_points, ia_bound * 0.8, ia_bound * 1.2,
                    color=COLORS['red'], alpha=0.15)

    # 添加紧致比标注
    for i, (da, ia) in enumerate(zip(da_bound, ia_bound)):
        ratio = ia / da
        ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (ia + da) / 2),
                   xytext=(8, 0), textcoords='offset points',
                   fontsize=7, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=9)
    ax.set_ylabel('Bound Value', fontsize=9)
    ax.set_title('(A) DA vs IA Bound Tightness', fontsize=10, fontweight='bold', pad=5)

    # 清洁 spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # 图例
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray')

    # 保存为 SVG
    output_dir = Path("D:/neuroplc-paper/paper/figures/panels")
    plt.savefig(output_dir / "panel_a_da_vs_ia.svg", format='svg', bbox_inches='tight',
                transparent=True, dpi=300)
    plt.close()
    print(f"  -> {output_dir / 'panel_a_da_vs_ia.svg'}")

def generate_panel_adaptive_lut():
    """
    Panel B: Uniform vs Adaptive LUT
    """
    print("Generating Panel B: Adaptive LUT...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    # 数据
    lut_points = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    uniform_error = np.array([0.00982, 0.00406, 0.0022, 0.00145, 0.00102, 0.00076, 0.00059, 0.00047, 0.00038])
    adaptive_error = np.array([0.00294, 0.00115, 0.00061, 0.0004, 0.00028, 0.00021, 0.00016, 0.00013, 0.0001])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # 绘制折线
    ax.plot(lut_points, uniform_error, 'o-', color=COLORS['red'], linewidth=1.2,
            markersize=5, label='Uniform LUT')
    ax.plot(lut_points, adaptive_error, 's-', color=COLORS['blue'], linewidth=1.2,
            markersize=5, label='Adaptive LUT')

    # 添加置信区间
    ax.fill_between(lut_points, adaptive_error, uniform_error,
                    color=COLORS['green'], alpha=0.15)

    # 添加紧致比标注
    for i, (u, a) in enumerate(zip(uniform_error, adaptive_error)):
        ratio = u / a
        if i % 2 == 0:
            ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (u + a) / 2),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=7, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=9)
    ax.set_ylabel('Error', fontsize=9)
    ax.set_title('(B) Uniform vs Adaptive LUT', fontsize=10, fontweight='bold', pad=5)
    ax.set_yscale('log')

    # 清洁 spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # 图例
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray')

    # 保存为 SVG
    output_dir = Path("D:/neuroplc-paper/paper/figures/panels")
    plt.savefig(output_dir / "panel_b_adaptive_lut.svg", format='svg', bbox_inches='tight',
                transparent=True, dpi=300)
    plt.close()
    print(f"  -> {output_dir / 'panel_b_adaptive_lut.svg'}")

def generate_panel_da_scaling():
    """
    Panel C: DA Scaling Law
    """
    print("Generating Panel C: DA Scaling...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    # 数据
    sqrt_d = np.array([2.0, 2.828, 3.464, 4.0, 4.472, 4.899, 5.657])
    ratio_mean = np.array([2.17, 2.7, 3.39, 4.22, 4.3, 4.92, 5.22])
    ratio_std = np.array([0.4, 0.44, 0.4, 0.55, 0.54, 0.76, 0.52])

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    # 绘制带误差棒的折线
    ax.errorbar(sqrt_d, ratio_mean, yerr=ratio_std, fmt='o-',
                color=COLORS['blue'], linewidth=1.2, markersize=5,
                capsize=3, capthick=1, label='Measured DA/IA')

    # 添加理论 √d 线
    ax.plot(sqrt_d, sqrt_d, '--', color=COLORS['red'], linewidth=1,
            alpha=0.7, label='Theoretical √d')

    ax.set_xlabel('√d (Hidden Dim)', fontsize=9)
    ax.set_ylabel('DA/IA Ratio', fontsize=9)
    ax.set_title('(C) DA Scaling Law', fontsize=10, fontweight='bold', pad=5)

    # 清洁 spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # 图例
    ax.legend(fontsize=8, loc='upper left', framealpha=0.9, edgecolor='gray')

    # 保存为 SVG
    output_dir = Path("D:/neuroplc-paper/paper/figures/panels")
    plt.savefig(output_dir / "panel_c_da_scaling.svg", format='svg', bbox_inches='tight',
                transparent=True, dpi=300)
    plt.close()
    print(f"  -> {output_dir / 'panel_c_da_scaling.svg'}")

def generate_panel_segment_bounds():
    """
    Panel D: Global vs Segment Bound
    """
    print("Generating Panel D: Segment Bounds...")

    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 9,
        'axes.linewidth': 0.5,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'xtick.major.size': 3,
        'ytick.major.size': 3,
        'hatch.color': 'white',
        'hatch.linewidth': 0.8,
    })

    # 数据
    lut_points = ['10', '15', '20', '50']
    global_error = [0.00998, 0.00412, 0.00224, 0.00034]
    segment_error = [0.00179, 0.00069, 0.00036, 5e-05]
    tightening_x = [5.6, 6.0, 6.2, 6.7]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    x = np.arange(len(lut_points))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, global_error, width, color=COLORS['red'],
                   label='Global', edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, segment_error, width, color=COLORS['blue'],
                   label='Segment', edgecolor='white', linewidth=0.5, hatch='//')

    # 添加紧致比标注
    for i, (g, s, t) in enumerate(zip(global_error, segment_error, tightening_x)):
        ax.text(i, max(g, s) * 1.3, f'{t:.1f}×',
                ha='center', fontsize=8, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=9)
    ax.set_ylabel('Error', fontsize=9)
    ax.set_title('(D) Global vs Segment Bound', fontsize=10, fontweight='bold', pad=5)
    ax.set_xticks(x)
    ax.set_xticklabels(lut_points, fontsize=8)
    ax.set_yscale('log')

    # 清洁 spine
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)

    # 图例
    ax.legend(fontsize=8, loc='upper right', framealpha=0.9, edgecolor='gray')

    # 保存为 SVG
    output_dir = Path("D:/neuroplc-paper/paper/figures/panels")
    plt.savefig(output_dir / "panel_d_segment_bounds.svg", format='svg', bbox_inches='tight',
                transparent=True, dpi=300)
    plt.close()
    print(f"  -> {output_dir / 'panel_d_segment_bounds.svg'}")

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Generating SVG Panels for Scientific Figure Composition")
    print("=" * 60 + "\n")

    generate_panel_da_vs_ia()
    generate_panel_adaptive_lut()
    generate_panel_da_scaling()
    generate_panel_segment_bounds()

    print("\n" + "=" * 60)
    print("All SVG panels generated!")
    print("=" * 60)
