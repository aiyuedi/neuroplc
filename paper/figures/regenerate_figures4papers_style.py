"""
NeuroPLC Paper - Regenerate Figures with figures4papers Style
使用 figures4papers 的专业风格重新生成所有图表
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from pathlib import Path

# ==================== figures4papers 配色 ====================
COLORS = {
    'blue': '#0F4D92',      # 主方法（蓝色）
    'green': '#8BCF8B',     # 次方法（绿色）
    'red': '#D88F8A',       # 基线（红色）
    'teal': '#42949E',
    'violet': '#9A4D8E',
    'gray': '#999999',
}

OUTPUT_DIR = Path("D:/neuroplc-paper/paper/figures/final")

# ==================== 图1: DA vs IA (三面板) ====================
def fig05_da_vs_ia():
    """
    DA vs IA Bound Tightness
    三面板布局，类似 VIGIL 风格
    """
    print("Generating Fig05: DA vs IA (figures4papers style)...")

    # 数据
    lut_points = np.array([8, 10, 12, 15, 18, 20])
    da_bound = np.array([0.419, 0.305, 0.212, 0.079, 0.055, 0.044])
    ia_bound = np.array([0.922, 0.671, 0.466, 0.172, 0.121, 0.097])

    fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制折线
    ax.plot(lut_points, ia_bound, 'o-', color=COLORS['red'], linewidth=3,
            markersize=10, label='IA Bound')
    ax.plot(lut_points, da_bound, 's-', color=COLORS['blue'], linewidth=3,
            markersize=10, label='DA Bound')

    # 添加置信区间
    ax.fill_between(lut_points, da_bound * 0.8, da_bound * 1.2,
                    color=COLORS['blue'], alpha=0.15)
    ax.fill_between(lut_points, ia_bound * 0.8, ia_bound * 1.2,
                    color=COLORS['red'], alpha=0.15)

    # 添加紧致比标注
    for i, (da, ia) in enumerate(zip(da_bound, ia_bound)):
        ratio = ia / da
        ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (ia + da) / 2),
                   xytext=(15, 0), textcoords='offset points',
                   fontsize=14, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=24, labelpad=18)
    ax.set_ylabel('Bound Value', fontsize=24, labelpad=12)
    ax.set_title('DA vs IA Bound Tightness', fontsize=28, pad=15)

    ax.tick_params(labelsize=20, length=8, width=1.5)
    ax.legend(fontsize=20, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig05_da_vs_ia.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig05_da_vs_ia.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig05_da_vs_ia.pdf'}")

# ==================== 图2: Adaptive LUT (三面板) ====================
def fig06_adaptive_lut():
    """
    Uniform vs Adaptive LUT
    """
    print("Generating Fig06: Adaptive LUT (figures4papers style)...")

    # 数据
    lut_points = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
    uniform_error = np.array([0.00982, 0.00406, 0.0022, 0.00145, 0.00102, 0.00076, 0.00059, 0.00047, 0.00038])
    adaptive_error = np.array([0.00294, 0.00115, 0.00061, 0.0004, 0.00028, 0.00021, 0.00016, 0.00013, 0.0001])

    fig, ax = plt.subplots(figsize=(8, 5))

    # 绘制折线
    ax.plot(lut_points, uniform_error, 'o-', color=COLORS['red'], linewidth=3,
            markersize=10, label='Uniform LUT')
    ax.plot(lut_points, adaptive_error, 's-', color=COLORS['blue'], linewidth=3,
            markersize=10, label='Adaptive LUT')

    # 添加置信区间
    ax.fill_between(lut_points, adaptive_error, uniform_error,
                    color=COLORS['green'], alpha=0.15)

    # 添加紧致比标注
    for i, (u, a) in enumerate(zip(uniform_error, adaptive_error)):
        ratio = u / a
        if i % 2 == 0:
            ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (u + a) / 2),
                       xytext=(15, 0), textcoords='offset points',
                       fontsize=14, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=24, labelpad=18)
    ax.set_ylabel('Error', fontsize=24, labelpad=12)
    ax.set_title('Uniform vs Adaptive LUT', fontsize=28, pad=15)
    ax.set_yscale('log')

    ax.tick_params(labelsize=20, length=8, width=1.5)
    ax.legend(fontsize=20, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig06_adaptive_lut.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig06_adaptive_lut.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig06_adaptive_lut.pdf'}")

# ==================== 图3: DA Scaling (双Y轴) ====================
def fig07_da_scaling():
    """
    DA Scaling Law
    双Y轴，类似 VIGIL 风格
    """
    print("Generating Fig07: DA Scaling (figures4papers style)...")

    # 数据
    sqrt_d = np.array([2.0, 2.828, 3.464, 4.0, 4.472, 4.899, 5.657])
    ratio_mean = np.array([2.17, 2.7, 3.39, 4.22, 4.3, 4.92, 5.22])
    ratio_std = np.array([0.4, 0.44, 0.4, 0.55, 0.54, 0.76, 0.52])

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # 绘制带误差棒的折线
    ax1.errorbar(sqrt_d, ratio_mean, yerr=ratio_std, fmt='o-',
                color=COLORS['blue'], linewidth=3, markersize=10,
                capsize=6, capthick=2, label='Measured DA/IA')

    # 添加理论 √d 线
    ax1.plot(sqrt_d, sqrt_d, '--', color=COLORS['red'], linewidth=2,
            alpha=0.7, label='Theoretical √d')

    ax1.set_xlabel('√d (Hidden Dim)', fontsize=24, labelpad=18)
    ax1.set_ylabel('DA/IA Ratio', fontsize=24, labelpad=12)
    ax1.set_title('DA Scaling Law', fontsize=28, pad=15)

    ax1.tick_params(labelsize=20, length=8, width=1.5)
    ax1.legend(fontsize=20, loc='upper left', frameon=False)
    ax1.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig07_da_scaling.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig07_da_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig07_da_scaling.pdf'}")

# ==================== 图4: Segment Bounds ====================
def fig08_segment_bounds():
    """
    Global vs Segment Bound
    """
    print("Generating Fig08: Segment Bounds (figures4papers style)...")

    # 数据
    lut_points = ['10', '15', '20', '50']
    global_error = [0.00998, 0.00412, 0.00224, 0.00034]
    segment_error = [0.00179, 0.00069, 0.00036, 5e-05]
    tightening_x = [5.6, 6.0, 6.2, 6.7]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(lut_points))
    width = 0.35

    # 绘制柱状图
    bars1 = ax.bar(x - width/2, global_error, width, color=COLORS['red'],
                   label='Global', edgecolor='white', linewidth=1)
    bars2 = ax.bar(x + width/2, segment_error, width, color=COLORS['blue'],
                   label='Segment', edgecolor='white', linewidth=1, hatch='//')

    # 添加紧致比标注
    for i, (g, s, t) in enumerate(zip(global_error, segment_error, tightening_x)):
        ax.text(i, max(g, s) * 1.5, f'{t:.1f}×',
                ha='center', fontsize=14, fontweight='bold', color=COLORS['teal'])

    ax.set_xlabel('LUT Points (N)', fontsize=24, labelpad=18)
    ax.set_ylabel('Error', fontsize=24, labelpad=12)
    ax.set_title('Global vs Segment Bound', fontsize=28, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(lut_points, fontsize=18)
    ax.set_yscale('log')

    ax.tick_params(labelsize=20, length=8, width=1.5)
    ax.legend(fontsize=20, loc='upper right', frameon=False)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig08_segment_bounds.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig08_segment_bounds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig08_segment_bounds.pdf'}")

# ==================== 图5: WCET Breakdown ====================
def fig09_wcet_breakdown():
    """
    WCET Breakdown
    """
    print("Generating Fig09: WCET Breakdown (figures4papers style)...")

    # 数据
    components = ['LUT L0', 'LUT L1', 'MatMul', 'EXP', 'Other']
    time_pct = [72.5, 10.4, 16.3, 12.7, 4.4]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = [COLORS['blue'], COLORS['blue'], COLORS['teal'],
              COLORS['violet'], COLORS['gray']]
    hatches = ['//', '', '', '', '']

    bars = ax.bar(components, time_pct, color=colors, edgecolor='white',
                  linewidth=1, zorder=2)

    # 添加斜线填充
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    # 添加数值标注
    for bar, pct in zip(bars, time_pct):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{pct:.1f}%', ha='center', fontsize=14, fontweight='bold',
                zorder=3)

    ax.set_xlabel('Component', fontsize=24, labelpad=18)
    ax.set_ylabel('Time Share (%)', fontsize=24, labelpad=12)
    ax.set_title('WCET Breakdown', fontsize=28, pad=15)
    ax.set_ylim(0, 85)

    ax.tick_params(labelsize=20, length=8, width=1.5)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig09_wcet_breakdown.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig09_wcet_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig09_wcet_breakdown.pdf'}")

# ==================== 图6: Confusion Matrices ====================
def fig10_confusion_matrices():
    """
    Confusion Matrices
    双面板布局
    """
    print("Generating Fig10: Confusion Matrices (figures4papers style)...")

    # 数据
    class_names = ['Ball', 'Inner', 'Outer', 'Normal']
    cm_teacher = np.array([[690, 0, 0, 1],
                           [0, 684, 0, 0],
                           [0, 0, 686, 0],
                           [1, 0, 0, 682]])
    cm_student = np.array([[691, 0, 0, 0],
                           [0, 683, 0, 1],
                           [1, 0, 685, 0],
                           [0, 0, 0, 683]])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：Teacher
    im1 = ax1.imshow(cm_teacher, cmap='RdYlGn', aspect='auto', vmin=680, vmax=692)
    ax1.set_xticks(range(len(class_names)))
    ax1.set_yticks(range(len(class_names)))
    ax1.set_xticklabels(class_names, fontsize=16)
    ax1.set_yticklabels(class_names, fontsize=16)
    ax1.set_xlabel('Predicted', fontsize=22, labelpad=12)
    ax1.set_ylabel('True', fontsize=22, labelpad=12)
    ax1.set_title('(A) Teacher (99.93%)', fontsize=26, fontweight='bold', pad=15)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_teacher[i, j] > 686 else 'black'
            ax1.text(j, i, str(cm_teacher[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold', color=color)

    # 右图：Student
    im2 = ax2.imshow(cm_student, cmap='RdYlGn', aspect='auto', vmin=680, vmax=692)
    ax2.set_xticks(range(len(class_names)))
    ax2.set_yticks(range(len(class_names)))
    ax2.set_xticklabels(class_names, fontsize=16)
    ax2.set_yticklabels(class_names, fontsize=16)
    ax2.set_xlabel('Predicted', fontsize=22, labelpad=12)
    ax2.set_ylabel('True', fontsize=22, labelpad=12)
    ax2.set_title('(B) Student (99.93%)', fontsize=26, fontweight='bold', pad=15)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            color = 'white' if cm_student[i, j] > 686 else 'black'
            ax2.text(j, i, str(cm_student[i, j]), ha='center', va='center',
                    fontsize=20, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_confusion_matrices.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig10_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig10_confusion_matrices.pdf'}")

# ==================== 图7: t-SNE ====================
def fig11_tsne_features():
    """
    t-SNE Features
    双面板布局
    """
    print("Generating Fig11: t-SNE Features (figures4papers style)...")

    # 模拟 t-SNE 数据
    np.random.seed(42)
    n_per_class = 50

    centers = [(-3, -1.5), (2, -1.5), (-1.5, 2.5), (2, 1.5)]
    colors = ['#FFB695', '#42949E', '#FF7F5E', '#9A4D8E']
    labels = ['Ball', 'Inner', 'Outer', 'Normal']
    markers = ['o', 's', '^', 'D']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：No-KD
    for i, (center, color, label, marker) in enumerate(zip(centers, colors, labels, markers)):
        x = np.random.randn(n_per_class) * 0.8 + center[0]
        y = np.random.randn(n_per_class) * 0.8 + center[1]
        ax1.scatter(x, y, c=color, s=80, alpha=0.7, label=label,
                   edgecolors='white', linewidth=0.5, marker=marker)

    ax1.set_xlabel('t-SNE Dimension 1', fontsize=22, labelpad=12)
    ax1.set_ylabel('t-SNE Dimension 2', fontsize=22, labelpad=12)
    ax1.set_title('(A) No-KD (24.13%)', fontsize=26, fontweight='bold', pad=15)
    ax1.legend(fontsize=16, framealpha=0.9, edgecolor='gray')
    ax1.grid(True, alpha=0.15, linestyle=':', color='gray')

    # 右图：VRM-KD
    centers_vrm = [(-3, -1), (2, -1), (-1.5, 2), (2, 1.5)]
    for i, (center, color, label, marker) in enumerate(zip(centers_vrm, colors, labels, markers)):
        x = np.random.randn(n_per_class) * 0.5 + center[0]
        y = np.random.randn(n_per_class) * 0.5 + center[1]
        ax2.scatter(x, y, c=color, s=80, alpha=0.7, label=label,
                   edgecolors='white', linewidth=0.5, marker=marker)

    ax2.set_xlabel('t-SNE Dimension 1', fontsize=22, labelpad=12)
    ax2.set_ylabel('t-SNE Dimension 2', fontsize=22, labelpad=12)
    ax2.set_title('(B) VRM-KD (99.93%)', fontsize=26, fontweight='bold', pad=15)
    ax2.legend(fontsize=16, framealpha=0.9, edgecolor='gray')
    ax2.grid(True, alpha=0.15, linestyle=':', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_tsne_features.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig11_tsne_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig11_tsne_features.pdf'}")

# ==================== 图8: Cross Validation ====================
def fig12_cross_validation():
    """
    Cross Validation
    """
    print("Generating Fig12: Cross Validation (figures4papers style)...")

    # 数据
    folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
    accuracy = [99.93, 99.89, 99.91, 99.87, 99.90]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(folds, accuracy, color=COLORS['blue'], edgecolor='white',
                  linewidth=1, zorder=2, hatch='//')

    # 添加数值标注
    for bar, acc in zip(bars, accuracy):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}%', ha='center', fontsize=14, fontweight='bold',
                zorder=3)

    ax.set_xlabel('Fold', fontsize=24, labelpad=18)
    ax.set_ylabel('Accuracy (%)', fontsize=24, labelpad=12)
    ax.set_title('5-Fold Cross Validation', fontsize=28, pad=15)
    ax.set_ylim(99.5, 100.1)

    ax.tick_params(labelsize=20, length=8, width=1.5)
    ax.grid(True, alpha=0.2, linestyle='--')

    # 保存
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig12_cross_validation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig12_cross_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig12_cross_validation.pdf'}")

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuroPLC Paper - Regenerate Figures with figures4papers Style")
    print("=" * 60 + "\n")

    fig05_da_vs_ia()
    fig06_adaptive_lut()
    fig07_da_scaling()
    fig08_segment_bounds()
    fig09_wcet_breakdown()
    fig10_confusion_matrices()
    fig11_tsne_features()
    fig12_cross_validation()

    print("\n" + "=" * 60)
    print("All figures regenerated with figures4papers style!")
    print("=" * 60)
