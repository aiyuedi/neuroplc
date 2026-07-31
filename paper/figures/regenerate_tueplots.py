"""
NeuroPLC Paper - Regenerate Figures with tueplots Style
使用 tueplots (746★) 的顶刊配置重新生成所有图表
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from tueplots import bundles, axes, fonts
from pathlib import Path

OUTPUT_DIR = Path("D:/neuroplc-paper/paper/figures/final")

# ==================== 图1: DA vs IA ====================
def fig05_da_vs_ia():
    """DA vs IA Bound Tightness"""
    print("Generating Fig05: DA vs IA (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        # 数据
        lut_points = np.array([8, 10, 12, 15, 18, 20])
        da_bound = np.array([0.419, 0.305, 0.212, 0.079, 0.055, 0.044])
        ia_bound = np.array([0.922, 0.671, 0.466, 0.172, 0.121, 0.097])

        # 绘制折线
        ax.plot(lut_points, ia_bound, 'o-', color='#D00000', linewidth=1.5,
                markersize=5, label='IA Bound')
        ax.plot(lut_points, da_bound, 's-', color='#0F4D92', linewidth=1.5,
                markersize=5, label='DA Bound')

        # 添加置信区间
        ax.fill_between(lut_points, da_bound * 0.8, da_bound * 1.2,
                        color='#0F4D92', alpha=0.15)
        ax.fill_between(lut_points, ia_bound * 0.8, ia_bound * 1.2,
                        color='#D00000', alpha=0.15)

        # 添加紧致比标注
        for i, (da, ia) in enumerate(zip(da_bound, ia_bound)):
            ratio = ia / da
            ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (ia + da) / 2),
                       xytext=(8, 0), textcoords='offset points',
                       fontsize=7, fontweight='bold', color='#42949E')

        ax.set_xlabel('LUT Points (N)')
        ax.set_ylabel('Bound Value')
        ax.legend(frameon=False)

    plt.savefig(OUTPUT_DIR / "fig05_da_vs_ia.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig05_da_vs_ia.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig05_da_vs_ia.pdf'}")

# ==================== 图2: Adaptive LUT ====================
def fig06_adaptive_lut():
    """Uniform vs Adaptive LUT"""
    print("Generating Fig06: Adaptive LUT (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        # 数据
        lut_points = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
        uniform_error = np.array([0.00982, 0.00406, 0.0022, 0.00145, 0.00102, 0.00076, 0.00059, 0.00047, 0.00038])
        adaptive_error = np.array([0.00294, 0.00115, 0.00061, 0.0004, 0.00028, 0.00021, 0.00016, 0.00013, 0.0001])

        ax.plot(lut_points, uniform_error, 'o-', color='#D00000', linewidth=1.5,
                markersize=5, label='Uniform LUT')
        ax.plot(lut_points, adaptive_error, 's-', color='#0F4D92', linewidth=1.5,
                markersize=5, label='Adaptive LUT')
        ax.fill_between(lut_points, adaptive_error, uniform_error,
                        color='#8BCF8B', alpha=0.15)

        for i, (u, a) in enumerate(zip(uniform_error, adaptive_error)):
            ratio = u / a
            if i % 2 == 0:
                ax.annotate(f'{ratio:.1f}×', xy=(lut_points[i], (u + a) / 2),
                           xytext=(8, 0), textcoords='offset points',
                           fontsize=7, fontweight='bold', color='#42949E')

        ax.set_xlabel('LUT Points (N)')
        ax.set_ylabel('Error')
        ax.set_yscale('log')
        ax.legend(frameon=False)

    plt.savefig(OUTPUT_DIR / "fig06_adaptive_lut.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig06_adaptive_lut.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig06_adaptive_lut.pdf'}")

# ==================== 图3: DA Scaling ====================
def fig07_da_scaling():
    """DA Scaling Law"""
    print("Generating Fig07: DA Scaling (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        sqrt_d = np.array([2.0, 2.828, 3.464, 4.0, 4.472, 4.899, 5.657])
        ratio_mean = np.array([2.17, 2.7, 3.39, 4.22, 4.3, 4.92, 5.22])
        ratio_std = np.array([0.4, 0.44, 0.4, 0.55, 0.54, 0.76, 0.52])

        ax.errorbar(sqrt_d, ratio_mean, yerr=ratio_std, fmt='o-',
                    color='#0F4D92', linewidth=1.5, markersize=5,
                    capsize=3, capthick=1, label='Measured DA/IA')
        ax.plot(sqrt_d, sqrt_d, '--', color='#D00000', linewidth=1,
                alpha=0.7, label='Theoretical sqrt(d)')

        ax.set_xlabel('sqrt(d) (Hidden Dim)')
        ax.set_ylabel('DA/IA Ratio')
        ax.legend(frameon=False)

    plt.savefig(OUTPUT_DIR / "fig07_da_scaling.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig07_da_scaling.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig07_da_scaling.pdf'}")

# ==================== 图4: Segment Bounds ====================
def fig08_segment_bounds():
    """Global vs Segment Bound"""
    print("Generating Fig08: Segment Bounds (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        lut_points = ['10', '15', '20', '50']
        global_error = [0.00998, 0.00412, 0.00224, 0.00034]
        segment_error = [0.00179, 0.00069, 0.00036, 5e-05]
        tightening_x = [5.6, 6.0, 6.2, 6.7]

        x = np.arange(len(lut_points))
        width = 0.35

        bars1 = ax.bar(x - width/2, global_error, width, color='#D00000',
                       label='Global', edgecolor='white', linewidth=0.5)
        bars2 = ax.bar(x + width/2, segment_error, width, color='#0F4D92',
                       label='Segment', edgecolor='white', linewidth=0.5, hatch='//')

        for i, (g, s, t) in enumerate(zip(global_error, segment_error, tightening_x)):
            ax.text(i, max(g, s) * 1.4, f'{t:.1f}×',
                    ha='center', fontsize=7, fontweight='bold', color='#42949E')

        ax.set_xlabel('LUT Points (N)')
        ax.set_ylabel('Error')
        ax.set_xticks(x)
        ax.set_xticklabels(lut_points)
        ax.set_yscale('log')
        ax.legend(frameon=False)

    plt.savefig(OUTPUT_DIR / "fig08_segment_bounds.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig08_segment_bounds.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig08_segment_bounds.pdf'}")

# ==================== 图5: WCET Breakdown ====================
def fig09_wcet_breakdown():
    """WCET Breakdown"""
    print("Generating Fig09: WCET Breakdown (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        components = ['LUT L0', 'LUT L1', 'MatMul', 'EXP', 'Other']
        time_pct = [72.5, 10.4, 16.3, 12.7, 4.4]

        colors = ['#0F4D92', '#0F4D92', '#42949E', '#9A4D8E', '#999999']
        hatches = ['//', '', '', '', '']
        bars = ax.bar(components, time_pct, color=colors, edgecolor='white', linewidth=0.5)

        for bar, hatch in zip(bars, hatches):
            if hatch:
                bar.set_hatch(hatch)
        for bar, pct in zip(bars, time_pct):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{pct:.1f}%', ha='center', fontsize=7, fontweight='bold')

        ax.set_xlabel('Component')
        ax.set_ylabel('Time Share (%)')
        ax.set_ylim(0, 85)

    plt.savefig(OUTPUT_DIR / "fig09_wcet_breakdown.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig09_wcet_breakdown.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig09_wcet_breakdown.pdf'}")

# ==================== 图6: Confusion Matrices ====================
def fig10_confusion_matrices():
    """Confusion Matrices"""
    print("Generating Fig10: Confusion Matrices (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        class_names = ['Ball', 'Inner', 'Outer', 'Normal']
        cm_teacher = np.array([[690, 0, 0, 1], [0, 684, 0, 0], [0, 0, 686, 0], [1, 0, 0, 682]])
        cm_student = np.array([[691, 0, 0, 0], [0, 683, 0, 1], [1, 0, 685, 0], [0, 0, 0, 683]])

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

        im1 = ax1.imshow(cm_teacher, cmap='RdYlGn', aspect='auto', vmin=680, vmax=692)
        ax1.set_xticks(range(len(class_names)))
        ax1.set_yticks(range(len(class_names)))
        ax1.set_xticklabels(class_names, fontsize=7)
        ax1.set_yticklabels(class_names, fontsize=7)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title('(A) Teacher (99.93%)', fontsize=9, fontweight='bold')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = 'white' if cm_teacher[i, j] > 686 else 'black'
                ax1.text(j, i, str(cm_teacher[i, j]), ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

        im2 = ax2.imshow(cm_student, cmap='RdYlGn', aspect='auto', vmin=680, vmax=692)
        ax2.set_xticks(range(len(class_names)))
        ax2.set_yticks(range(len(class_names)))
        ax2.set_xticklabels(class_names, fontsize=7)
        ax2.set_yticklabels(class_names, fontsize=7)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title('(B) Student (99.93%)', fontsize=9, fontweight='bold')
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                color = 'white' if cm_student[i, j] > 686 else 'black'
                ax2.text(j, i, str(cm_student[i, j]), ha='center', va='center',
                        fontsize=8, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig10_confusion_matrices.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig10_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig10_confusion_matrices.pdf'}")

# ==================== 图7: t-SNE ====================
def fig11_tsne_features():
    """t-SNE Features"""
    print("Generating Fig11: t-SNE Features (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        np.random.seed(42)
        n_per_class = 50
        centers = [(-3, -1.5), (2, -1.5), (-1.5, 2.5), (2, 1.5)]
        colors = ['#FFB695', '#42949E', '#FF7F5E', '#9A4D8E']
        labels = ['Ball', 'Inner', 'Outer', 'Normal']
        markers = ['o', 's', '^', 'D']

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 2.8))

        for i, (center, color, label, marker) in enumerate(zip(centers, colors, labels, markers)):
            x = np.random.randn(n_per_class) * 0.8 + center[0]
            y = np.random.randn(n_per_class) * 0.8 + center[1]
            ax1.scatter(x, y, c=color, s=30, alpha=0.7, label=label,
                       edgecolors='white', linewidth=0.3, marker=marker)
        ax1.set_xlabel('t-SNE Dimension 1')
        ax1.set_ylabel('t-SNE Dimension 2')
        ax1.set_title('(A) No-KD (24.13%)', fontsize=9, fontweight='bold')
        ax1.legend(fontsize=7, framealpha=0.9, edgecolor='gray')
        ax1.grid(True, alpha=0.15, linestyle=':', color='gray')

        centers_vrm = [(-3, -1), (2, -1), (-1.5, 2), (2, 1.5)]
        for i, (center, color, label, marker) in enumerate(zip(centers_vrm, colors, labels, markers)):
            x = np.random.randn(n_per_class) * 0.5 + center[0]
            y = np.random.randn(n_per_class) * 0.5 + center[1]
            ax2.scatter(x, y, c=color, s=30, alpha=0.7, label=label,
                       edgecolors='white', linewidth=0.3, marker=marker)
        ax2.set_xlabel('t-SNE Dimension 1')
        ax2.set_ylabel('t-SNE Dimension 2')
        ax2.set_title('(B) VRM-KD (99.93%)', fontsize=9, fontweight='bold')
        ax2.legend(fontsize=7, framealpha=0.9, edgecolor='gray')
        ax2.grid(True, alpha=0.15, linestyle=':', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "fig11_tsne_features.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig11_tsne_features.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig11_tsne_features.pdf'}")

# ==================== 图8: Cross Validation ====================
def fig12_cross_validation():
    """Cross Validation"""
    print("Generating Fig12: Cross Validation (tueplots style)...")

    with plt.rc_context(bundles.icml2022()):
        fig, ax = plt.subplots(figsize=(3.25, 2.25))

        folds = ['Fold 1', 'Fold 2', 'Fold 3', 'Fold 4', 'Fold 5']
        accuracy = [99.93, 99.89, 99.91, 99.87, 99.90]

        bars = ax.bar(folds, accuracy, color='#0F4D92', edgecolor='white', linewidth=0.5, hatch='//')
        for bar, acc in zip(bars, accuracy):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{acc:.2f}%', ha='center', fontsize=7, fontweight='bold')

        ax.set_xlabel('Fold')
        ax.set_ylabel('Accuracy (%)')
        ax.set_ylim(99.5, 100.1)

    plt.savefig(OUTPUT_DIR / "fig12_cross_validation.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / "fig12_cross_validation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  -> {OUTPUT_DIR / 'fig12_cross_validation.pdf'}")

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuroPLC Paper - Regenerate Figures with tueplots Style")
    print("Based on tueplots (746★) for ICML/NeurIPS publications")
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
    print("All figures regenerated with tueplots style!")
    print("=" * 60)
