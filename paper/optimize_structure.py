"""
NeuroPLC Paper - File Structure Optimization
优化论文项目文件结构
"""
import os
import shutil
from pathlib import Path

# ==================== 配置 ====================
BASE_DIR = Path("D:/neuroplc-paper/paper")
FIGURES_DIR = BASE_DIR / "figures"
FINAL_DIR = FIGURES_DIR / "final"

# ==================== 1. 清理 figures 根目录 ====================
def clean_figures_root():
    """清理 figures 根目录，只保留必要文件"""
    print("=" * 60)
    print("1. Cleaning figures root directory...")
    print("=" * 60)

    # 要保留的文件
    keep_files = [
        "FIGURE_CONTRACT.md",
        "photo.jpg",
    ]

    # 要保留的目录
    keep_dirs = [
        "final",
        "source_data",
    ]

    # 删除的文件
    deleted_files = []
    deleted_dirs = []

    for item in FIGURES_DIR.iterdir():
        if item.is_file():
            if item.name not in keep_files and item.suffix in ['.py', '.tex', '.png', '.pdf', '.eps']:
                # 删除临时生成的文件
                if item.name not in ['fig01_c2bv_basis.png', 'fig02_verification.png',
                                     'fig03_da_tightness.png', 'fig04_sharp_bound.png',
                                     'fig05_da_vs_ia.png', 'fig06_adaptive_lut.png',
                                     'fig07_da_scaling.png', 'fig08_segment_bounds.png',
                                     'fig09_wcet_breakdown.png', 'fig1_overview.png',
                                     'fig10_confusion_matrices.png', 'fig11_tsne_features.png',
                                     'fig12_cross_validation.png', 'fig13_model_comparison.png',
                                     'fig14_cross_domain.png', 'fig15_safety_monitor.png',
                                     'fig16_scl_code.png', 'fig2_compiler_arch.png',
                                     'neuroplc_architecture.png', 'overview.pdf']:
                    item.unlink()
                    deleted_files.append(item.name)

        elif item.is_dir():
            if item.name not in keep_dirs:
                # 删除临时目录
                shutil.rmtree(item)
                deleted_dirs.append(item.name)

    print(f"  Deleted {len(deleted_files)} files")
    print(f"  Deleted {len(deleted_dirs)} directories")
    print(f"  Kept: {keep_files}")
    print()

# ==================== 2. 清理 final 目录 ====================
def clean_final_dir():
    """清理 final 目录，只保留论文引用的图表"""
    print("=" * 60)
    print("2. Cleaning final directory...")
    print("=" * 60)

    # 论文中引用的图表
    keep_figures = [
        "fig1_overview",
        "fig2_compiler_arch",
        "fig01_c2bv_basis",
        "fig02_verification",
        "fig03_da_tightness",
        "fig04_sharp_bound",
        "fig05_da_vs_ia",
        "fig06_adaptive_lut",
        "fig07_da_scaling",
        "fig08_segment_bounds",
        "fig09_wcet_breakdown",
        "fig10_confusion_matrices",
        "fig11_tsne_features",
        "fig12_cross_validation",
        "fig13_model_comparison",
        "fig14_cross_domain",
        "fig15_safety_monitor",
        "fig16_scl_code",
        # 附录中的新图表
        "fig01_bar_grouped",
        "fig02_line_confidence",
        "fig03_scatter_tsne",
        "fig07_model_comparison",
        "fig08_heatmap",
    ]

    # 删除的文件
    deleted_files = []

    for item in FINAL_DIR.iterdir():
        if item.is_file():
            # 检查文件名是否在保留列表中
            stem = item.stem
            if stem not in keep_figures:
                item.unlink()
                deleted_files.append(item.name)

    print(f"  Deleted {len(deleted_files)} files")
    print(f"  Kept {len(keep_figures)} figures")
    print()

# ==================== 3. 清理编译临时文件 ====================
def clean_latex_temp():
    """清理 LaTeX 编译临时文件"""
    print("=" * 60)
    print("3. Cleaning LaTeX temporary files...")
    print("=" * 60)

    temp_extensions = ['.aux', '.log', '.out', '.bbl', '.blg', '.toc', '.lof', '.lot']

    deleted_files = []

    for item in BASE_DIR.iterdir():
        if item.is_file():
            if item.suffix in temp_extensions and item.name != 'main.aux':
                item.unlink()
                deleted_files.append(item.name)

    print(f"  Deleted {len(deleted_files)} temporary files")
    print()

# ==================== 4. 创建 README ====================
def create_readme():
    """创建 README 文件说明文件结构"""
    print("=" * 60)
    print("4. Creating README...")
    print("=" * 60)

    readme_content = """# NeuroPLC Paper - File Structure

## 目录结构

```
neuroplc-paper/paper/
├── main.tex                    # 主论文（英文版）
├── main_cn.tex                 # 中文版
├── appendix_coq_spec.tex       # 附录：形式化验证
├── section_*.tex               # 各章节文件
├── references.bib              # 参考文献
├── Makefile                    # 编译脚本
├── figures/                    # 图表目录
│   ├── final/                  # 最终图表（论文引用）
│   │   ├── fig1_overview.*     # 系统总览图
│   │   ├── fig2_compiler_arch.* # 编译器架构图
│   │   ├── fig01_c2bv_basis.*  # B样条基函数
│   │   ├── fig02_verification.* # 验证结果
│   │   ├── ...                 # 其他图表
│   │   └── fig16_scl_code.*    # SCL代码截图
│   ├── source_data/            # 图表源数据（CSV）
│   ├── FIGURE_CONTRACT.md      # 图表命名规范
│   └── photo.jpg               # 作者照片
└── fig_tikz/                   # TikZ 图表源码
```

## 图表命名规范

- `fig{N}_{descriptive_name}.{ext}`
- N: 图表编号（1-16）
- descriptive_name: 描述性名称
- ext: pdf/png/eps

## 编译方法

```bash
# 编译英文版
xelatex main.tex
bibtex main
xelatex main.tex
xelatex main.tex

# 或使用 Makefile
make
```

## 图表生成

- **Origin**: 专业科研绘图（主文图表）
- **Python**: IEEE TII 规范图表（附录图表）
- **TikZ**: LaTeX 原生图表

## 最后更新

2026-07-10: 优化文件结构，清理临时文件
"""

    readme_path = BASE_DIR / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print(f"  Created {readme_path}")
    print()

# ==================== 5. 更新 .gitignore ====================
def update_gitignore():
    """更新 .gitignore 文件"""
    print("=" * 60)
    print("5. Updating .gitignore...")
    print("=" * 60)

    gitignore_content = """# LaTeX temporary files
*.aux
*.log
*.out
*.bbl
*.blg
*.toc
*.lof
*.lot
*.fls
*.fdb_latexmk
*.synctex.gz

# Python temporary files
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Generated figures (keep only final)
figures/fig*_*.png
figures/fig*_*.eps
figures/*.py
figures/*.tex
figures/neuroplc_architecture.png
figures/overview.pdf

# IDE files
.vscode/
.idea/
*.swp
*.swo

# OS files
.DS_Store
Thumbs.db
"""

    gitignore_path = BASE_DIR / ".gitignore"
    with open(gitignore_path, 'w', encoding='utf-8') as f:
        f.write(gitignore_content)

    print(f"  Updated {gitignore_path}")
    print()

# ==================== Run All ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("NeuroPLC Paper - File Structure Optimization")
    print("=" * 60 + "\n")

    clean_figures_root()
    clean_final_dir()
    clean_latex_temp()
    create_readme()
    update_gitignore()

    print("\n" + "=" * 60)
    print("Optimization completed!")
    print("=" * 60)
