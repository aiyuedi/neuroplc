# NeuroPLC Paper - File Structure

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
