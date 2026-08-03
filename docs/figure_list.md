# NeuroPLC Figure List (TNNLS submission, 2026-08-04)

> TNNLS requires figures uploaded as separate files with captions. All figures in `paper/figures/`.
> 87-page manuscript: fig1–fig13 + tables (figures counted from `\begin{figure}` in main.tex and section files).

| # | File | Caption (abridged) | Source |
|---|------|--------------------|--------|
| fig1 | fig1_overview.png | NeuroPLC end-to-end pipeline (28-D feature extraction → certified compilation → PLC) | main.tex:804 |
| fig2 | fig2_compiler_arch.png | Compiler architecture: four-stage IR-based pipeline | main.tex:821 |
| fig3 | fig3_lut_ablation.png | IR design ablation: removing BsplineLUT → 47.4× node explosion | main.tex:887 |
| fig4 | fig4_bspline_lut.png | B-spline evaluation via LUT (SCL: FC2) | main.tex:972 |
| fig5 | fig5_curvature_lut.png | Uniform vs curvature-aware adaptive B-spline LUT | main.tex:1020 |
| fig6 | fig6_sign_structure.png | Sign structure of trained KAN weights (DA tightening analysis) | main.tex:1390 |
| fig7 | fig7_ratio_scaling.png | DA/IA tightening ratio scaling law (105 random KAN architectures) | main.tex:1569 |
| fig8 | fig8_da_ia_validation.png | DA/IA tightening ratio ∝ √d: empirical validation | main.tex:1604 |
| fig9 | fig9_da_bound.png | DA vs IA bound comparison with LUT resolution | main.tex:1634 |
| fig10 | fig10_seg_aware.png | Segment-aware bound comparison across LUT resolutions | main.tex:1697 |
| fig11 | fig11_greedy_density.png | Greedy adaptive LUT density allocation | main.tex:1731 |
| fig12 | fig12_scl_excerpt.png | Generated SCL excerpt (FB_Inference) | main.tex:2609 |
| fig13 | fig13_compilation_matrix.png | Multi-target compilation matrix in TIA Portal V21 | main.tex:2633 |
| fig14 | fig14_block_sizes.png | TIA Portal V21 measured block sizes | main.tex:2670 |
| fig15 | fig15_cross_project.png | Cross-project TIA Portal V21 compilation coverage | main.tex:2702 |
| fig16 | fig16_wcet.png | Z3-verified WCET for KAN [28,16,4] | main.tex:2843 |
| fig17 | fig17_wcet_breakdown.png | WCET breakdown: KAN [28,16,4] on S7-1200 | main.tex:3061 |
| fig18 | fig18_confusion.png | Confusion matrices (Teacher 1D-CNN / student KAN) | main.tex:3297 |
| fig19 | fig19_tsne.png | Feature embeddings (t-SNE): No-KD vs KD | main.tex:3363 |
| fig20 | fig20_instr_time.png | Instruction-level inference time | main.tex:3462 |
| fig21 | fig21_width_scaling.png | Scalability: width scaling | main.tex:3520 |
| fig22 | fig22_depth_scaling.png | Scalability: depth scaling | main.tex:3542 |
| fig23 | fig23_grid_resolution.png | Scalability: grid resolution | main.tex:3563 |
| fig24 | fig24_z3_translation.png | Z3 SMT translation validation per node | main.tex:3611 |

**Note (2026-08-04)**: fig02/fig04/fig05/fig09/fig1_overview carry FIXME tags (numbers changed after the P0 recomputation; regeneration pending). Verify figure numbers against the PDF before submission.
