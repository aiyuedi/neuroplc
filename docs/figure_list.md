# NeuroPLC Figure List (TNNLS submission, v2 — 2026-08-04)

> v2 rewrite after pre-submission review (A5): the previous version listed 24 entries
> (16 pointed at tables/algorithms); the manuscript contains **15 figure environments**.
> All figure PDFs were regenerated 2026-08-03 22:31-22:32 (post-recomputation batch);
> FIXME/TODO comments in the .tex sources are residuals and are being cleared.
> All 15 figures are now referenced in the main text (references added 2026-08-04).

## Figure environments (15)

| # | File (figures/final/) | Label | Caption (abridged) | Referenced |
|---|----------------------|-------|--------------------|-----------|
| 1 | fig1_overview.pdf | fig:overview | End-to-end pipeline (28-D features → certified compilation → PLC) | ✓ main.tex |
| 2 | fig2_compiler_arch.pdf | fig:compiler | Compiler architecture: four-stage IR pipeline | ✓ |
| 3 | fig01_c2bv_basis.pdf | fig:c2bv_basis_functions | C²-BV architecture family: activation basis functions | ✓ |
| 4 | fig02_verification.pdf | fig:c2bv_verification | Per-family verification results | ✓ (added 08-04) |
| 5 | fig03_da_tightness.pdf | fig:da_tightness | DA tightness: numerical attainment (Thm C) | ✓ (added 08-04) |
| 6 | fig04_sharp_bound.pdf | fig:sharp_lower_bound | Sharp lower-bound construction (Thm A) | ✓ (added 08-04) |
| 7 | fig05_da_vs_ia.pdf | fig:da_vs_ia | DA vs IA bound comparison across resolutions | ✓ (added 08-04) |
| 8 | fig06_adaptive_lut.pdf | fig:bspline_adaptive | Uniform vs curvature-aware adaptive B-spline LUT | ✓ |
| 9 | fig07_da_scaling.pdf | fig:da_ratio_dist | DA/IA tightening ratio ∝ √d (105 architectures) | ✓ |
| 10 | fig08_segment_bounds.pdf | fig:segment_bounds | Segment-aware bounds across LUT resolutions | ✓ |
| 11 | fig09_wcet_breakdown.pdf | fig:wcet_breakdown | WCET breakdown KAN [28,16,4] on S7-1200 | ✓ (added 08-04) |
| 12 | fig10_confusion_matrices.pdf | fig:confusion_matrices | Confusion matrices (teacher CNN / student KAN) | ✓ |
| 13 | fig11_tsne_features.pdf | fig:tsne_features | Feature embeddings (t-SNE) No-KD vs KD | ✓ |
| 14 | fig12_cross_validation.pdf | fig:cross_validation | Cross-validation results | ✓ |
| 15 | fig16_scl_code.pdf | fig:scl_snippet | Generated SCL excerpt (FB_Inference) | ✓ |
| 16 | fig17_capacity_plane.pdf | fig:capacity_plane | (V_B, ε) capacity plane: three strata + packing line + verification cost (E-T4/E-T5) | ✓ (added 08-04) |

## Orphan files in figures/final/ (not included in the manuscript)

- fig13_model_comparison.pdf — orphan (no \includegraphics; delete before packaging)
- fig14_cross_domain.pdf — orphan
- fig15_safety_monitor.pdf — orphan

## Notes

- All figure captions use IEEE sentence-case style; figures are in `paper/figures/final/`.
- Figure content (WCET 3.86 ms, safety ~1.0×, γ honest values) is from the
  2026-08-03 22:31+ regeneration batch; fig1_overview no longer contains 22.67 ms.
- The capacity section (Thm 17-19) has no figures yet; the (V_B, ε) plane figure
  is a suggested addition before final submission (A5/A6).
- Delete the 3 orphan files and re-verify the figure count (15) when packaging.
