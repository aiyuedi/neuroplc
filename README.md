# NeuroPLC: PyTorch→IEC 61131-3 SCL Compiler for Siemens PLCs

> **An IR-Based Compiler from PyTorch to IEC 61131-3 SCL for Siemens PLCs with Bearing Fault Diagnosis**

Started: 2026-07-03 | Status: **Final — 35 pages, 0e/0w, polished, submission-ready**

Author: 刘甫悦 (板板) + Claude

---

## Quick Status

| Dimension | State |
|-----------|-------|
| Paper | ~4,438 lines, 35 pages, **0 errors, 0 warnings** |
| Experiments | E1–E53 + 7 validation experiments (V1–V7) |
| Algorithms | DA + Segment-Aware Bounds + Adaptive LUT (3 contributions) |
| TIA Portal compile | ✅ MCP-verified 4 targets × 0e 0w, DB+FB 90.4% (45.2/50 KB) |
| PLCSIM Advanced | ✅ Python ctypes bridge: RegisterInstance+PowerOn (SREC_OK), <100ms |
| SCL generation | KAN + MLP, S7-1200 + S7-1500, 12 output variants + feature extraction FB |
| Model | KAN [28,16,4], 512 B-spline fn, VRM-KD distilled, 99.93% CWRU |
| Verification | Z3 3-tier: 512/512 functions, certificate VALID, ~200-line TCB |
| Safety | 5,000 adversarial inputs → 100/100 worst-case preserved |
| DA scaling | 105 architectures, Pearson r=0.9872, √d law confirmed |

---

## One-Paragraph Summary

NeuroPLC is the **first compiler** that translates PyTorch neural networks (KAN/MLP) to IEC 61131-3 Structured Control Language (SCL) for Siemens S7 PLCs with **machine-checkable end-to-end correctness guarantees**. It introduces an Intermediate Representation (IR) graph that decouples model semantics from PLC dialect, enabling 6 substantive optimization passes (plus 2 structural) before code generation. The SVNN framework formalizes the architectural conditions under which a compiler CAN provide such guarantees (Theorem 2): KAN satisfies them; standard MLPs provably do not (Proposition 1, validated by 512/512 vs 0/48 Z3-verifiable components). Three novel algorithms guarantee correctness: (1) **Doubleton Arithmetic**—3.1× tighter than interval arithmetic, with √d scaling law confirmed across 105 random architectures (Pearson r=0.987); (2) **Segment-Aware de Boor Bounds**—6.0× per-segment tightening; (3) **Adaptive Mixed-Precision LUT Allocation**—71.6% worst-ε reduction. The compiler is validated on bearing fault diagnosis: KAN [28,16,4] (6,148 params) distilled via VRM-KD (7.9× compression from 48K-param CNN teacher), deploys on S7-1200 CPU 1211C at 45.2 KB work memory (90.4% of 50 KB budget, TIA V21 MCP-measured), with 0 errors, 0 warnings.

---

## Algorithmic Contributions (3 novel algorithms)

### 1. Segment-Aware Analytical Error Bounds
Exploits the piecewise-linear structure of cubic B-spline second derivatives φ''(x). Computes per-LUT-segment M₂_j instead of a single global M₂.

| N | Global ε | Mean Segment ε | Tightening | DA Safety (uniform→segment) |
|---|----------|----------------|------------|------------------------------|
| 10 | 0.00998 | 0.00179 | **5.6×** | 2.5× → 3.5× |
| 15 | 0.00412 | 0.00069 | **6.0×** | 6.1× → 8.4× |
| 20 | 0.00224 | 0.00036 | **6.2×** | 11.3× → 15.6× |
| 50 | 0.00034 | 0.00005 | **6.7×** | 75.3× → 103.5× |

96.7% of LUT segments have ε < 50% of the global bound. Combined with DA: **~11.9× safety factor** at N=15.

### 2. Adaptive Mixed-Precision LUT Density Allocation
Greedy max-heap algorithm: allocates more LUT points to high-curvature B-spline functions, fewer to flat ones.

| Budget | Uniform Worst ε | Adaptive Worst ε | Reduction | N Range |
|--------|-----------------|-------------------|-----------|---------|
| N=10 | 0.00982 | 0.00294 | **70.0%** | [3, 18] |
| N=15 | 0.00406 | 0.00115 | **71.6%** | [3, 28] |
| N=20 | 0.00220 | 0.00061 | **72.2%** | [3, 38] |
| N=50 | 0.00033 | 0.00009 | **73.1%** | [4, 96] |

Quality parity: adaptive needs **41.8% less storage** (17,888 vs 30,720 bytes) for same worst-ε as uniform N=15.

### 3. Doubleton Arithmetic (DA) with Sign-Structural Analysis
Affine arithmetic preserving weight-matrix sign structure. Random-walk model explains 3.1× tightening over Interval Arithmetic. Forms the base error-propagation framework that both algorithms above compose with.

---

## Compiler Architecture

```
                        NEUROPLC COMPILER
                              |
    ┌──────────┬──────────┬───┴───┬──────────┬──────────┐
    │          │          │       │          │          │
    ▼          ▼          ▼       ▼          ▼          ▼
┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐ ┌───────┐
│FRONTEND│ │  IR   │ │OPTIMIZ│ │ANALYZE│ │BACKEND│ │VALIDATE│
│PyTorch│▶│ GRAPH │▶│  ER   │▶│   R   │▶│  SCL  │▶│Python │
│→ IR   │ │       │ │       │ │       │ │S7-1200│ │vs SCL │
│       │ │MatMul │ │Adapt. │ │Memory │ │S7-1500│ │1e-4 ok│
│KAN -┐ │ │Bspline│ │Bspline│ │FLOPs  │ │       │ │       │
│MLP -┼─┘│ReLU   │ │DeadNod│ │Budget%│ │       │ │       │
└───────┘ └───────┘ └───────┘ └───────┘ └───────┘ └───────┘
```

**6 optimization passes**: adaptive B-spline LUT sampling, dead node elimination, constant folding, and 3 backend-specific transforms.

**DB+FB split** for S7-1200 64KB work memory limit: parameters in DATA_BLOCK, inference logic in FUNCTION_BLOCK.

---

## Technical Stack

| Layer | Technology |
|-------|-----------|
| Model training | PyTorch, VRM-KD (ICCV 2025 Highlight) |
| IR + compiler | Custom IR graph (6 op types), 6 substantive optimization passes |
| Code generation | Siemens SCL (IEC 61131-3), S7-1200 + S7-1500 targets |
| Verification | DA + Segment-Aware Bounds + IA (3 methods) |
| Validation | TIA Portal V21 Openness API (184 MCP tools) |
| Dataset | CWRU 12kHz DE, 4 classes × 4 fault diameters |
| Features | 28-D: 10 time-domain + 10 frequency-domain + 8 dispersion entropy |

| Parameter | Value |
|-----------|-------|
| CWRU | 12kHz DE, 4 fault types × 4 diameters × 4 loads |
| Teacher | 1D-CNN(16→32→64) + 4-head SA, 48,708 params |
| Student KAN | [28,16,4], grid=8, k=3, 6,148 params |
| Distillation | VRM-KD: τ=4.0, α=0.3, λ_rel=0.5 → 99.93% CWRU |
| S7-1200 | 15 LUT pts, 45.2 KB work memory / 50 KB budget (90.4%, TIA V21 measured) |
| S7-1500 | 50 LUT pts, 110.8 KB / 1.5 MB budget (7.4%) |
| PLCSIM Adv | Auto instance creation: InitializeApi → RegisterInstance → PowerOn (<100ms) |

---

## File Structure

```
D:/neuroplc-paper/
├── README.md                          ← you are here
├── paper/
│   ├── main.tex                       (~4,005 lines, 35 pages)
│   ├── section_svnn.tex                (SVNN framework)
│   ├── references.bib
│   ├── figures/                        (9 PDF figures)
│   └── fig_tikz/                       (TikZ source for overview + arch)
├── code/
│   ├── models/
│   │   ├── student_kan.py             KAN [28,16,4], B-spline basis
│   │   ├── student_mlp.py             MLP baseline [28,32,16,4]
│   │   └── teacher_cnn.py             Teacher CNN, 48K params
│   ├── neuroplc/
│   │   ├── ir.py                      IR graph (6 op types)
│   │   ├── frontend.py                PyTorch → IR
│   │   ├── optimizer.py               6 passes + adaptive B-spline
│   │   ├── backend_s7.py              IR → SCL (single-file, S7-1200/1500)
│   │   ├── backend_s7_db.py           IR → SCL (DB+FB split, TIA-compatible)
│   │   ├── analyzer.py                Memory/FLOPs budget analysis
│   │   ├── compiler.py                Orchestrator (Frontend→Optimizer→Backend)
│   │   ├── affine_verify.py           Doubleton Arithmetic verification
│   │   ├── interval_verify.py         Interval Arithmetic baseline
│   │   ├── validator.py               Python vs SCL cross-validation
│   │   └── scl_templates.py           SCL code templates
│   ├── segment_bound.py               ★ Algorithm A: segment-aware bounds
│   ├── adaptive_lut.py                ★ Algorithm B: adaptive LUT allocation
│   ├── analyze_da_depth.py            DA sign-structural analysis
│   ├── evaluate.py                    Experiments E1–E16
│   ├── train_teacher.py / train_student_kd.py
│   ├── preprocess.py                  28-D feature extraction
│   ├── visualize.py                   7 figures + plots
│   ├── regenerate_scl.py              Quick SCL regen with compiler
│   ├── regenerate_db.py               DB+FB SCL regen (TIA-compatible)
│   └── tests/                         42 tests (38 pass, 4 skip)
├── results/
│   ├── student/
│   │   ├── kan_kd_vrmKD_best.pt       Active KAN checkpoint (VRM-KD)
│   │   └── mlp_kd_vrmKD_best.pt       Active MLP checkpoint (VRM-KD)
│   ├── teacher/teacher_best.pt
│   ├── scl_output/                    12 SCL files (KAN+MLP × 2 targets × 3 formats)
│   ├── adaptive_lut.json              Algorithm B results
│   ├── da_analysis.json               DA sign-structural analysis results
│   └── evaluation/evaluation_results.json
├── data/                              CWRU + XJTU-SY datasets
├── tia_project/                       TIA Portal V21 validation projects
└── docs/                              Gap report + ModelScope guide
```

---

## Reproducing Results

### Algorithm A: Segment-Aware Bounds
```bash
cd code
python segment_bound.py
# → outputs per-N statistics + DA composition results
```

### Algorithm B: Adaptive LUT Allocation
```bash
cd code
python adaptive_lut.py
# → outputs 4-budget comparison + quality parity + saves results/adaptive_lut.json
```

### SCL Generation
```bash
cd code
python regenerate_scl.py   # single-file SCL (compiler pipeline)
python regenerate_db.py    # DB+FB split SCL (TIA Portal compatible)
```

### Full Paper Compile
```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
# → 0 errors, 0 warnings, 18 pages
```

### TIA Portal Validation
```bash
# Requires TIA Portal V21 + Openness + MCP server
# Projects in tia_project/NeuroPLC_Verify/
# Import SCL → Compile → Verify 0 errors
```

---

## Experiment Index (E1–E53, 7 Validation Experiments)

| # | Experiment | Key Finding |
|---|-----------|-------------|
| E1–E16 | (core compiler experiments) | DA + Segment-Aware + Adaptive LUT |
| E17 | RTNNIgen comparison | NeuroPLC: native B-spline, formal guarantees |
| E18 | Paderborn cross-dataset | Domain shift quantified |
| E21 | Theorem 1 tightness | Adversarial lower bound |
| E25 | Z3-verified WCET | ≤2.86 ms, 2.9% of scan cycle |
| E28 | Compiler scalability | Memory is binding constraint |
| E29 | PLCSIM resource analysis | TIA-measured block sizes |
| E37 | Two-Tier verification (DA+Z3) | 512/512 UNSAT |
| E40 | Compositional verification | 9-step cert, ~200-line TCB |
| E41 | MLP verification gap | 512/512 vs 0/48 |
| E42 | MNIST cross-domain | Identical pipeline, 98.6% |
| **E43** | **TIA auto multi-target validation** | **4 targets, MCP 0e 0w** |
| **E48** | **KAN vs MLP verification gap** | **512/512 vs 0/48, Prop 1 validated** |
| **E49** | **DA √d scaling law** | **105 archs, r=0.9872, p<10⁻⁵** |
| **E50** | **Adversarial safety proof** | **5,000 inputs, 100/100 preserved** |
| **E51** | **SCL feature extraction front-end** | **10-D FB, IEEE 754 equivalent** |
| **E52** | **Verification blind spot** | **Test passes but SVNN SF<1; adversary finds flips** |
| **E53** | **Sound in-domain worst-case** | **Real compiler LUT, strict domain, certifies at N≥15** |
| **V1** | **Worst-case adversarial safety** | **5,000 inputs, 100/100 worst-case preserved** |
| **V2** | **LLM vs NeuroPLC SCL generation** | **LLM: 6 defects, 0 weights; NeuroPLC: 0e 0w** |
| **V3** | **DA √d scaling law** | **105 archs, Pearson r=0.987, p<10⁻⁵** |
| **V4** | **KAN vs MLP verification gap** | **512/512 vs 0/48; 38× worse error propagation** |
| **V5** | **ONNX vs NeuroPLC IR** | **Export fails; 763× node explosion; 450× memory overshoot** |
| **V6** | **Z3-verified WCET** | **Total ≤2.86 ms, 2.9% of cycle** |
| **V7** | **Verification blind spot** | **Accuracy 99.93% but SF<1 at N≤7; 225 adversarial flips** |

---

## Session History

| Session | Date | Key Achievement |
|---------|------|----------------|
| S1 | 07-03 | Theorem 1 proof + references fix |
| S2 | 07-04 | DB+FB split → TIA Portal 0 errors |
| S3 | 07-04 | DA sign analysis + paper restructure |
| S4 | 07-05 | Algorithm A + B: segment-aware bounds + adaptive LUT |
| S5 | 07-07 | ★ Final: 4 killer experiments, TIA MCP validation, SCL front-end, IEC 61508 SIL mapping, verification certificate bundle, PLCSIM API pipeline |

---

## Environment

- Python 3.14.3 (system), venv: `D:\dev-tools\research\venv\`
- PyTorch 2.7.1+cpu, NumPy, SciPy, scikit-learn
- TIA Portal V21 + Openness API (MCP 184 tools)
- Windows 11, Git Bash

---

*Last updated: 2026-07-07 CST*
*Author: 刘甫悦 (板板) + Claude*
*Paper: 0 errors, 0 warnings, 35 pages, 4438 lines + 780 lines (section_svnn.tex), polished & submission-ready*
