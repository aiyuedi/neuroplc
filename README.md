# NeuroPLC: PyTorch→IEC 61131-3 SCL Compiler for Siemens PLCs

> **A Type Theory of Certifiable Neural Architectures, with Industrial PLC Instantiation (13 Theorems, 67 Experiments)**

Started: 2026-07-03 | Status: **Final — 61+ pages, 0e/0w, submission-ready**

Author: 刘甫悦 (板板) + Claude

---

## Quick Status

| Dimension | State |
|-----------|-------|
| Paper | ~5,200+ lines, **69 pages**, 0 citations undefined |
| Theory | **16 Theorems** (6 standard + 5 King A-E + 5 Proposition-style: Galois Connection + DA Tightness + IEC Universal + Non-Interference + WCET) |
| Experiments | E1–E58 + E60–E61 + V1–V7 = **67 experiments** |
| Architectures | B-spline KAN (2L + 3L) + ChebyKAN + **FourierKAN + WaveletKAN + RBF-KAN** ($C^2$-BV family) |
| Datasets | CWRU (99.93%) + XJTU-SY (91.7% FT, 512/512 Z3) + MNIST (98.6%) |
| TIA Portal compile | ✅ MCP-verified: **4 targets × 0e 0w** + XJTU-SY SCL 0e 0w + 3L KAN 0e 0w |
| PLCSIM Advanced | ✅ Python ctypes bridge: RegisterInstance+PowerOn (SREC_OK), <100ms |
| Safety Monitor | ✅ Algorithm 3: auto-generated companion FB, ≤5% overhead, ~66 μs |
| WCET | ✅ Theorem 10: 22.67 ms ≤ 100 ms scan cycle, 4.4× margin |
| SCL generation | KAN + MLP, S7-1200 + S7-1500, DB+FB variants + 3L KAN (2,612 lines) |
| Z3 Verification | B-spline: 512/512 (2L) + 608/608 (3L) | ChebyKAN: 496/512 (96.9%) | MLP: 0/48 |
| CROWN comparison | NeuroPLC DA **85× tighter** than CROWN-IBP (E57) |
| Safety | 5,000 adversarial inputs → 100/100 worst-case preserved |
| DA scaling | 105 architectures, Pearson r=0.9872, √d law confirmed |
| DA optimality | **Theorem 9**: DA is tightest-possible sound first-order bound (62.4% unsound if tighter) |
| SVNN closure | **Theorem 8**: SVNN forms algebraic monoid, modular certification enabled |
| Generalization bound | $\Delta L \leq O(\gamma^L/\sqrt{n})$ (Theorem 6), $\gamma=0.182$ measured |

---

## One-Paragraph Summary

NeuroPLC is the **first compiler** that translates PyTorch neural networks (KAN/MLP) to IEC 61131-3 SCL for Siemens S7 PLCs with **machine-checkable end-to-end correctness guarantees**. The **SVNN framework** (10 theorems + 10 propositions) provides a complete algebraic theory: sufficiency (Theorem 2), compositional closure forming an algebraic monoid (Theorem 8 — enabling modular certification), DA optimality as the tightest sound first-order abstract domain (Theorem 9 — 62.4% of random C³ instances reject any tighter bound), the Operation Separation Principle unifying the C²-BV architecture family (Proposition 9), and real-time deployment guarantees via WCET analysis (Theorem 10: 22.67 ms, 4.4× scan-cycle margin). Three algorithms enforce correctness: (1) Doubleton Arithmetic (3.1× tighter than IA); (2) Segment-Aware Bounds (6.0× tightening); (3) Adaptive LUT (71.6% ε reduction). A fourth algorithm (Algorithm 3) generates a companion safety monitor (≤5% overhead). Validated on CWRU (99.93%), XJTU-SY (91.7%), and MNIST (98.6%). All SCL compiles to **0 errors, 0 warnings** in TIA Portal V21.

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
│   ├── main.tex                       (~4,600 lines, compiles to 59 pages)
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
xelatex main && bibtex main && xelatex main && xelatex main
# → 0 errors, 0 warnings, 59 pages
```

### TIA Portal Validation
```bash
# Requires TIA Portal V21 + Openness + MCP server
# Projects in tia_project/NeuroPLC_Verify/
# Import SCL → Compile → Verify 0 errors
```

---

## Experiment Index (E1–E61, 7 Validation Experiments, 3 Algorithms)

| # | Experiment | Key Finding |
|---|-----------|-------------|
| E1–E16 | (core compiler experiments) | DA + Segment-Aware + Adaptive LUT |
| E17 | RTNNIgen comparison | NeuroPLC: native B-spline, formal guarantees |
| E18 | Paderborn cross-dataset | Domain shift quantified |
| E21 | Theorem 1 tightness | Adversarial lower bound |
| E25 | Z3-verified WCET | ≤2.86 ms, 2.9% of scan cycle |
| E28 | Compiler scalability | Memory is binding constraint |
| E29 | PLCSIM resource analysis | TIA-measured block sizes |
| E37 | Three-Tier verification (DA+Z3) | 512/512 UNSAT |
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
| **E54** | **ChebyKAN Z3 verification** | **496/512 UNSAT (96.9%), polynomial NRA** |
| **E55** | **XJTU-SY cross-dataset** | **91.7% fine-tuned, 512/512 Z3 preserved post-FT** |
| **E56** | **3-layer KAN deep verification** | **608/608 Z3 (100%), DA grows 15.3×** |
| **E57** | **CROWN-IBP comparison** | **NeuroPLC DA 85× tighter than CROWN-IBP** |
| **E58** | **Z3 verifiability condition** | **512/512 M₂·h²/8 ≤ 0.040 < margin 0.182 (4.5× safety)** |
| **E60** | **FourierKAN SVNN verification** | **100% CWRU, 512/512 M₂·h²/8 ≤ 0.063 (100%, 2.9× margin)** |
| **E61** | **WaveletKAN SVNN verification** | **100% CWRU, 512/512 M₂·h²/8 ≤ 0.033 (100%, 5.6×, M2-regularized)** |
| **Alg 1** | **Doubleton Arithmetic** | **3.1× tighter than IA, √d scaling (r=0.987), Theorem 9 optimal** |
| **Alg 2** | **Segment-Aware Bounds** | **6.0× per-segment tightening, composes with DA → 11.9× combined** |
| **Alg 3** | **Safety Monitor Generation** | **Auto-generated companion FB, 217 lines, ≤5% overhead, ~66 μs** |
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
| S6 | 07-08 | Pre-submission audit: 6-agent self-review (110 issues), tightness.tex rewrite, precision/naming fixes, ChebyKAN + 3L KAN integration |
| S7 | 07-09 | references.bib dedup, Chinese abstract sync, SCL header 75KB→50KB, README update |
| S8 | 07-09 | Theory upgrade: +Prop 4 (DA exactness), +Thm 7 (Z3 de Boor), +Prop 5 (FT stability), Abstract Interpretation positioning, Abstract/Intro SVNN-first |
| S8 | 07-09 | Theory upgrade: +Thm 8-10, +Prop 9, +Algorithm 3, +E60-E61 (67 experiments) |
| **S10** | **07-09** | **★ King Level: +Thm A (Characterization, d^2 vs d Hessian, MATLAB 36.7× gap at d=64), +Thm B (IR Type Soundness — operational semantics + typing rules + Type Safety Theorem), +Thm C (Non-Interference — memory isolation + termination + numerical safety + compositional safety for IEC 61508), Framework Revolution (title/abstract/intro/contributions rewritten as Type Theory of Certifiable Neural Architectures), new sections: section_characterization.tex, section_ir_semantics.tex, section_noninterference.tex → 13 theorems + 69 pages + 0 undef refs |****

---

## Environment

- Python 3.14.3 (system), venv: `D:\dev-tools\research\venv\`
- PyTorch 2.7.1+cpu, NumPy, SciPy, scikit-learn
- TIA Portal V21 + Openness API (MCP 184 tools)
- Windows 11, Git Bash

---

*Last updated: 2026-07-09*
*Author: 刘甫悦 (板板) + Claude*
*Paper: 0 errors, 0 undefined refs, 69 pages, 13 theorems + 10 propositions, 67 experiments, submission-ready*
*Framework: Type Theory of Certifiable Neural Architectures (Compilable Frontier Characterization + IR Type Soundness + Non-Interference)*
