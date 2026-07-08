# Pre-Submission Referee Report

**Paper**: NeuroPLC — Structurally Verifiable Compilation from PyTorch to IEC 61131-3 SCL  
**Authors**: 刘甫悦 (板板) + Claude  
**Date**: 2026-07-08  
**Review Standard**: Leading Field Journal (IEEE TII / TIE level)

---

## Overall Assessment

This paper introduces the first PyTorch-to-IEC 61131-3 SCL compiler with machine-checkable design-time correctness guarantees, backed by the Structurally Verifiable Neural Network (SVNN) theoretical framework (6 theorems + 2 propositions). The combination is genuinely novel: no prior work provides end-to-end compilation + formal verification for Siemens PLCs. The principal strength is the three-tier compositional verification pipeline (Z3 template proofs + 512 per-function certificates + ~200-line composition checker) and the theoretical closure from sufficiency (Theorem 2) through computational necessity (Theorem 5) to generalization (Theorem 6). The single most critical issue is the **absence of physical PLC measurement** — all timing and fidelity evidence is simulation-based (PLCSIM, Z3 WCET, instruction counting), which a TII/TIE reviewer will immediately flag as the gap between "design-time correctness" claims and hardware deployment evidence.

**Preliminary Recommendation**: Major Revision. The bones of a TII/TIE paper are present, but acceptance requires closing the physical PLC measurement gap, reconciling conflicting numerical values, and fixing 6 theorem/equation-level errors.

---

## 1. Contribution & Referee Assessment

### Part 1 — Central Contribution

The paper introduces the Structurally Verifiable Neural Network (SVNN) framework, which formalizes the architectural conditions (operation-type closure, univariate boundedness) under which a compiler can compute, from model parameters alone, a finite design-time correctness guarantee for PyTorch-to-IEC 61131-3 SCL compilation — and it proves that KAN satisfies these conditions while standard MLPs provably do not.

**Is it genuinely new?** Yes.
1. No prior compiler translates PyTorch models to Siemens S7-1200/1500 IEC 61131-3 SCL with TIA Portal V21 0-error/0-warning compilation.
2. The SVNN framework is a genuinely new theoretical lens: reframing compiler correctness from "can we verify an arbitrary architecture?" to "which architectures are inherently verifiable?"
3. The three-tier compositional verification has no precedent in PLC compilation.
4. The sign-structural affine arithmetic (Doubleton Arithmetic) with probabilistic tightening lemma is novel.

**Closest prior work:** RTNNIgen (IECON 2024) for Keras→IEC 61131-3 ST; Corrêa (USP 2024) for KAN-on-PLC via Snap7 distributed execution; Schwartz et al. (2026) for PWA/MILP KAN verification. None provides the end-to-end compilation + formal verification combination.

**Rating: Significant.** Shifts the conversation from "how to compile ML to PLCs" to "which architectures can be compiled with design-time guarantees." Not Transformative because: (a) validation is confined to 2-3 layer KANs on CWRU; (b) no physical PLC measurement; (c) SVNN currently admits only a narrow class of architectures.

### Part 2 — Evidence and Credibility

| Claim | Evidence | Assessment |
|-------|----------|------------|
| Compiler semantic preservation | Theorem 1 + Z3 SMT (9/11 exact, 2/11 bounded) + E6 100% agreement | Strong |
| SVNN sufficiency (Theorem 2) | Formal proof via structural induction + de Boor bound | Strong |
| KAN satisfies SVNN | Verification of Conditions 1-3 on KAN[28,16,4] | Strong |
| MLP does NOT satisfy SVNN | 0/16 or 0/48 Z3-verifiable components (E41) | Architecture-dependent count needs clarification |
| DA 3.1× tightening | Lemma 3 + 105-architecture scaling law (r=0.987) | Strong |
| TIA Portal compilation | 4 model-PLC combinations, all 0e/0w | Strong, but offline only |
| Cross-dataset | XJTU-SY 91.7% FT, MNIST 98.6% | Adequate, limited |
| Verification blind spot | Adversarial search: 225 flips at N=4 | Honest and valuable |

**Main threats:**
1. **No physical PLC measurement** — acknowledged, not resolved
2. **PLCSIM-only validation** — acknowledged, not resolved
3. **CWRU dataset age** — adequately addressed (XJTU-SY + cross-load splits + leakage mitigation)
4. **Soundness gap (Szász et al. 2025)** — cited correctly, scoped honestly

### Part 3 — Required and Suggested Analyses

**Required:**

1. [CRITICAL] **Physical PLC measurement of inference latency and floating-point discrepancy.** Z3 WCET (2.86 ms) and static instruction timing (13.4 ms) are proxies. A reviewer will demand wall-clock measurement on S7-1200 CPU 1211C. This is the single most important missing evidence.

2. [CRITICAL] **Clarify and elevate the per-element vs. aggregate bound distinction.** The DA bound (0.079 per-element) is 46× smaller than empirical MaxAE (3.65). The correlation-aware aggregate bound (6.4, safety factor 1.75×) is the honest, defensible number but appears only buried in a Remark. This must be front-loaded to preempt the most obvious reviewer challenge.

3. [CRITICAL] **Deeper KAN validation (L ≥ 3 layers).** Accuracy + Z3 verification results exist only for L=2. The depth-scaling claim is theoretically validated but empirically unsubstantiated.

4. [CRITICAL] **CROWN/DeepPoly/α-β-CROWN comparison on the same KAN.** The positioning claim ("compiler-computed bounds without LP solver") needs quantitative backing. Is the 11.9× safety factor competitive with state-of-the-art linear relaxation methods?

5. [CRITICAL] **SVNN condition ablation.** Experimentally isolate which condition matters more: e.g., KAN with transcendental base activation vs. KAN with coupled linear+nonlinear node.

**Suggested:**

1. [MAJOR] OPC UA live data demonstration (simulated loop: Python OPC UA → PLCSIM → SCL → OPC UA)
2. [MAJOR] TFLite Micro comparison on the same KAN model
3. [MAJOR] Sensitivity analysis of DA bound to weight perturbation
4. [MAJOR] Formal soundness argument for the Tier-3 checker
5. [MAJOR] Verification cost vs. model training cost contextualization

### Part 4 — Literature Positioning

**Citation coverage:** Thorough and current (many 2025–2026 references). Key works correctly cited: Liu et al. (KAN), RTNNIgen, MLconverter, Corrêa (KAN-on-PLC), Schwartz et al. (PWA+MILP), Tankman (KAN Lipschitz), KANELÉ + LUT-KAN (hardware), Szász et al. (soundness), Katz et al. (Reluplex/NP-hardness), de Boor (spline theory), ISO/IEC TS 22440.

**Missing:**
- Mohri et al. (2018) "Foundations of Machine Learning" — standard reference for Rademacher bounds
- Arcade.PLC (Biallas et al., 2014) — early PLC verification for completeness
- DeepPoly/zonomotope work (Singh et al., 2019; Mirman et al., 2018) — strengthen NN verification Related Work

**Differentiation from prior work:** Strong. Competitive positioning tables clearly differentiate from RTNNIgen, MLconverter, KANELÉ, LUT-KAN, KAN-SoC, Corrêa. ONNX incompatibility (V5) and LLM comparison (V2, 6 defects) provide empirical evidence for "why not" arguments.

### Part 5 — Journal Fit and Recommendation

**Best realistic targets (in order):**
1. **IEEE TII** — Best fit: industrial AI + embedded systems + formal methods
2. **IEEE TIE** — Strong fit: PLC research + embedded AI deployment
3. **IEEE T-ASE** — Good fit: formal methods for automation
4. **MSSP** — Weak fit unless reframed around diagnostic application

**Recommendation: Major Revision.** Acceptance requires physical PLC measurement, reconciliation of conflicting numerical values, and closing the per-element/aggregate bound communication gap.

### Part 6 — Questions to the Authors

1. **Physical hardware gap.** Have you run even a single-inference test on a physical S7-1200? If not, what is the specific engineering blocker and what is your timeline?

2. **Per-element bound vs. aggregate MaxAE.** The DA bound (0.079) is 46× smaller than MaxAE (3.65). If I am a safety engineer certifying this system, which number should I use as my deployment gate — the per-element bound (0.079, safety factor 8.5×) or the correlation-aware aggregate bound (6.4, safety factor 1.75×)? Under what conditions does the per-element bound fail to bound actual per-element error?

3. **SVNN Condition 2 for SiLU on compact domains.** If LUTizeEXP bounds SiLU analytically (ε ≤ 0.00346), why can't the same segment-aware analysis apply to SiLU MLP, making it partially SVNN-compliant? Is the barrier truly transcendental nature, or the coupled MatMul+SiLU structure?

4. **Compositional certificate checker correctness.** What is the argument that the ~200-line checker is correct? Have you tested it on deliberately corrupted certificates to verify it rejects invalid compositions?

5. **Theorem 5 NP-hardness dependence on ReLU.** The reduction uses Katz et al. (2017) for ReLU NP-completeness. But Condition 2 already excludes ReLU. The architectures satisfying Condition 2 but violating Condition 1 are Sigmoid/Tanh MLPs. Does Theorem 5 hold for smooth activations? Can you provide a reduction for the smooth case?

6. **Generalization bound practical significance.** Theorem 6 gives ΔL ≤ 0.0136 (L=2, γ=0.182). At typical industrial training set sizes (<10,000 samples), how many samples would be needed for Theorem 6 to provide a non-vacuous bound for a 5-layer KAN with γ=0.5?

7. **The compilable frontier for hybrid architectures.** What about CNN feature extraction + KAN classification? Would such a hybrid be interior or exterior? Is the frontier binary or a continuous spectrum?

---

## 2. Unsupported Claims & Identification Integrity

### Causal/Claim Overclaiming

1. [MAJOR] Abstract L105 vs. Experiments (V4, E41) | "identically-sized KAN vs. MLP yield 512/512 vs. 0/16 Z3-verifiable components" | Abstract says 0/16; experiments use MLP [28,32,16,4] yielding 0/48. Unify denominators.

2. [MAJOR] main.tex L2982 | "first application of translation validation to a neural network-to-PLC compiler" | TV has been used on NN compilers before (TVM, Glow). Add qualifier: "first for PLC target."

3. [MAJOR] main.tex L4036 | "first neural network compiler whose correctness is formally verified at compiler-template level" | Depends on Z3 soundness. Acknowledge "verified modulo Z3's soundness (UNSAT results)."

4. [MINOR] main.tex L2272 | "production Siemens environment" → "Siemens TIA Portal V21 engineering environment" (not a live factory floor).

5. [MINOR] main.tex L318 | "inherits the LUT compilation paradigm" → "converges on" (not direct lineage from KANELÉ).

### Generalization Issues

6. [MAJOR] Cross-Domain Generality | MNIST uses fundamentally different pipeline (PCA, training from scratch, 4-class subset). The "identical pipeline" claim is true only for compiler/verification stage. Clarify.

7. [MINOR] Abstract L113 | Three accuracies listed in sequence (99.93%, 91.7%, 98.6%) imply comparable rigor. Add parenthetical qualification for each training regime.

### Missing Caveats

8. [CRITICAL] 4/6 IR op types Z3-verified | StandardAct and Softmax use "analytic proof" (identity argument: "SCL and PyTorch evaluate identical formulas"). 33% of IR ops are NOT mechanically verified. Flag prominently.

9. [MAJOR] Szász soundness gap | S7-1200 lacks hardware FMA — this *reduces* the gap. Mention as a mitigating factor.

10. [MAJOR] Conclusion does not carry the "no physical PLC" caveat with same prominence as Limitations section. Add to conclusion.

11. [MAJOR] S7-1500 "~100× faster" | This is estimated, not measured. Add qualification.

12. [MINOR] XJTU-SY 91.7% | 8.3% misclassification rate may be insufficient for safety-critical deployment. Note this is a domain gap property, not a compiler limitation.

---

## 3. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. [CRITICAL] section_svnn_theorems.tex L102 vs. Abstract L111 | Remark says "Theorem~3 establishes computational complexity separation" but computational necessity is **Theorem~5**. Theorem~3 is greedy-LUT minimax optimality. **Replace Theorem~3 → Theorem~5.**

2. [CRITICAL] main.tex L3711 vs. E12-FT L3093-L3112 | `E12 (XJTU-SY zero-shot: 37.3%, fine-tuned: 79.4%)` — these numbers do NOT belong to XJTU-SY. XJTU-SY = 29.8%/91.7%. These are Paderborn (E18) numbers. **Fix attribution and numbers.**

### Terminology Drift

3. [MAJOR] IA bound: 0.242 vs. 0.172 | Two conflicting "sound IA worst-case" bounds. DA tightening 3.1× uses 0.242/0.079; reconciliation table shows 0.172 only. Add 0.242 to table, label each clearly.

4. [MAJOR] Two "Proposition 1" — IR Minimality (main.tex L500) and MLP non-SVNN (section_svnn.tex L642). Two "Proposition 2" — Adversarial Lower Bound (main.tex L1924) and ChebyKAN SVNN (section_svnn_chebykan.tex L21). Rename non-SVNN ones to "Claim/Observation" or use distinct labels.

5. [MAJOR] MLP component count: 0/16 vs. 0/48 | Always specify architecture when citing verifiability count.

### Minor Inconsistencies

6. [MINOR] FP32 accuracy: 99.99% (E10) vs. 99.93% (E1) — different numbers for same quantity.
7. [MINOR] Compilation time: <20s (prop:complexity) vs. ~30s (tab:engineering).
8. [MINOR] FLOPs count: 7,388 (tab:compiler) vs. 4,308 (tab:wcet).
9. [MINOR] Memory: 46,332 bytes → 45.25 KB, rounded to 45.2 KB. Utilization should be 90.5%, not 90.4%.
10. [MINOR] Hoisting ratio example uses [28,32,16,4] (L=3) but primary KAN is [28,16,4] (L=2).

---

## 4. Mathematics, Equations & Notation

### Mathematical Errors

1. [CRITICAL] section_svnn_theorems.tex L102 | Remark `rem:separation`: "Theorem~3" → **"Theorem~5"**.

2. [CRITICAL] section_svnn.tex Eq. `eq:convergent_bound` (L277) | `max_ℓ(ε_f·d)/(1-γ) = O(L·M_max·h²·d_max)` is **algebraically false**. The LHS is O(1) in L (geometric series converges); the RHS is O(L). Remove `= O(L·M_max·h²·d_max)` or change to `O(M_max·h²·d_max/(1-γ))`.

3. [CRITICAL] IA safety factor: 2.8× (L1032, Δ=0.242) vs. 3.9× (L1666, Δ=0.172). Both claim to be the sound IA bound for the same model at N=15. Reconcile to a single value.

4. [CRITICAL] Two "Proposition 1" labels and two "Proposition 2" labels — numbering collision across sections.

5. [CRITICAL] Theorems 2, 5, 6 have **no `\label`** — cannot be cross-referenced. Add `\label{thm:svnn_conditions}`, `\label{thm:necessity}`, `\label{thm:generalization}`.

6. [MAJOR] Theorem 2 "depth-uniform" claim contradicts "grows only linearly in L." Under Condition 3 (γ<1), the bound is trivially depth-independent. State clearly: `O(M_max·h²·d_max/(1-γ))`.

7. [MAJOR] Theorem 6 Eq. `eq:svnn_gen_bound`: factor `4` in Rademacher term replaces standard `2`. Justification ("multi-output networks with d_out outputs") is hand-wavy. Provide rigorous argument.

8. [MAJOR] Theorem 5 NP-hardness for Sigmoid/Tanh: reduction path not provided. Restrict claim to piecewise-linear activations or provide explicit reduction.

9. [MAJOR] Eq. `eq:layer_recurrence`: `d_{ℓ-1}` multiplier may be incorrect depending on layer ordering (element-wise-then-linear vs. linear-then-element-wise). Clarify.

### Notation Inconsistencies

10. [MAJOR] Δ overload: grid spacing AND error bound. Use distinct notation.
11. [MAJOR] Remark `rem:six_theorems` conflates Theorem 4's probabilistic bound with Theorem 2's deterministic bound.
12. [MINOR] γ used for per-layer product (Theorem 2) and network-wide majorant (Theorem 4/6). Clarify.

### Undefined Notation

14. [MAJOR] `M_2^{char}` used in Theorem 1 before definition (first defined in E11). Move definition earlier.
15. [MINOR] `L_net^{IA}`, `L_net^{DA}` need explicit definitional paragraph.
16. [MINOR] DA random walk `R ∝ √d_1`: constant of proportionality not specified.

### LaTeX Math Formatting

21-23. [MINOR] Absolute values need scaling (`\bigl|...\bigr|`), text-mode fragments, redundant wording.

---

## 5. Tables, Figures & Documentation

### Cross-Reference Issues (CRITICAL)

1. [CRITICAL] `tab:opt_soundness` (L984) — **defined but never referenced in text**. Add citation.
2. [CRITICAL] `tab:compositional` (L4000) — **defined but never referenced in text**. Add citation.
3. [CRITICAL] `tab:cross_domain` (L4181) — **defined but never referenced in text**. Add citation.
4. [CRITICAL] `tab:wcet` (L2501) — **defined but never referenced in text**. Add citation.

### Tables with Missing/Incomplete Notes

5. [MAJOR] `tab:template_verify` — No notes. Add: Z3 version, which are mechanized vs. analytic, that proofs are one-time.
6. [MAJOR] `tab:compositional` — No notes. Add: IA vs. DA distinction, what 1,716 ms covers.
7. [MAJOR] `tab:verify_gap_propagation` — No notes. Add: which methods produce which values.
8. [MAJOR] `tab:cross_domain` caption — 6 lines long, too verbose per IEEE style. Shorten.

### Formatting Inconsistencies

9. [MINOR] 11 tables use `\small`, 22 use `\footnotesize`. Unify to `\footnotesize`.
10. [MINOR] Number formatting inconsistent (comma-separated vs. raw). Unify.
11. [MINOR] `tab:cycle_count` — circular caption dependency with `tab:instr_timing`.

---

## 6. Spelling, Grammar & Style

### Critical Issues

1. [CRITICAL] `section_svnn.tex` L147 | `nolinear` → **`nonlinear`** | Spelling error.
2. [CRITICAL] Duplicate Proposition numbering: "Proposition 1" used for both IR Minimality (main.tex L500) AND MLP non-SVNN (section_svnn.tex L642). "Proposition 2" used for Adversarial Lower Bound (main.tex L1924) AND ChebyKAN SVNN (section_svnn_chebykan.tex L21).
3. [CRITICAL] Hard-coded section references: `{\S}IV-D`, `{\S}IV-F`, `{\S}V-G` will break if sections are reordered. Replace with `\ref{}` labels.
4. [CRITICAL] Terminology inconsistency: "sign-structural affine arithmetic" vs. "doubleton arithmetic (DA)" used interchangeably. Pick one consistent name.

### Major Issues

5. [MAJOR] Theorem 2 depth claim ambiguity: text says "depth-independent geometric-series form" but proof gives `O(L·...)`.
6. [MAJOR] `ΔL` collision: uses L for depth AND generalization gap. Use distinct notation.
7. [MAJOR] Scalar norm `\|\Delta z_k\|` should be `|\Delta z_k|`.
8. [MAJOR] section_svnn_chebykan.tex L198: `8.5 vs. 45.6` needs qualifier (mean empirical vs. Markov-bound M₂).
9. [MAJOR] FB1 memory: 13,360 bytes ≠ 13.4 KB (it's 13.36 KB).
10. [MAJOR] "Critically"/"Crucially" overused (~25 times). Most instances unnecessary.

### Minor Issues (Selected)

- Missing space before `\square` (L1546)
- `nolinear` → `nonlinear` (L147 section_svnn)
- "Two-Tier verification" vs. "Three-Tier" (L2528)
- `$+=$` should be `$\mathrel{+}=$` (L994)
- "interestingly" should be removed (L2722)
- MCU → MCUs (L360)

---

## Priority Action Items

### CRITICAL (must fix — these could cause desk rejection or major referee objections):

| # | Agent | Issue | Location |
|---|-------|-------|----------|
| 1 | A6 | **No physical PLC measurement** — the primary evidence gap | §VI-D |
| 2 | A6 | **Per-element vs. aggregate bound confusion** — 46× gap unexplained | §IV-D Remark 1 |
| 3 | A3 | **4/6 IR op types Z3-verified, not 6/6** — StandardAct+Softmax unverified | Abstract + §V Tier 1 |
| 4 | A2 | **Theorem~3 → Theorem~5 mislabel** in rem:separation | section_svnn_theorems.tex L102 |
| 5 | A2 | **XJTU-SY numbers (37.3%/79.4%) are actually Paderborn's** | main.tex L3711 |
| 6 | A4 | **O(1) = O(L) algebraic error** in eq:convergent_bound | section_svnn.tex L277 |
| 7 | A4 | **IA safety factor: 2.8× vs. 3.9× conflict** | main.tex L1032 vs L1666 |
| 8 | A4 | **Two "Proposition 1" + two "Proposition 2"** — numbering collision | Multiple |
| 9 | A4 | **Theorems 2, 5, 6 lack \label** — uncross-referenceable | section_svnn.tex + section_svnn_theorems.tex |
| 10 | A5 | **4 tables defined but never referenced in text** | tab:opt_soundness, tab:compositional, tab:cross_domain, tab:wcet |
| 11 | A1 | **`nolinear` → `nonlinear`** spelling error | section_svnn.tex L147 |
| 12 | A3 | **0/16 vs. 0/48 denominator mismatch** across abstract/experiments | Abstract L105 vs E41 |

### MAJOR (should fix — will likely be raised by referees):

| # | Agent | Issue |
|---|-------|-------|
| 13 | A6 | Deeper KAN validation (L ≥ 3 layers) with accuracy + Z3 |
| 14 | A6 | CROWN/DeepPoly comparison on same KAN model |
| 15 | A6 | SVNN condition ablation experiment |
| 16 | A4 | Theorem 6 factor-4 inflation lacks rigorous justification |
| 17 | A4 | Theorem 5 NP-hardness for smooth activations: reduction path needed |
| 18 | A4 | Eq. layer_recurrence: d_{ℓ-1} multiplier may be wrong per layer ordering |
| 19 | A2 | IA bound 0.242 vs. 0.172: reconcile and document |
| 20 | A2 | MLP architecture always specified with verifiability count |
| 21 | A3 | Translation validation novelty claim: add "for PLC target" qualifier |
| 22 | A3 | Correctness guarantee scope: physical PLC caveat must appear in conclusion |
| 23 | A1 | "sign-structural affine arithmetic" vs "doubleton arithmetic" — pick one |
| 24 | A1 | Hard-coded `{\S}IV-D` etc. → replace with `\ref{}` |
| 25 | A5 | 4 tables need notes (template_verify, compositional, verify_gap_propagation, cross_domain caption) |

### MINOR (polish — improves paper quality):

| # | Agent | Issue |
|---|-------|-------|
| 26 | A1 | Remove "interestingly" (L2722) |
| 27 | A1 | "Two-Tier" → "Three-Tier" (L2528) |
| 28 | A1 | `$+=$` → `$\mathrel{+}=$` (L994) |
| 29 | A1 | MCU → MCUs (L360); missing space before `\square` (L1546) |
| 30 | A5 | Unify table font size: `\small` → `\footnotesize` throughout |
| 31 | A5 | Unify number formatting in tables |
| 32 | A2 | FP32 accuracy: 99.99% (E10) vs 99.93% (E1) |
| 33 | A2 | Compilation time: <20s vs ~30s |
| 34 | A2 | FLOPs: 7,388 vs 4,308 — explain methodology difference |
| 35 | A2 | Memory utilization: 90.4% should be 90.5% (46,332/51,200) |
| 36 | A6 | Add Mohri et al. (2018) as standard Rademacher reference |
| 37 | A6 | Add Arcade.PLC (Biallas 2014) for PLC verification completeness |

---

## Counts by Category

| Category | CRITICAL | MAJOR | MINOR |
|----------|:---:|:---:|:---:|
| Agent 1 (Spelling/Grammar/Style) | 4 | 8 | 24 |
| Agent 2 (Internal Consistency) | 2 | 3 | 6 |
| Agent 3 (Unsupported Claims) | 2 | 6 | 6 |
| Agent 4 (Math/Equations/Notation) | 6 | 7 | 11 |
| Agent 5 (Tables/Figures) | 4 | 4 | 7 |
| Agent 6 (Contribution) | 5 | 5 | — |
| **Total** | **23** | **33** | **54** |

---

*Report generated 2026-07-08 by 6-agent parallel review of 4 .tex source files (main.tex, section_svnn.tex, section_svnn_chebykan.tex, section_svnn_theorems.tex).*
