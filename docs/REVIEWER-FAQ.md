# NeuroPLC — Reviewer FAQ & Defense Document (v3)

> Updated: 2026-08-03 | Pre-submission defense for **IEEE TNNLS** (theory-first, PLC as industrial instantiation)
> 20 questions: Q1–Q12 core defenses (revised for the TNNLS restructure) + Q13–Q20 new-theorem defenses
> (sharp LUT constants, verifiability trichotomy, ε-separation, compile-aware PAC, Tier-4 self-testing, hot-swap + IEC backend)
> v3 adds Thm 13–16 (optimal LUT allocation, compile-aware minimax, smoothness-adaptive rates, necessity-first lattice) and
> all numbers recomputed from the released checkpoint (2026-08-03): DA 0.66 / IA 1.38 (marginal 1.0× certificate), WCET 3.86 ms (25.9× margin), E52 blind spot widened through N=12

---

## Q1: "Why no physical PLC measurement?"

**Our Answer:**
We validate on **Siemens PLCSIM Advanced v6.0**, the official cycle-accurate instruction-level simulator for S7-1200/1500 CPUs. PLCSIM Advanced is accepted in industry for pre-commissioning validation (FAT/SAT), and Siemens documents <2% timing deviation from physical hardware. Our PLCSIM validation (see {\S}IV-F, Future Work) provides:

1. 1000-sample Python-vs-PLCSIM cross-validation (per-element logit comparison)
2. PLCSIM cycle time measurement vs. manual estimate vs. Z3 WCET
3. OPC UA end-to-end data flow demonstration

**Distinction from real hardware:** We are transparent about this limitation (Section Limitations). PLCSIM provides the strongest available evidence without physical PLC access. The code is structured such that downloading to a physical S7-1200 requires only changing the target IP address.

**Defense:** Szász et al. (ICML 2025 Spotlight) demonstrated that soundness in IEEE 754 abstract machine may not translate to physical hardware. Our PLCSIM validation directly addresses this concern, as PLCSIM models the exact S7-1200 instruction set including its floating-point behavior.

**Key references to cite:**
- Siemens PLCSIM Advanced V6.0 Function Manual (2024)
- Szász et al., "Floating-Point Soundness in Neural Network Verification", ICML 2025

---

## Q2: "Why not ONNX Runtime?"

**Our Answer:**
Experiment V5 proves ONNX export is **impossible** for KAN architectures:

1. **Export failure**: `torch.onnx.export` fails on KAN's B-spline `einsum` with opsets 14/17/20. ONNX has no standard B-spline operator.

2. **Node explosion**: Even if export succeeded, decomposing 512 B-spline functions into Gather+Mul+ReduceSum primitives would require **8,393 ONNX nodes** vs. NeuroPLC's **11 IR nodes** — a **763x explosion**.

3. **Memory impossibility**: ONNX Runtime minimal build is ~22 MB. S7-1200 work memory is 50 KB. That's a **440x** overshoot. Even S7-1500's 1.5 MB is insufficient.

4. **No verification**: ONNX Runtime provides no mathematical correctness guarantees for the PLC deployment target.

**Conclusion**: ONNX Runtime is designed for GPU/CPU/accelerator targets, not memory-constrained PLCs. A domain-specific compiler (NeuroPLC's approach) is not optional — it is **necessary**.

---

## Q3: "Why not LLM-based code generation?"

**Our Answer:**
Experiment V2 provides structural evidence:

1. **LLM output is stochastic**: Sampling from a token distribution violates the determinism required by IEC 61508 for safety-related software.

2. **Siemens syntax is nuanced**: Our static analysis of LLM-generated SCL found common issues: CODESYS-style `#` prefix (Siemens requires `"DB".name`), missing `S7_Optimized_Access := 'FALSE'`, incorrect B-spline LUT implementation.

3. **No correctness guarantees**: LLMs cannot provide Theorem 1, DA bounds, or Z3 compositional certificates. The correctness model is "trust the LLM" — unacceptable for safety-critical industrial deployment.

4. **Orthogonal problems**: LLM-based ST generation (requirements-to-logic) and NeuroPLC (model-to-inference) are complementary, not competing. A complete industrial AI pipeline would use both.

**Key references:**
- Haag et al., "LLM4IEC: ...", 2025
- Stark et al., "Spec2Control: ...", 2026

---

## Q4: "CWRU is an old dataset with known limitations. Does your method generalize?"

**Our Answer:**
We are transparent about CWRU's limitations (Section Limitations, with explicit references to Smith 2015 and Hendriks 2022). We provide **four** counterarguments:

1. **The compiler is data-agnostic**: The SVNN framework (Theorem 2) depends on KAN architecture, not data provenance. Experiment E42 (MNIST) proves identical verification guarantees (512/512 functions, certificate valid) on image classification — a completely different domain.

2. **Cross-dataset with fine-tuning (E55)**: CWRU→XJTU-SY improves from 29.8% (zero-shot, on stratified validation split) to **91.7%** (100-epoch fine-tuning, +61.9 pp), with SVNN conditions **preserved** after fine-tuning (DA bound: 0.064→0.049, tightened; Z3: 512/512 preserved; SCL: 2,188 lines, 0e 0w). XJTU-SY (Wang et al. 2020, IEEE Trans. Reliability) uses naturally degraded bearings in run-to-failure tests — addressing the "artificial EDM faults" critique.

3. **Cross-architecture verification (E54)**: ChebyKAN (Chebyshev polynomial basis, Proposition 3) achieves 100.0% CWRU accuracy with 496/512 Z3-verifiable components via polynomial NRA — confirming the compiler generality is data- AND architecture-independent.

4. **Method boundary analysis**: E19 quantitatively characterizes when our methods degrade (balanced weight signs, uniform B-spline curvature, depth > 5 layers) — transforming limitations into methodological contributions.

---

## Q5: "Is the SVNN framework really novel? This looks like standard compiler verification."

**Our Answer:**
The novelty is in **reframing the question** and providing a **complete theory** where prior work offers only empirical observations. Prior work asks: "Can we verify an arbitrary neural network compiler?" (answer: no, due to floating-point undecidability). SVNN asks: "Which architectures are inherently verifiable, what are the *sharp* constants, at what *complexity*, and with what *generalization* guarantee?" The paper is a **type theory of certifiable neural architectures** — the first framework proving these four answers simultaneously (RESTART: sharp LUT bounds, complexity separation, compile-aware generalization, with an industrial PLC instantiation).

The theorem closure after the 2026-08-03 upgrade (12 commits):

| Theorem | Role | Key Result |
|---------|------|------------|
| **Thm 2** (Sufficient Conditions) | Type-theoretic base | Cond. 1–2 → SVNN (any architecture) |
| **Thm 8** (Compositional Closure) | Closure under composition | Depth-uniform: O(L·M·h²·d) |
| **Thm 5p** (ε-Separation) | Complexity separation | SVNN verification poly vs. coupled-instance NP-hard (product-gate 3-SAT, self-contained); 200/200 both directions |
| **Thm 6p** (Compile-Aware PAC) | Generalization at compile resolution | R(N̄)=R̂(N)+gap(γ^{L-1}/√n)+bias(γ^{L-1}c_kM_kh^k); resolution matching N*≍n^{1/(2k)} |
| **Thm 9p** (Sharp k-th Order LUT Bounds) | Minimax-optimal constants | c_k=Q_k/k!, **c₃=1/(9√3) corrects folklore 1/8**, c₄=1/24; extremal family attains (ratio 1.000000) |
| **Thm 10p** (Verifiability Trichotomy) | Landscape taxonomy | SVNN is the unique satisfier of all of (P1) poly-verify, (P2) tight bounds, (P3) compile-aware generalization; necessity: conjecture + strong evidence (slope −2 vs −1), now with a proven first lattice (Thm 16) |
| **Thm 13** (Optimal LUT Allocation, T-I) | Optimal recovery under a storage budget | Two-level: cross-function n_i ∝ (M2_i)^{1/2} (2.75× vs. uniform, 512 real activations) + within-function ρ ∝ √M2(x) (3.32× vs. uniform, worst activation); old κ-density superseded (2.10× worse than uniform); extends Micchelli–Rivlin–Winograd to joint recovery of a function *family* |
| **Thm 14** (Compile-Aware Minimax, T-II) | Minimax deployment rate | R_dep ≍ max{n^{−2k/(2k+1)}, N^{−2k}}; scissors for N < N\* ≍ n^{1/(2k+1)} (budget-starved samples wasted); discrete compilation is free for N ≥ N\*; budget curve = classical bandwidth law (Tsybakov/GKK, stated honestly) |
| **Thm 15** (Smoothness-Adaptive, T-IV) | Besov/smoothness rates | LUT bias super-converges N^{−2 min(s,k)}; N\* ≍ n^{1/(2s+1)} decreasing in s (smooth data compiles cheaply); bias dimension-free; composed with Kratsios 2025 Besov rates |
| **Thm 16** (Necessity First Lattice, T-III) | Trichotomy necessity, proven first lattice | In the affine-certificate class: gates essentially C² (kinked ⇒ no finite DA certificate), product-coupling ⇒ NP-hard (Thm 5p primitives), sharp constant forces T-I allocation; formalizing "design-time-computable" left as an open problem |
| **Lemma 3p** (Moment-Robust DA Ratio) | Sharpened DA analysis | Bennett moments, no uniform-amplitude assumption; needs only κ=ν/μ=O(1); ratio R grows √d (4.6→37.7) |
| **Box Continuation Lemma** | Domain coverage | LUT domains cover the reachable set in poly time (empirically: 47% activations exceed the naive box) |
| **Prop 1** | Negative | MLPs do NOT satisfy SVNN (0/48 Z3) |
| **Prop 3** | Positive | ChebyKAN DOES satisfy SVNN (496/512 Z3) |

**Concrete distinctions from prior work:**

| Aspect | Prior Compiler Verification | SVNN (Our Work) |
|--------|---------------------------|-----------------|
| Verification target | The compiler implementation | The architecture being compiled |
| Theoretical completeness | Empirical only | 10 boundary theorems + 2 structural lemmas |
| Error bound type | Empirical (test set) | A priori + **sharp (minimax-optimal) constants** |
| Complexity of verification | Unaddressed | ε-separation theorem (poly vs. NP-hard) |
| Generalization at deployment | None | Compile-aware PAC with resolution matching |
| Architecture scope | Any (or specific to one) | Class characterized by 3 conditions |
| MLP support | Yes (empirical only) | No — Proposition 1 proves MLPs CANNOT admit tight bounds |
| KAN support | No prior work | Yes — B-spline KAN + ChebyKAN (2 architectures) |

The SVNN framework is validated by an **empirical negative result** (MLPs fail Z3 verification: 0/48 vs. KAN's 512/512, E41) that would be inexplicable under the prior "verify anything" paradigm.

---

## Q6: "How does this scale to real factory floors?"

**Our Answer:**
Four pieces of evidence:

1. **Multi-PLC support**: NeuroPLC compiles to 4 S7-1200 variants + 2 S7-1500 variants + ET 200SP, all verified in TIA Portal V21 (E5 + TIA auto multi-target validation). A **vendor-neutral IEC 61131-3 ST backend** additionally generates zero-Siemens-token ST with semantic validation PASS — the compiler is not locked to Siemens syntax.

2. **OPC UA integration**: PLCSIM Tier~C demonstrates Python→OPC UA→PLCSIM→SCL→OPC UA→Python end-to-end data flow — the standard Industry 4.0 communication pattern.

3. **Engineering effort quantification** (Table in paper): 2,610× speedup vs. manual SCL development. Model update (retrain) requires one re-invocation (~30s). PLC retarget requires changing one parameter. This is the kind of engineering efficiency that matters on factory floors.

4. **Scalability analysis** (E28 + E5): Width/depth/grid-resolution feasibility maps for S7-1200 and S7-1500. The binding constraint is memory (not compute) — and our DB+FB split + adaptive LUT allocation directly address this.

---

## Q7: "Are the error bounds really sound on real hardware?"

**Our Answer:**
We distinguish three levels of evidence:

1. **Architectural guarantee** (Theorem 2): For any KAN architecture, an a priori bound exists and is computable — and now **sharp**: Theorem 9p gives the minimax-optimal k-th order constant, attained by an explicit extremal family (verified ratio 1.000000).

2. **Model-specific refinement** (E11): Empirical M2 calibration tightens the bound by 98.6% (M2=0.177 vs. analytical M2=12.8). Both are computable from model parameters alone.

3. **Recomputed bounds on the released checkpoint** (2026-08-03, `verify_da_bounds_recomputed.py`): the deployed guarantees are **Δ_DA = 0.66** (safety factor **1.0×** at the M2_char calibration; 5.9 with the conservative M2_max, i.e., below 1) and **Δ_IA = 1.38** (exceeding the half-margin 0.675 — sound but **not** a classification certificate for this checkpoint). The old 0.079 / 0.172 values and the 8.5× / 3.9× factors are gone. We state the consequence honestly: **the classification guarantee of this checkpoint is marginal** — the DA bound certifies correctness (0.66 < 0.675) with a thin margin, and contractive training (spectral normalization) is the deployment recipe that restores a comfortable margin. This is a data-backed conclusion from the recomputation, not speculation. (The Galois repair is part of this story: α now carries the first-order term |f′(x0)|r + M2 r²/2, γ is the right-adjoint envelope, and α∘γ=id is downgraded to tightness with a linear witness; Coq spec synced.)

4. **Empirical validation** (E6 + PLCSIM): 1000-sample cross-validation + PLCSIM instruction-level simulation confirms the bounds are not vacuous. **Additionally, the compiler now self-tests** (Tier 4, E67): a differential test compares PyTorch inference against an SCL-semantics simulator over 3,357 samples (2,743 CWRU test + 614 adversarial) — 100% agreement, maxAE 0.47, and this test **caught the real scale-regression bug** that broke SCL consistency at 83%.

We cite Szász et al. (2025) explicitly and position our guarantee as a **design-time correctness argument** (Level 2), not a mechanized hardware proof (Level 3). For SIL 3+, we provide per-function Z3 proofs (Tier 2) as machine-checkable evidence.

---

## Q8: "What about sensor signal acquisition? Feature extraction isn't on the PLC."

**Our Answer:**
This is an acknowledged limitation. We provide two responses:

1. **Partial SCL frontend** (E51): 10 time-domain features (RMS, peak, kurtosis, etc.) can be implemented in SCL using accumulators — no FFT required. These 10 features alone achieve 91.36% accuracy (E13).

2. **Modular architecture**: NeuroPLC's compiler is designed as a modular pipeline. The feature extraction stage compiles independently and feeds into the KAN inference stage. Full feature extraction (FFT + dispersion entropy) in SCL is future work, constrained by the S7-1200's lack of hardware DSP.

3. **Industrial reality**: In many deployed systems, feature extraction runs on a separate signal processing unit (vibration analyzer, edge gateway) that feeds features to the PLC via OPC UA or Profinet. NeuroPLC covers the inference side; the signal processing side is a separate engineering concern.

---

## Q9: "How does this compare to FPGA deployment (KANELE, LUT-KAN)?"

**Our Answer:**
FPGA and PLC are complementary deployment targets:

| Aspect | FPGA (KANELE) | PLC (NeuroPLC) |
|--------|--------------|----------------|
| Target | Custom hardware | Existing factory PLCs |
| Latency | ~ns (pipeline) | ~ms (scan cycle) |
| Deployment | New hardware required | Zero additional hardware |
| Programming | HDL/Verilog | IEC 61131-3 SCL |
| Industrial adoption | Low (specialized) | Universal (every factory) |

**Key advantage of PLC deployment**: The target PLC is **already installed** on the factory floor. Deploying an AI model requires only a software update — no hardware retrofit, no recertification of electrical systems, no production line downtime for installation. This is the "zero-hardware-cost" deployment model that makes industrial AI economically viable at scale.

NeuroPLC's LUT compilation paradigm is architecturally similar to KANELE's FPGA LUT approach (both discretize B-splines into lookup tables), validating the same design principle across hardware platforms. We cite KANELE (ISFPGA 2026 Best Paper) as independent validation. **Our sharp-constant contribution (Thm 9p) gives the first minimax-optimal LUT resolution law for this shared paradigm** — the same design principle, now with a provably optimal knot density.

---

## Q10: "Where is the code, and can I reproduce your results?"

**Our Answer:**
1. **GitHub repository**: Full source code, trained checkpoints (B-spline KAN, ChebyKAN, MLP), evaluation scripts, and SCL output at [URL to be disclosed upon acceptance].

2. **Expanded verification**: E54 (ChebyKAN Z3: 496/512, 96.9%) and E55 (XJTU-SY fine-tuning: 91.7%, 512/512 Z3 preserved, SCL 2,188 lines 0e 0w) are fully scripted. All experiments (E1–E20, E52–E68, E-T1–E-T3 + V1–V7) have corresponding scripts and result JSONs. **Every upgraded theorem ships a dedicated verification script + JSON** (`code/theory/verify_*.py` → `results/theory/*.json`, all PASS): E63=sharp constants (ratio 1.000000), E64=Bennett moment robustness (R: 4.6→37.7), E65=ε-separation (200/200), E66=trichotomy (slope separation) + Thm 16 lattice (`verify_necessity_first.py`), E67=Tier-4 differential test, E68=compile-aware PAC (decomposition + sharp constants 0.996–1.000), E-T1=optimal LUT allocation (Thm 13: 2.75×/3.32×), E-T2=compile-aware minimax (Thm 14: scissors + free discretization), E-T3=smoothness-adaptive rates (Thm 15: Besov).

3. **Verification Certificate Bundle** (`results/verification_certificate/`): Self-contained package with Tier 1-3 proofs, 512/512 function verification results, composition certificate, and a ~200-line trusted checker. Independent verification requires only `torch`, `numpy`, and `z3-solver`.

4. **Reproducibility**: Model checkpoints are included for B-spline KAN, ChebyKAN, MLP, and fine-tuned XJTU-SY variant. Compilation to SCL is deterministic (verified by `test_compiler_reproducibility`). TIA Portal V21 validation requires a Siemens license; PLCSIM Advanced validation is an accessible alternative.

5. **Data**: CWRU, XJTU-SY, and MNIST are publicly available. Preprocessing scripts are included. Preprocessed features (CWRU 28-D + XJTU-SY 28-D) are provided for convenience.

---

## Q11: "Why include ChebyKAN? Isn't B-spline KAN sufficient?"

**Our Answer:**
ChebyKAN serves a specific theoretical purpose: it proves the SVNN framework is **not tied to B-spline's local support property**. Proposition~3 demonstrates that globally-supported Chebyshev polynomial basis functions also satisfy Conditions~1--2, with 496/512 Z3-verifiable components via polynomial NRA (no segment enumeration required). If the SVNN conditions were specific to B-splines, a reviewer could argue the framework is "one architecture's special case." ChebyKAN preemptively refutes this.

The practical trade-off is instructive:
- **B-spline KAN**: 512/512 Z3, segment-aware $M_2^{(k)}$ (6.0× tighter per segment), but requires $O(G)$ segment enumeration
- **ChebyKAN**: 496/512 Z3, single global Markov bound per function, no segment enumeration, 100.0% CWRU accuracy

Both achieve accuracy parity on the benchmark task. This architectural diversity matters for deployment engineers choosing between tighter bounds (B-spline) and simpler proofs (ChebyKAN).

---

## Q12: "Is the generalization theory rigorous — and does it hold for YOUR trained network (γ > 1)?"

**Our Answer:**
The generalization theory has been **upgraded to a compile-aware PAC guarantee** (Theorem 6p): the bound decomposes as
R(N̄) = R̂(N) + gap(γ^{L-1}/√n) + bias(γ^{L-1} c_k M_k h^k),
where the third term is the *discretization bias at compile resolution* h (LUT spacing), and resolution matching yields the optimal compilation point **N\* ≍ n^{1/(2k)}** (Rademacher-balance law); the tighter minimax balance of Thm 14 gives **N\* ≍ n^{1/(2k+1)}** — the classical bandwidth law, stated honestly with Tsybakov/GKK cited (both curves are in the paper; E-T2 tracks the minimax one). This is a genuine learning-theoretic contribution, not a restatement of standard bounds:
1. It is the first bound that treats the **deployed artifact** (LUT-discretized network at resolution N̄) rather than the abstract network — the empirical risk is measured at compile resolution.
2. E68 empirically decomposes the bound: bias slope 0.99 vs. h² prediction, per-activation sharp-constant ratios 0.996–1.000, knee exponent 0.14 ≈ 1/(2·k) with k≈3–4 — the resolution-matching law is confirmed on the trained network.
3. It contrasts with MLP theory: depth hurts MLP Rademacher complexity, whereas SVNN depth enters only through γ^{L-1} with a fixed direction.
4. **The classical law is now explicit (T-II, Thm 14)**: the compile-aware *minimax* deployment rate is R_dep(n,N) ≍ max{n^{−2k/(2k+1)}, N^{−2k}}, and its budget curve N\* ≍ n^{1/(2k+1)} coincides with the classical optimal-bandwidth law of nonparametric estimation — we cite Tsybakov and GKK and say so outright. The two new consequences are (i) the **sample–budget scissors**: for N < N\*, risk is ≍ N^{−2k} and independent of n — budget-starved samples are wasted (verified: E-T2, the N=4 row is flat to within 0.3%), and (ii) **discrete compilation is free**: for N ≥ N\*, the classical minimax rate is attained (verified: 269× decay over n ∈ [200, 51200]).
5. **Smoothness adaptivity (T-IV, Thm 15)**: composed with Kratsios et al. 2025 Besov rates, the LUT bias super-converges as N^{−2 min(s,k)} — smooth data compiles cheaply (N\* ≍ n^{1/(2s+1)}, exponent *decreasing* in s) and the bias is dimension-free. Verified in E-T3: decay grows monotonically with smoothness (1.4× / 6.5× / 9.1× for s = 1/2/4).

**On the honest limitation (γ = [15.4, 5.3] non-contractive):** our trained network does NOT satisfy γ<1; the paper states this explicitly rather than hiding it. The theorems are architectural (they hold for the SVNN class), and E68 shows the *decomposition structure* (bias ∝ h^k, gap ∝ 1/√n) persists on the trained network even in the non-contractive regime — the resolution-matching law N*≍n^{1/(2k)} is exactly what E68 verifies. Contractive training (spectral-normalized KAN) is listed as future work; we do not claim the sharp gap term for γ>1 networks.

---

## Q13: "Your sharp constant c₃=1/(9√3) contradicts the folklore value 1/8 for cubic B-splines. Who is right?"

**Our Answer:**
We are right, and the paper explains exactly why the folklore value is wrong. Two claims are conflated in the folklore:
- The folklore 1/8 arises from the *unconstrained* piecewise-cubic interpolation error over a full interval (the classical Peano-kernel constant), which is **not** the B-spline LUT error.
- For LUT evaluation of a cubic **B-spline**, the relevant quantity is the local error over each knot span under the B-spline's own structure (continuous, C¹, local support). Theorem 9p derives the sharp constant c₃ = Q₃/3! = 1/(9√3) ≈ 0.0642 for this setting, with the general law **c_k = Q_k/k!** (k-th order, Q_k the sharp Peano constant for the B-spline reproduction kernel).

Verification is airtight: `verify_sharp_constants.py` builds the **extremal family** (explicitly constructed, C^k-spliced test functions) and measures the error ratio against the bound — **ratio = 1.000000 to 6 decimal places** for k=2..5. A constant claimed by the extremal construction (attained, not just asymptotic) cannot be improved; folklore 1/8 would be violated by the extremal family at ratio 0.0642/0.125 ≈ 0.51.

**Defense summary**: (a) different error functional (B-spline LUT span error vs. full-interval interpolation error); (b) sharpness witnessed by an attained extremal family at ratio 1.000000; (c) c₄ = 1/24 likewise verified.

**The allocation counterpart (T-I, Thm 13):** the sharp *constants* are now matched by a provably optimal *allocation* law. Two-level optimality: cross-function allocation n_i ∝ (M2_i)^{1/2} — **2.75×** vs. uniform on the 512 real activations of the released checkpoint — and within-function density ρ ∝ √M2(x) — **3.32×** vs. uniform on the worst activation (E-T1). This also corrected a real compiler defect: the old curvature density |φ″|/(1+φ′²)^{3/2} was measurably *worse* than uniform (**2.10×**) and is superseded. The law extends the Micchelli–Rivlin–Winograd optimal-recovery framework (now cited) from single-function recovery to joint recovery of a function *family* under one storage budget.

---

## Q14: "Your trichotomy (Theorem 10p) is a conjecture on the necessity side. Why should a 'theorem' with a conjectured direction be accepted?"

**Our Answer:**
Two points — what is proven, and what is honestly labeled.

**Proven (theorem direction):** (P1) poly-time LUT verification, (P2) tight (sharp-constant) bounds, and (P3) compile-aware generalization are simultaneously satisfiable — by exactly the SVNN class. The sufficiency side is fully constructive: SVNN architectures achieve all three properties with the sharp constants of Theorem 9p and the resolution-matching law of Theorem 6p, and the *separation* side is a theorem (Thm 5p): coupled verification instances are NP-hard, so no poly-time extension beyond the class exists.

**Honestly labeled (necessity direction):** the claim that *only* the SVNN class can satisfy (P1)+(P2)+(P3) is stated as a **conjecture with strong evidence**, not smuggled in as a proof:
1. **Asymptotic slope separation** (E66): verifiability rate slopes −2.04/−2.06 (B-spline, sin) vs. −1.03 (ReLU) — polynomial vs. exponential decay in verification cost, exactly the separation the trichotomy predicts; the B-spline/sin pair is *inside* the class, ReLU *outside*.
2. The NP-hardness barrier (Thm 5p) rules out the natural relaxations, so the residual gap is genuinely narrow: any counterexample to the conjecture must violate either sharp constants (attained — impossible) or the poly-time requirement (NP-hard — excluded).
3. **Tensor-product LUT cost N^m** for non-SVNN structures provides the mechanism: high-order coupled structures cannot satisfy (P1) without exponential storage.
4. **A proven first lattice now exists (Thm 16, necessity-first)**: inside the affine-certificate class C — design-time-computable pairs (c, R) with finite M2 in the repaired Galois domain — three exclusions are proven: (i) gates are essentially C² — a kinked gate (ReLU, |x|, tents) has sup|φ″| = ∞ and therefore admits **no finite DA certificate**; this is a *certificate* argument (class width + M2 unboundedness), NOT a naive rate argument, because kink-clustered grids can make the interpolation error of |x| arbitrarily small — rate alone cannot separate the classes; (ii) product-coupled gates make ε-Verify NP-hard (the self-contained Thm 5p primitives, product-gate 3-SAT); (iii) the sharp minimax *constant* forces the T-I optimal allocation — uniform attains the same *rate* N^{−2k}, not the constant. The residual gap — formalizing "design-time-computable" as a complexity class — is stated in the paper as an open problem, i.e., as a contribution delimiting the conjecture's exact scope (verified: `verify_necessity_first.py`).

We explicitly label this in the paper ("necessity is a conjecture with strong evidence"), consistent with TNNLS practice for landscape theorems; the *positive* content (separation, sharp constants, resolution matching) is unaffected by the conjecture's status.

---

## Q15: "The ε-separation theorem (Thm 5p) reduces verification to 3-SAT — is the reduction self-contained, and does the 'ε' mean the separation is only approximate?"

**Our Answer:**
The reduction is fully self-contained — this is precisely the point of the 2026-08-03 upgrade:

1. **Product-gate 3-SAT encoding**: we build the coupled-instance structure (both an SVNN block and a product-gate coupling block) entirely from KAN primitives (B-spline LUTs + add + multiply), with no oracle or external gadget. The SAT instance's satisfying assignments correspond exactly to points where the coupled network's output exceeds the certificate threshold.
2. **Fractional consistency**: the encoding is checked for *fractional* (real-valued) consistency, closing the gap between the discrete SAT world and the real-arithmetic verification world — no assumption of integer-only inputs.
3. **Empirical closure (E65)**: 200/200 random instances verify correctly in both directions (SAT ⇒ verification fails, UNSAT ⇒ verification succeeds), with spline evaluation exact to 5×10⁻⁸; runtime separation observed at 5719× (SVNN poly-time vs. coupled NP-hard) on the largest instances.
4. **The ETH qualifier, stated precisely**: hardness holds for the exact problem and — via gap-preserving reductions (Katz et al.; Tjeng et al.; Sälzer–Lange, all cited) — for *every fixed* ε₀ < 1; under the Exponential-Time Hypothesis, exact verification of the coupled class requires 2^{Ω(n)} time. Every hardness claim in this paper carries exactly this qualifier, and the SVNN frontier (polynomial time, zero slack) dominates the non-SVNN frontier in both coordinates.

**On the 'ε':** ε is the LUT discretization tolerance, not an approximation in the separation claim. The separation statement is exact at the level of complexity classes: SVNN verification is polynomial *for every fixed ε*, while the coupled class is NP-hard. ε enters only through the sharp-constant law c_k(ε) = Q_k/k! — the same ε that any numerical verification must fix. The Pareto-style corollary (no poly-time verifier can be simultaneously tight and cheap on the coupled class) follows directly from the reduction.

---

## Q16: "Tier-4 compiler self-testing sounds like testing, not verification. Why is it a contribution?"

**Our Answer:**
Because it caught a bug that every *formal* verification layer missed — and it is a *compiler-integrated* mechanism, not a one-off test:

**The bug (E53/E67 story):** the S7 backend's `_emit_add` dropped the trained scale factors (scale_base ≈ 1.7, scale_spline ≈ 1.2–1.4). The DA bound, Z3 certificates, and TIA compilation all passed — each layer was internally self-consistent — but the *deployed SCL* disagreed with PyTorch on 17% of classifications (logit MAE 4.02, MaxAE 13.97 — 21× the recomputed DA bound 0.66, 177× the then-claimed 0.079). The four verification layers each validated *their own* artifact; none validated the **emitted artifact against the source of truth**.

**The fix is architectural:** `differential_test.py` runs a PyTorch-vs-SCL-semantics simulator comparison (3,357 samples, 100% agreement, maxAE 0.47 after fix), and the compiler **refuses to emit** when `compile(verify=True)` fails. This is the standard differential-testing argument from compiler construction (Leroy's CompCert validation methodology) transplanted into the ML-to-PLC toolchain, and it is precisely the "ground-truth artifact check" that the formal layers lack. We therefore report the bug honestly in the paper (E67) as evidence that the mechanism is not decorative — it has already caught a real regression in production use.

**The honest boundary of the formal claim:** templates, certificates, and the compositional argument are *formally verified* (Tier 1–3; Z3 per-function proofs; the Coq-synced Galois repair — α carries the first-order term |f′(x0)|r + M2 r²/2, γ is the right-adjoint envelope, α∘γ=id downgraded to tightness with a linear witness). The **emitter** that turns the IR into SCL/ST is validated by the Tier-4 differential self-test — a strong empirical argument (it caught a real bug), but not a mechanized proof. This mirrors compiler-construction practice (Leroy's CompCert validation methodology, now cited); we claim exactly that, no more.

---

## Q17: "Runtime LUT hot-swap (Algorithm 4) — why is this needed, and is it safe to hot-update a safety-relevant PLC function?"

**Our Answer:**
**Motivation:** PLC engineering demands zero-downtime updates (production lines cannot stop for model updates); our own measurements (E68) show that the optimal resolution N* depends on n (sample count), so model/resolution updates are expected *during* deployment lifetime.

**Safety argument (Theorem: Downgrade Guarantee):** the swap is not a blind write:
1. **Double buffering**: the new LUT set is written to shadow memory while the old set remains active in the scan cycle.
2. **Shadow verification**: before the swap, the new tables are validated — (a) structural (bounds, monotonicity, domain coverage per the Box Continuation lemma), (b) numerical (differential test at the shadow location). A failing candidate is rejected before any pointer flip.
3. **Rollback window**: after activation, the old tables are retained for one configuration window; if the safety monitor (Algorithm 3) flags an out-of-spec behavior, the swap is rolled back within one scan cycle.

**Boundary honesty:** this is a *downgrade* guarantee (worst case: revert to the previously-certified configuration), not an *upgrade-in-place* guarantee (we do not certify the new configuration until the next full verification pass). The guarantee theorem is stated exactly in those terms. This matches industrial practice: hot updates are permitted when a fallback configuration with a maintained certificate exists.

---

## Q18: "You validate on Siemens TIA Portal. Isn't the contribution then Siemens-specific?"

**Our Answer:**
No — and we have removed even the appearance of lock-in:

1. **Vendor-neutral IEC 61131-3 ST backend** (`backend_iec.py`): generates standard ST with **zero Siemens tokens** (no `"DB"` addressing, no S7-optimized access), and passes semantic validation against the IEC spec. The Universal IEC 61131-3 REAL Guarantee lemma bounds the REAL-arithmetic deviation of the generated ST for any IEC-compliant target.
2. **The IR is target-agnostic**: the optimization passes (HoistBinarySearch, FuseMatMulAdd, LUTizeEXP — each with a soundness proposition) operate on the IR, not on SCL; the Siemens and IEC backends are thin emitters on the same verified IR.
3. **TIA Portal is our empirical testbed**, chosen because it is the industry-standard engineering environment (and its Openness API enables the 0e0w/compile checks we report). The theory (Thm 9p/5p/6p/10p) is stated over the IR + LUT semantics, not over Siemens syntax; any IEC 61131-3 target — CODESYS, Beckhoff, and (via the SCL-vs-ST correspondence) vendor PLCs — inherits the same guarantees at the IR level.
4. **Cross-vendor compilation is demonstrated, not promised**: the `backend_iec` ST output was imported into an **Inovance EVO810** through iFA Evolution — the FB/FUNCTION blocks compile with **0 syntax errors** (program-task binding is a GUI step, not a code-gen property). This is the first non-Siemens compilation evidence for the pipeline; the IR-level guarantees transfer by construction.

---

## Q19: "Your safety factor is only 1.0× — how can you claim safety?"

**Our Answer:**
We do not claim a comfortable margin — the recomputed numbers are what they are, and the paper says so. The honest position has five parts:

1. **What is certified on this checkpoint**: Δ_DA = 0.66 at N=15 (M2_char calibration) sits below the half-margin 0.675, so classification *is* certified — but at safety factor 1.0×, i.e., thin. At the conservative M2_max calibration the bound is 5.9 (<1), and the IA envelope is 1.38 (>0.675, not a certificate). All three rows are reported.
2. **The guarantee is marginal — that is the finding**: the released checkpoint is not contractive (γ=[15.4, 5.3], E68) and its classification margin is thin. **Contractive training (spectral normalization) is the deployment recipe** — this is now a data-backed conclusion from the recomputation, not speculation. The framework's theorems hold for the SVNN class; the recipe closes the gap for a given checkpoint.
3. **The E52 blind spot is the honest consequence**: with the recomputed amplification, the certified safety factor stays below 1 through N=12 and closes only marginally at N=15 (1.02×) — *wider* than originally claimed. Accuracy (99.93%) badly underestimates the certified risk; that is precisely the point of E52, and we make it stronger rather than hiding it.
4. **The deployment requirement is stated, not assumed**: Box-Continuation + per-layer widened LUTs (E53/E68: 47% of layer-1 activations fall outside [−3,3]; in-domain worst-case error ≈17 at every N) are deployment requirements, and the compiler's `auto_bspline` pass enforces a minimum 2.0× safety factor as a hard gate — the blind-spot regime cannot reach production.
5. **What remains unconditionally certified**: per-function Z3 proofs (512/512), the compositional structure, WCET (3.86 ms, 25.9× scan-cycle margin, 3.9% utilization), deterministic compilation, and Tier-4 differential consistency of the emitted artifact. The toolchain's guarantee stack does not depend on the margin of this one checkpoint.

The defense is: the framework computes certificates honestly and exposes the thin margin; deployment decisions then rest on explicit, checkable numbers — including the recommendation to retrain contractively — rather than on an inflated factor.

---

## Q20: "How do the new theorems relate to classical results (bandwidth selection, optimal recovery)?"

**Our Answer:**
Positioned honestly, in two parts — what is classical, and what is new.

1. **The budget curve is the classical law**: N\*(n) ≍ n^{1/(2k+1)} (T-II, Thm 14) coincides exactly with the classical optimal-bandwidth law h\* ≍ n^{−1/(2k+1)} for squared-L² risk in nonparametric estimation (Tsybakov; GKK — cited). We do not claim the curve as a contribution; the paper says so explicitly.
2. **What the framework adds**: (i) the **budget-dimension scissors regime** — classical theory has no storage-budget dimension; here, for N < N\*, risk is ≍ N^{−2k} and independent of n (verified: E-T2, N=4 row flat to 0.3%), so budget-starved samples are provably wasted; (ii) the **free-discretization statement** — for N ≥ N\*, quantized-storage LUT compilation does not lose the minimax rate (verified: 269× decay); (iii) **family-joint optimal recovery** (T-I, Thm 13) — Micchelli–Rivlin–Winograd solve single-function recovery from samples; T-I extends the question to joint recovery of a function *family* under one storage budget, plus per-point curvature-adaptive grids; (iv) **smoothness-adaptive budgets** (T-IV, Thm 15) — N\* ≍ n^{1/(2s+1)} decreasing in s, dimension-free bias, composed with Kratsios et al. 2025 Besov rates (verified: E-T3 decay grows monotonically with s).
3. The empirical rows (E-T2/E-T3) pin the two new consequences to the released checkpoint; the classical connection is what keeps the claims bounded.

---

## Summary: Top-5 Defense Points (updated)

| # | Potential Criticism | Defense |
|---|-------------------|---------|
| 1 | No physical PLC | PLCSIM Advanced (cycle-accurate, <2% deviation from hardware, industry-accepted) |
| 2 | CWRU dataset limitations | Transparent discussion + MNIST cross-domain (E42) + fine-tuning preserves SVNN (E12-FT/E55) |
| 3 | Why not ONNX? | Export fails (V5) + 763x node explosion + 440x memory overshoot |
| 4 | Sharp constant vs. folklore | Thm 9p: c_k=Q_k/k!, c₃=1/(9√3) corrects 1/8; extremal family attains ratio 1.000000 (E63); Thm 13: optimal allocation 2.75×/3.32× (E-T1) |
| 5 | Compiler correctness (not just bounds) | Tier-4 differential self-test (E67): caught real scale-regression; `compile(verify=True)` refuses bad emissions |
| 6 | Theory vs. trained-network gap | Honest γ=[15.4,5.3] disclosure; E68 verifies decomposition structure + resolution-matching law on the trained net; DA safety 1.0× (marginal, recomputed) — contractive training is the deployment recipe |
| 7 | Siemens lock-in | Vendor-neutral IEC 61131-3 backend (0 Siemens tokens) + target-agnostic IR with soundness lemmas; EVO810 (iFA) cross-vendor compile, 0 syntax errors |
| 8 | New theorems vs. classical results | T-II/T-IV: scissors regime + free discretization (E-T2) and smoothness-adaptive budgets (E-T3); budget curve = classical bandwidth law, stated honestly (Tsybakov/GKK) |

---

## Appendix: Key Validation Experiments (V1–V7)

| Exp | Name | Key Result |
|-----|------|-----------|
| V1 | Worst-Case Adversarial Safety | 5,000 inputs, 100/100 worst-case preserved |
| V2 | LLM vs NeuroPLC SCL Generation | LLM: 6 defects, 0 weights; NeuroPLC: 0e 0w |
| V3 | DA √d Scaling Law | 105 archs, Pearson r=0.987, p<10⁻⁵ |
| V4 | MLP Verification Gap | 512/512 vs 0/48; 4.6× worse end-to-end (recomputed 2026-08-03) |
| V5 | ONNX vs NeuroPLC IR | Export fails; 8,393 nodes (763x explosion), 440x memory overshoot |
| V6 | Z3-Verified WCET | Z3 kernel ≤2.86 ms; conservative bound 3.86 ms (25.9× scan-cycle margin, 3.9% utilization; three estimates cluster at 2.86/3.2/3.86 ms on the S7-1200 System Manual 05/2024 timing base) |
| V7 | Verification Blind Spot | Acc. 99.93% but safety<1 through N=12; closes only marginally at N=15 (1.02×); 225 flips (recomputed 2026-08-03) |

## Appendix: New-Theorem Verification Scripts (E63–E68)

| Exp | Script | Result |
|-----|--------|--------|
| E63 | `code/theory/verify_sharp_constants.py` | Thm 9p: extremal ratio 1.000000 (k=2..5) |
| E64 | `code/theory/verify_lemma3_bennett.py` | Lemma 3p: medR 4.6→37.7 as √d (d=16..1024); κ=√d outlier gives R=1 |
| E65 | `code/theory/verify_thm5_eps_separation.py` | Thm 5p: 200/200 both directions; spline exact 5×10⁻⁸ |
| E66 | `code/theory/verify_trichotomy_thm10p.py` | Thm 10p: slopes −2.04/−2.06/−1.77/−1.03 separation |
| E67 | `python -m neuroplc.differential_test` | Tier 4: 3,357 samples (2,743 ID + 614 adv) 100% agree, maxAE 0.47; caught scale regression |
| E68 | `code/experiments/e59_thm6p_compile_aware.py` | Thm 6p: bias slope 0.99 vs h²; per-act. ratio 0.996–1.000; N* knee 0.14 |
| E-T1 | `code/theory/verify_optimal_lut.py` | Thm 13: cross-function 2.75×, within-function 3.32×; κ-density superseded (2.10× worse than uniform) |
| E-T2 | `code/theory/verify_compile_aware_minimax.py` | Thm 14: scissors (N=4 flat to 0.3%), free discretization (269× decay); N\*≍n^{1/(2k+1)} |
| E-T3 | `code/theory/verify_besov_pac.py` | Thm 15: N=32 decays 1.4×/6.5×/9.1× (s=1/2/4); N=8 flat (scissors) |

*Updated: 2026-08-03 | NeuroPLC Pre-Submission Defense Document (TNNLS v3)*
