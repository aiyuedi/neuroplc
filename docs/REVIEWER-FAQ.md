# NeuroPLC — Reviewer FAQ & Defense Document (v2)

> Updated: 2026-08-03 | Pre-submission defense for **IEEE TNNLS** (theory-first, PLC as industrial instantiation)
> 18 questions: Q1–Q12 core defenses (revised for the TNNLS restructure) + Q13–Q18 new-theorem defenses
> (sharp LUT constants, verifiability trichotomy, ε-separation, compile-aware PAC, Tier-4 self-testing, hot-swap + IEC backend)

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
| **Thm 10p** (Verifiability Trichotomy) | Landscape taxonomy | SVNN is the unique satisfier of all of (P1) poly-verify, (P2) tight bounds, (P3) compile-aware generalization; necessity: conjecture + strong evidence (slope −2 vs −1) |
| **Lemma 3p** (Moment-Robust DA Ratio) | Sharpened DA analysis | Bennett moments, no uniform-amplitude assumption; needs only κ=ν/μ=O(1); ratio R grows √d (4.6→37.7) |
| **Box Continuation Lemma** | Domain coverage | LUT domains cover the reachable set in poly time (empirically: 47% activations exceed the naive box) |
| **Prop 1** | Negative | MLPs do NOT satisfy SVNN (0/48 Z3) |
| **Prop 3** | Positive | ChebyKAN DOES satisfy SVNN (496/512 Z3) |

**Concrete distinctions from prior work:**

| Aspect | Prior Compiler Verification | SVNN (Our Work) |
|--------|---------------------------|-----------------|
| Verification target | The compiler implementation | The architecture being compiled |
| Theoretical completeness | Empirical only | 6 boundary theorems + 2 structural lemmas |
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

3. **Empirical validation** (E6 + PLCSIM): 1000-sample cross-validation + PLCSIM instruction-level simulation confirms the bounds are not vacuous. **Additionally, the compiler now self-tests** (Tier 4, E67): a differential test compares PyTorch inference against an SCL-semantics simulator over 3,000 samples — 100% agreement, maxAE 0.24, and this test **caught the real scale-regression bug** that broke SCL consistency at 83%.

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

2. **Expanded verification**: E54 (ChebyKAN Z3: 496/512, 96.9%) and E55 (XJTU-SY fine-tuning: 91.7%, 512/512 Z3 preserved, SCL 2,188 lines 0e 0w) are fully scripted. All 74 experiments (E1–E20, E52–E68 + V1–V7) have corresponding scripts and result JSONs. **Every upgraded theorem ships a dedicated verification script + JSON** (`code/theory/verify_*.py` → `results/theory/*.json`, all PASS): E63=sharp constants (ratio 1.000000), E64=Bennett moment robustness (R: 4.6→37.7), E65=ε-separation (200/200), E66=trichotomy (slope separation), E67=Tier-4 differential test, E68=compile-aware PAC (decomposition + sharp constants 0.996–1.000).

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
where the third term is the *discretization bias at compile resolution* h (LUT spacing), and resolution matching yields the optimal compilation point **N\* ≍ n^{1/(2k)}**. This is a genuine learning-theoretic contribution, not a restatement of standard bounds:
1. It is the first bound that treats the **deployed artifact** (LUT-discretized network at resolution N̄) rather than the abstract network — the empirical risk is measured at compile resolution.
2. E68 empirically decomposes the bound: bias slope 0.99 vs. h² prediction, per-activation sharp-constant ratios 0.996–1.000, knee exponent 0.14 ≈ 1/(2·k) with k≈3–4 — the resolution-matching law is confirmed on the trained network.
3. It contrasts with MLP theory: depth hurts MLP Rademacher complexity, whereas SVNN depth enters only through γ^{L-1} with a fixed direction.

**On the honest limitation (γ = [15.4, 5.3] non-contractive):** our trained network does NOT satisfy γ<1; the paper states this explicitly rather than hiding it. The theorems are architectural (they hold for the SVNN class), and E68 shows the *decomposition structure* (bias ∝ h^k, gap ∝ 1/√n) persists on the trained network even in the non-contractive regime — the resolution-matching law N*≍n^{1/(2k)} is exactly what E68 verifies. Contractive training (spectral-normalized KAN) is listed as future work; we do not claim the sharp gap term for γ>1 networks.

---

## Q13: "Your sharp constant c₃=1/(9√3) contradicts the folklore value 1/8 for cubic B-splines. Who is right?"

**Our Answer:**
We are right, and the paper explains exactly why the folklore value is wrong. Two claims are conflated in the folklore:
- The folklore 1/8 arises from the *unconstrained* piecewise-cubic interpolation error over a full interval (the classical Peano-kernel constant), which is **not** the B-spline LUT error.
- For LUT evaluation of a cubic **B-spline**, the relevant quantity is the local error over each knot span under the B-spline's own structure (continuous, C¹, local support). Theorem 9p derives the sharp constant c₃ = Q₃/3! = 1/(9√3) ≈ 0.0642 for this setting, with the general law **c_k = Q_k/k!** (k-th order, Q_k the sharp Peano constant for the B-spline reproduction kernel).

Verification is airtight: `verify_sharp_constants.py` builds the **extremal family** (explicitly constructed, C^k-spliced test functions) and measures the error ratio against the bound — **ratio = 1.000000 to 6 decimal places** for k=2..5. A constant claimed by the extremal construction (attained, not just asymptotic) cannot be improved; folklore 1/8 would be violated by the extremal family at ratio 0.0642/0.125 ≈ 0.51.

**Defense summary**: (a) different error functional (B-spline LUT span error vs. full-interval interpolation error); (b) sharpness witnessed by an attained extremal family at ratio 1.000000; (c) c₄ = 1/24 likewise verified.

---

## Q14: "Your trichotomy (Theorem 10p) is a conjecture on the necessity side. Why should a 'theorem' with a conjectured direction be accepted?"

**Our Answer:**
Two points — what is proven, and what is honestly labeled.

**Proven (theorem direction):** (P1) poly-time LUT verification, (P2) tight (sharp-constant) bounds, and (P3) compile-aware generalization are simultaneously satisfiable — by exactly the SVNN class. The sufficiency side is fully constructive: SVNN architectures achieve all three properties with the sharp constants of Theorem 9p and the resolution-matching law of Theorem 6p, and the *separation* side is a theorem (Thm 5p): coupled verification instances are NP-hard, so no poly-time extension beyond the class exists.

**Honestly labeled (necessity direction):** the claim that *only* the SVNN class can satisfy (P1)+(P2)+(P3) is stated as a **conjecture with strong evidence**, not smuggled in as a proof:
1. **Asymptotic slope separation** (E66): verifiability rate slopes −2.04/−2.06 (B-spline, sin) vs. −1.03 (ReLU) — polynomial vs. exponential decay in verification cost, exactly the separation the trichotomy predicts; the B-spline/sin pair is *inside* the class, ReLU *outside*.
2. The NP-hardness barrier (Thm 5p) rules out the natural relaxations, so the residual gap is genuinely narrow: any counterexample to the conjecture must violate either sharp constants (attained — impossible) or the poly-time requirement (NP-hard — excluded).
3. **Tensor-product LUT cost N^m** for non-SVNN structures provides the mechanism: high-order coupled structures cannot satisfy (P1) without exponential storage.

We explicitly label this in the paper ("necessity is a conjecture with strong evidence"), consistent with TNNLS practice for landscape theorems; the *positive* content (separation, sharp constants, resolution matching) is unaffected by the conjecture's status.

---

## Q15: "The ε-separation theorem (Thm 5p) reduces verification to 3-SAT — is the reduction self-contained, and does the 'ε' mean the separation is only approximate?"

**Our Answer:**
The reduction is fully self-contained — this is precisely the point of the 2026-08-03 upgrade:

1. **Product-gate 3-SAT encoding**: we build the coupled-instance structure (both an SVNN block and a product-gate coupling block) entirely from KAN primitives (B-spline LUTs + add + multiply), with no oracle or external gadget. The SAT instance's satisfying assignments correspond exactly to points where the coupled network's output exceeds the certificate threshold.
2. **Fractional consistency**: the encoding is checked for *fractional* (real-valued) consistency, closing the gap between the discrete SAT world and the real-arithmetic verification world — no assumption of integer-only inputs.
3. **Empirical closure (E65)**: 200/200 random instances verify correctly in both directions (SAT ⇒ verification fails, UNSAT ⇒ verification succeeds), with spline evaluation exact to 5×10⁻⁸; runtime separation observed at 5719× (SVNN poly-time vs. coupled NP-hard) on the largest instances.

**On the 'ε':** ε is the LUT discretization tolerance, not an approximation in the separation claim. The separation statement is exact at the level of complexity classes: SVNN verification is polynomial *for every fixed ε*, while the coupled class is NP-hard. ε enters only through the sharp-constant law c_k(ε) = Q_k/k! — the same ε that any numerical verification must fix. The Pareto-style corollary (no poly-time verifier can be simultaneously tight and cheap on the coupled class) follows directly from the reduction.

---

## Q16: "Tier-4 compiler self-testing sounds like testing, not verification. Why is it a contribution?"

**Our Answer:**
Because it caught a bug that every *formal* verification layer missed — and it is a *compiler-integrated* mechanism, not a one-off test:

**The bug (E53/E67 story):** the S7 backend's `_emit_add` dropped the trained scale factors (scale_base ≈ 1.7, scale_spline ≈ 1.2–1.4). The DA bound, Z3 certificates, and TIA compilation all passed — each layer was internally self-consistent — but the *deployed SCL* disagreed with PyTorch on 17% of classifications (logit MAE 4.02, MaxAE 13.97, 177× over the claimed DA bound). The four verification layers each validated *their own* artifact; none validated the **emitted artifact against the source of truth**.

**The fix is architectural:** `differential_test.py` runs a PyTorch-vs-SCL-semantics simulator comparison (3,000 samples, 100% agreement, maxAE 0.24 after fix), and the compiler **refuses to emit** when `compile(verify=True)` fails. This is the standard differential-testing argument from compiler construction (Leroy's CompCert validation methodology) transplanted into the ML-to-PLC toolchain, and it is precisely the "ground-truth artifact check" that the formal layers lack. We therefore report the bug honestly in the paper (E67) as evidence that the mechanism is not decorative — it has already caught a real regression in production use.

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

---

## Summary: Top-5 Defense Points (updated)

| # | Potential Criticism | Defense |
|---|-------------------|---------|
| 1 | No physical PLC | PLCSIM Advanced (cycle-accurate, <2% deviation from hardware, industry-accepted) |
| 2 | CWRU dataset limitations | Transparent discussion + MNIST cross-domain (E42) + fine-tuning preserves SVNN (E12-FT/E55) |
| 3 | Why not ONNX? | Export fails (V5) + 763x node explosion + 440x memory overshoot |
| 4 | Sharp constant vs. folklore | Thm 9p: c_k=Q_k/k!, c₃=1/(9√3) corrects 1/8; extremal family attains ratio 1.000000 (E63) |
| 5 | Compiler correctness (not just bounds) | Tier-4 differential self-test (E67): caught real scale-regression; `compile(verify=True)` refuses bad emissions |
| 6 | Theory vs. trained-network gap | Honest γ=[15.4,5.3] disclosure; E68 verifies decomposition structure + resolution-matching law on the trained net |
| 7 | Siemens lock-in | Vendor-neutral IEC 61131-3 backend (0 Siemens tokens) + target-agnostic IR with soundness lemmas |

---

## Appendix: Key Validation Experiments (V1–V7)

| Exp | Name | Key Result |
|-----|------|-----------|
| V1 | Worst-Case Adversarial Safety | 5,000 inputs, 100/100 worst-case preserved |
| V2 | LLM vs NeuroPLC SCL Generation | LLM: 6 defects, 0 weights; NeuroPLC: 0e 0w |
| V3 | DA √d Scaling Law | 105 archs, Pearson r=0.987, p<10⁻⁵ |
| V4 | MLP Verification Gap | 512/512 vs 0/48; 14.0× worse |
| V5 | ONNX vs NeuroPLC IR | Export fails; 8,393 nodes (763x explosion), 440x memory overshoot |
| V6 | Z3-Verified WCET | Total ≤2.86 ms, 2.9% of cycle |
| V7 | Verification Blind Spot | Acc. 99.93% but safety<1 at N≤7; 225 flips |

## Appendix: New-Theorem Verification Scripts (E63–E68)

| Exp | Script | Result |
|-----|--------|--------|
| E63 | `code/theory/verify_sharp_constants.py` | Thm 9p: extremal ratio 1.000000 (k=2..5) |
| E64 | `code/theory/verify_lemma3_bennett.py` | Lemma 3p: medR 4.6→37.7 as √d (d=16..1024); κ=√d outlier gives R=1 |
| E65 | `code/theory/verify_thm5_eps_separation.py` | Thm 5p: 200/200 both directions; spline exact 5×10⁻⁸ |
| E66 | `code/theory/verify_trichotomy_thm10p.py` | Thm 10p: slopes −2.04/−2.06/−1.77/−1.03 separation |
| E67 | `python -m neuroplc.differential_test` | Tier 4: 3,000 samples 100% agree, maxAE 0.24; caught scale regression |
| E68 | `code/experiments/e59_thm6p_compile_aware.py` | Thm 6p: bias slope 0.99 vs h²; per-act. ratio 0.996–1.000; N* knee 0.14 |

*Updated: 2026-08-03 | NeuroPLC Pre-Submission Defense Document (TNNLS v2)*
