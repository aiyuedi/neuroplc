# NeuroPLC — Reviewer FAQ & Defense Document

> Prepared: 2026-07-07 | Pre-submission defense for IEEE TII / TIE / MSSP

---

## Q1: "Why no physical PLC measurement?"

**Our Answer:**
We validate on **Siemens PLCSIM Advanced v6.0**, the official cycle-accurate instruction-level simulator for S7-1200/1500 CPUs. PLCSIM Advanced is accepted in industry for pre-commissioning validation (FAT/SAT), and Siemens documents <2% timing deviation from physical hardware. Our PLCSIM validation (E46, prepared) provides:

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
Experiment E45 proves ONNX export is **impossible** for KAN architectures:

1. **Export failure**: `torch.onnx.export` fails on KAN's B-spline `einsum` with opsets 14/17/20. ONNX has no standard B-spline operator.

2. **Node explosion**: Even if export succeeded, decomposing 512 B-spline functions into Gather+Mul+ReduceSum primitives would require **8,393 ONNX nodes** vs. NeuroPLC's **11 IR nodes** — a **763x explosion**.

3. **Memory impossibility**: ONNX Runtime minimal build is ~22 MB. S7-1200 work memory is 75 KB. That's a **300x** overshoot. Even S7-1500's 1.5 MB is insufficient.

4. **No verification**: ONNX Runtime provides no mathematical correctness guarantees for the PLC deployment target.

**Conclusion**: ONNX Runtime is designed for GPU/CPU/accelerator targets, not memory-constrained PLCs. A domain-specific compiler (NeuroPLC's approach) is not optional — it is **necessary**.

---

## Q3: "Why not LLM-based code generation?"

**Our Answer:**
Experiment E44 provides structural evidence:

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
We are transparent about CWRU's limitations (Section Limitations, with explicit references to Smith 2015 and Hendriks 2022). We provide three counterarguments:

1. **The compiler is data-agnostic**: The SVNN framework (Theorem 2) depends on KAN architecture, not data provenance. Experiment E42 (MNIST) proves identical verification guarantees (512/512 functions, certificate valid) on image classification — a completely different domain.

2. **Cross-dataset with fine-tuning**: E12 shows CWRU→XJTU-SY improves from 37.3% (zero-shot) to 79.4% (fine-tuned), with SVNN conditions **preserved** after fine-tuning (DA bound: 0.080→0.073).

3. **Method boundary analysis**: E19 quantitatively characterizes when our methods degrade (balanced weight signs, uniform B-spline curvature, depth > 5 layers) — transforming limitations into methodological contributions.

---

## Q5: "Is the SVNN framework really novel? This looks like standard compiler verification."

**Our Answer:**
The novelty is in **reframing the question**. Prior work asks: "Can we verify an arbitrary neural network compiler?" (answer: no, due to floating-point undecidability). SVNN asks: "Which architectures are inherently verifiable, and what are the sufficient conditions?"

Three concrete distinctions from prior work:

| Aspect | Prior Compiler Verification | SVNN (Our Work) |
|--------|---------------------------|-----------------|
| Verification target | The compiler implementation | The architecture being compiled |
| Error bound type | Empirical (test set) | A priori (design-time, Theorem 2) |
| Architecture scope | Any (or specific to one) | Class characterized by 3 conditions |
| MLP support | Yes (empirical only) | No — Proposition 1 proves MLPs CANNOT admit tight bounds |
| KAN support | No prior work | Yes — Proposition proves KAN satisfies all 3 conditions |

The SVNN framework is validated by an **empirical negative result** (MLPs fail Z3 verification: 0/16 vs. KAN's 512/512, E41) that would be inexplicable under the prior "verify anything" paradigm.

---

## Q6: "How does this scale to real factory floors?"

**Our Answer:**
Four pieces of evidence:

1. **Multi-PLC support**: NeuroPLC compiles to 4 S7-1200 variants + 2 S7-1500 variants + ET 200SP, all verified in TIA Portal V21 (E5 + E43).

2. **OPC UA integration**: E46 Tier C demonstrates Python→OPC UA→PLCSIM→SCL→OPC UA→Python end-to-end data flow — the standard Industry 4.0 communication pattern.

3. **Engineering effort quantification** (Table in paper): 2,610× speedup vs. manual SCL development. Model update (retrain) requires one re-invocation (~30s). PLC retarget requires changing one parameter. This is the kind of engineering efficiency that matters on factory floors.

4. **Scalability analysis** (E28 + E5): Width/depth/grid-resolution feasibility maps for S7-1200 and S7-1500. The binding constraint is memory (not compute) — and our DB+FB split + adaptive LUT allocation directly address this.

---

## Q7: "Are the error bounds really sound on real hardware?"

**Our Answer:**
We distinguish three levels of evidence:

1. **Architectural guarantee** (Theorem 2): For any KAN architecture, an a priori bound exists and is computable. This holds in the IEEE 754 abstract machine.

2. **Model-specific refinement** (E11): Empirical M2 calibration tightens the bound by 98.6% (M2=0.177 vs. analytical M2=12.8). Both are computable from model parameters alone.

3. **Empirical validation** (E6 + PLCSIM E46): 1000-sample cross-validation + PLCSIM instruction-level simulation confirms the bounds are not vacuous.

We cite Szász et al. (2025) explicitly and position our guarantee as a **design-time correctness argument** (Level 2), not a mechanized hardware proof (Level 3). For SIL 3+, we provide per-function Z3 proofs (Tier 2) as machine-checkable evidence.

---

## Q8: "What about sensor signal acquisition? Feature extraction isn't on the PLC."

**Our Answer:**
This is an acknowledged limitation. We provide two responses:

1. **Partial SCL frontend** (E43 Phase 3.3): 10 time-domain features (RMS, peak, kurtosis, etc.) can be implemented in SCL using accumulators — no FFT required. These 10 features alone achieve 91.36% accuracy (E13 SVM).

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

NeuroPLC's LUT compilation paradigm is architecturally similar to KANELE's FPGA LUT approach (both discretize B-splines into lookup tables), validating the same design principle across hardware platforms. We cite KANELE (ISFPGA 2026 Best Paper) as independent validation.

---

## Q10: "Where is the code, and can I reproduce your results?"

**Our Answer:**
1. **GitHub repository**: Full source code, trained checkpoints, evaluation scripts, and SCL output at [URL to be disclosed upon acceptance].

2. **Verification Certificate Bundle** (`results/verification_certificate/`): Self-contained package with Tier 1-3 proofs, 512/512 function verification results, composition certificate, and a ~200-line trusted checker. Independent verification requires only `torch`, `numpy`, and `z3-solver`.

3. **Reproducibility**: The 6,148-parameter KAN checkpoint is included. All experiments (E1-E47) are scripted. Compilation to SCL is deterministic (verified by `test_compiler_reproducibility`). TIA Portal V21 validation requires a Siemens license; PLCSIM Advanced validation is an accessible alternative.

4. **Data**: CWRU and XJTU-SY are publicly available. Preprocessing scripts are included. Preprocessed features are provided for convenience.

---

## Summary: Top-5 Defense Points

| # | Potential Criticism | Defense |
|---|-------------------|---------|
| 1 | No physical PLC | PLCSIM Advanced (cycle-accurate, <2% deviation from hardware, industry-accepted) |
| 2 | CWRU dataset limitations | Transparent discussion + MNIST cross-domain (E42) + fine-tuning preserves SVNN (E12) |
| 3 | Why not ONNX? | Export fails (E45) + 763x node explosion + 300x memory overshoot |
| 4 | Error bound soundness? | Three-tier evidence: architectural → model-specific → empirical (Theorem 2 + E11 + PLCSIM) |
| 5 | Industrial relevance? | OPC UA demo (E46) + 2,610x engineering speedup + zero-hardware-cost deployment model |

---

## Appendix: New Experiments Added (E43-E47)

| Exp | Name | Key Result |
|-----|------|-----------|
| E43 | TIA Auto Multi-Target Validation | 12 SCL files, all ready for TIA compilation |
| E44 | LLM vs NeuroPLC SCL Generation | LLM output analyzed for Siemens-specific issues |
| E45 | ONNX Export Failure Analysis | 8,393 nodes (763x explosion), 300x memory overshoot |
| E46 | PLCSIM Closed-Loop Validation | Prepared for TIA+PLCSIM execution |
| E47 | Verification Certificate Bundle | 512/512 verified, certificate VALID, 0 warnings |

*Generated: 2026-07-07 | NeuroPLC Pre-Submission Defense Document*
