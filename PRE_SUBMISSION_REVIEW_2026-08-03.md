# Pre-Submission Referee Report

**Paper**: A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation
**Authors**: [TBD]
**Date**: 2026-08-03
**Review Standard**: IEEE TNNLS (theory-first; 7-agent parallel review: contribution, claims, consistency, mathematics, tables/figures, language, project health)

---

## Overall Assessment

The paper claims a "type theory of certifiable neural architectures": the SVNN conditions characterize which architectures admit design-time certified LUT compilation, anchored by four results — sharp k-th order LUT constants c_k=Q_k/k! (c₃=1/(9√3), "correcting folklore 1/8"), an ε-separation complexity theorem (poly vs NP-hard via product-gate 3-SAT), a compile-aware PAC bound with resolution-matching law N*≍n^{1/(2k)}, and a verifiability trichotomy with SVNN as the unique optimum — instantiated in a PyTorch→IEC 61131-3 compiler with a bearing-fault case study (74 experiments).

**Principal strength**: unusual intellectual honesty — self-correcting over-claims (ε-separation framing, Rademacher arithmetic, trichotomy dichotomy), the E52 verification blind-spot experiment, reconciliation tables, and honest E68 reporting. All verified mathematics is correct: sharp constants (Q₂=0.25, Q₃=2/(3√3), Q₄=1, Q₅=3.6314 recomputed independently and confirmed), Bennett Lemma 3, the 3-SAT reduction, and Box Continuation all check out numerically and logically.

**Single most critical issue**: the paper's headline guarantee chain is broken at three independent points that a referee will hit in the first hour — (1) Theorem B (DA Galois Connection) is mathematically unsound as stated (fails soundness, optimality, and adjunction; the Coq appendix lemmas are unprovable as written), (2) Lemma 6's own measurements (ℓ∞ logit error 14.17, in-domain ≈17 at E53) exceed the worst-case bounds the paper states for the same configuration (0.71, 6.4) by 20–24×, with the root cause (47% of layer-1 activations outside the [−3,3] LUT grid, violating the Box-Continuation assumption that Thm 1 and Thm 6 explicitly require) disclosed only in E68 and nowhere in the abstract/intro/conclusion that advertise the guarantees, and (3) the same trained checkpoint is claimed both contractive (γ=0.182, Condition 3 satisfied, depth-uniform guarantee) and non-contractive (γ=[15.4,5.3], E68) without reconciliation.

**Preliminary Recommendation**: **Revise before sending to referees** (not a desk reject — the paper is internally coherent, the claimed mathematics is correct, and the honesty is a genuine asset; but sending the current version would generate predictable rejection letters).

---

## 1. Contribution & Referee Assessment

### Part 1 — Central Contribution

**Claim**: a "type theory" (SVNN conditions: operation-type closure + C² computable-curvature gates) characterizing which architectures admit design-time certified LUT compilation, anchored by sharp LUT constants (c₃=1/(9√3) "correcting folklore 1/8"), ε-separation, compile-aware PAC with resolution matching, and a verifiability trichotomy — instantiated in a PyTorch→IEC 61131-3 compiler with an industrial bearing-fault case study.

**Genuinely new or replication?** Each headline result is *correct but substantially pre-existing*:
- **Thm 9**: the sharp-constant law follows in one line from the classical interpolation remainder theorem; the extremal family is the standard tightness argument. All values verified numerically (Q₂=1/4→c₂=1/8; Q₃=2/(3√3)→c₃=1/(9√3)≈0.0642; Q₄=1→c₄=1/24; Q₅≈3.63→c₅≈0.0303). But c₃=√3/27·M₃h³ is the *textbook* constant for quadratic interpolation (in the paper's own cited Atkinson) — the "correction of folklore 1/8" is largely a straw man with no cited source asserting 1/8. Thm 9(3) (minimax optimality for general k) is asserted with a proof sketch that does not establish it (optimal-recovery literature, Micchelli–Rivlin / Golomb–Weinberger, uncited).
- **Thm 5**: the product-gate 3-SAT reduction is valid and self-contained (verified 200/200 both directions, E65). But the paper's own Remark concedes the single-fused-layer (ReLU) case is polynomial — the hardness lives in the *contrived product-gate class*, not the MLP class it contrasts KAN against.
- **Thm 6**: standard Lipschitz–Rademacher PAC + additive LUT-bias term is sound; N*≍n^{1/(2k)} is precisely the classical optimal-bandwidth law of nonparametric estimation (h*≍n^{−1/(2k)}; Tsybakov; Györfi–Kohler–Krzyżak–Walk), uncited. "No existing PAC bound states this" is an over-claim.
- **Thm 10**: useful synthesis, but the "unique optimum" reading rests on a necessity direction that is a conjecture — whose stated evidence is partly wrong ((P2) does not force C² gates; bounded-second-variation C¹ gates achieve O(h²)).
- **Type-theory framing**: lightweight classification; the CompCert analogy is invoked without citing Leroy (absent from the bibliography); the Coq appendix states mechanization is "ongoing" — it is a specification, not a proof.

**Rating: [Incremental]**. The genuine strengths — honest synthesis, E52 blind-spot experiment, self-correcting over-claim history — are systems-level, not theory-level.

### Part 2 — Identification and Credibility

Theorem–proof chain credible *at the level claimed*, aided by unusual honesty (self-downgrades of ε-separation framing, factor-4 Rademacher correction, trichotomy dichotomy correction, DA uniform-amplitude counterexample). Sharp constants are proven (remainder theorem + extremal) with E63 corroboration; the attribution is the problem, not the numbers. Trichotomy is an honest conjecture with evidence. **Weakest evidentiary links (ranked):**
1. **Deployed certificate does not cover the deployed model** — 47% of layer-1 activations outside [−3,3] (range [−23.9,+25.1]); E53 measures N-independent error ≈17 at depth≥2; Box-Continuation — which Thm 6 explicitly requires — is violated at depth≥2 for the certified checkpoint.
2. **γ ambiguity** — 0.182 (contractivity, Condition 3 "satisfied") vs [15.4,5.3] (measured amplification, E68) for the same checkpoint; three γ values (0.182, 0.459/0.502, [15.4,5.3]) never reconciled.
3. **"Verified compiler" vs the emitter bug** — Tier 1–3 formal chain did not catch a real emitter regression (83% deployed agreement); only the empirical Tier-4 differential test caught it. "First verified compiler" overstates what is formally verified.
4. **Trichotomy necessity** — honestly flagged, but one evidential pillar (C² forcing) is incorrect.
5. **Universal IEC guarantee** — format-level argument (IEC 60559 binary32 data format) does not imply per-operation IEEE-754 semantics (FMA, subnormals, rounding modes are implementation-defined) — inherits exactly the Szász et al. critique the paper cites.
6. **Thm 9(3)** — asserted without proof and without optimal-recovery literature.

### Part 3 — Analyses: Required and Suggested

**Required:**
1. [CRITICAL] Independent verification of the sharp-constant claims and the folklore attribution — produce a published source for "folklore 1/8" or restate the contribution as "explicit window-product constants with a certified extremal family for LUT compilation" and delete the folklore-correction framing.
2. [CRITICAL] Prove or formally downgrade the trichotomy necessity — the stated evidence ((P2) forces C²) is false; either prove under a formalized "design-time-computable" notion or remove "unique satisfier" from the abstract.
3. [CRITICAL] Reconcile γ and give a contractive-trained demonstration — spectral-normalized or weight-clipped KAN with γ<1, accuracy ≥99%, and the Thm 6 bound evaluated honestly.
4. [CRITICAL] Close the deployed-certificate gap — re-compile with per-layer widened LUTs (47% out-of-domain) and re-run 512/512 verification + E53/E68; or state the certified operating envelope exactly.
5. [CRITICAL] Define the formal boundary of the "verified compiler" — which components carry formal proofs (templates, composition rules, LUT tables) and which do not (the emitter itself); add emitter-level verification or re-scope the claim.

**Suggested:**
1. [MAJOR] Higher-order and non-uniform sharp constants (k>5; adaptive grids; minimax over node placement).
2. [MAJOR] A genuinely contractive trained network, end-to-end (one training run).
3. [MAJOR] Physical PLC measurement — at least one S7-1200 run; the four WCET figures (2.86/3.15/13.4/22.68 ms) need reconciliation with measured values.
4. [MAJOR] Ablation of the type-theory framing — a compiler pass that type-checks arbitrary models and rejects ill-typed ones, or a second architecture newly discovered SVNN by type-checking.
5. [MAJOR] Validate the resolution-matching law on a non-saturated task (e.g., the E51 10-D front-end) and cite the nonparametric-estimation literature.

### Part 4 — Literature Positioning

Right papers cited (KAN, KANELE, Schwartz, CROWN line, Szász, de Boor, Bartlett–Mendelson). **Material gaps:** CompCert/Leroy (the paper leans on the analogy repeatedly, never cites it); quantized/fixed-point NN verification (Giacobbe–Henzinger–Lechner; QVIP/ILP line); Tsybakov and Györfi et al. (resolution law = classical two-scale); Micchelli–Rivlin / Golomb–Weinberger (optimal recovery, underpinning Thm 9(3)). Bibliography under-disciplined: multiple placeholder/empty entries, ~24 entries never cited (fastkan2025, quantkan2025, sharekan2025, memnilli2025plc, chen2021byoc, blott2018finn, ...).

### Part 5 — Journal Fit and Recommendation

TNNLS bar is theory depth: sharp interpolation constants are textbook, the separation concerns a contrived gate class, the generalization bound is classical two-scale restated, the trichotomy's uniqueness is a conjecture with one incorrect pillar. **Recommendation: Revise before sending to referees.** Concretely needed: (1) prove or properly downgrade trichotomy necessity; (2) fix the folklore narrative and prove/cite Thm 9(3); (3) contractive-trained demonstration with honest γ and valid Box-Continuation; (4) precise "what is formally verified" statement; (5) at least one physical PLC datapoint; (6) discipline the manuscript (single theorem stack, clean bibliography with CompCert and nonparametric citations).

**Best alternatives:** IEEE TII (certified NN-on-PLC + E52 blind-spot + IEC 61508 alignment — strong fit, tolerates breadth); IEEE TC (compiler-correctness angle); NeurIPS/ICML route for a rebuilt theory piece; IEEE TCAD for the LUT-allocation machinery. If trichotomy necessity is genuinely proved, return to TNNLS or JAIR.

### Part 6 — Questions to the Authors

1. Name the published source that asserts "folklore" 1/8 for cubic B-spline LUT error; c₃=1/(9√3) is the classical quadratic-interpolation constant (Atkinson). Second, Thm 9(3) asserts minimax optimality among ALL sound affine schemes for general k; the sketch only shows the LUT cannot be improved on the extremal σ* — give the full argument or cite Micchelli–Rivlin, or restrict (3) to the LUT scheme.
2. Your necessity evidence says "(P2) forces C² gates" — but bounded-second-variation C¹ classes achieve O(h²); only kinked gates (ReLU) are rate-limited. What exactly is the excluded class, and can the "unique optimum" reading survive this counterexample?
3. Which γ? 0.182 (Condition 3 certified) vs [15.4,5.3] (E68 measured) for the same checkpoint; Thm 6's bias uses γ^{L−1}=15.4 while the abstract trades on contractivity. Define both precisely and re-evaluate Thm 6 honestly. Would a spectral-normalized contractive run change any claim?
4. The certified region: 47% of layer-1 activations outside [−3,3]; Box-Continuation (required by Thm 6 and the Tier-2 certificates) violated at depth≥2. For exactly which layers do the stated certificates hold? Have you re-compiled with widened LUTs?
5. The Tier-4 differential test caught an emitter regression (83% agreement) that Tier 1–3 formal proofs did not. Does the formal chain verify templates while the emitter is unverified? What correctness argument beyond the empirical differential test covers the current emitter? (Note: as of 2026-08-03, `python -m neuroplc.differential_test` crashes with a numpy broadcast error — the documented entry point does not run.)
6. The measured knee exponent is 0.14±0.09 vs predicted 0.25, the 0-1 knee is statistically indistinguishable from zero on a 0.04% test-error task, and N*≍n^{1/(2k)} is the classical bandwidth law (Tsybakov). What is new beyond the classical two-scale tradeoff?
7. (a) Confirm the normative status of IEC 61131-3:2025 §2.3.2 and vendor compliance with the IEEE-754 binary32 arithmetic you rely on (FMA, subnormal flushing, rounding modes are implementation-defined); (b) what does "universal industrial certification" mean operationally given all validation is PLCSIM/Z3-based?

---

## 2. Unsupported Claims & Identification Integrity

### Claim Overreach (must address)

1. [CRITICAL] **Theorem 1 universal bound vs. the paper's own measurements** (main.tex:1876–1936, 2167–2209, 3905–3963, 4000–4004) — Thm 1 states "sound worst-case bound valid for all x∈X" Δ≤0.172 (IA)/0.079 (DA); the method section asserts "Classification correctness is … mathematically guaranteed" (1238–1239). Own measurements: Lemma 6/E21 ℓ∞ logit error up to **14.17** (2206); E53 in-domain worst-case ≈**17** at every LUT density; E52 37/20,000 adversarial flips at N=15. "Δ≤0.71 using M₂^max" (2196) does not bound 14.17. Fix: state the Box-Continuation assumption inside Thm 1, mark the trained [28,16,4] as violating it at depth≥2, qualify every "guaranteed correctness" claim as conditional on margin>2Δ AND Box-Continuation; reconcile 0.71 vs 14.17.
2. [CRITICAL] **Trichotomy necessity labeled conjecture in one place only** (section_trichotomy.tex:84–97 vs abstract 147–151, intro 255–262, conclusion 5219–5224, section title, Consequences (iii)) — "conjecture" appears exactly once in the entire English paper; everywhere else "unique regime" is asserted as fact. Fix: qualify with "(conjecturally)" in abstract/intro/conclusion/title; conclusion must flag the open necessity problem.
3. [CRITICAL] **Theorem B (DA Galois Connection) mathematically unsound** (section_galois.tex:34–51, 77–93; appendix_coq_spec.tex:96–103) — α(f)=(f(x₀), M₂(f)r²) with γ(c,R)={f : |f(x)−c|≤R}: (i) soundness fails for f(x)=x (M₂=0, |f(±r)|=r>0); (ii) α∘γ=id fails (constant f≡c∈γ(c,R) gives α=(c,0)≠(c,R)); (iii) adjunction ⟸ fails (x²∈γ(0,1) but α=(0,2)≰(0,1)). Correct form needs first-order term: γ(c,R)={f : |f(x₀)−c|+M₂(f)r²≤R} or α=(f(x₀), |f′(x₀)|r+M₂r²/2). Coq lemmas galois_soundness/galois_optimality are unprovable as written (optimality holds only at r=0). Fix and re-verify; update the Coq spec.
4. [CRITICAL] **Contractivity asserted both ways** (section_svnn.tex:529–531, 649–672 "L_B·‖W‖=0.182≪1", "satisfies the contractivity refinement"; taxonomy footnote "Contractivity verified" vs section_svnn_theorems.tex:319–322 "not contractive: γ=[15.4,5.3]") — same checkpoint, three γ values (0.182, 0.459/0.502, [15.4,5.3]), no reconciliation. Fix: define γ unambiguously (per-layer Lipschitz product vs L_B‖W‖₁,∞ vs error-amplification), state which E68 measured, reconcile or report Condition 3 fails for the deployed checkpoint.
5. [CRITICAL] **"SVNN uniquely attains (polynomial time, zero slack)" presented as unconditional** (abstract 155–156, intro 266–269, conclusion 5226–5229) — Thm 5's Pareto corollary is conditional on the cited gap-preserving reductions and ETH. Fix: append "(under ETH)" / "(conditional on the cited gap reductions)" to the three summary statements.
6. [MAJOR] **Theorem A necessity exceeds its proof** (section_characterization.tex:33–48) — "‖W‖₁,∞=Θ(√d) under standard initialization" and "B_total=Θ(d^{L/2}r²)" conflate "there exists" (one adversarial MLP W=J_d/√d) with "for all"; the 22× comparison contrasts a worst-case adversarial MLP against the *measured* γ=0.182 of a trained KAN (apples vs oranges). Fix: restate as "there exist ill-typed architectures amplifying Θ(√d)…".
7. [MAJOR] **Corollary Fixed-Depth Tradeoff logically self-contradictory** (section_svnn_theorems.tex:267–280) — for γ<1 both gap factor γ√((L+1)/L) and bias factor γ(L+1)/L are <1: gap/bias *shrink* with depth; the derived condition ΔR̂≥(γ−1)·C·… is vacuous ((γ−1)<0). Non-monotonicity is the γ>1 case. Fix: state "when γ>1 non-monotone; when γ<1 both decay and depth helps unless training-risk degrades," re-derive the threshold.
8. [MAJOR] **"Contrast with standard MLP" uses the discredited 0.0019** (section_svnn_theorems.tex:339–350 vs 287–299) — the same section corrected 0.0019→0.0038→gap≈1.5; after correction the KAN's gap term (≈1.5) is LARGER than the MLP's (0.217), inverting the paragraph's conclusion. Recompute or delete.
9. [MAJOR] **E56 presented as confirming depth-uniformity while data contradicts it** (main.tex:4484–4513) — 0.98/0.064=15.3× jump for one added layer (text says 14.6×, "consistent with linear growth"); under Condition 3 (γ<1) Thm 2 predicts convergence, not a 15× increase; "≈7.3× per added layer" is an arithmetic error (one layer added). Fix or remove "confirms" language.
10. [MAJOR] **IR Minimality universal claims from one ablation** (main.tex:787–817, 864–880) — "no third alternative exists" is inferred from a single ablation on one architecture. Downgrade to "empirically minimal for B-spline KAN on S7-1200 (47.4× measured)".
11. [MAJOR] **Downgrade-Guarantee invariant exceeds what the monitor enforces; no experiment validates Algorithm 4** (main.tex:3037–3086) — case (a) lets decisions run on uncertified T_new under heuristic domain/confidence/range checks; a wrong-but-in-range output is not detected. Abstract claims "maintains the certified safety state" as fact; "first runtime-update protocol" with no E-number exercising it. Fix: restrict invariant ("certified bound OR monitor detection OR one-cycle rollback"), add differential-test assumption, add an experiment or mark as untested.
12. [MAJOR] **WCET instruction timings internally contradictory** (main.tex:2847–2856 vs 3345–3347 vs 2711–2714) — REAL MUL 3.5μs (Thm 11) vs 0.60μs (Z3 table): 6–7× discrepancy behind the three "levels" (2.86/13.4/22.67 ms). Use one authoritative table; state none is hardware-measured.
13. [MAJOR] **Remark (Near-Necessity)(b) false mathematical fact** (section_svnn.tex:401–410) — "sup|SiLU″|=∞ on ℝ" is false: SiLU″ is bounded, sup≈0.21 (the remark's own next sentence concedes max≈0.21 on [−3,3]); contradicts Prop (mlp_negative)(a) ("SiLU″ bounded, Condition 2 formally satisfiable") and the taxonomy table. Delete the claim; the exclusion mechanism is Z3-undecidability only.
14. [MAJOR] **Remark (Near-Necessity)(a) restates the disavowed claim** (section_svnn.tex:386–396 vs section_svnn_theorems.tex:9–20) — "separates architectures admitting O(L·d²) bounds from exponential-time SMT" was explicitly superseded (interval propagation and CROWN are poly-time for MLPs). Rewrite to state the valid ε-dimension separation.
15. [MAJOR] **Theorem 4 probabilistic depth-scaling: unproven key step** (main.tex:2297–2319) — 𝔼[Δ]=ε_LUT·Σd_{ℓ−1}γ^{L−ℓ} equals the worst-case unrolled IA bound under random signs (cancellation should reduce expectation); asserted without derivation; the martingale bound ignores cumulative amplification; "typically 0.1–0.3 for trained KANs" conflicts with E68's [15.4,5.3]. Prove or relabel "worst-case envelope".
16. [MAJOR] **NC.3 overflow claim contradicts E68 measured ranges** (section_noninterference.tex:123–130) — "clamping [−3,3] ensures z_i∈[−20,20]" is false: layer-1 activations reach [−23.9,+25.1] (E68), so softmax inputs can exceed 20 (exp(25.1)≈8×10¹⁰ > 4.85×10⁸). Conclusion survives; the stated bound is wrong. Fix with propagated-error analysis or add Box-Continuation to NC.3.
17. [MAJOR] **Non-interference temporal argument assumes P's load is zero** (section_noninterference.tex:169–179, 227–229) — "22.67ms < 100ms leaves 77.33ms for P" assumes WCET(P)≤77.33ms; if P used 90ms the composition overruns and the watchdog resets. Add the premise to the theorem; carry provisos into abstract/intro/conclusion.

### Unverified Priority Assertions (all [MAJOR] unless noted — verify against literature before submission)

1. main.tex:306 "the first IR-based compiler from PyTorch to Siemens S7-1200/1500 SCL" (also abstract 163–164, conclusion 5203–5204).
2. section_galois.tex:123–136 "first neural network compiler formally grounded in abstract interpretation theory" + "first Galois connection defined for NN activation functions" (AI²/DeepPoly/CROWN/αβ-CROWN are formally grounded in Cousot's framework — likely contestable).
3. main.tex:285–289, section_noninterference.tex:235–236 "first such guarantee for NN-on-PLC deployment" / "first formal proof that an NN inference module can be added without compromising pre-existing safety properties".
4. main.tex:573 "No prior work compiles KANs to IEC 61131-3 SCL" — absolute negative claim.
5. main.tex:278, section_svnn_theorems.tex:261 "a directly actionable compiler rule no existing PAC bound states".
6. main.tex:2812–2822, 2911–2920 "first parameterized WCET analysis for SVNN-compiled networks" / "first formal guarantee of timeliness".
7. main.tex:3086–3087 "first runtime-update protocol for certified NN-on-PLC deployment".
8. section_characterization.tex:157–159 "first *exact* characterization of the certifiable architecture class".
9. section_tightness.tex:99–106 "DA is the unique optimal abstract domain" (many domains achieve the same bound; only the bound is shown tightest) — soften to "attains the tightest bound among the considered affine domains".
10. main.tex:4698–4704 "first NN compiler for industrial controllers whose correctness is formally verified at the compiler-template level".
11. main.tex:586–588 "first VRM-KD to a KAN student".
12. [MINOR] main.tex:3505–3508 "first application of translation validation to a NN-to-PLC compiler" (hedged — acceptable if qualification survives).
13. [MINOR] main.tex:1617–1618, 4242–4244 "strongest formal guarantee … to date" (E16's "that we are aware of" is acceptable).
14. [MINOR] main.tex:4643–4644 "first formal, machine-checkable proof that every B-spline activation satisfies its LUT error bound".
15. [MINOR] main.tex:846–850 "first instance of a broader compiler design paradigm: structurally verifiable compilation".

### Generalization Issues

1. [MAJOR] **Universal IEC guarantee overclaims** (section_iec_universal.tex:61–137; main.tex:290–292) — IEC 61131-3:2025 §2.3.2 defines the REAL *data format* (IEC 60559 binary32); compliance does not mandate per-operation round-to-nearest-even semantics on every controller. "Entire industrial PLC ecosystem" is un-tested phrasing. Fix: "controllers whose REAL arithmetic conforms to IEC 60559 operation semantics"; add residual per-platform validation; remove "entire ecosystem".
2. [MAJOR] **Vendor-neutral backend never compiled on any non-Siemens toolchain** (main.tex:906–916, 333–334) — E67 validates PyTorch-vs-SCL-simulation only. Fix: qualify "target-independent in semantics (uncompiled on third-party toolchains)" or add a CODESYS/TwinCAT compilation experiment.
3. [MAJOR] **RBF-KAN and MNIST claimed but no results exist** (abstract 171–172; conclusion 5280–5281) — "Five C²-BV architectures … and three datasets validate the framework": RBF-KAN appears only in a proposition and a basis-illustration figure (no experiment; E60/E61 are Fourier/Wavelet); MNIST is described but no MNIST result, table row, or experiment number exists anywhere. Fix: add the experiments or change to "four architectures and two datasets were experimentally validated; RBF-KAN and MNIST remain theoretical/planned".
4. [MAJOR] **"Empirically validated on physical hardware (TIA Portal V21, S7-1200)"** (main.tex:336–338, 5085–5093; taxonomy footnote) — TIA Portal V21 is offline compilation, not physical hardware; the paper itself says so (2501–2502, Limitations, conclusion). Replace "physical hardware" with "the TIA Portal V21 engineering toolchain".
5. [MINOR] main.tex:2602 "hardware-level compilation evidence (0e 0w)" — misleading for an offline check; harmonize.
6. [MINOR] main.tex:474–475, 5275–5278 "confirming … cross-architecture fine-tuning stability" — 2 architectures × 1 dataset; soften to "consistent with".
7. [MINOR] Resolution-matching law stated without carrying the Box-Continuation hypothesis; E68 support is weak (knee 0.14±0.09 vs 0.25; 0-1 variant unmeasurable). Add "under the two-scale upper bound".

### Missing Caveats

1. [CRITICAL] **Box-Continuation violation for the deployed network** (abstract 156–161, intro 271–279, conclusion 5231–5235, Thm 1 "for all x∈X") — the deployed KAN violates the assumption underpinning the two-scale law and Thm-1 certificate (47% out-of-domain; clamping error ≈17 at every N; E68: "violated at depth≥2 for this checkpoint"). Disclosed in E68/E53 only. Suggested abstract text: "Under the Box-Continuation condition (reachable per-layer signals inside the LUT domains — certified in polynomial time, violated by the as-trained [28,16,4] checkpoint at depth≥2, where per-layer domains are a deployment requirement), the deployed risk decomposes as …".
2. [MAJOR] **Non-contractive trained γ** — the headline "gap O(γ^{L−1}/√n)" never notes that for the deployed network γ=[15.4,5.3]>1 makes the gap term ≈1.5 (order 1) and the knee slope (0.14±0.09) disagrees with 0.25. Add the qualifier.
3. [MAJOR] **Experiment-vs-bug chronology** — which experiments ran with the buggy (83%-agreement) vs fixed emitter is never stated (E21's 14.17 and E53's ≈17 may predate the fix; E9/E11/E40 values may be pre-fix). Add: "All error-bound and differential-test experiments (E6, E9, E11, E21, E53, E67, E68) were rerun with the post-fix emitter".
4. [MAJOR] **Flips at the deployment default** (E14-S, E52: 37 flips/20,000 at N=15; 1238–1239 "mathematically guaranteed") — the guarantee is conditional on margin>2Δ (0.158), never stated in the "guaranteed" sentence or abstract.
5. [MAJOR] **Trichotomy necessity unproven** — conclusion should say "The necessity direction of the trichotomy is stated as a conjecture (P≠NP); establishing it formally is open."
6. [MINOR] **Hot-swap protocol never exercised** — no E-number runs Algorithm 4; mark "specified and proved under stated assumptions; deployment validation is future work".
7. [MINOR] 0-1 decision knee unmeasurable — already honestly reported; ensure abstract carries the same honesty.

### Minor Language Issues

1. [MINOR] section_galois.tex:122–124 garbled leftover draft sentence ("establishes NeuroPLC as the first Grounds a neural network compiler…").
2. [MINOR] main.tex:4349 "a actionable" → "an actionable".
3. [MINOR] main.tex:4243 "strongest correctness guarantee bound" double noun.
4. [MINOR] section_svnn_theorems.tex:64 "Tjeng et al.; Sälzer–Lange" have no \cite entries.
5. [MINOR] E52 safety factor at N=15: 5.02× vs computed (0.675/0.134)=5.04; Δ_DA=0.13 at N=15 conflicts with 0.079 (Thm 1/E9) and 0.064 (E56) — reconciliation table omits 0.13 and 0.064.
6. [MINOR] "‖ΔW‖≈0.003 — 50× below safety margin δ_max≈0.007": 0.007/0.003≈2.3×, not 50×.
7. [MINOR] Conclusion item 5 "Non-SVNN architectures degrade by Θ(d) per layer" vs Theorem A's Θ(√d) per layer.
8. [MINOR] Intro contribution 6 "O(γ^L/√n)" vs Theorem 6's O(γ^{L−1}/√n) — exponent mismatch; bound also does not require Condition 3.
9. [MINOR] FT-Stability: "safety factor γ₀>1" vs table's γ₀=0.459/0.502<1 (conflates safety factor with contractivity).
10. [MINOR] E52 "per-activation LUT error bound 0.15 (analytical M₂≤0.3)" vs paper's own analytical M₂≤12.8.
11. [MINOR] Abstract "degraded deployed accuracy to 83%" vs "83% classification agreement" — different quantities; unify.
12. [MINOR] "Trichotomy" with four Regimes (1–4) — rename or note "three properties, four regimes".
13. [MINOR] Abstract "ReLU architectures admit only first-order bounds" vs trichotomy's own "corrects the overstrong dichotomy" — reconcile framing.
14. [MINOR] N*≈10.3 vs deployed 15 ("within 1.5×") — the law is a bound-minimizer, not a validated scaling law; add "under the two-scale upper bound".

---

## 3. Internal Consistency & Cross-Reference Verification

### Critical Inconsistencies

1. [CRITICAL] **Lemma 6 self-contradicts** (main.tex:2194–2208) — states worst case "bounded by Δ≤0.71 using M₂^max"; same lemma's validation reports ℓ∞ logit error up to **14.17** (E21) and ≈17 (E53). 14.17 exceeds 0.71 by 20× and even the aggregate 6.4 (claimed to cover MaxAE 3.65 with 1.75× safety). Three worst-case figures (MaxAE 3.65 E6 / 14.17 E21 / ≈17 E53) never reconciled.
2. [CRITICAL] **E52 table contradicts E9/Theorem 1 at the same configuration** (main.tex:3992–4013 vs 1904–1921, 3789–3801) — N=15: Δ_DA=0.13, safety 5.02× vs Δ=0.079, safety 8.5×; the reconciliation table (tab:da_bounds_summary, 3814–3841) omits 0.13 (E52), 0.064 (E12-FT pre-FT, E56, E57, CROWN), 0.049 (post-FT); its caption "safety<1 at N=4,5" conflicts with its own data (safety<1 at N=4,5,6,7: 0.23/0.41/0.64/0.92).
3. [CRITICAL] **"5.6× (IA, E9) to 17.0× (DA, E9)" contradicts canonical E9 safety factors** (section_svnn.tex:954) — E9/Thm 1/E14/abstract all report IA 3.9× and DA 8.5×; 5.6× is actually Tier-3 compositional DA safety (0.675/0.1196, 3827); 17.0×=1.35/0.079 uses a different (no /2) safety definition — both misattributed to E9.
4. [CRITICAL] **Same model declared both contractive and non-contractive** (section_svnn.tex:656–672 "Both satisfy the contractive condition … contractivity is not an artifact" γ=0.182 vs section_svnn_theorems.tex:288–321 "not contractive: γ=[15.4,5.3]>1" + 47% OOD violating Box-Continuation) — the 0.65×0.28 product vs measured amplification gap never explained; γ also given as 0.459/0.502 (E55) and safety-factor >1 (FT-Stability Prop) for the same checkpoint.
5. [CRITICAL] **"Lemma 1.6" referenced 4 times, defined nowhere** (main.tex:1991, 2035, 4010, 4034) — denotes the Argmax criterion, i.e., item (vi) of Lemma 4 (lem:per_op). Stale hardcoded numbering. Fix: "Lemma~\ref{lem:per_op}(vi)".

### Terminology Drift

1. [MAJOR] **γ** — (a) contractivity <1 = L_B·‖W‖₁,∞ = 0.182; (b) amplification γ^{L−1} measured [15.4,5.3]; (c) depth-scaling constant 0.18 (Thm 4); (d) per-layer contractivity γ₀=0.459/0.502 (E55), 0.404/0.318 (E56); (e) **safety factor γ₀>1** (FT-Stability Prop — one proposition, two opposite meanings). Recommend separate symbol (SF or η) for safety factor.
2. [MAJOR] **"Margin"** — inter-class logit margin 1.35 (E6) vs "deployment margin m=0.182" (the contraction constant!) in the Z3 section (section_svnn.tex:1036, fig02) — "safety margin 4.5×"=0.182/0.040 while classification uses 0.675. Safety factor computed three ways: m/(2Δ) 8.5×, m/Δ 17.0×, 0.182/ε_max 4.5–5.6×. Standardize.
3. [MAJOR] **M₂^max** — 1.586 (E11, Lemma 6), 0.178 (E68), 2.0 (analytical, E6-SMT). Three values under one name.
4. [MAJOR] **h / h_LUT / Δ** — LUT spacing at N=15 is 6/14≈0.4286, but Z3 section states 0.75 (that's the B-spline knot spacing h=6/G): its max/mean M₂h²/8 (0.040/0.017) are consistent with h≈0.43 and M₂≈1.74, NOT with h=0.75 (which gives 0.11) nor M₂^max=1.586.
5. [MAJOR] **DA/IA ratio** — 3.1× (E9) vs 2.2× (E2E) vs 2.4× (per-segment) vs 2.82 (MATLAB symbolic) vs random-architecture distributions mutually disjoint (mean 1.41, range [1.15,1.69] E19 vs mean 3.71 [2.11,5.11] 30-seed vs 4.22±0.55 at d=16); E19 claims "trained KANs achieve 2.2×" while E9 says 3.1×.
6. [MINOR] **C(3)/C(k)** — used in Z3 remark ("UNSAT when M₂h²/8≤C(3)"), intro ("≤C(k)"), "95th-percentile C(3)=0.030"; never defined (not c₃=0.0642).
7. [MINOR] **κ** — moment ratio ν/μ (Lemma 3) vs plane-curve curvature (adaptive sampling).

### Theorem/Lemma Numbering Issues

1. [CRITICAL] **"Lemma 1.6"** — 4 hardcoded occurrences; undefined; should be Lemma~\ref{lem:per_op}(vi). (See Critical #5.)
2. [MAJOR] **"Remark [The Six SVNN Theorems]" proposition numbering wrong** (section_svnn_theorems.tex:400–411) — lists Prop 1=MLP (actually Prop 2), Prop 2=ChebyKAN (actually 3), "Prop 9 (c2bv)" (actually 5), "Props 3–8" (actually 4,3,11,Lemma 2,10,7–9). Every named proposition number is wrong.
3. [MAJOR] **Prop NC.1/NC.2/NC.3 are not proposition environments** (section_noninterference.tex:30–92) — \textbf paragraphs with \label; \ref{prop:…} resolves to whatever counter was last stepped, not "NC.x".
4. [MINOR] Lemma "IR Minimality" (main.tex:787) has no \label, consumes Lemma 2, miscounted as a "Proposition" in the Six Theorems remark.
5. [MINOR] Lemma 6 (main.tex:2167) has no \label; referenced only by hardcoded "Lemma~6".
6. [MINOR] Hardcoded "Remark 2/3/4" (section_svnn.tex:425,436; section_svnn_chebykan.tex:280) drift from the auto counter (rem:near_necessity=2, so hardcoded 2/3/4 render as 3/4/5).
7. [Verified OK] All \ref{thm:…}/\ref{lem:…}/\ref{prop:…}/\ref{cor:…} resolve correctly (Thm 1–12, Lemmas 1–7, Props 1–11, Cors 1–3, King A–E); all hardcoded "Theorem~X" prose references currently match the scheme.

### Minor Inconsistencies

1. [MINOR] Split "70/10/20" (3143) vs n=10,971 train / 2,743 test (exactly 80/20); "~12,000 windows" (2435) vs implied 13,714.
2. [MINOR] E5 "[28,32,4] at 65.0 KB … widest feasible" vs "[28,64,4] exceeds 100 KB" — the S7-1200 budget is 50 KB everywhere else (65 KB already exceeds).
3. [MINOR] "MLP→S7-1200 uses only 13.2 KB (17.6%)" (3270) vs its own table 26.4% (4071).
4. [MINOR] E18 "79.4%" appears only in the frozen CN version, never in EN prose (only 25.0% zero-shot).
5. [MINOR] E21 referenced twice (2171, 2202) never defined; E48 (section_characterization.tex:175) and E40 (3827) never defined/absent from tab:summary; E-gaps E21–39, E42–47, E49–50, E59 — "74 experiments (E1–E58+E60–E68+V1–V7)" (486–487) overstates the defined set (~47).
6. [MINOR] Thm 4 numerical instantiation: stated formula with M_max=1.59, γ=0.18, W_max≈0.3 gives expected≈1.21, concentration≈1.19, total≈2.4 — text claims 0.71/0.15/0.86 (0.71 is copied from Lemma 6); "concentration≤0.05" vs computed ≈0.22.
7. [MINOR] E4 "93.2%/96.5% of DP-optimal" vs computed 11.6/18.3=63.4% and 4.9/8.1=60.5% from the same paragraph's own percentages.
8. [MINOR] E56 "14.6× increase" vs 0.98/0.064=15.3×; "≈7.3× per added layer" inconsistent.
9. [MINOR] FT-stability "‖ΔW‖≈0.003 — 50× below δ_max≈0.007" — 2.3×; C_FT=28·0.65/0.67=27.2 vs formula L·d_max·L_B/(1−γ)²=54.4.
10. [MINOR] ε-separation "124×/641×/5,719×" vs stated 3ⁿ model (81/729/6,561).
11. [MINOR] E10 "FP32 baseline 99.99%" (3859) vs 99.93% everywhere else.
12. [MINOR] DP-LUT cost three ways: ~8ms/function (E4), ~20ms (1001), ~0.03s (~17s total, 2413).
13. [MINOR] fig04 caption: "d=28: per-layer ≈31×, 2-layer ≈967×" are the d=32 row values (d=28: √28/0.182≈29.1×, 29.1²≈847×).
14. [MINOR] Trichotomy slopes −2.11/−1.82 (Regime 3) absent from E66 summary (−2.04/−2.06/−1.77/−1.03); "vs −2.04 for C² activations" ignores the other two C² slopes.
15. [MINOR] SCL line counts for the same KAN: 3,818 (tab:compiler) vs 2,187 (V2) / 2,188 (E55) / 2,185 (E60/E61).
16. [MINOR] Verify-gap table caption "KAN vs. MLP [28,16,4]" vs text's MLP [28,32,16,4]; E41 cited for both "0/16 (MLP [28,16,4] SiLU)" and "0/48 (MLP [28,32,16,4])".
17. [MINOR] E6-SMT Z3 times sum 559ms, stated total 599ms.
18. [MINOR] "Z3 WCET (2,862μs) is 30% tighter than static FLOPs estimates" — 2.86 vs 13.4ms is a 79% reduction; paper uses four WCET models (2.86/3.15/13.4/22.68ms), documents three.
19. [MINOR] Paderborn "features 9,7 account for 54% of shift" vs (7.40+3.83)/(28×0.84)=47.7%.
20. [MINOR] FourierKAN "M₂h²/8≤0.063<0.182 (2.9×)" vs cross_domain M₂max=2.298, h=6/14 → ε=0.0528, 3.4× (4856).
21. [MINOR] Engineering "37.8 expected errors at 0.5%/value" vs 8,243×0.5%=41.2.
22. [MINOR] ONNX "40.3 KB of IEC native code" vs measured 45.2 KB; "30.2 KB LUT figure (tab:compiler)" — no 30.2 KB appears there.
23. [MINOR] MLP contrast reuses the retracted 0.0019.
24. [MINOR] Abstract "three datasets" vs four used (CWRU, XJTU-SY, Paderborn, MNIST cross-domain).
25. [MINOR] E51 prose "compared to 99.93% with full 28-D" vs its table "LUT accuracy (full 28-D) = 1.0000".
26. [MINOR] E68 "test error 0.04%" vs 99.93% accuracy (0.07%).
27. [MINOR] Prop 4(iv) "(0.41)^32≈2.2×10⁻¹³" — 0.41³²=4.1×10⁻¹³ (0.39³²=2.2×10⁻¹³).
28. [MINOR] "E17 … 2,610× speedup" — 2,610× is the engineering-effort speedup, not RTNNIgen comparison.
29. [MINOR] Citation: all \cite keys resolve; ~24 bib entries unused; in-text "Tankman" (main.tex:566), "Tjeng et al.; Sälzer–Lange" (section_svnn_theorems.tex:63–65) have no bib entries.
30. [MINOR] Z3 "safety margin 4.5× (6.0× at P95)" internally consistent but uses the 0.182 "margin" definition; max/mean M₂h²/8 (0.040/0.017) imply per-function M₂≈1.74/0.74, not 1.586/0.607.
31. [MINOR] tab:summary header "(E1–E20, E52–E57, E60–E62)" doesn't match its own rows (E53, E63–E68 present).
32. [MINOR] ChebyKAN "M₂ max … 5.4× M₂" mixes a ratio into an absolute-value column.

---

## 4. Mathematics, Equations & Notation

**Positive verification (balanced record):** Thm 9's mathematics entirely correct — all constants recomputed numerically (Q₂=0.25→c₂=1/8; Q₃=0.38490=2/(3√3)→c₃=1/(9√3)≈0.06415; Q₄=1.0→c₄=1/24; Q₅=3.6314→c₅=0.03026); extremal family σ*(x)=(M_k/k!)∏(x−x_i) has k-th derivative ≡ M_k, zero interpolant, error exactly (M_k/k!)Q_kh^k; argmax of ∏|u−i| strictly interior for all k≥2. Lemma 3 (Bennett) sound (upper tail via Bennett with V=d₁e²ν², b=w_max·e; lower tail via one-sided Hoeffding; R≥μ/(2√2Cκ)·√(d₁/log(2/p)); κ=√d₁ outlier family correct). Thm 5 fractional-consistency argument sound. Box Continuation poly-time claim fine. Lemma 6's 0.131-vs-0.079 *is* correctly resolved (0.71≥0.131 with M₂^max) — but the lemma then introduces a worse contradiction (below). Thm 6's decomposition dimensionally consistent; h*/N* formulas mutually consistent.

### Mathematical Errors

1. [CRITICAL] **Theorem B (DA Galois Connection), section_galois.tex** — false as stated on three points: (i) soundness f∈γ(α(f)) fails for f(x)=x (M₂=0, α=(0,0), |f(±r)|=r>0); the proof relies on the false "|f(x)−f(x₀)|≤M₂(f)r²" (correct: |f(x)−f(x₀)−f′(x₀)(x−x₀)|≤M₂r²/2); (ii) α∘γ=id fails (constant f≡c: α=(c,0)≠(c,R), R_min=0); (iii) adjunction ⟸ fails (x²∈γ(0,1), α=(0,2)≰(0,1)). Fix: γ(d):={f : |f(x₀)−c|+M₂(f)r²≤R}, or add first-order term to the doubleton. Coq lemmas galois_soundness/galois_optimality false as written (optimality only at r=0).
2. [CRITICAL] **Lemma 6 (Adversarial Lower Bound), main.tex:2167–2209** — bound "Δ≤0.71 using M₂^max" (2196) vs own measurement ℓ∞ logit error up to 14.17 (2206) and aggregate 6.4 (2154): 14.17 exceeds both — "correctly bounds even pathological cases" (2207–2208) contradicted by the lemma's own data. Fix: reconcile (14.17 must be labeled ℓ₁ if it is, or the bound recomputed to cover it).
3. [MAJOR] **Lemma 6 resolution vs Table 11** — "segment-aware yields Δ≤0.079" (2193–2194) but segment-aware+DA is 0.056 (tab:da_bounds_summary, E16) and 0.079 is the plain-DA value; additionally 0.064 (E57, E12-FT) omitted from the reconciliation table; E52 uses yet another calibration (0.13, analytical M₂=0.3, safety 5.02×). Reconcile 0.079/0.064/0.056/0.13/0.1196 explicitly.
4. [MAJOR] **Prop SVNN Fine-Tuning Stability, main.tex:3657–3691** — (i) C_FT: formula L·d_max·L_B/(1−γ)² with L=2, d=28, L_B=0.65, γ=0.182 gives 54.4, text evaluates "28·0.65/0.67≈27.2" (drops L, uses (1−γ)¹, 0.67≠1−0.182=0.818); (ii) "‖ΔW‖≈0.003 is 50× below δ_max≈0.007" — actual ratio ≈2.3×; (iii) "safety factor γ₀>1 (Definition def:svnn)" — Definition 1 defines no safety factor.
5. [MAJOR] **Corollary Fixed-Depth Tradeoff, section_svnn_theorems.tex:267–280** — condition "ΔR̂≥(γ−1)·C·√(Ld/γ^{L−1}n)" vacuous for γ<1 ((γ−1)<0) in the very regime discussed; "gap amplifies with depth" false for γ<1 (γ^{L−1}√L decreases). Consistent version: ΔR̂≥C·γ^{L−1}·√(Ld/n).
6. [MAJOR] **Thm 6 bias/gap derivation, section_svnn_theorems.tex:196–222** — "∏_{j>ℓ}‖W_j‖∞≤γ^{L−1}" does not follow from Condition 3 (which bounds products L_f^(ℓ)·‖W^(ℓ+1)‖, not ‖W‖ products); when activation Lipschitz <1 (L_B≈0.65), ‖W‖-products exceed γ^{L−1} — the bias term can be unsoundly tight for contractive networks; gap-term Lipschitz Λ=L_ℓγ^{L−1} omits the leading ‖W^(1)‖₁,∞ factor. Fix: state per-layer ‖W^(ℓ)‖₁,∞≤γ explicitly or define γ as max per-layer error amplification.
7. [MAJOR] **Corollary Resolution Matching, section_svnn_theorems.tex:245–264** — "minimizing the two-scale bound over h" is not the minimizer: the gap term is h-independent, so the infimum is at h→0 (N→∞); N*≍n^{1/(2k)} is a bias-variance-style heuristic balance, not the minimizer of the stated bound. Present as a design law (balanced-bias criterion).
8. [MAJOR] **Thm 4 numeric instantiation, main.tex:2252–2259, 2375–2393** — with stated parameters (M_max=1.59, h=0.43, W_max=0.3, d_max=28, L=2, δ=0.05): expected≈1.22, concentration≈1.19, total≈2.4 — text claims 0.71/0.15/0.86 (0.71 copied from Lemma 6; line 2357 confirms reuse); "concentration adds ≤0.05" vs formula 0.13–0.21.
9. [MAJOR] **Thm 7/E58 numbers, section_svnn.tex:1031–1039** — "h_LUT≈0.75" wrong (N=15 LUT on [−3,3]: 6/14≈0.429; 0.75 is the knot spacing); with h=0.75 and M₂^max=1.586 the bound is 0.1115>0.040 — "all 512 satisfy M₂h²/8≤0.040" is consistent only with h≈0.43 (0.0364); "deployment margin m=0.182" is the contractivity constant, not the classification margin (1.35/0.675); C(k)/C(3) never defined.
10. [MAJOR] **SiLU curvature claims, section_svnn.tex:402–406, 576–577; main.tex:1152; Table 5 footnote** — the stated SiLU″ formula σ(x)(2−2xσ(x)+xσ(x)(1−σ(x))) is wrong; correct: σ(x)(1−σ(x))[2+x(1−2σ(x))], sup over ℝ = 0.5 (attained at x=0, verified numerically). Three claims consequently false: "sup|SiLU″|=∞ on ℝ", "max≈0.21 on [−3,3]", "M₂^SiLU≈1.1" — and 0.21 vs 1.1 contradict each other.
11. [MAJOR] **Thm 5 (ε-Separation), section_svnn_theorems.tex:45–60** — (a) "F=0 iff φ satisfiable" imprecise (iff the induced assignment satisfies φ — proof says the correct version); (b) "knot-aligned grids reproduce the spline exactly … verified to 5×10⁻⁸" — exact reproduction requires every 4-node window inside one polynomial segment, not guaranteed by knot inclusion; 5×10⁻⁸ is not "exact"; (c) "ε-decision O(1) given the LUT" is a YES-certificate, not a decision procedure for sup≤ε.
12. [MAJOR] **Theorem A (Compilable Frontier), section_characterization.tex** — statement claims per-layer "Ω(d²r²)" and "Θ(d) gap"; proof derives Θ(d^{L/2}r²) — Eq. mlp_total_sharp omits the per-layer fresh-error dimension factor d (which the SVNN recurrence includes); correct total Θ(ε·d^{1+L/2}); at L=2 the stated Θ(d) gap vanishes (d·r² vs L·d·r²). Statement and proof disagree on the width exponent.
13. [MAJOR] **Thm 12 (Downgrade Guarantee), main.tex:3037–3076** — invariant "every control decision covered by a sound error bound or the monitor's fallback" not preserved during the rollback window: case (a) runs uncertified T_new under only domain/confidence/range checks (a "monitored envelope" is not a sound bound). Weaken invariant or restrict to cycles where T_old (or certified T_new) is in use.
14. [MAJOR] **WCET instruction timings contradict, main.tex:2714, 2855, 3345–3347** — same datasheet quoted with REAL MUL 3.5μs/ADD 1.8μs/DIV 8.0μs (Thm 11) vs ADD 0.50μs/MUL 0.60μs/DIV 1.20μs (Z3-WCET and table): 6–7× discrepancy behind the three "levels"; never flagged.
15. [MAJOR] **Trichotomy E66 slopes, section_trichotomy.tex:67 vs 106/133** — "measured slopes −2.11, −1.82, E66" (Regime 3) contradict "slope −1.034 (E66) vs −2.04" (Regime 2) and "{−2.04,−2.06,−1.77} for tanh/sin/B-spline" (Consequences (iv)) and the E66 summary row; the B-spline slope −1.77 is not a second-order rate, undercutting the Regime-1 claim for the flagship architecture.
16. [MAJOR] **"Folklore formula" framing, section_lut_sharp.tex:110–115** — for k=2 the "folklore value" 1/4 contradicts the classical de Boor constant 1/8 which the paper itself treats as optimal recovery (Thm 9 item 3); the "folklore 1/8" for k=3 has no citation. The mathematics is fine; the framing needs a reference or softening.
17. [MINOR] Lemma 3, main.tex:1367–1370 — "C=4 suffices when w_max≤3ν" fails for very small p (additive Bennett term dominates); harmless (claim is "an absolute constant C").
18. [MINOR] Trichotomy Regime 3, section_trichotomy.tex:70–72 — misattributes arity-2 hardness to Thm 5's arity-3 embedding; arity-2 needs its own argument.
19. [MINOR] main.tex:2271, 2354 — "O(√(L log(L/δ)))" vs the Azuma-derived √(2L log(2/δ)).
20. [MINOR] Prop NC.3, section_noninterference.tex:118–130 — "z_i well within [−20,20]" not justified by clamping alone (row-norm bound gives |z|≲27; conclusion survives: exp(27)≈5×10¹¹≪3.4×10³⁸); "no NaN" requires the LUT grid strictly increasing (t=(x−G[lo])/(G[hi]−G[lo]) divides by zero otherwise) — unstated.

### Notation Inconsistencies

1. [MAJOR] γ — Condition-3 contractivity <1 vs Thm-6 amplification vs "safety factor γ₀>1" in FT-Stability (both meanings in one proposition). Rename safety factor (e.g., SF).
2. [MAJOR] W_max — max_ℓ‖W^(ℓ)‖₁,∞ (Thm 4) vs max_c|w_c| control-point magnitude (Prop kan-svnn). Rename one (W_ctrl).
3. [MINOR] κ — moment ratio ν/μ (Lemma 3) vs plane-curve curvature.
4. [MINOR] h — LUT cell width 6/14≈0.43 vs B-spline knot spacing 6/G=0.75 (E58 prints the knot value as h_LUT).
5. [MINOR] N — LUT points vs "Let N be an SVNN" (Thm 6).
6. [MINOR] ε — LUT error bound / ε-Verify threshold / DA noise symbol ε∈[−1,1] / generic perturbation.
7. [MINOR] L — depth vs Lipschitz constants L₁, L₂, L=L₂L₁ vs L_f, L_B, L_ℓ.
8. [MINOR] m — gate arity (N^m) vs classification margin 1.35 vs "deployment margin 0.182".
9. [MINOR] C — output bound [−C,C] vs absolute constants vs undefined C(k).
10. [MINOR] B — input box [−B,B] vs storage budget B (bytes).
11. [MINOR] M₂ — curvature bound vs "global M₂=0.177" vs M₂^max=1.586 vs analytical M₂=0.3 (E52) — four values under one symbol; only char/max superscripts distinguished.

### Undefined Notation

1. [MAJOR] ε_fp, ‖scale‖ — first used in Thm 6 Eq. compile_aware, never defined. Add definitions.
2. [MAJOR] C(k)/C(3) — main.tex:424, section_svnn.tex:1035, section_svnn_theorems.tex:371 — never defined; Thm 7 itself uses margin m. Define or replace.
3. [MAJOR] "deployment margin m=0.182" — section_svnn.tex:1036 — conflicts with margin 1.35/half-margin 0.675; 0.182 is the contractivity constant. State what it represents.
4. [MINOR] ‖W‖_prod — Eq. mlp_bound, never defined.
5. [MINOR] P(KAN) — main.tex:557, Table 3 footnote — undefined (Lipschitz product? probability?).
6. [MINOR] "safety factor γ₀ (Definition def:svnn)" — Definition 1 defines no safety factor.

### Equation Numbering / Hardcoded Reference Issues

1. [CRITICAL] **"Lemma~1.6" hardcoded** at main.tex:1991, 4010, 4034 — actual target is Lemma 4 (lem:per_op, item 6 = Argmax); stale numbering. Replace with \ref{lem:per_op}.
2. [MAJOR] **Hardcoded proposition numbers in Remark (Six SVNN Theorems), section_svnn_theorems.tex:399–411 are wrong** — actual order: Prop 1=KAN SVNN, 2=MLP-negative, 3=ChebyKAN, 4=DA Segment-Exactness, 5=Operation Separation, 6=C²-BV, 7–9=pass soundness, 10=Compiler Complexity, 11=FT Stability, 12–14=NC.1–3; "IR Minimality" is a lemma, not a proposition.
3. [MAJOR] **Hardcoded "Remark 2/3/4" off by one** — rem:near_necessity is Remark 2, so hardcoded "Remark 2/3/4" (section_svnn.tex:425,436; section_svnn_chebykan.tex:280) are actually 3/4/5.
4. [MAJOR] **~50 hardcoded "Theorem~N" instances instead of \ref** — all currently correct (Thm 1=Compiler … 12=Downgrade, verified), but any insert will silently break them under the manual \setcounter scheme. Convert to \ref.
5. [MAJOR] **thm:greedy, thm:deep, thm:hot_swap_safety never \ref'd** — all mentions hardcoded; Thm 12 never referenced at all.
6. [MINOR] **36 numbered equations never referenced** (candidates for removal or cross-reference): eq:alpha_da, eq:bspline_basis_bounds, eq:bspline_second_deriv, eq:cheby_f_m2_total, eq:cheby_f_second, eq:chebykan_def, eq:chebykan_layer, eq:convexity, eq:correlation_aware, eq:dp_lut, eq:element_error, eq:error_at_midpoint, eq:error_quadratic, eq:expected_error, eq:galois_condition, eq:gamma_da, eq:greedy_optimal, eq:hoisting_ratio, eq:iec_universal, eq:interp_remainder, eq:ir_bnf, eq:kan_decomposition, eq:l_net_ia, eq:lagrangian, eq:lower_tail, eq:lut_error_def, eq:lut_quadratic, eq:mlp_bound, eq:resolution_matching, eq:sharp_amplification, eq:sharp_compositional, eq:sharp_constants, eq:sharp_l1inf, eq:svnn_closure, eq:svnn_def, eq:theorem1.
7. [MINOR] No broken \refs (verified programmatically). All equations referenced in text exist and are numbered.
8. [MINOR] Theorems appear out of numerical order in the compiled document (2, 8, 5, 6, A–E, 3, 1, 4, 11, 12) — consequence of the manual \setcounter scheme; disorienting and fragile.
9. [MINOR] Stray duplicate label thm:frontier (section_characterization.tex:47) — unreferenced.

### LaTeX Math Formatting

1. [MINOR] main.tex:836–837, 3384–3385 — "\small" immediately followed by "\footnotesize" (duplicate size commands).
2. [MINOR] Table 6 "M$_2$ max" — text-mode math fragment.
3. [MINOR] Eq. sharp_constants: c₅ printed only numerically (≈0.0303) while c₂–c₄ as fractions — give c₅=Q₅/120.
4. [MINOR] main.tex:471 "M2-regularized" — M₂ not typeset.
5. [MINOR] Thm 4 proof display duplicates Eq. eq:l_net without label/cross-reference.
6. [MINOR] Eq. compile_aware — \tfrac for the log ratio and \bigl(\bigr) would improve readability.

---

## 5. Tables, Figures & Documentation

Scope: 42 table floats (41 in main.tex + tab:svnn_taxonomy), 15 figure floats, tab:summary experiment matrix, all in-text references. All 42 tables have at least one \ref; all \ref'd labels exist; no hard-coded "Figure N"/"Table N" strings.

### Tables with Missing or Incomplete Notes

1. [MAJOR] Table tab:summary (main.tex:4119) — no sample-size (N) per row; no footnote explaining the E21–E51 gap; no "what is reported" note; no significance-stars convention.
2. [MINOR] Table tab:instr_timing (main.tex:3323) — EXP row percentage arithmetically wrong (2,880/13,371=21.5%, table states 13.5%); column percentages sum to 91.9% but Total row claims 100%.
3. [MINOR] Table tab:cycle_count (main.tex:2665) — caption "Static operation counts" but body has dynamic timing rows (13.4ms/58ms/0.21ms); "2.7× dynamic ops → ~58ms" inconsistent (2.7×13.4=36.2≠58; static ratio 6,221/2,699=2.3×).
4. [MINOR] Table tab:compiler (main.tex:4058) — no footnote for the "Budget (%)" basis (90.4% of 50KB vs 7.4% of 1.5MB).
5. [MINOR] Table tab:xjtu_ft (main.tex:3705) — dangling "$^{*}$" marker in footnote; no per-class N.
6. [MINOR] Table tab:kd_ablation / tab:models — CI method given, no N; no test-split note in tab:models.
7. [MINOR] Table tab:tia_multitarget (main.tex:2513) — caption "all verified 0e0w" but first row reads "63.9 (exceeds)" (work-memory budget); clarify "0e0w (TIA compile diagnostics); DB2+FB2 variant exceeds the 50KB budget".
8. [MINOR] Table tab:cert_thresholds (main.tex:5142) — "Safety" values (8.5×/2.9×/1.4×/2.3×) have no source cross-references.

### Figures with Missing or Incomplete Notes

1. [MAJOR] Figure fig04_sharp_bound (fig:sharp_lower_bound, section_characterization.tex:135) — fully captioned but **never cited** (below); caption values for d=28 ("≈31×, ≈967×") are the d=32 row values (d=28: ≈29×, ≈845×) — caption disagrees with its own math and inline table.
2. [MINOR] Figure fig09_wcet_breakdown (main.tex:2923) — caption percentages (73.4/16.3/0.5/0.3) sum to ~90.5%; LUT 73.4% conflicts with Thm 11's per-edge arithmetic (512×36.7μs/22,683μs≈82.8%).
3. [MINOR] Figure fig07_da_scaling (main.tex:1498) — no data-source/code pointer (unlike fig03).
4. [MINOR] Coverage note: E63–E68 results have **no figure coverage** — sharp-constant attainment (E63) and trichotomy slopes (E66) are natural single-panel figures if space permits; no trichotomy figure exists.

### Cross-Reference Issues

1. [MAJOR] **Figures defined but never cited (5 of 15):** fig:c2bv_verification (fig02), fig:da_tightness (fig03), fig:sharp_lower_bound (fig04_sharp_bound — the new 2026-08-03 sharp-bound figure!), fig:da_vs_ia (fig05), fig:wcet_breakdown (fig09). Cite each at its natural location or remove the floats.
2. [MAJOR] **Table tab:summary omits E58** (reported experiment, section_svnn.tex:1031) — caption claims "E1–E20, E52–E68" but rows contain only E52–E57, E60–E68 (15 of 17 numbers in the claimed range; E58, E59 missing). Add an E58 row (de Boor-bound certification of 512 LUTs, max M₂h²/8=0.040≤margin 0.182, 4.5× margin) or drop E58 from prose.
3. [MAJOR] **E21–E51 gap unexplained** — intro claims "74 experiments (E1–E58+E60–E68+V1–V7)" (main.tex:486) but tab:summary lists 42 rows; no sentence explains the E21–E51 renumbering; prose cites E21 (2171, 2202), E40 (3827), E41 (318), E48 (section_characterization.tex:175), E51 (3511) — none in the table. Add a one-line note.
4. [MAJOR] **V7 duplicates E52 verbatim** (identical row content); V1 duplicates E14-S. Two of seven V-rows re-report core experiments — keep one occurrence each or retitle.
5. [MINOR] E14-S (main.tex:3905) absent from tab:summary (covered only via duplicate V1).
6. [MINOR] Prose-vs-table instruction timing conflict: prose (2695, 3311) reports DB array 37.4%, REAL ADD 29.3%, REAL MUL 16.6% — tab:instr_timing reports 59.7%/10.2%/6.9% for the same quantity; one set is stale.
7. [MINOR] E59 skipped in numbering with no explanation.
8. [MINOR] tab:summary sub-header "Core Experiments (E1–E20, E52–E57, E60–E62)" contradicts the rows it introduces (E63–E68 present).

### Formatting Inconsistencies

1. [MINOR] tab:summary row order breaks numeric sequence (E53 placed after E62).
2. [MINOR] Duplicate size commands (\small+\footnotesize) in tab:ir_ablation, tab:da_scaling, tab:scalability_*, tab:compiler.
3. [MINOR] Mixed float placement specifiers ([t], [tb], [ht], [!t]).
4. [MINOR] Uncaptioned inline tabular in section_characterization.tex:107–116 (data table with no caption/label; prose ">10,000× for 3-layer d=28" not backed by the shown 2-layer rows).
5. [MINOR] fig12 caption carries "E6:" prefix, no other figure does.
6. [MINOR] Orphan figure assets never \includegraphics'd: figures/final/fig13_model_comparison.*, fig14_cross_domain.*, fig15_safety_monitor.* (+ legacy eps/older results/figures set).
7. [MINOR] tab:wcet Total row (2,862μs) vs sum of rows (2,861.2μs) — rounding; FLOPs total matches exactly.

**Bottom line:** No broken labels or wrong-numbered in-text references. Submission-blocking: five never-cited figures (incl. the new fig04_sharp_bound), the missing E58 row and unexplained E21–E51 gap (intro's "74 experiments" unreconcilable against 42 reported rows), V7/V1 duplicate rows, and prose-vs-table numeric conflicts (instr_timing percentages; fig09/fig04 caption numbers).

---

## 6. Spelling, Grammar & Style

### Critical Issues (must fix before submission)

1. [CRITICAL] section_galois.tex:123–125 — "Theorem~\ref{thm:galois} establishes NeuroPLC as the first \textbf{Grounds a neural network compiler in the standard formal framework of abstract interpretation}." — verb fragment where a noun phrase is required; unreadable as written. → "establishes NeuroPLC as the first neural network compiler grounded in the standard formal framework of abstract interpretation." (The following lead-in also dangles: "The DA abstract domain (α,γ) is:" → "has three distinguishing properties:")
2. [CRITICAL] main.tex:2116–2118 — "the 16 hidden-dimension error symbols are (all biased positive or negative by the boundary curvature), defeating the DA cancellation mechanism" — parentheses turn the predicate into a fragment. → "…are all biased positive or negative by the boundary curvature, defeating…"

### Major Issues

1. [MAJOR] **"fixed-point" is factually wrong terminology** (main.tex:139, 194, 212, 732, 5201) — "design-time correctness certification under fixed-point (LUT) compilation" — no fixed-point arithmetic exists in the paper; deployed SCL uses IEEE-754 binary32 REAL (Lemma 7). → "under lookup-table (LUT) compilation" / "discretized (LUT) compilation".
2. [MAJOR] **"Tankman's Lipschitz result"** (main.tex:566–567) — no citation and no antecedent; the frozen CN translation retains `Tankman~\cite{tankman2026lipschitz}` (section_svnn_cn.tex:145). Restore the cite (and add the bib entry) or remove the sentence.
3. [MAJOR] section_svnn_chebykan.tex:290–292 — "Fourier-KAN … faces the same Z3-undecidability issue as SiLU" directly contradicts main.tex:466–470/E60–E62 ("FourierKAN … 512/512 Z3-equivalent"). Reconcile (if E60's verification uses a bounded/LUT encoding of sin/cos, say so).
4. [MAJOR] section_svnn_chebykan.tex:284–286 — Wavelet-KAN citekey is the ChebyKAN paper (bozorgasl2024chebykan) — wrong citation; and contradicts the rest of the paper (WaveletKAN uses a Mexican-hat mother wavelet: section_svnn.tex:1217–1221, E61 "8 Mexican hat scales" vs "Daubechies or Haar").
5. [MAJOR] **"three-tier" vs "four-tier"** (main.tex:2757, 3836, 4551, 4686, 5103 vs 164, 4556) — the system is "Four-Tier" with Tier 4 (differential self-test) explicitly included; five prose passages call it "three-tier". Pick one framing.
6. [MAJOR] section_svnn_theorems.tex:350 — "Rademacher … ≈0.217 — an order of magnitude larger than the KAN's $0.0019$" re-introduces the value the same section corrected (0.0038; corrected gap ≈1.5). Update.
7. [MAJOR] section_svnn.tex:473–474, 655 — "depth-uniform ($\gamma<1$, $O(L)$ growth)" — with Condition 3 the bound is depth-independent O(M_maxh²d_max/(1−γ)) (Eq. 13), not O(L). Read "$O(1)$ growth" / "depth-independent". Self-contradiction.
8. [MAJOR] section_svnn.tex:769–770 — "decreases cubically with LUT resolution h" — the de Boor bound is M₂h²/8: quadratic in h. "Cubically" is wrong.
9. [MAJOR] main.tex:4349 — "a actionable insight" → "an actionable insight".
10. [MAJOR] **Remark-numbering collisions** (section_svnn.tex:425 "Remark 2", 436 "Remark 3", section_svnn_chebykan.tex:280 "Remark 4") — hand-numbered bold paragraphs collide with auto-numbered remark environments (auto 1=Compilable Frontier, 2=Near-Necessity, 3=Probabilistic Tightening, 4=Tightness Conditions, 5=Six Theorems) — PDF contains two "Remark 2"s, two "Remark 3"s, two "Remark 4"s. Rename hand-numbered ones.
11. [MAJOR] main.tex:4243–4244 — "the strongest correctness guarantee bound" — garbled double noun; near-duplicates main.tex:3851–3852 ("strongest correctness guarantee for compiled NN inference on industrial controllers that we are aware of"). Keep one.
12. [MAJOR] main.tex:592–593 — "RTNNIgen … converts Keras models … and demonstrated real-time inference on Beckhoff PLCs" — mixed tense (present + past). → "converts … and demonstrates".

### Minor Issues

1. [MINOR] main.tex:1907; section_svnn.tex:308 — sentence/list items begin lowercase ("**High-probability DA bound.** sign-structural…"; "Interval" capitalized mid-sentence at 3789 while "sign-structural" lowercase).
2. [MINOR] main.tex:2134 — "$3\times \times 5\times \times 2\times$" double "× ×" → "$3{\times}5{\times}2$".
3. [MINOR] "de Boor" vs "de~Boor" (1574, 1577, 3845, 4229 lack the tie); "Cox--de Boor" vs "Cox-de~Boor" — normalize to "de~Boor"/"Cox--de~Boor".
4. [MINOR] Thousands separators inconsistent ("2,700×" vs "2{,}700×", "1,000-point" vs "1{,}000-point", "2185-line" vs "2,185", …). Use `{,}` consistently.
5. [MINOR] Raw "§" characters (main.tex:2626, 2633) vs "{\S}"/"$\S$" elsewhere.
6. [MINOR] "0e 0w" informal shorthand in prose (main.tex:388, 743, 4154, 4170, 4502, 4898, 4928; "0e/0w" at 4272) — expand on first use; keep spelled-out form in running text.
7. [MINOR] "vs" vs "vs." — use "vs." consistently.
8. [MINOR] main.tex:4039 — "EMSE" never defined (does not match "Empirical M₂ Calibration" word order).
9. [MINOR] main.tex:4121, 4130 — tab:summary header "Core Experiments (E1–E20, E52–E57, E60–E62)" omits E58 and the E63–E68 rows that follow.
10. [MINOR] main.tex:3127 vs 486 — "all 20 experiment scripts" vs "74 experiments" — clarify ("20 core-experiment scripts").
11. [MINOR] main.tex:4827–4829 — "a diagnostic tool demonstration" hyphenation.
12. [MINOR] main.tex:4946–4947 — "Four independent lines of work---FPGA, MCU, and PLC" — three categories listed. Either enumerate four works or write "Three".
13. [MINOR] main.tex:3869–3874 — "three-way ablation using SVM…: (a) time, (b) frequency, (c) dispersion entropy, and (d) all 28" — three removals plus baseline labeled (a)–(d); call it "per-group ablation" or "four-condition".
14. [MINOR] main.tex:3499 — "The compiler correctness is partially formally verified" → "The compiler's correctness is partially verified formally".
15. [MINOR] main.tex:171–172 — "Five C²-BV architectures … and three datasets" — four datasets are used (CWRU, XJTU-SY, Paderborn, MNIST); five architectures claim includes RBF-KAN with no experiment.
16. [MINOR] Global LUT error at N=15 appears as 0.00412 (1601–1602, 4232), 0.00406 (1675, 4189), 0.0041 (3454, 4729); DA bound 0.064 (4523–4524, 3639) and 0.049 absent from the reconciliation table.
17. [MINOR] Dash typography mixed: "---" vs " — " vs unicode "—" (4492, 4527, section_characterization.tex:127, 4357/4398, 3967). Normalize.
18. [MINOR] main.tex:2963, 2965 — ASCII arrows "-->" in Algorithm 3 → $\rightarrow$.
19. [MINOR] main.tex:415–416 — "99.60% acc" / "acc" informal abbreviations in prose; spell out.
20. [MINOR] section_svnn_theorems.tex:353–354 — "The Six SVNN Theorems" actually lists eleven numbered theorems (1–9, 11) plus A–E; title stale.
21. [MINOR] main.tex:1203 — "All nine claims across three passes are proved sound" — prefer active voice.
22. [MINOR] main.tex:5135–5136 — "We emphasize this outlines" → "We emphasize that this outlines".
23. [MINOR] main.tex:5011–5012 — "…requirements (2D ARRAY initialization…) that the LLM was unaware of" → "…of which the LLM was unaware".
24. [MINOR] "M2" vs "$M_2$" in figure captions — use $M_2$ consistently.

### Style Patterns to Fix Throughout

1. [MAJOR] **"Notably," — 4 instances** (main.tex:1602, 4232, 4404, 4731) — replace with the substantive claim.
2. [MINOR] "This paper contributes:" (main.tex:340) — banned construction; recast as "The contributions are:".
3. [MINOR] Dense self-correction meta-commentary in the 08-03 additions ("This corrects the earlier 'factor-4' inflation…", "Earlier framing corrected", "Honest correction of the SVNN/MLP dichotomy", "(E68, honest report)", "superseding the earlier cubic-restricted… claim") — honest, but the running retraction narrative is excessive for a referee; consolidate into one "Corrections relative to prior claims" remark or footnotes.
4. [MINOR] First-person consistency — passed ("we" uniform; no "interestingly/importantly/obviously/clearly/it is worth noting" found — good).

---

## 7. Project Health Check (2026-08-03)

**Summary: PASS 12 | WARN 3 | FAIL 1**

### ❌ FAIL (1)

1. **E67 differential test broken at HEAD** — `python -m neuroplc.differential_test` (and `compile(verify=True)` without `verify_features`) crashes at code/neuroplc/differential_test.py:183 with numpy broadcast error `(2000,4) vs (2000,)`. The paper's flagship "caught the scale regression" self-test is not runnable from its documented entry point; no committed E67 result JSON. Fix is one line (`diff_id.max(axis=1) <= margin_frac * per_in` or `per_in[:, None]`). Also: the pre-fix SCL agreement was 83%; verify the E67 result numbers after the fix.

### ⚠️ WARN (3)

1. **Commit 671e7b7 (FAQ v2 + lemma-numbering fix) unpushed** — both origin/master and github/master sit at e623754, one commit behind local.
2. **SCL directory inconsistency** — 25 files regenerated on 2026-08-03 (documented claim: 24); 4 stale pre-fix July files remain: feature_extraction_s7_1200.scl (7/7), kan_reg_s7-1200_db.scl (7/6), kan_reg_s7-1200_db_fb.scl (7/10), neuroplc_safety_monitor.scl (7/10); IEC ST comments show encoding mojibake.
3. **Minor TODOs** — code/experiments/e19_paderborn.py:427 (PU data loader pending); code/neuroplc/ir.py:445 (test-only placeholder, documented).

### ✅ PASS (12)

- E63/E64/E65/E66/E68 verification scripts run and PASS; JSON values match the paper exactly (ratio 1.000000; medR 4.6→37.7; 200/200; slopes −2.04/−2.06/−1.77/−1.03; bias slope 0.99, knee 0.1377).
- frontend.py scale folding present (base_weight×scale_base, table×scale_spline, with "2026-08-01 audit fix" comments).
- backend_iec.py generates ST with zero Siemens tokens (grep verified).
- compiler.py compile(verify=True) integrated (subject to the E67 crash above).
- Verification certificate bundle complete (7 files; trusted checker 214 lines; tier2 512/512; tier3 "Certificate VALID: True").
- Git: 18 commits, clean tree, remotes origin=Gitee + github=GitHub configured.
- No TODO/FIXME/placeholder in paper/*.tex (English side).
- Data/model assets correct: features_X.npy (13714, 28) float32; kan_kd_vrmKD_best.pt 31,519 B.

---

## Priority Action Items

**CRITICAL** (must fix — these could cause desk rejection or major referee objections):

1. **Theorem B (DA Galois Connection) is unsound as stated** — fails soundness (f(x)=x), optimality (α∘γ=id), and adjunction; Coq lemmas unprovable. Fix α/γ definitions (add first-order term or redefine γ), re-verify, update the Coq spec. [Agent 3 #3, Agent 4 #1]
2. **Lemma 6 / Thm 1 bound vs. own measurements** — Δ≤0.71/6.4/0.079 (Thm 1 "for all x") vs measured 14.17 (E21) and ≈17 (E53); three worst-case figures never reconciled; Box-Continuation assumption missing from Thm 1's statement; classification "mathematically guaranteed" unqualified. State the assumption, qualify the guarantees (margin>2Δ AND Box-Continuation), reconcile 0.71 vs 14.17, label ℓ₁ vs ℓ∞. [Agent 3 #1, Agent 2 #1, Agent 4 #2]
3. **Trichotomy necessity "conjecture" appears exactly once; "unique regime" asserted everywhere else** (abstract/intro/conclusion/title/Consequences) — carry "(conjecturally)" everywhere; conclusion must flag the open problem. The stated evidence pillar "(P2) forces C² gates" is false (bounded-second-variation C¹ classes achieve O(h²)); either prove under a formalized notion or withdraw "unique satisfier". [Agent 3 #2 #5, Agent 6 Req #2, Agent 4 #15]
4. **Same checkpoint declared both contractive (γ=0.182, Condition 3, depth-uniform) and non-contractive (γ=[15.4,5.3], E68)** — define γ unambiguously (per-layer Lipschitz product vs L_B‖W‖₁,∞ vs error amplification), state which E68 measured, reconcile all claims, and give a contractive-trained demonstration (spectral-normalized, γ<1, ≥99% acc) or explicitly report Condition 3 fails for the deployed checkpoint. [Agent 3 #4, Agent 2 #4, Agent 6 Req #3]
5. **Deployed-certificate gap** — 47% of layer-1 activations outside [−3,3] (range [−23.9,+25.1]); Box-Continuation (required by Thm 6 and Tier-2 certificates) violated at depth≥2 for the certified checkpoint; abstract/intro/conclusion advertise guarantees without this caveat. Re-compile with per-layer widened LUTs and re-verify 512/512 + E53/E68, or state the exact certified operating envelope. [Agent 6 Req #4, Agent 3 Missing #1]
6. **"Verified compiler" boundary** — Tier 1–3 formal chain did not catch the emitter scale regression (Tier 4 empirical test did); "first verified compiler" overstates. State precisely which components carry formal proofs vs the emitter; and E67 is currently broken at HEAD (differential_test.py:183 broadcast crash — `python -m neuroplc.differential_test` does not run; fix the one line and commit an E67 result JSON). [Agent 6 Req #5, Agent 7 FAIL]
7. **"Lemma 1.6" hardcoded 4×, does not exist** — replace with \ref{lem:per_op}(vi). [Agent 2 #5, Agent 4 #1]
8. **E52 table vs E9/Thm 1 at N=15** — Δ_DA 0.13 vs 0.079, safety 5.02× vs 8.5×; the "reconciliation" table omits 0.13/0.064/0.049 and its caption misstates the N at which safety<1. Reconcile all DA values (0.079/0.064/0.056/0.13/0.1196) with stated M₂ calibration. [Agent 2 #2 #3, Agent 4 #3]
9. **SiLU curvature claims are mathematically wrong** — SiLU″ formula incorrect (correct: σ(1−σ)[2+x(1−2σ)], sup=0.5 at x=0); "sup=∞", "max≈0.21", "M₂^SiLU≈1.1" all false and mutually contradictory. Fix formula and all three claims. [Agent 4 #10, Agent 3 #13]
10. **WCET three conflicting per-instruction timing tables** (MUL 3.5μs vs 0.60μs — 6–7×) behind the 2.86/13.4/22.68ms "levels"; four WCET models documented as three; none hardware-measured. Use one authoritative table; reconcile or relabel. [Agent 3 #12, Agent 4 #14, Agent 2 Mi #18]
11. **Fixed-Depth Tradeoff corollary self-contradictory** (vacuous condition for γ<1; "gap amplifies with depth" false in the regime discussed). Re-derive. [Agent 3 #7, Agent 4 #5]
12. **Thm 4 numeric instantiation wrong** — formula gives ≈2.4 total, text claims 0.86 (copied from Lemma 6); "concentration ≤0.05" vs ≈0.22. [Agent 4 #8, Agent 2 Mi #6]
13. **Folklore-correction narrative** — c₃=1/(9√3) is the classical quadratic-interpolation constant (Atkinson, cited by the paper itself); no source asserts "folklore 1/8"; Thm 9(3) asserted without proof and without optimal-recovery literature (Micchelli–Rivlin). Produce a source or restate the contribution; cite or prove (3). [Agent 6 Req #1, Agent 4 #16]
14. **Resolution-matching law over-claimed** — N*≍n^{1/(2k)} is the classical bandwidth law (Tsybakov/Györfi, uncited); it is not the minimizer of the stated two-scale bound (gap is h-independent); knee 0.14±0.09 vs 0.25; 0-1 variant unmeasurable. Cite the literature; present as a design law; add the Box-Continuation qualifier. [Agent 6 Part 1/Req #5, Agent 4 #7, Agent 3 Gen #7]
15. **RBF-KAN and MNIST claimed, no results exist**; "physical hardware (TIA Portal V21)" is false (offline toolchain); "three datasets" is four. Fix abstract/conclusion. [Agent 3 Gen #3 #4, Agent 1 Mi #15]
16. **Universal IEC guarantee overclaims** (format-level ≠ operational semantics; no non-Siemens compilation ever run; Szász critique applies). Re-label and add residual per-platform validation; qualify "entire ecosystem". [Agent 3 Gen #1 #2, Agent 6 Q7]
17. **"first/unique/no-prior-work" priority assertions ×15** — verify each against the literature (esp. "first Galois connection for NN activations" — AI²/DeepPoly/CROWN are formally grounded; CompCert/Leroy must be cited given the repeated analogy). [Agent 3, Agent 6 Part 4]

**MAJOR** (should fix — will likely be raised by referees):

18. Six-Theorems remark: every named proposition number is wrong; NC.1–3 not real environments; IR Minimality lemma miscounted; Lemma 6 unlabeled; hardcoded Remark 2/3/4 off-by-one; ~50 hardcoded "Theorem~N" (convert to \ref); thm:greedy/deep/hot_swap never \ref'd. [Agent 2, Agent 4]
19. 36 numbered equations never referenced; duplicate label thm:frontier; theorems render out of numeric order (setcounter scheme). [Agent 4]
20. "Contrast with MLP" reuses the retracted 0.0019 (post-correction the KAN gap ≈1.5 exceeds the MLP's 0.217 — conclusion inverts); delete or recompute. [Agent 3 #8, Agent 2 Mi #23]
21. E56 "confirms depth-uniformity" contradicts its own data (15× jump vs predicted convergence); fix numbers (14.6→15.3; drop "≈7.3× per added layer"). [Agent 3 #9, Agent 2 Mi #8]
22. Theorem A statement vs proof disagree on the width exponent (Θ(d^{L/2}) vs Θ(εd^{1+L/2})); "Θ(√d)" conflates existence with universality; apples-vs-oranges comparison vs measured KAN γ. [Agent 3 #6, Agent 4 #12]
23. Thm 6 bias/gap derivation: ‖W‖-product ≤ γ^{L−1} does not follow from Condition 3; Λ omits ‖W^(1)‖; define ε_fp, ‖scale‖. [Agent 4 #6]
24. Remark Near-Necessity (a) restates a disavowed separation claim; (b) asserts a false mathematical fact (SiLU″). [Agent 3 #13 #14]
25. Thm 12 downgrade invariant not preserved in the rollback window; no experiment exercises Algorithm 4. [Agent 3 #11, Agent 4 #13]
26. Non-interference proof needs WCET(P)+22.67ms ≤ cycle premise; abstract/intro omit the stated provisos. [Agent 3 #17]
27. NC.3 overflow bound wrong ([−20,20] not justified; exp(25.1)>4.85×10⁸); "no NaN" needs strictly-increasing-grid assumption. [Agent 3 #16, Agent 4 #20]
28. Thm 4 unproven expectation step (random-sign cancellation); "typical γ 0.1–0.3" conflicts with E68. [Agent 3 #15]
29. FT-Stability proposition: C_FT arithmetic wrong (27.2 vs 54.4); "50×" is 2.3×; "safety factor γ₀ (Definition)" undefined; γ₀>1 vs γ₀<1 in one proposition. [Agent 2 Ti #1, Agent 4 #4]
30. IR Minimality "no third alternative" from a single ablation — downgrade to empirical claim. [Agent 3 #10]
31. tab:summary: missing E58 row; E21–E51 gap unexplained ("74 experiments" vs 42 rows); E53 row out of order; sub-header mismatch; V7/V1 duplicate rows; no per-row N; instr_timing percentages wrong (21.5% not 13.5%; sums to 91.9%). [Agent 5]
32. Five figures never cited (fig02/03/04/05/09 — incl. the new sharp-bound figure fig04); fig04 caption uses d=32 values for d=28; fig09 percentages don't sum and conflict with Thm 11. [Agent 5]
33. Terminology: γ (5 meanings), margin (2 meanings), M₂^max (3 values), h (0.75 vs 0.429), DA/IA ratio (disjoint distributions), C(k), κ, W_max, ε, L, m, C, B — add a notation table and unify. [Agent 2 Ti, Agent 4 Notation]
34. "fixed-point" wrong terminology ×5 → "discretized (LUT)"; "three-tier" vs "four-tier" ×5; "depth-uniform O(L)" → O(1); "cubically" → quadratically. [Agent 1]
35. Tankman citation lost in EN (restore \cite or delete); Wavelet-KAN wrong citekey + mother-wavelet contradiction; Fourier-KAN Z3-undecidability vs E60 contradiction; Tjeng/Sälzer–Lange uncited; ~24 unused bib entries + placeholder entries (clean the bibliography). [Agent 1, Agent 2 Mi #29, Agent 6 Part 4]
36. E18 "79.4%" exists only in the frozen CN version; E51 prose/table conflict (99.93% vs 1.0000); E68 "0.04% test error" vs 0.07%; Prop 4(iv) 0.41³² arithmetic; Z3 times 559 vs 599ms; ONNX 40.3 vs 45.2KB; SCL line counts 3,818 vs 2,185–2,188; Paderborn 54% vs 47.7%; split 70/10/20 vs 80/20; E5 50KB vs 65/100KB; MLP 13.2KB 17.6% vs 26.4%. [Agent 2 Mi]
37. Experiment-vs-emitter-bug chronology never stated — which experiments ran pre-fix vs post-fix (E21/E53 ≈17 may predate the fix). [Agent 3 Missing #3]
38. E63–E68 have no figure coverage; trichotomy slopes inconsistent across Regime 2/3/Consequences (−2.11/−1.82 vs −2.04/−2.06/−1.77/−1.03); B-spline slope −1.77 undercuts the Regime-1 claim. [Agent 5, Agent 4 #15]
39. Project: push commit 671e7b7 (dual remotes); regenerate/remove the 4 stale pre-fix SCL files; fix IEC ST mojibake; document figures/final orphans. [Agent 7]

**MINOR** (polish — improves paper quality):

40. Agent 1's 24+ minor language items (dashes, separators, "0e 0w" expansion, § vs \S, vs/vs., EMSE undefined, "a actionable", "guarantee bound", sentence fragments in galois/2116, "Notably," ×4, self-correction meta-commentary consolidation, "This paper contributes:", "Four lines of work" with three items, hyphenation, Algorithm-3 ASCII arrows, "The Six SVNN Theorems" stale title, active voice).
41. Agent 2's 32 minor numerical inconsistencies (0.41³², split, E4 percentages 93.2% vs 63.4%, Z3 totals, DP-LUT cost ×3, fig04 d=32 values, etc.).
42. Agent 4's LaTeX formatting items (duplicate size commands, M₂ typesetting, c₅ as fraction, unreferenced equation cleanup ×36).
43. Agent 5's formatting items (size commands, float specifiers, uncaptioned tabular, E6: prefix, orphan figures, tab:wcet rounding).

---

*Report generated 2026-08-03 | 7-agent parallel review (contribution / claims / consistency / mathematics / tables-figures / language / project health) | All agents read the full ~9,000-line manuscript; Agent 4 independently recomputed all sharp constants numerically; Agent 7 executed the verification scripts against the repository.*
