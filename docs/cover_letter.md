# Cover Letter — IEEE TNNLS (v2, 2026-08-04)

**Manuscript title**: A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

Dear Editor,

We submit a manuscript that unifies the theory of *which* neural architectures are certifiable by construction under a single quantity — the **verifiable deployment capacity** C(B, V): the best certifiable deployment error achievable under a storage budget B and a certificate-verification budget V. The paper answers a question the neural-network verification literature has not posed: not "can we verify a given network" but "which architecture classes are certifiable, with what sharp constants, at what complexity, at what verification cost — and what cannot be certified, no matter how clever the compiler."

**Theoretical contributions (five theorem groups, all with numerical verification scripts):**

1. **Verifiable deployment capacity (Thm 17–19, new).** A universal packing line (metric entropy) bounds every certified deployment scheme; three verification strata — closed-form (certificate-free, verification cost decoupled from storage), kinked (rate-degraded), and structure-optimal (verification-exponential at the class level) — place every architecture class; a certificate proof system (following the proof-carrying-code and validated-numerics lineage: Gappa, Sollya, LeanCert) formalizes the verification budget under perfect deterministic soundness and yields a theorem-level information-theoretic necessity within the calculus. An architecture-as-code law decides between table and spectral codes from target smoothness (exponential slope matches the smoothness rate to 2%). The approximation-side rates (entropy, width, rate-distortion) are explicitly inputs, not claims — the contribution is the verification side.

2. **Sharp LUT constants and curvature-aware optimality (Thm 9, Thm 13).** Sharp k-th-order lookup-table constants c_k = Q_k/k! (extremal family attains ratio 1.000000) and the two-level optimal allocation for LUT compilation: cross-function n_i ∝ M₂_i^{1/2} (2.75×) and within-function ρ ∝ √M₂(x) (3.32×), correcting a provably-suboptimal compiler density. These reappear in the capacity framework as the stratum-1 capacity limit of the interpolation code.

3. **Complexity separation (Thm 5).** Self-contained product-gate 3-SAT reduction: ε-verification is polynomial for SVNN and NP-hard for coupled architectures — the stratum-3 mechanism of the capacity theorem.

4. **Compile-aware minimax deployment rates (Thm 14, Thm 15).** R(n,N) ≍ max{n^{−2k/(2k+1)}, N^{−2k}} with the sample-budget scissors and free-discretization statements; smoothness-adaptive extension N^{−2 min(s,k)}. These are the sample-dimension coordinate of the capacity surface.

5. **Verifiability trichotomy with necessity first lattice (Thm 10, Thm 16).** SVNN is the unique regime where second-order rate, linear LUT cost, sharp constants, and reachability-freedom coexist; the first lattice of the necessity direction is proved in the affine-certificate class, and the capacity framework adds the CP-relative information-theoretic form (Thm 17(5)). The absolute necessity is stated honestly as an open problem.

**Industrial instantiation**: a verified PyTorch→IEC 61131-3 compiler with four-tier self-verification (templates/certificates/composition formally verified; emitter validated by differential self-testing, which caught a real regression), vendor-neutral ST validated on a third-party toolchain (Inovance EVO810), and 74+ experiments on bearing-fault diagnosis. All bounds recomputed and verified against the released checkpoint; the new capacity experiments (E-T4/E-T5) are script-reproducible.

**Deployment certificates (closing the loop)**: adaptive LUT allocation lifts the DA safety factor to 3.6× at unchanged storage; soft-contractive training (E-T9, 98.5% acc) reaches a **two-tier sound certificate** — a first-principles theoretical tier of 2.34× (pure propagation, no empirical floor; IEEE-754 δ_fp32 ≤ 3×10⁻⁵) and a calibrated tier of 11.6× (float32, measured floor 0.058 vs 0.053) / 26× (float64 backend), expected tier 493×, Box-Continuation coverage 99.9%, Z3 512/512, full SCL compile chain, and depth extension to L=3 (6.1×). Bounded-amplitude bases purchase certificates nearly free: FourierKAN 4.9× at 99.96% accuracy, WaveletKAN 2.8× at 99.67% (0.03/0.26 pp cost vs. the B-spline 1.4 pp). The released checkpoint remains non-contractive (γ = [15.4, 5.3]) with a thin 1.0× expected-tier margin — the empirical motivation for, and contrast against, the contractive recipe.
**Honest limitations** (stated in the paper): the trichotomy's full necessity remains a conjecture (first lattice + CP-relative theorem proven); validation is PLCSIM/Z3-based without physical PLC measurement; layer-wise full contractivity (γ<1 in every layer) remains unreached — bounded-amplitude bases (Fourier/Wavelet) reach first-layer γ<1 at 99.96%/99.67% accuracy with near-free certificates, but the second layer's γ stays > 1.

We believe the paper's unified capacity framework — with an unusually honest boundary between its contributions and the classical approximation side — is a strong fit for TNNLS's theory-oriented scope.

Sincerely,
Fuyue Liu
Guilin University of Electronic Technology
