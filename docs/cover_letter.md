# Cover Letter — IEEE TNNLS

**Manuscript title**: A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation

Dear Editor,

We submit our manuscript on a type-theoretic characterization of *which* neural architectures are certifiable by construction, and at what cost. The paper answers a question that the neural-network verification literature has not posed: rather than verifying a given network, we characterize the architecture class for which **design-time certification is possible, sharp, and cheap** — with an industrial instantiation that compiles such networks to programmable logic controllers (PLCs).

**Theoretical contributions (four new theorems, all with numerical verification scripts):**

1. **Sharp LUT constants and curvature-aware optimality (Thm 9, Thm 13).** We derive the sharp k-th order lookup-table constants c_k = Q_k/k! (c₃ = 1/(9√3), c₄ = 1/24; extremal family attains ratio 1.000000), and prove the two-level optimal allocation for LUT compilation: cross-function node allocation n_i ∝ M₂_i^{1/2} (2.75× improvement on the released 512-activation checkpoint) and within-function node density ρ ∝ √M₂(x) (3.32×), with a measurable correction of the compiler's previous curvature density (which was provably worse than uniform).

2. **Complexity separation (Thm 5).** Self-contained product-gate 3-SAT reduction: ε-verification is polynomial for the SVNN class and NP-hard for coupled architectures — with the Pareto point (poly, zero slack) under ETH.

3. **Compile-aware minimax deployment rate (Thm 14, Thm 15).** The deployment risk obeys R(n,N) ≍ max{n^{−2k/(2k+1)}, N^{−2k}}: a sample-budget *scissors* regime (budget-starved samples are wasted — verified flat to 0.3%) and the *free-discretization* statement (LUT compilation with adequate budget does not degrade the statistical minimax rate). The budget curve extends to smoothness-adaptive classes: bias super-converges N^{−2 min(s,k)} and smooth data compiles cheaply. The relation to the classical bandwidth law is stated explicitly and honestly.

4. **Verifiability trichotomy with necessity first lattice (Thm 10, Thm 16).** SVNN is the unique regime where second-order rate, linear LUT cost, sharp constants, and reachability-freedom coexist; we prove the first lattice of the necessity direction in the affine-certificate class (gates essentially C², no product coupling, sharp constant forces optimal allocation), and state the residual formalization as an open problem.

**Industrial instantiation**: a verified PyTorch→IEC 61131-3 compiler with four-tier self-verification (templates/certificates/composition formally verified; emitter validated by differential self-testing — which caught a real regression), vendor-neutral ST validated on a third-party toolchain (Inovance EVO810), and 74 experiments on bearing-fault diagnosis. All reported bounds were recomputed and verified against the released checkpoint.

**Honest limitations** (stated in the paper): the deployed checkpoint is non-contractive (γ = [15.4, 5.3]), giving a thin 1.0× DA safety margin — which we report as the empirical motivation for contractive training as the deployment recipe; the trichotomy's full necessity is a conjecture (first lattice proven); validation is PLCSIM/Z3-based without physical PLC measurement.

We believe the paper is a strong fit for TNNLS's theory-oriented scope, and we have taken unusual care to ensure every number in the manuscript is reproducible from the released artifacts.

Sincerely,
[Author names]
