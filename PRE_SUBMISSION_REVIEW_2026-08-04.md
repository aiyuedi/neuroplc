# Pre-Submission Referee Report

**Paper**: A Type Theory of Certifiable Neural Architectures: Sharp LUT Bounds, Complexity Separation, and Compile-Aware Generalization, with an Industrial PLC Instantiation
**Authors**: Fuyue Liu (GUET)
**Date**: 2026-08-04
**Review Standard**: Leading Field Journal (IEEE TNNLS-tier, theory-first; 6-agent parallel review + external lit-scout bib verification)

---

## Overall Assessment

The upgraded manuscript (87 pages, 19 theorems, capacity framework Thm 17-19 as the new apex) passed the hardest consistency check: all three capacity verification scripts match the paper's numbers exactly (KT ratio 1.000000; ordering 0.000288<0.000365<0.000558; slopes -1.11/-2.05; exp_slope=-ρ within 2.5%; V_B flat 14.6μs), the theorem numbering chain is intact end-to-end, and the necessity conjecture status is preserved at every mention. The principal strengths are the honest claim discipline (domain split yielding the approximation side to KT/Zhu-Lafferty/Shannon; conjecture labels everywhere) and the numerical integrity. The single most critical issue: the capacity section's mathematical core has four repairable but real defects — a parameterization mixup (bits vs nodes), a constant misattribution (c_fix < c_k contradicts the paper's own Thm 9'(3)), a partially circular necessity step with an unproven premise (class-level reduction asserted, not derived), and an invalid PCP inference — plus a submission-blocking stale-number block (4 tables + 5 figures + E56/E57 claims on pre-recomputation constants) and a hard format blocker (87 single-column pages vs TNNLS ~14-page limit).

**Preliminary Recommendation**: Revise before sending to referees (conditionally; two blockers: the capacity-section mathematical defects and the stale-number block; the length constraint is a separate submission-format blocker).

---

## 1. Contribution & Referee Assessment (A6)

Rating: **Significant** (paper as a whole) / **Incremental** (capacity layer alone). Decomposition of Thm 17: item 1 (packing line) classical-attributed; item 2 = paper's own sharp constants (elementary but correct; c_4=1/24 verified by hand) + a *measured* width comparison; item 3 classical C¹ width; item 4 hardness *asserted* to transfer (defect); item 5 (CP-relative necessity) is the only genuinely new item, with a premise gap. The unification is real but shallow: strata = trichotomy regimes coordinatized; V_B is a real new coordinate but carries no new exclusion; C(B,V) never formally defined; no theorem states what the capacity formulation yields that the trichotomy + packing + hardness do not.

Required [CRITICAL]: (1) formal definition of C(B,V) as a minimax over schemes; (2) closed-form identification of c_* (Euler-spline/n-width constants) or relabel as numerically suggested; (3) repair Thm 17(5) premise-computability + write class-level reduction or downgrade to conjecture; (4) (V_B, ε) plane figure (apex section has zero figures); (5) Thm 19 full smoothness range + closed-form crossover + experimental protocol.

Suggested [MAJOR]: locate existing verifiers (CROWN/DeepPoly/SOS) inside CP with V_B profiles; verify stratum profiles on real activations; theorem-pyramid figure; explicit boundary with information-based complexity; resolve numeric inconsistencies.

## 2. Unsupported Claims & Claim Discipline (A3)

No [CRITICAL]. Four [MAJOR] precision issues in the capacity section: (1) PCP[0,poly]=NP ⟹ V≥|π| does not follow (perfect soundness permits subset-checking verifiers) — replace with modeling convention + P≠NP weak statement; (2) **parameterization mixup**: B defined as bits but B^{-k} is the N-node rate — bits give exponentially stronger entropy bound; unify on node parameterization with explicit bit mapping (B = w·N); (3) c_* conflates entropy constant and free-node width constant — separate notation (c_ent / c_*), mark ordering as classical-consistent; (4) Thm 19 "for all budgets" too strong — "eventually dominates + at all measured budgets". Plus: headline classification-guarantee sentence (main.tex:1298) lacks in-domain + M₂-calibration qualifiers; stale-number block (tab:cert_thresholds rows 2-4, E56 0.064→0.66 argument collapse, E57 CROWN 5.45, fig02 "safety≥2×" on γ=0.182, Tier-3 0.2473/0.1196, 11.9×, E58) blocks submission.

## 3. Internal Consistency & Cross-Reference (A2)

No [CRITICAL]. All script-vs-text numbers verified exact (listed in Overall). Three [MAJOR]: (1) "4.4× scan-cycle margin" (4 locations) contradicts 25.9× (wcet_analysis.json confirms 25.9); (2) **Regime 4 (e^x-type) never placed in any stratum** — trichotomy has 4 regimes, capacity 3 strata, so "three strata place every architecture" overstates; (3) E-numbering: E-T4 missing, E-T series undefined in summary table, experiment count 74 excludes 5 E-T experiments. Minors: "2% match" → "within 2.5%"; intro 22×/62× on stale γ unqualified; Conclusion items 2/13 duplicate Thm 9 (13 should be Corollary 3); main_cn.tex mirror stale.

## 4. Mathematics, Equations & Notation (A4)

Four [MAJOR] mathematical errors: (1) **c_fix < c_k contradicts Thm 9'(3)**: LS projection is affine; for k=2 interpolation IS the optimal-recovery operator from exact node data — no affine scheme (LS included) has sup-norm constant below c_k; the width constant is smaller but not attained by any sample-based affine scheme. Also "Thm 13 attains c_fix" is wrong (Thm 13 attains the interpolation constant c_k). Fix: stratum-1 constant = c_k (affine-class minimax, Thm 9'(3)); c_* (width) in stratum 3. (2) **Thm 17(5) step (iv) false as stated**: interval propagation IS reachability-based AND closed-form AND polynomial; the true statement needs the tightness qualifier ("tight bound generation at constant c_* via exact reachability is NP-hard"); Condition 1 (univariate gates) never derived — conclusion overreaches. (3) **V defined twice inconsistently** (|π| vs steps); closure lemma is definitional/axiomatic, not a derived lemma. (4) **Class-level reduction asserted, never demonstrated** (Thm 5 hardens ε-Verify of a specific product-gate network, not free-node class-level bound generation); Thm 18(i) inherits unproven premise; "V_gen or V_B exponential" disjunction vague; V_arch=poly must be added to hypotheses.

Notation conflicts: **B** (capacity bits vs Thm 6 input bound B=3); **γ** (Galois concretization vs amplification [15.4,5.3]); minors N/k/M/m/n/c/ρ. Undefined: E-T* labels never defined in main.tex; Bronshtein no cite; "OSB in preparation" undefined; {f_M} M undefined. Theorem numbering scrambled in document order (Thm 12 does not exist; 5/6/14/15 appear before 7/9/13); recommend sequential renumbering (deferred, high-cost).

Verified correct: packing entropy argument (1-D); spectral tail e^{-cρN}; crossover logic; steps (ii)(iii) given (i); class-level qualifier of Thm 18(i) enforced; sharp constants consistent; all 17 capacity citation keys resolve.

## 5. Tables, Figures & Documentation (A5)

15 figures + 41 tables all exist, have captions, no undefined refs. [CRITICAL]: 4 tables conflict with recomputed constants — tab:cert_thresholds (IA 1.38 not a certificate; N≥22 needed), tab:scalability_grid (safety 584× vs ~1.0×, 30× discrepancy), tab:cross_domain ("Cond 3: Yes" vs γ>1), tab:compositional (0.1196/0.2473 vs 0.66/1.38); 5 figures carry stale FIXMEs (fig1_overview "WCET 22.67ms" vs 3.86ms), 4 of them never referenced; figure_list.md declares 24 figures but 15 exist (16 entries point to tables/algorithms; 5 real figures missing; filename mismatches) — must be rewritten. ~30 TODO/FIXME comments must be cleared. Capacity section (apex) has zero figures — strongly suggested: (V_B, ε) plane + decision-law phase transition.

## 6. Spelling, Grammar & Style (A1)

Capacity section prose is journal-grade (only MINOR polish). 4 [CRITICAL] in main.tex: Abstract "uniquely attains (polynomial time...)" ungrammatical parenthetical; Intro missing article "a structural necessity"; **"RBF-KAN and MNIST remain theoretical/planned" contradicts tab:cross_domain (MNIST 0.9859, E42)**; Conclusion capacity item fragment chain. 7 [MAJOR]: tautology; 4.4× vs 25.9×; **E41 architecture conflict (MLP [28,32,16,4] 0/48 vs [28,16,4] 0/16)**; zeugma "detected and validated the fix"; "first verified compiler" unqualified (emitter verified by differential testing, not formal proofs); "Six SVNN Theorems" heading stale; "Two findings" vs three items. ~35 MINOR + 9 style patterns (first-claim inventory ~30 instances; literal dashes; de Boor variants; TODO/FIXME residues; "honest" meta-voice 6×; number formats).

---

## Priority Action Items

**CRITICAL** (must fix — desk-reject or major-referee risk):
1. Capacity parameterization: bits vs nodes mixup (A3/A4) — unify on node parameterization with explicit bit mapping; fixes the B-symbol conflict with Thm 6 too (A4).
2. c_fix < c_k contradicts Thm 9'(3) (A4): stratum-1 constant = c_k (affine-class minimax); c_* (width) in stratum 3; delete "uniform-node projection gives c_fix" claim; re-scope verify_capacity.py interpretation.
3. Thm 17(5) repairs: premise-computability assumption explicit; step (iv) tightness qualifier; Condition 1 not claimed; V_arch=poly in hypotheses; V definition unified (proof length); closure lemma marked as modeling axiom (A4).
4. Class-level reduction downgraded to conjecture (A6/A3/A4, red line 3 executed): stratum-3 exponentiality conditional; Thm 18(i) inherits the conjecture.
5. PCP sentence removed/replaced with modeling-convention + P≠NP weak statement (A3/A4/A6).
6. Stale-number block: tab:cert_thresholds rows 2-4, tab:scalability_grid, tab:cross_domain, tab:compositional recomputed; E56 (0.064→0.66 collapses 15.3× claim), E57 CROWN 5.45, Tier-3 0.2473/0.1196, 11.9×, E58 margins; fig02 caption (A3/A5).
7. figure_list.md rewritten to the real 15-figure inventory; 4 unreferenced stale figures fixed or removed; fig1_overview WCET 22.67→3.86ms (A5).
8. Abstract: "RBF-KAN and MNIST remain theoretical/planned" corrected (MNIST validated E42); grammar fixes (A1).
9. C(B,V) formally defined as minimax over schemes; either used (C(B,V) ≍ min{packing, stratum}) or the "unifying quantity" framing adjusted (A4/A6).
10. "4.4×" → "25.9×" in 4 locations (A2/A1); E41 architecture adjudicated from experiment records (A1).

**MAJOR** (should fix):
11. Regime 4 placement in the strata (A2).
12. E-T numbering: E-T4 gap, summary-table rows, count 74→79 (A2/A4).
13. c_ent vs c_* notation separation; packing-line dimension (1-D stated) (A3/A4).
14. Thm 19: "for all budgets" softening; certified-error framing on A_ρ (interpolation is exponential there — Bernstein); N ∝ B note; crossover closed form (A4/A3).
15. Headline guarantee sentence qualifiers (in-domain, M₂ calibration) (A3).
16. Notation: γ (Galois→γ†), N/k/M/m/n/c/ρ conflicts (A4).
17. Literature: +8 entries (IBC/Rump/Neumaier/Trefethen/Boyd/Pinkus/Scaman/Fazlyab) + LeanCert @software; related-work boundary expanded; Bronshtein cite; "OSB in preparation" defined; E-T definitions in main.tex (A6/A4/lit-scout).
18. (V_B, ε) plane figure + decision-law figure for the apex section (A6/A5).
19. "first verified compiler" → qualified; first-claim inventory sweep (~30, A1).
20. Capacity-coordinates paragraph drops M_k — restore c·M_k B^{-k} forms (A4).

**MINOR**: A1 polish list (~35), A2 minor list (5), A4 minor list (~12), A3 minor list (6). See agent sections.

**DEFERRED (open)**: theorem sequential renumbering (A4 — high cost, low risk); 87-page length constraint vs TNNLS ~14 pages (A6 — submission-format blocker, strategy needed); 58 overfull hboxes in existing content.

*Report generated by 6-agent parallel review (A1-A6) + external bib verification, 2026-08-04.*
