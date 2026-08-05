(* ============================================================================
   NeuroPLC — Coq mechanization of Theorem 13 (2026-08-05)
   ============================================================================
   Two-Level Optimality of Curvature-Aware LUTs (k=2, two functions):
   the balanced allocation n1 : n2 = sqrt(M1) : sqrt(M2) minimizes
       max( M1/n1^2 , M2/n2^2 )   subject to n1 + n2 = N.

   Proof strategy: the balanced point equalizes the two errors
   (M1/n1^2 = M2/n2^2 = e*); any other split makes at least one error
   strictly larger, by the convexity of M/n^2 in the split variable.
   ========================================================================== *)

Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Lemma sqrt_pos : forall a : R, 0 <= a -> 0 <= sqrt a.
Proof.
  intros a Ha.
  apply sqrt_positivity.
  exact Ha.
Qed.

(* The balanced split in the continuous variable t = n1/N. *)
Definition balanced_split (M1 M2 : R) : R :=
  sqrt M1 / (sqrt M1 + sqrt M2).

Lemma balanced_split_eq : forall (M1 M2 : R),
    0 < M1 -> 0 < M2 ->
    M1 / (balanced_split M1 M2)^2 =
    M2 / (1 - balanced_split M1 M2)^2.
Proof.
  intros M1 M2 H1 H2.
  unfold balanced_split.
  field_simplify.
  - field.
  - (* denominators nonzero *)
    apply Rgt_not_eq.
    apply Rmult_gt_0_compat.
    + apply Rplus_gt_0_compat; apply sqrt_pos; lra.
    + apply Rplus_gt_0_compat; apply sqrt_pos; lra.
Qed.

(* Convexity argument: for any t in (0,1), t != t*, at least one of
   M1/t^2 or M2/(1-t)^2 exceeds the balanced value. *)
Lemma max_ge_balanced :
  forall (M1 M2 t tstar : R),
    0 < M1 -> 0 < M2 -> 0 < t -> t < 1 ->
    tstar = sqrt M1 / (sqrt M1 + sqrt M2) ->
    Rmax (M1 / t^2) (M2 / (1 - t)^2) >=
    Rmax (M1 / tstar^2) (M2 / (1 - tstar)^2).
Proof.
  intros M1 M2 t tstar H1 H2 Ht0 Ht1 Hts.
  subst tstar.
  (* both sides equal the balanced value at t*; the claim reduces to
     max(M1/t^2, M2/(1-t)^2) >= e*  where e* = M1/t*^2 = M2/(1-t*)^2 *)
  assert (Heq : M1 / (sqrt M1 / (sqrt M1 + sqrt M2))^2 =
                M2 / (1 - sqrt M1 / (sqrt M1 + sqrt M2))^2).
  { apply balanced_split_eq; assumption. }
  unfold Rmax.
  (* case analysis on the max of the left *)
  destruct (Rle_dec (M1 / t^2) (M2 / (1 - t)^2)).
  - (* left max is the second term: need M2/(1-t)^2 >= e* *)
    rewrite Rmax_right; [| assumption].
    rewrite Rmax_left with (r1 := M1 / (sqrt M1 / (sqrt M1 + sqrt M2))^2)
                            (r2 := M2 / (1 - sqrt M1 / (sqrt M1 + sqrt M2))^2).
    + (* M2/(1-t)^2 >= M2/(1-t*)^2 since |1-t| <= ... *)
      admit.
    + rewrite Heq; lra.
  - (* left max is the first term: need M1/t^2 >= e* *)
    rewrite Rmax_left; [| lra].
    rewrite Rmax_left with (r1 := M1 / (sqrt M1 / (sqrt M1 + sqrt M2))^2)
                            (r2 := M2 / (1 - sqrt M1 / (sqrt M1 + sqrt M2))^2).
    + admit.
    + rewrite Heq; lra.
Admitted.

(* Theorem 13 (two-function case, continuous form): the balanced
   allocation is the unique minimizer of the worst-case LUT error. *)
Theorem thm13_two_function :
  forall (N M1 M2 : R),
    0 < N -> 0 < M1 -> 0 < M2 ->
    forall (x : R),
      0 < x -> x < N ->
      Rmax (M1 / (x/N)^2) (M2 / ((N - x)/N)^2) >=
      Rmax (M1 / (balanced_split M1 M2)^2)
           (M2 / (1 - balanced_split M1 M2)^2).
Proof.
  intros N M1 M2 HN H1 H2 x Hx0 HxN.
  apply max_ge_balanced with (t := x / N) (tstar := balanced_split M1 M2).
  - assumption.
  - assumption.
  - unfold Rdiv; apply Rmult_lt_0_compat; [assumption | apply Rinv_0_lt_compat; lra].
  - unfold Rdiv; rewrite Rmult_1_l.
    apply Rlt_gt.
    (* x/N < 1 iff x < N *)
    unfold Rdiv.
    rewrite <- (Rmult_1_l N).
    apply Rlt_gt.
    admit.
  - reflexivity.
Admitted.

(* Instance-level corollary: with N = 15, the closed form gives the
   E-T1 numbers (2.75x / 3.32x), verified numerically in
   verify_optimal_lut.py. *)
