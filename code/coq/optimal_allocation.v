(* ============================================================================
   NeuroPLC — Coq mechanization of Theorem 13 (2026-08-05)
   ============================================================================
   Two-Level Optimality of Curvature-Aware LUTs (k=2, two functions):
   the balanced allocation n1 : n2 = sqrt(M1) : sqrt(M2) minimizes
       max( M1/n1^2 , M2/n2^2 )   subject to n1 + n2 = N.

   Proof: the balanced point equalizes the two errors; the functions
   t |-> M1/t^2 and t |-> M2/(1-t)^2 are strictly monotone on opposite
   sides of t*, so any other split makes at least one error strictly
   larger (convexity/monotonicity argument).
   ========================================================================== *)

Require Import Reals.
Require Import Psatz.

Open Scope R_scope.

Definition balanced_split (M1 M2 : R) : R :=
  sqrt M1 / (sqrt M1 + sqrt M2).

(* denominators are positive *)
Lemma sqrt_gt0 : forall a : R, 0 < a -> 0 < sqrt a.
Proof.
  intros a Ha.
  apply sqrt_lt_R0.
  exact Ha.
Qed.

Lemma denom_gt0 : forall M1 M2 : R, 0 < M1 -> 0 < M2 ->
  0 < sqrt M1 + sqrt M2.
Proof.
  intros M1 M2 H1 H2.
  apply Rplus_lt_0_compat; apply sqrt_gt0; assumption.
Qed.

Lemma balanced_split_gt0 : forall M1 M2 : R, 0 < M1 -> 0 < M2 ->
  0 < balanced_split M1 M2.
Proof.
  intros M1 M2 H1 H2.
  unfold balanced_split.
  apply Rmult_lt_0_compat.
  - apply sqrt_gt0; assumption.
  - apply Rinv_0_lt_compat.
    apply denom_gt0; assumption.
Qed.

Lemma balanced_split_lt1 : forall M1 M2 : R, 0 < M1 -> 0 < M2 ->
  balanced_split M1 M2 < 1.
Proof.
  intros M1 M2 H1 H2.
  unfold balanced_split.
  apply Rlt_gt.
  rewrite Rmult_1_l.
  apply Rlt_gt.
  (* sqrt M1 < sqrt M1 + sqrt M2 *)
  apply Rlt_le_trans with (sqrt M1 + sqrt M2).
  - apply Rlt_le_trans with (sqrt M1 + 0).
    + rewrite Rplus_0_r; lra.
    + apply Rplus_le_compat_l.
      apply Rlt_le.
      apply sqrt_gt0; assumption.
  - apply Rinv_le_contravar.
    + apply denom_gt0; assumption.
    + field.
      apply Rgt_not_eq.
      apply denom_gt0; assumption.
Qed.

(* The balanced point equalizes the two errors. *)
Lemma balanced_split_eq : forall (M1 M2 : R),
    0 < M1 -> 0 < M2 ->
    M1 / (balanced_split M1 M2)^2 =
    M2 / (1 - balanced_split M1 M2)^2.
Proof.
  intros M1 M2 H1 H2.
  unfold balanced_split.
  field_simplify.
  - field.
  - apply Rgt_not_eq.
    apply Rmult_gt_0_compat.
    + apply Rplus_gt_0_compat; apply sqrt_gt0; assumption.
    + apply Rplus_gt_0_compat; apply sqrt_gt0; assumption.
Qed.

(* Monotonicity: t < t*  ->  M1/t^2 > M1/t*^2  (t^2 < t*^2, M1 > 0). *)
Lemma mono_left : forall (M1 t tstar : R),
    0 < M1 -> 0 < t -> 0 < tstar -> t < tstar ->
    M1 / t^2 > M1 / tstar^2.
Proof.
  intros M1 t tstar H1 Ht Hts Hlt.
  unfold Rdiv.
  apply Rgt_gt_gt.
  - apply Rmult_lt_compat_l.
    + apply Rinv_0_lt_compat; apply Rinv_0_lt_compat; lra.
    + apply Rinv_lt_contravar.
      * apply Rmult_lt_0_compat; lra.
      * apply Rmult_lt_compat_l; lra.
  - apply Rmult_lt_compat_l.
    + apply Rinv_0_lt_compat; apply Rinv_0_lt_compat; lra.
    + apply Rinv_lt_contravar.
      * apply Rmult_lt_0_compat; lra.
      * apply Rmult_lt_compat_l; lra.
Qed.

(* Monotonicity: t > t*  ->  M2/(1-t)^2 > M2/(1-t*)^2. *)
Lemma mono_right : forall (M2 t tstar : R),
    0 < M2 -> t < 1 -> tstar < 1 -> t > tstar ->
    M2 / (1 - t)^2 > M2 / (1 - tstar)^2.
Proof.
  intros M2 t tstar H2 Ht Hts Hgt.
  assert (H1 : 0 < 1 - t) by lra.
  assert (H2s : 0 < 1 - tstar) by lra.
  assert (Hlt : 1 - t < 1 - tstar) by lra.
  unfold Rdiv.
  apply Rgt_gt_gt.
  - apply Rmult_lt_compat_l.
    + apply Rinv_0_lt_compat; apply Rinv_0_lt_compat; lra.
    + apply Rinv_lt_contravar.
      * apply Rmult_lt_0_compat; lra.
      * apply Rmult_lt_compat_l; lra.
  - apply Rmult_lt_compat_l.
    + apply Rinv_0_lt_compat; apply Rinv_0_lt_compat; lra.
    + apply Rinv_lt_contravar.
      * apply Rmult_lt_0_compat; lra.
      * apply Rmult_lt_compat_l; lra.
Qed.

(* Core theorem: the balanced split is the minimizer of the max error. *)
Theorem thm13_two_function :
  forall (M1 M2 t : R),
    0 < M1 -> 0 < M2 -> 0 < t -> t < 1 ->
    Rmax (M1 / t^2) (M2 / (1 - t)^2) >=
    Rmax (M1 / (balanced_split M1 M2)^2)
         (M2 / (1 - balanced_split M1 M2)^2).
Proof.
  intros M1 M2 t H1 H2 Ht0 Ht1.
  set (tstar := balanced_split M1 M2).
  assert (Hts0 : 0 < tstar) by (unfold tstar; apply balanced_split_gt0; assumption).
  assert (Hts1 : tstar < 1) by (unfold tstar; apply balanced_split_lt1; assumption).
  assert (Heq : M1 / tstar^2 = M2 / (1 - tstar)^2).
  { unfold tstar; apply balanced_split_eq; assumption. }
  (* the balanced value e* *)
  set (e := M1 / tstar^2).
  assert (Heq2 : M2 / (1 - tstar)^2 = e) by (rewrite <- Heq; reflexivity).
  (* show: Rmax(M1/t^2, M2/(1-t)^2) >= e  by dichotomy on t vs tstar *)
  destruct (Rlt_le_dec t tstar) as [Hlt | Hge].
  - (* t < t*: M1/t^2 > e *)
    assert (Hm1 : M1 / t^2 > e).
    { unfold e; unfold tstar.
      apply mono_left; assumption. }
    unfold Rmax.
    destruct (Rle_dec (M1 / t^2) (M2 / (1 - t)^2)).
    + (* max = M2/(1-t)^2; need >= e: by transitivity of >= via M1/t^2 <= ... *)
      rewrite Rmax_right; [| assumption].
      (* M2/(1-t)^2 >= M1/t^2 (from the Rle_dec) >= e *)
      apply Rle_trans with (M1 / t^2).
      * apply Rlt_le; assumption.
      * apply Rge_le; lra.
    + (* max = M1/t^2 *)
      rewrite Rmax_left; [| lra].
      apply Rlt_le; assumption.
  - (* t >= t* *)
    destruct (Req_dec t tstar) as [Heqt | Hne].
    + (* t = t*: max = e = e *)
      subst t.
      unfold Rmax.
      rewrite <- Heq.
      lra.
    + (* t > t*: M2/(1-t)^2 > e *)
      assert (Hgt : t > tstar) by lra.
      assert (Hm2 : M2 / (1 - t)^2 > e).
      { unfold e; unfold tstar.
        rewrite <- Heq.
        apply mono_right with (t := t) (tstar := tstar); assumption. }
      unfold Rmax.
      destruct (Rle_dec (M1 / t^2) (M2 / (1 - t)^2)).
      * rewrite Rmax_right; [| assumption].
        apply Rlt_le; assumption.
      * rewrite Rmax_left; [| lra].
        (* M1/t^2 >= M2/(1-t)^2 > e *)
        apply Rle_trans with (M2 / (1 - t)^2).
        -- apply Rge_le; lra.
        -- apply Rlt_le; assumption.
Qed.

(* Instance-level corollary: with N = 15, the closed form gives the
   E-T1 numbers (2.75x / 3.32x), verified numerically in
   verify_optimal_lut.py. *)
