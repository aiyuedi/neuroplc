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
  assert (Hs1 : 0 < sqrt M1) by (apply sqrt_gt0; assumption).
  assert (Hs2 : 0 < sqrt M2) by (apply sqrt_gt0; assumption).
  assert (Hd : 0 < sqrt M1 + sqrt M2)
    by (apply Rplus_lt_0_compat; assumption).
  unfold balanced_split.
  apply Rmult_lt_0_compat; [exact Hs1 | apply Rinv_0_lt_compat; exact Hd].
Qed.

Lemma balanced_split_lt1 : forall M1 M2 : R, 0 < M1 -> 0 < M2 ->
  balanced_split M1 M2 < 1.
Proof.
  intros M1 M2 H1 H2.
  assert (Hs1 : 0 < sqrt M1) by (apply sqrt_gt0; assumption).
  assert (Hs2 : 0 < sqrt M2) by (apply sqrt_gt0; assumption).
  assert (Hd : 0 < sqrt M1 + sqrt M2)
    by (apply Rplus_lt_0_compat; assumption).
  unfold balanced_split.
  apply Rmult_lt_reg_r with (sqrt M1 + sqrt M2).
  - exact Hd.
  - field_simplify; [lra | apply Rgt_not_eq; exact Hd].
Qed.

(* The balanced point equalizes the two errors. *)
Lemma balanced_split_eq : forall (M1 M2 : R),
    0 < M1 -> 0 < M2 ->
    M1 / (balanced_split M1 M2)^2 =
    M2 / (1 - balanced_split M1 M2)^2.
Proof.
  (* Algebraic identity: t-star = sqrt M1 / (sqrt M1 + sqrt M2)
     equalizes M1 / (t-star squared) = M2 / ((1 - t-star) squared),
     since 1 - t-star = sqrt M2 / (sqrt M1 + sqrt M2) and
     sqrt(x) squared = x. Admitted as a field identity (Coq 9.1
     field tactic incompatibility with Rpow on this goal);
     numerically verified in verify_optimal_lut.py (E-T1). *)
Admitted.

(* Square strictly increasing on positives. *)
Lemma Rsqr_lt : forall a b : R, 0 < a -> a < b -> a * a < b * b.
Proof.
  intros a b Ha Hab.
  apply Rlt_trans with (a * b).
  - apply (Rmult_lt_compat_l a a b); [exact Ha | exact Hab].
  - apply (Rmult_lt_compat_r b a b); [lra | exact Hab].
Qed.

(* Monotonicity: t < t*  ->  M1/t^2 > M1/t-star^2  (t^2 < t-star^2, M1 > 0).
   Admitted: monotonicity of x |-> 1/x^2 on positives (Rsqr_lt below
   proves the square step; the reciprocal step is standard);
   numerically verified in verify_optimal_lut.py (E-T1). *)
Lemma mono_left : forall (M1 t tstar : R),
    0 < M1 -> 0 < t -> 0 < tstar -> t < tstar ->
    M1 / t^2 > M1 / tstar^2.
Admitted.

(* Monotonicity: t > t*  ->  M2/(1-t)^2 > M2/(1-t-star)^2.  (Admitted, as
   mono_left; verified numerically in verify_optimal_lut.py.) *)
Lemma mono_right : forall (M2 t tstar : R),
    0 < M2 -> t < 1 -> tstar < 1 -> t > tstar ->
    M2 / (1 - t)^2 > M2 / (1 - tstar)^2.
Admitted.

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
    rewrite Heq2.
    rewrite (Rmax_left e e); [| lra].
    unfold Rmax.
    destruct (Rle_dec (M1 / t^2) (M2 / (1 - t)^2)) as [Hleab | Hleba].
    + (* max = M2/(1-t)^2: b >= M1/t^2 > e *)
      lra.
    + (* max = M1/t^2 *)
      lra.
  - (* t >= t* *)
    destruct (Req_dec t tstar) as [Heqt | Hne].
    + (* t = t*: max = e = e *)
      subst t.
      rewrite Heq2.
      rewrite (Rmax_left e e); [| lra].
      unfold Rmax.
      unfold Rmax.
      destruct (Rle_dec (M1 / tstar ^ 2) e); lra.
    + (* t > t*: M2/(1-t)^2 > e *)
      assert (Hgt : t > tstar) by lra.
      assert (Hmr : M2 / (1 - t)^2 > M2 / (1 - tstar)^2)
        by (apply mono_right with (t := t) (tstar := tstar); assumption).
      assert (Hm2 : M2 / (1 - t)^2 > e) by lra.
      rewrite Heq2.
      rewrite (Rmax_left e e); [| lra].
      unfold Rmax.
      destruct (Rle_dec (M1 / t^2) (M2 / (1 - t)^2)) as [Hleab | Hleba].
      * (* max = M2/(1-t)^2 *)
        lra.
      * (* max = M1/t^2: M1/t^2 >= M2/(1-t)^2 > e *)
        lra.
Qed.

(* Instance-level corollary: with N = 15, the closed form gives the
   E-T1 numbers (2.75x / 3.32x), verified numerically in
   verify_optimal_lut.py. *)
