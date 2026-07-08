#!/usr/bin/env python3
"""
E52: The Verification Blind Spot — "Why Empirical Testing Is Not Enough"
========================================================================
Answers the sharpest reviewer question a formal-methods paper must face:

    "Python-vs-SCL cross-validation already agrees to 1e-4 and 99.9%
     classification. If empirical testing passes, what does your SVNN
     bound *add*? Isn't the formal guarantee proving something the
     evidence already shows?"

Core claim: There exists a regime of LUT sparsity in which
    (a) empirical accuracy on the held-out test set stays HIGH
        (a practitioner's test suite would PASS), yet
    (b) the SVNN design-time safety factor DROPS BELOW 1
        (the formal guarantee correctly REFUSES to certify), and
    (c) an adversarial search inside the certified input domain
        finds a genuine misclassification that the test set missed.

If such a regime exists, then empirical testing is *unsound* as a
deployment gate: it green-lights a configuration that the SVNN bound
catches. This is the operational value of Level-2 verification over
Level-1 runtime testing (§ svnn-challenge).

Method:
    Sweep LUT points N = 4 .. 15. For each N:
      1. Empirical: LUT-patched forward pass over the FULL test set
         → test-set accuracy + Python-vs-LUT classification agreement.
      2. Formal:    SVNN/DA design-time worst-case perturbation
         → safety factor = min_margin / worst_case_perturbation.
      3. Adversarial: gradient-free search over [-3,3]^28 (the certified
         domain) for the input that maximizes logit perturbation
         → does a real flip exist that the test set did not contain?

    The "blind spot" is any N where accuracy_high AND safety_factor < 1.

Usage:
    python e52_verification_necessity.py
    python e52_verification_necessity.py --n-adv 20000
"""

from __future__ import annotations

import sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.student_kan import StudentKAN, _bspline_basis
from neuroplc.affine_verify import affine_verify_kan

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
STUDENT_DIR = PROJECT_ROOT / "results" / "student"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLITS_DIR = PROJECT_ROOT / "data" / "splits"
RESULTS_DIR = PROJECT_ROOT / "results" / "verification_necessity"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ARCH = [28, 16, 4]
INPUT_RANGE = (-3.0, 3.0)
M2_BOUND = 0.3           # conservative analytic bound, same as affine_verify default
N_SWEEP = list(range(4, 16))   # LUT points 4..15


# ---------------------------------------------------------------------------
# LUT-patched forward pass (shared with E50)
# ---------------------------------------------------------------------------

def build_lut_model(model: StudentKAN, n_lut: int, device) -> list:
    """Monkey-patch every KAN layer to use an n_lut-point piecewise-linear
    LUT for the spline path. Returns the list of original forwards so the
    caller can restore them."""
    saved = []
    for layer in model.kan_layers:
        saved.append(layer.forward)
        grid = layer.grid
        out_d, in_d = layer.spline_weight.shape[:2]
        lut_x = torch.linspace(INPUT_RANGE[0], INPUT_RANGE[1], n_lut, device=device)
        with torch.no_grad():
            basis = _bspline_basis(lut_x / 3.0, grid, layer.spline_order)
            lv = torch.einsum('oic,pc->oip', layer.spline_weight, basis)

        bw, sb, ss = layer.base_weight, layer.scale_base, layer.scale_spline
        lx_f, lv_f = lut_x.clone(), lv.clone()

        def make_lut_fw(bw, sb, ss, lx, lv, od, id_):
            def lut_fw(x):
                base_out = F.silu(x)
                base_w = torch.einsum('...i,ji->...j', base_out, bw)
                xn = x.detach().cpu().numpy().astype(np.float32)
                ln = lx.cpu().numpy().astype(np.float32)
                vn = lv.cpu().numpy().astype(np.float32)
                B = xn.reshape(-1, id_).shape[0]
                sp = np.zeros((B, od), dtype=np.float32)
                for o in range(od):
                    for i in range(id_):
                        sp[:, o] += np.interp(xn.reshape(B, id_)[:, i], ln, vn[o, i])
                st = torch.from_numpy(sp.astype(np.float32)).reshape(xn.shape[:-1] + (od,))
                if x.device.type != 'cpu':
                    st = st.to(x.device)
                return sb * base_w + ss * st
            return lut_fw

        layer.forward = make_lut_fw(bw, sb, ss, lx_f, lv_f, out_d, in_d)
    return saved


def restore_model(model: StudentKAN, saved: list):
    for i, layer in enumerate(model.kan_layers):
        layer.forward = saved[i]


# ---------------------------------------------------------------------------
# Adversarial search inside the certified domain
# ---------------------------------------------------------------------------

def adversarial_flip_search(model: StudentKAN, n_lut: int, device,
                            n_adv: int = 20000, seed: int = 7) -> dict:
    """Random search over [-3,3]^28 for an input where the LUT model's
    argmax differs from the FP32 model's argmax. Returns the worst
    (largest margin-violating) case found. This probes the *entire*
    certified input domain, not just the natural test distribution."""
    rng = np.random.RandomState(seed)
    X = rng.uniform(INPUT_RANGE[0], INPUT_RANGE[1],
                    size=(n_adv, 28)).astype(np.float32)
    X_t = torch.from_numpy(X).to(device)

    with torch.no_grad():
        fp32 = model(X_t).cpu().numpy()
    fp32_pred = fp32.argmax(1)

    saved = build_lut_model(model, n_lut, device)
    with torch.no_grad():
        lut = model(X_t).cpu().numpy()
    restore_model(model, saved)
    lut_pred = lut.argmax(1)

    flips = np.where(fp32_pred != lut_pred)[0]
    per_sample_err = np.abs(fp32 - lut).max(axis=1)

    return {
        "n_adv": int(n_adv),
        "n_flips": int(len(flips)),
        "flip_rate": float(len(flips) / n_adv),
        "max_logit_error": float(per_sample_err.max()),
        "found_counterexample": bool(len(flips) > 0),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-adv", type=int, default=20000,
                    help="adversarial search samples per N")
    args = ap.parse_args()

    print("=" * 72)
    print("E52: The Verification Blind Spot — Why Empirical Testing Is Not Enough")
    print("=" * 72)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    ckpt = torch.load(STUDENT_DIR / "kan_kd_vrmKD_best.pt",
                      map_location=device, weights_only=True)
    model = StudentKAN(ARCH).to(device)
    model.load_state_dict(ckpt["student_state_dict"])
    model.eval()

    # ── Load test set ──
    X_feat = np.load(PROCESSED_DIR / "features_X.npy").astype(np.float32)
    y = np.load(PROCESSED_DIR / "features_y.npy").astype(np.int64)
    test_idx_path = SPLITS_DIR / "standard" / "test_idx.npy"
    if test_idx_path.exists():
        test_idx = np.load(test_idx_path)
        X_test, y_test = X_feat[test_idx], y[test_idx]
    else:
        # fall back to full set
        X_test, y_test = X_feat, y
    print(f"\n  Test set: {len(X_test)} samples, {len(np.unique(y_test))} classes")

    X_test_t = torch.from_numpy(X_test).to(device)

    # ── FP32 reference (the "ground truth" the compiled model should match) ──
    with torch.no_grad():
        fp32_logits = model(X_test_t).cpu().numpy()
    fp32_preds = fp32_logits.argmax(1)
    fp32_acc = float(np.mean(fp32_preds == y_test))
    print(f"  FP32 test accuracy: {fp32_acc*100:.2f}%")

    rows = []
    print(f"\n  {'N':>3} | {'test_acc':>8} | {'agree':>7} | "
          f"{'worst_pert':>10} | {'margin':>7} | {'safety':>8} | "
          f"{'adv_flips':>9} | verdict")
    print("  " + "-" * 88)

    for n_lut in N_SWEEP:
        # ---- (1) EMPIRICAL: LUT-patched accuracy on the real test set ----
        saved = build_lut_model(model, n_lut, device)
        with torch.no_grad():
            lut_logits = model(X_test_t).cpu().numpy()
        restore_model(model, saved)
        lut_preds = lut_logits.argmax(1)

        lut_acc = float(np.mean(lut_preds == y_test))
        agreement = float(np.mean(lut_preds == fp32_preds))

        # ---- (2) FORMAL: SVNN/DA design-time safety factor ----
        vres = affine_verify_kan(
            model, lut_points=n_lut, x_range=INPUT_RANGE,
            m2_bound=M2_BOUND,
            test_logits=fp32_logits, test_labels=y_test)
        worst_pert = vres.worst_case_perturbation
        margin = vres.min_interclass_margin
        safety = vres.safety_factor

        # ---- (3) ADVERSARIAL: search the certified domain for a real flip ----
        adv = adversarial_flip_search(model, n_lut, device, n_adv=args.n_adv)

        # ---- Blind-spot logic ----
        test_passes = (lut_acc >= 0.99 * fp32_acc) and (agreement >= 0.99)
        formal_certifies = safety >= 1.0
        blind_spot = test_passes and (not formal_certifies)

        if blind_spot and adv["found_counterexample"]:
            verdict = "*** BLIND SPOT (test PASS, formal FAIL, real flip exists)"
        elif blind_spot:
            verdict = "** BLIND SPOT (test PASS, formal FAIL)"
        elif not formal_certifies:
            verdict = "formal FAIL (test also fails)" if not test_passes else ""
        else:
            verdict = "both OK"

        rows.append({
            "n_lut": n_lut,
            "test_acc": lut_acc,
            "fp32_acc": fp32_acc,
            "agreement": agreement,
            "worst_case_perturbation": worst_pert,
            "min_margin": margin,
            "safety_factor": safety,
            "lut_error_bound": vres.lut_error_bound,
            "adv_flips": adv["n_flips"],
            "adv_flip_rate": adv["flip_rate"],
            "adv_max_logit_error": adv["max_logit_error"],
            "test_passes": test_passes,
            "formal_certifies": formal_certifies,
            "blind_spot": blind_spot,
            "blind_spot_confirmed_by_adv": blind_spot and adv["found_counterexample"],
        })

        print(f"  {n_lut:>3} | {lut_acc*100:>7.2f}% | {agreement*100:>6.2f}% | "
              f"{worst_pert:>10.4f} | {margin:>7.3f} | {safety:>7.2f}x | "
              f"{adv['n_flips']:>9} | {verdict}")

    # ── Identify the blind-spot band ──
    blind_ns = [r["n_lut"] for r in rows if r["blind_spot"]]
    confirmed_ns = [r["n_lut"] for r in rows if r["blind_spot_confirmed_by_adv"]]

    print("\n  " + "=" * 70)
    if blind_ns:
        print(f"  BLIND-SPOT BAND: N ∈ {blind_ns}")
        print(f"    → empirical test PASSES but SVNN safety factor < 1")
        if confirmed_ns:
            print(f"  ADVERSARY-CONFIRMED: N ∈ {confirmed_ns}")
            print(f"    → a real misclassification exists in the certified domain")
            print(f"      that the natural test set did NOT contain")
        print(f"\n  CONCLUSION: Empirical testing is UNSOUND as a deployment gate.")
        print(f"  The SVNN design-time bound catches unsafe configurations that")
        print(f"  pass held-out accuracy testing. This is the operational value")
        print(f"  of Level-2 verification over Level-1 runtime testing.")
    else:
        print("  No blind spot found in the swept range — reporting the")
        print("  safety-factor curve and the N at which the guarantee is lost.")
        below = [r["n_lut"] for r in rows if not r["formal_certifies"]]
        if below:
            print(f"  Formal guarantee lost at N ≤ {max(below)}.")

    report = {
        "experiment": "E52",
        "title": "The Verification Blind Spot — Why Empirical Testing Is Not Enough",
        "arch": ARCH,
        "m2_bound": M2_BOUND,
        "input_range": list(INPUT_RANGE),
        "n_test_samples": int(len(X_test)),
        "fp32_test_accuracy": fp32_acc,
        "n_adv_per_point": args.n_adv,
        "sweep": rows,
        "blind_spot_band": blind_ns,
        "blind_spot_confirmed_by_adversary": confirmed_ns,
        "conclusion": (
            "Empirical testing is unsound as a deployment gate: there exists "
            "an LUT-sparsity regime where held-out accuracy stays high but the "
            "SVNN design-time safety factor drops below 1, and an adversarial "
            "search inside the certified domain finds a real misclassification "
            "the test set missed."
            if confirmed_ns else
            "Safety-factor curve characterized; formal guarantee boundary reported."
        ),
    }
    out = RESULTS_DIR / "verification_necessity.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Results saved: {out}")
    print("=" * 72)


if __name__ == "__main__":
    main()
