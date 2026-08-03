#!/usr/bin/env python3
"""
E59: Thm 6' (Compile-Aware Generalization Bound) Verification
=============================================================
Verifies, empirically, the claim

    R(N-bar) = R_hat(N) + O(gamma^{L-1} / sqrt(n)) + O(gamma^{L-1} . c_k . M_k . h^k)

for a KAN[28,16,4] compiled to PLC LUTs (linear interpolation on N points of
[-3,3], cell width h = 6/(N-1)), where:
  R(N-bar)      true risk of the DEPLOYED (LUT-compiled) network,
  R_hat(N)      empirical risk of the trained (uncompiled) network,
  gamma         per-layer contractivity (empirical Lipschitz),
  n             number of training samples,
  k = 2         linear interpolation, c_2 = 1/8,
  M_k           max |phi''| of the B-spline activations.

Key corollary under test — resolution matching law:
  N* ~ n^{1/(2k)} = n^{1/4}    (bias(N) balances gap ~ 1/sqrt(n))

Part A — Decomposition validity (pretrained VRM-KD student, held-out test)
Part B — Resolution-matching law (freshly trained at n in {100,400,1600,6400})
Part C — Depth scaling: 1-layer vs 2-layer gap/bias growth (gamma^{L-1})

Everything numpy/torch only. Honest reporting: measured numbers, no fudging.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sklearn.model_selection import train_test_split           # only for the split
from models.student_kan import StudentKAN, _bspline_basis      # noqa: E402

PROJECT_ROOT = ROOT.parent
CKPT = PROJECT_ROOT / "results" / "student" / "kan_kd_vrmKD_best.pt"
DATA = PROJECT_ROOT / "data" / "processed"
OUT_DIR = PROJECT_ROOT / "results" / "theory"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = OUT_DIR / "thm6p_compile_aware.json"

ARCH = [28, 16, 4]
XR = (-3.0, 3.0)
C_K = 1.0 / 8.0          # c_2 for linear interpolation error bound
H_DOM = XR[1] - XR[0]    # 6.0
NS_A = [8, 16, 32, 64, 128, 256]                      # Part A sweep
NS_B = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]     # Part B sweep
N_LIST = [100, 400, 1600, 6400]
SEEDS = {100: [0, 1, 2], 400: [0, 1], 1600: [0, 1], 6400: [0, 1]}
EPOCHS = {100: 400, 400: 300, 1600: 220, 6400: 180}
BATCH = {100: 64, 400: 64, 1600: 128, 6400: 128}
PATIENCE = 50

torch.set_num_threads(max(4, torch.get_num_threads()))


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load_and_split():
    X = np.load(DATA / "features_X.npy").astype(np.float32)
    y = np.load(DATA / "features_y.npy").astype(np.int64)
    idx = np.arange(len(y))
    # replicate create_dataloaders (train_student_kd.py): test 20%, val 10% of rest
    train_val, test_idx = train_test_split(idx, test_size=0.2, stratify=y,
                                           random_state=42)
    train_idx, val_idx = train_test_split(
        train_val, test_size=0.125, stratify=y[train_val], random_state=42)
    return X, y, train_idx, val_idx, test_idx


def clip_x(X):
    return np.clip(X, XR[0], XR[1])


# ----------------------------------------------------------------------------
# Risk metrics
# ----------------------------------------------------------------------------
def risks(logits_np, y_np):
    """0-1 misclassification risk and mean softmax cross-entropy risk."""
    pred = logits_np.argmax(1)
    err01 = float((pred != y_np).mean())
    p = np.exp(logits_np - logits_np.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    ce = float(-np.log(p[np.arange(len(y_np)), y_np] + 1e-12).mean())
    return {"err01": err01, "ce": ce}


def fp32_logits(model, X_np, batch=512):
    """Reference (uncompiled) forward, batched fp32, matching deployed training."""
    out = []
    with torch.no_grad():
        for b in range(0, len(X_np), batch):
            out.append(model(torch.from_numpy(X_np[b:b + batch])).numpy())
    return np.concatenate(out, 0)


# ----------------------------------------------------------------------------
# LUT compilation simulation (mirrors e53_sound_worstcase.lut_forward semantics)
# ----------------------------------------------------------------------------
def build_lut_tables(model, n_lut):
    """Uniform grid on [-3,3] with N points; per-activation table y = phi(grid/3).
    Backend semantics: largest grid[j] <= x, linear interpolate (np.interp)."""
    tables = []
    for layer in model.kan_layers:
        grid = layer.grid.detach().double()
        coeffs = layer.spline_weight.detach().double()
        lut_grid = np.linspace(XR[0], XR[1], n_lut)
        lut_grid_s = torch.from_numpy(lut_grid).double() / 3.0
        basis = _bspline_basis(lut_grid_s, grid, k=layer.spline_order)
        ys = torch.einsum('n c, o i c -> n o i', basis, coeffs).numpy()
        tables.append({"grid": lut_grid, "ys": ys,
                       "sb": float(layer.scale_base.detach()),
                       "ss": float(layer.scale_spline.detach()),
                       "wb": layer.base_weight.detach().numpy()})
    return tables


def lut_forward_batch(model, X_np, tables):
    """Vectorized LUT-compiled forward over a full data matrix (n, d0).
    X must be inside [-3,3] (caller clips). fp64 arithmetic like e53."""
    h = X_np.astype(np.float64)
    for t in tables:
        base = F.silu(torch.from_numpy(h).float()).numpy() @ t["wb"].T
        out_d, in_d = t["wb"].shape
        spline = np.zeros((len(h), out_d))
        for o in range(out_d):
            acc = np.zeros(len(h))
            for i in range(in_d):
                acc += np.interp(h[:, i], t["grid"], t["ys"][:, o, i])
            spline[:, o] = acc
        h = t["sb"] * base + t["ss"] * spline
    return h


# ----------------------------------------------------------------------------
# M_2 and per-activation interpolation error
# ----------------------------------------------------------------------------
def measure_m2(model, n_pts=4001):
    """Per-activation max |phi''| via central differences on a fine grid."""
    xs = torch.linspace(*XR, n_pts, dtype=torch.float64)
    xs_s = xs / 3.0
    dx = float(xs[1] - xs[0])
    m2_layers = []
    for layer in model.kan_layers:
        grid = layer.grid.detach().double()
        coeffs = layer.spline_weight.detach().double()
        basis = _bspline_basis(xs_s, grid, k=layer.spline_order)
        phi = torch.einsum('n c, o i c -> n o i', basis, coeffs).numpy()
        d1 = np.gradient(phi, dx, axis=0)
        d2 = np.gradient(d1, dx, axis=0)
        m2_layers.append(np.abs(d2).max(0))          # (out, in)
    return m2_layers


def phi_all(model, X_np):
    """Exact spline sum per activation for all samples (float64)."""
    h = X_np.astype(np.float64)
    out = []
    for layer in model.kan_layers:
        grid = layer.grid.detach().double()
        coeffs = layer.spline_weight.detach().double()
        basis = _bspline_basis(torch.from_numpy(h).double() / 3.0, grid,
                               k=layer.spline_order)
        out.append(torch.einsum('n i c, o i c -> n o i',
                                basis, coeffs).numpy())
    return out


# ----------------------------------------------------------------------------
# Empirical per-layer Lipschitz (contractivity) on the domain
# ----------------------------------------------------------------------------
def layer_map(layer, x):
    sb = layer.scale_base
    ss = layer.scale_spline
    base = torch.einsum('n i, o i -> n o', F.silu(x), layer.base_weight)
    basis = _bspline_basis(x / 3.0, layer.grid, layer.spline_order)
    spl = torch.einsum('n i c, o i c -> n o', basis, layer.spline_weight)
    return sb * base + ss * spl


def layer_lipschitz(model, n_pts=200, seed=0):
    """Sup over sampled domain points of the L_inf induced Jacobian norm:
    max_j sum_i |d f_j / d x_i|. torch.func if available, else central diff."""
    lips = []
    for layer in model.kan_layers:
        d = layer.in_features
        rng = np.random.RandomState(seed)
        pts = rng.uniform(XR[0], XR[1], (n_pts, d)).astype(np.float32)
        x = torch.from_numpy(pts)
        try:
            from torch.func import jacrev, vmap
            jac = vmap(jacrev(lambda z: layer_map(layer, z)))(x)
            rowsum = jac.abs().sum(-1)                 # (n_pts, out)
            lips.append(float(rowsum.max()))
        except Exception:
            # central differences along coordinate axes at sampled points
            best = 0.0
            with torch.no_grad():
                f0 = layer_map(layer, x)
                for i in range(min(d, 40)):
                    xp = x.clone(); xp[:, i] += 1e-3
                    xm = x.clone(); xm[:, i] -= 1e-3
                    fd = (layer_map(layer, xp) - layer_map(layer, xm)) / 2e-3
                    best = max(best, float(fd.abs().sum(-1).max()))
            lips.append(best)
    return lips


# ----------------------------------------------------------------------------
# Propagated LUT bias bound at logits: O(gamma^{L-1} c_k M_k h^k)
# ----------------------------------------------------------------------------
def propagated_bias_bound(model, m2_layers, n_lut):
    """Sound-ish bound: per-activation eps = c_2*M_2*h^2, propagate through the
    remaining layers with their empirical Lipschitz (linf)."""
    h = H_DOM / (n_lut - 1)
    eps2 = C_K * h * h
    lips = layer_lipschitz(model)                       # per layer
    d1_max = 0.0
    d2_max = 0.0
    for li, layer in enumerate(model.kan_layers):
        ss = float(layer.scale_spline.detach())
        e = eps2 * m2_layers[li]                        # (out, in)
        per_out = ss * e.sum(1)                         # (out,)
        if li == 0:
            d1_max = float(per_out.max())
        else:
            d2_max = float(per_out.max())
    logit_bound = lips[-1] * d1_max + d2_max            # gamma * layer-1 err + layer-2 err
    return {"h": h, "eps_per_activation_bound": eps2,
            "layer1_err_max": d1_max, "layer2_err_max": d2_max,
            "gamma": lips,
            "logit_bias_bound": logit_bound,
            "ce_bias_bound": 2.0 * logit_bound}         # CE is 2-Lipschitz in linfinf


# ----------------------------------------------------------------------------
# Training (plain CE, Adam, cosine LR, early stop on val acc)
# ----------------------------------------------------------------------------
def train_model(Xt, yt, Xv, yv, arch, n_epochs, batch, seed, patience=PATIENCE):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = StudentKAN(arch)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)
    rng = np.random.RandomState(seed)
    Xt_t = torch.from_numpy(Xt); yt_t = torch.from_numpy(yt)
    Xv_t = torch.from_numpy(Xv); yv_t = torch.from_numpy(yv)
    best_val, best_sd, bad, n = -1.0, None, 0, len(Xt)
    for ep in range(n_epochs):
        m.train()
        perm = rng.permutation(n)
        for b in range(0, n, batch):
            idx = perm[b:b + batch]
            opt.zero_grad()
            loss = F.cross_entropy(m(Xt_t[idx]), yt_t[idx])
            loss.backward()
            opt.step()
        sched.step()
        m.eval()
        with torch.no_grad():
            va = float((m(Xv_t).argmax(1) == yv_t).float().mean())
        if va > best_val:
            best_val = va
            best_sd = {k: v.clone() for k, v in m.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    m.load_state_dict(best_sd)
    m.eval()
    return m, best_val


# ----------------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------------
def loglog_fit(xs, ys):
    """slope, se, r2 of log10(ys) ~ a + b*log10(xs) (xs, ys > 0)."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    lx, ly = np.log10(xs), np.log10(ys)
    A = np.vstack([lx, np.ones_like(lx)]).T
    (b, a), *_ = np.linalg.lstsq(A, ly, rcond=None)
    yhat = A @ np.array([b, a])
    resid = ly - yhat
    dof = max(len(lx) - 2, 1)
    s2 = float(resid @ resid) / dof
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[0, 0])) if len(lx) > 2 else np.nan
    r2 = 1.0 - float(resid @ resid) / max(float((ly - ly.mean()) @ (ly - ly.mean())), 1e-30)
    return b, se, r2


# ============================================================================
# Part A — decomposition validity (pretrained student)
# ============================================================================
def part_a(model, X_tr, y_tr, X_te, y_te, X_te_unc, y_te):
    print("=" * 72)
    print("PART A — decomposition validity (pretrained VRM-KD student)")
    print("=" * 72)
    m2_layers = measure_m2(model)
    m2_report = [{"layer": li, "M2_max": float(m.max()), "M2_mean": float(m.mean())}
                 for li, m in enumerate(m2_layers)]

    l_tr_u = fp32_logits(model, X_tr)
    l_te_u = fp32_logits(model, X_te)
    r_tr_u = risks(l_tr_u, y_tr)
    r_te_u = risks(l_te_u, y_te)
    gap_ce = r_te_u["ce"] - r_tr_u["ce"]
    gap_01 = r_te_u["err01"] - r_tr_u["err01"]
    # unclipped reference (honest note: deployment domain is [-3,3])
    l_te_u_unc = fp32_logits(model, X_te_unc)
    r_te_u_unc = risks(l_te_u_unc, y_te)

    rows = []
    for N in NS_A:
        tables = build_lut_tables(model, N)
        l_tr_lut = lut_forward_batch(model, X_tr, tables)
        l_te_lut = lut_forward_batch(model, X_te, tables)
        r_tr_lut = risks(l_tr_lut, y_tr)
        r_te_lut = risks(l_te_lut, y_te)
        logit_bias = np.abs(l_te_lut - l_te_u).max(1)
        h = H_DOM / (N - 1)
        rows.append({
            "N": N, "h": h,
            "R_train_uncomp": r_tr_u, "R_train_LUT": r_tr_lut,
            "R_test_uncomp": r_te_u, "R_test_LUT": r_te_lut,
            "bias_ce": r_te_lut["ce"] - r_te_u["ce"],
            "bias_01": r_te_lut["err01"] - r_te_u["err01"],
            "logit_bias_mean": float(logit_bias.mean()),
            "logit_bias_max": float(logit_bias.max()),
        })
        print(f"  N={N:4d} h={h:.4f} | R̂tr={r_tr_u['err01']:.4f} "
              f"Rte_LUT={r_te_lut['err01']:.4f} (CE {r_te_lut['ce']:.5f}) "
              f"bias_ce={rows[-1]['bias_ce']:+.2e} "
              f"logit_bias mean/max={rows[-1]['logit_bias_mean']:.2e}/"
              f"{rows[-1]['logit_bias_max']:.2e}")

    # --- fits: bias vs h^2 (log-log) ---
    Ns = np.array([r["N"] for r in rows])
    hs2 = np.array([(H_DOM / (N - 1)) ** 2 for N in Ns])
    lbm = np.array([r["logit_bias_mean"] for r in rows])
    lbx = np.array([r["logit_bias_max"] for r in rows])
    bias_ce = np.array([r["bias_ce"] for r in rows])
    excess_ce = np.maximum(bias_ce - bias_ce[-1], 1e-12)   # remove plateau (gap)

    fit_logit_mean = loglog_fit(hs2, lbm)
    fit_logit_max = loglog_fit(hs2, lbx)
    fit_excess = loglog_fit(hs2, excess_ce) if excess_ce[-1] > 0 else (np.nan, np.nan, np.nan)

    # --- sound decomposition inequality: R_ce(N) <= R_ce_uncomp + gap + 2*logit_bound ---
    bound = propagated_bias_bound(model, m2_layers, max(NS_A))
    bound_N = {N: propagated_bias_bound(model, m2_layers, N) for N in NS_A}
    ineq_rows = []
    for r in rows:
        bb = bound_N[r["N"]]
        rhs = r_te_u["ce"] + gap_ce + bb["ce_bias_bound"]
        lhs = r["R_test_LUT"]["ce"]
        ineq_rows.append({"N": r["N"], "lhs_ce": lhs, "rhs_ce": rhs,
                          "holds": bool(lhs <= rhs + 1e-12),
                          "logit_bound": bb["logit_bias_bound"],
                          "margin": rhs - lhs})

    a1_flat = all(abs(r["R_train_uncomp"]["ce"] - rows[0]["R_train_uncomp"]["ce"]) < 1e-12
                  for r in rows)
    a2_decr = all(bias_ce[i] <= bias_ce[i - 1] + 1e-12 for i in range(1, len(bias_ce))) \
              and bias_ce[-1] <= 0.15 * bias_ce[0] + 1e-12
    a3_slope = 0.5 <= fit_logit_mean[0] <= 1.5 and 0.5 <= fit_logit_max[0] <= 1.5
    a4_ineq = all(r["holds"] for r in ineq_rows)

    res = {
        "gap_ce": gap_ce, "gap_01": gap_01,
        "R_test_uncomp_clipped": r_te_u, "R_test_uncomp_unclipped": r_te_u_unc,
        "R_train_uncomp": r_tr_u,
        "m2": m2_report,
        "sweep": rows,
        "fits": {
            "logit_bias_mean_vs_h2": {"slope": fit_logit_mean[0],
                                       "se": fit_logit_mean[1], "r2": fit_logit_mean[2]},
            "logit_bias_max_vs_h2": {"slope": fit_logit_max[0],
                                      "se": fit_logit_max[1], "r2": fit_logit_max[2]},
            "excess_ce_risk_vs_h2": {"slope": fit_excess[0],
                                      "se": fit_excess[1], "r2": fit_excess[2]},
        },
        "bound_at_largest_N": bound,
        "inequality": {"rows": ineq_rows,
                        "all_hold_ce": bool(a4_ineq),
                        "note": "CE-risk inequality: R_ce(N) <= R_ce(uncomp) + measured_gap + 2*logit_bound"},
        "assertions": {
            "A1_Rhat_flat_in_N": bool(a1_flat),
            "A2_decreases_then_flattens": bool(a2_decr),
            "A3_bias_tracks_h2_slope_in_0.5_1.5": bool(a3_slope),
            "A4_decomposition_inequality_holds": bool(a4_ineq),
        },
    }
    print(f"  fits (slope of log bias vs log h^2): mean-logit={fit_logit_mean[0]:.2f} "
          f"max-logit={fit_logit_max[0]:.2f} excess-CE={fit_excess[0]:.2f}")
    print(f"  gap (CE) = {gap_ce:.4f} | A1 flat={a1_flat} A2 decr+flatten={a2_decr} "
          f"A3 slope={a3_slope} A4 ineq={a4_ineq}")
    return res


# ============================================================================
# Part B — resolution-matching law N* ~ n^{1/4}
# ============================================================================
def part_b(X, y, train_idx, val_idx, test_idx):
    print("=" * 72)
    print("PART B — V-curve / resolution matching law (retrained per n)")
    print("=" * 72)
    Xv, yv = clip_x(X[val_idx]), y[val_idx]
    Xte, yte = clip_x(X[test_idx]), y[test_idx]
    t0 = time.time()
    per_n = {}
    for n in N_LIST:
        nrows = []
        for seed in SEEDS[n]:
            rng = np.random.RandomState(seed)
            sub, _ = train_test_split(train_idx, train_size=n, stratify=y[train_idx],
                                      random_state=seed)
            Xt, yt = clip_x(X[sub]), y[sub]
            m, best_val = train_model(Xt, yt, Xv, yv, ARCH, EPOCHS[n], BATCH[n], seed)
            l_tr_u = fp32_logits(m, Xt)
            l_te_u = fp32_logits(m, Xte)
            r_tr_u = risks(l_tr_u, yt)
            r_te_u = risks(l_te_u, yte)
            gap_ce = r_te_u["ce"] - r_tr_u["ce"]
            gap_01 = r_te_u["err01"] - r_tr_u["err01"]
            series = []
            for N in NS_B:
                tables = build_lut_tables(m, N)
                l_te_lut = lut_forward_batch(m, Xte, tables)
                r_lut = risks(l_te_lut, yte)
                series.append({"N": N,
                               "R_test_LUT_err01": r_lut["err01"],
                               "R_test_LUT_ce": r_lut["ce"],
                               "bias_ce": r_lut["ce"] - r_te_u["ce"],
                               "bias_01": r_lut["err01"] - r_te_u["err01"]})
            bias_ce = np.array([s["bias_ce"] for s in series])
            err01 = np.array([s["R_test_LUT_err01"] for s in series])
            # knee N*: smallest N where LUT bias drops to (or below) the gap
            knee = NS_B[int(np.argmax(bias_ce <= max(gap_ce, 0.0)))]
            # 0-1 plateau N*: smallest N achieving the best (largest-N) 0-1 error
            best01 = err01[-1]
            nstar01 = NS_B[int(np.argmax(err01 == best01))]
            nrows.append({"seed": seed, "val_acc": best_val,
                          "R_train_uncomp": r_tr_u, "R_test_uncomp": r_te_u,
                          "gap_ce": gap_ce, "gap_01": gap_01,
                          "knee_N": knee, "nstar01": nstar01,
                          "series": series})
            print(f"  n={n:5d} seed={seed} | val={best_val:.3f} "
                  f"Rte={r_te_u['err01']:.4f} gap_ce={gap_ce:.4f} "
                  f"knee_N*={knee:3d} N*01={nstar01:3d}")
        per_n[n] = {"rows": nrows,
                    "gap_ce_mean": float(np.mean([r["gap_ce"] for r in nrows])),
                    "gap_01_mean": float(np.mean([r["gap_01"] for r in nrows])),
                    "knee_N_mean": float(np.mean([r["knee_N"] for r in nrows])),
                    "nstar01_mean": float(np.mean([r["nstar01"] for r in nrows]))}

    ns = np.array(N_LIST, float)
    knee = np.array([per_n[n]["knee_N_mean"] for n in N_LIST])
    n01 = np.array([per_n[n]["nstar01_mean"] for n in N_LIST])
    gap = np.array([max(per_n[n]["gap_ce_mean"], 1e-6) for n in N_LIST])

    fit_knee = loglog_fit(ns, knee)
    fit_n01 = loglog_fit(ns, n01)
    fit_gap = loglog_fit(ns, 1.0 / gap)   # gap ~ n^{-a}  =>  log(1/gap) ~ a log n
    print(f"  fit: knee N* ~ n^{fit_knee[0]:.2f}  (theory 0.25) | "
          f"N*01 ~ n^{fit_n01[0]:.2f} | gap ~ n^{-{fit_gap[0]:.2f}} (theory 0.5)")
    print(f"  elapsed {time.time() - t0:.0f}s")

    res = {"n_list": N_LIST,
           "per_n": {str(n): per_n[n] for n in N_LIST},
           "fits": {
               "knee_N_vs_n": {"slope": fit_knee[0], "se": fit_knee[1],
                               "r2": fit_knee[2], "theory": 0.25},
               "nstar01_vs_n": {"slope": fit_n01[0], "se": fit_n01[1],
                                "r2": fit_n01[2], "theory": 0.25},
               "gap_vs_n": {"slope_abs": fit_gap[0], "se": fit_gap[1],
                            "r2": fit_gap[2], "theory": 0.5},
           },
           "assertions": {
               "B1_knee_exponent_in_0.1_0.45": bool(0.10 <= fit_knee[0] <= 0.45),
               "B2_nstar01_exponent_in_0.1_0.45": bool(0.10 <= fit_n01[0] <= 0.45),
               "B3_gap_scales_n_minus_half_in_0.25_0.75": bool(0.25 <= fit_gap[0] <= 0.75),
           }}
    return res


# ============================================================================
# Part C — depth scaling (1-layer vs 2-layer)
# ============================================================================
def part_c(X, y, train_idx, val_idx, test_idx):
    print("=" * 72)
    print("PART C — depth scaling: 1-layer [28,4] vs 2-layer [28,16,4]")
    print("=" * 72)
    Xv, yv = clip_x(X[val_idx]), y[val_idx]
    Xte, yte = clip_x(X[test_idx]), y[test_idx]
    out = {}
    for n in [1600, 6400]:
        rng = np.random.RandomState(0)
        sub, _ = train_test_split(train_idx, train_size=n, stratify=y[train_idx],
                                  random_state=0)
        Xt, yt = clip_x(X[sub]), y[sub]
        m1, _ = train_model(Xt, yt, Xv, yv, [28, 4], EPOCHS[n], BATCH[n], 0)
        l_te_u1 = fp32_logits(m1, Xte)
        r_te_u1 = risks(l_te_u1, yte)
        m2, _ = train_model(Xt, yt, Xv, yv, ARCH, EPOCHS[n], BATCH[n], 1)
        l_te_u2 = fp32_logits(m2, Xte)
        r_te_u2 = risks(l_te_u2, yte)
        lips2 = layer_lipschitz(m2)
        series = []
        for N in [8, 16, 32, 64, 128, 256]:
            t1 = build_lut_tables(m1, N)
            t2 = build_lut_tables(m2, N)
            l1 = lut_forward_batch(m1, Xte, t1)
            l2 = lut_forward_batch(m2, Xte, t2)
            b1 = risks(l1, yte)["ce"] - r_te_u1["ce"]
            b2 = risks(l2, yte)["ce"] - r_te_u2["ce"]
            series.append({"N": N, "bias_ce_1layer": b1, "bias_ce_2layer": b2,
                           "ratio_2over1": b1 / b2 if abs(b2) > 1e-9 else np.nan})
        out[str(n)] = {"gamma_layer2": lips2[1],
                       "bias_ratio_mean": float(np.nanmean([s["ratio_2over1"]
                                                            for s in series[:4]])),
                       "series": series}
        print(f"  n={n}: gamma2={lips2[1]:.3f} | bias ratio (2l/1l) N=8..64: "
              f"{[round(s['ratio_2over1'], 3) for s in series[:4]]}")
    res = {"assertions": {
        "C1_bias2_over_bias1_within_3x_of_gamma2": bool(
            all(abs(out[str(n)]["bias_ratio_mean"] / out[str(n)]["gamma_layer2"]) < 3
                for n in out))}}
    res["per_n"] = out
    return res


# ============================================================================
# main
# ============================================================================
def main():
    X, y, train_idx, val_idx, test_idx = load_and_split()
    print(f"Data: {len(y)} samples | train {len(train_idx)} / val {len(val_idx)} "
          f"/ test {len(test_idx)} (seed-42 stratified split)")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    model = StudentKAN(ARCH)
    model.load_state_dict(ckpt["student_state_dict"], strict=False)
    model.eval()
    print(f"Pretrained ckpt: val_acc={ckpt.get('val_acc', '?')}")

    Xc_tr, Xc_te = clip_x(X[train_idx]), clip_x(X[test_idx])
    y_tr, y_te = y[train_idx], y[test_idx]

    report = {
        "experiment": "E59",
        "theorem": "thm6p_compile_aware",
        "title": "Compile-aware generalization bound for LUT-compiled KAN",
        "arch": ARCH, "domain": list(XR), "k": K, "c_k": C_K,
        "h_definition": "h = 6/(N-1)  (N points on [-3,3])",
        "n_list": N_LIST,
        "model_note": ("Part A uses the pretrained VRM-KD student. Parts B/C use "
                       "freshly CE-trained students (no teacher), to vary n. "
                       "In-domain evaluation: features clipped to [-3,3]; "
                       "unclipped numbers reported separately in Part A."),
        "gamma": None, "partA": None, "partB": None, "partC": None,
        "assertions": {}, "verdict": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Part A
    report["partA"] = part_a(model, Xc_tr, y_tr, Xc_te, y_te,
                             X[test_idx], y_te)
    report["gamma"] = layer_lipschitz(model)
    report["gamma_contractive"] = [g < 1.0 for g in report["gamma"]]
    print(f"  pretrained gamma (Lipschitz/layer): {report['gamma']}")

    # Part B
    report["partB"] = part_b(X, y, train_idx, val_idx, test_idx)

    # Part C
    report["partC"] = part_c(X, y, train_idx, val_idx, test_idx)

    # aggregate assertions
    for part, key in [("partA", "assertions"), ("partB", "assertions"),
                      ("partC", "assertions")]:
        report["assertions"].update(report[part][key])
    n_pass = sum(report["assertions"].values())
    report["verdict"] = {
        "n_pass": n_pass, "n_total": len(report["assertions"]),
        "summary": "PASS" if n_pass == len(report["assertions"]) else "PARTIAL",
    }

    (OUT_DIR / "thm6p_compile_aware.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print("=" * 72)
    print(f"Saved: {OUT_JSON}")
    print(f"Verdict: {report['verdict']}")
    for k, v in report["assertions"].items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")


if __name__ == "__main__":
    main()
