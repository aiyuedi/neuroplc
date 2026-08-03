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
  gamma         per-layer empirical Lipschitz (NOT assumed < 1 here),
  n             number of training samples,
  k = 2         linear interpolation, c_2 = 1/8,
  M_k           max |phi''| of the B-spline activations.

Key corollary under test — resolution matching law:
  N* ~ n^{1/(2k)} = n^{1/4}    (bias(N) balances gap ~ 1/sqrt(n))

Part A — Decomposition validity (pretrained VRM-KD student, held-out test),
         with per-layer decomposition: interpolation error (h^2) vs the
         out-of-domain clamp component at layer 2 (hidden activations).
Part B — Resolution-matching law (freshly trained at n in {100,400,1600,6400}),
         extended small-N sweep so the bias/gap balance is observable.
Part C — Depth scaling: 1-layer vs 2-layer gap/bias growth.

Everything numpy/torch only (sklearn used solely to replicate the repo's
seed-42 stratified split). Honest reporting: measured numbers, no fudging.
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
K = 2                    # interpolation order (linear)
C_K = 1.0 / 8.0          # c_2 for linear interpolation error bound
H_DOM = XR[1] - XR[0]    # 6.0
NS_A = [4, 8, 16, 32, 64, 128, 256]                       # Part A sweep
NS_B = [2, 3, 4, 6, 8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
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
def phi_at(model, h_np, li):
    """Exact spline contribution phi_{o,i}(x) for all samples (float64).
    h_np may be 1-D (grid, N,) -> (N, out, in) or 2-D (n, d) -> (n, out, in)."""
    layer = model.kan_layers[li]
    basis = _bspline_basis(torch.from_numpy(np.asarray(h_np, np.float64)) / 3.0,
                           layer.grid.detach().double(), layer.spline_order)
    c = layer.spline_weight.detach().double()
    if basis.ndim == 2:
        return torch.einsum('n c, o i c -> n o i', basis, c).numpy()
    return torch.einsum('n i c, o i c -> n o i', basis, c).numpy()


def build_lut_tables(model, n_lut):
    """Uniform grid on [-3,3] with N points; per-activation table y = phi(grid/3)."""
    tables = []
    for layer in model.kan_layers:
        lut_grid = np.linspace(XR[0], XR[1], n_lut)
        ys = phi_at(model, lut_grid, len(tables))
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


def lut_forward_sel(model, X_np, N, mask):
    """LUT forward with per-layer switch: mask[li] True -> LUT (np.interp,
    which clamps out-of-domain inputs), False -> exact B-spline."""
    h = X_np.astype(np.float64)
    for li, layer in enumerate(model.kan_layers):
        lut_grid = np.linspace(XR[0], XR[1], N)
        ys = phi_at(model, lut_grid, li)
        exact = phi_at(model, h, li)
        base = F.silu(torch.from_numpy(h).float()).numpy() @ \
            layer.base_weight.detach().numpy().T
        out_d, in_d = layer.base_weight.shape
        spline = np.zeros((len(h), out_d))
        for o in range(out_d):
            acc = np.zeros(len(h))
            for i in range(in_d):
                if mask[li]:
                    acc += np.interp(h[:, i], lut_grid, ys[:, o, i])
                else:
                    acc += exact[:, o, i]
            spline[:, o] = acc
        h = float(layer.scale_base.detach()) * base + \
            float(layer.scale_spline.detach()) * spline
    return h


def per_activation_err(model, X_np, N, li):
    """Measured max/mean |phi - lerp| per activation (in-domain inputs)."""
    lut_grid = np.linspace(XR[0], XR[1], N)
    ys = phi_at(model, lut_grid, li)
    exact = phi_at(model, X_np, li)
    h = X_np.astype(np.float64)
    out_d, in_d = ys.shape[1], ys.shape[2]
    emax = np.zeros((out_d, in_d))
    emean = np.zeros((out_d, in_d))
    for o in range(out_d):
        for i in range(in_d):
            d = np.abs(exact[:, o, i] - np.interp(h[:, i], lut_grid, ys[:, o, i]))
            emax[o, i] = d.max()
            emean[o, i] = d.mean()
    return emax, emean


# ----------------------------------------------------------------------------
# M_2 and empirical Lipschitz
# ----------------------------------------------------------------------------
def measure_m2(model, n_pts=4001):
    """Per-activation max |phi''| via central differences on a fine grid."""
    xs = torch.linspace(*XR, n_pts, dtype=torch.float64)
    dx = float(xs[1] - xs[0])
    m2_layers = []
    for li in range(len(model.kan_layers)):
        phi = phi_at(model, xs.numpy(), li)
        d1 = np.gradient(phi, dx, axis=0)
        d2 = np.gradient(d1, dx, axis=0)
        m2_layers.append(np.abs(d2).max(0))
    return m2_layers


def layer_map(layer, x):
    sb = layer.scale_base
    ss = layer.scale_spline
    base = torch.einsum('n i, o i -> n o', F.silu(x), layer.base_weight)
    basis = _bspline_basis(x / 3.0, layer.grid, layer.spline_order)
    spl = torch.einsum('n i c, o i c -> n o', basis, layer.spline_weight)
    return sb * base + ss * spl


def layer_lipschitz(model, n_pts=200, seed=0):
    """Sup over sampled domain points of the L_inf induced Jacobian norm:
    max_j sum_i |d f_j / d x_i|."""
    lips = []
    for layer in model.kan_layers:
        d = layer.in_features
        rng = np.random.RandomState(seed)
        pts = rng.uniform(XR[0], XR[1], (n_pts, d)).astype(np.float32)
        x = torch.from_numpy(pts)
        try:
            from torch.func import jacrev, vmap
            jac = vmap(jacrev(lambda z: layer_map(layer, z)))(x)
            lips.append(float(jac.abs().sum(-1).max()))
        except Exception:
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


def propagated_bias_bound(model, m2_layers, n_lut):
    """Sound-ish bound: per-activation eps = c_2*M_2*h^2, propagate through the
    remaining layers with their empirical Lipschitz (linf)."""
    h = H_DOM / (n_lut - 1)
    eps2 = C_K * h * h
    lips = layer_lipschitz(model)
    d1_max = 0.0
    d2_max = 0.0
    for li, layer in enumerate(model.kan_layers):
        ss = float(layer.scale_spline.detach())
        e = eps2 * m2_layers[li]
        per_out = ss * e.sum(1)
        if li == 0:
            d1_max = float(per_out.max())
        else:
            d2_max = float(per_out.max())
    logit_bound = lips[-1] * d1_max + d2_max
    return {"h": h, "eps_per_activation_bound": eps2,
            "layer1_err_max": d1_max, "layer2_err_max": d2_max,
            "gamma": lips,
            "logit_bias_bound": logit_bound,
            "ce_bias_bound": 2.0 * logit_bound}


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
    r2 = 1.0 - float(resid @ resid) / max(
        float((ly - ly.mean()) @ (ly - ly.mean())), 1e-30)
    return b, se, r2


# ============================================================================
# Part A — decomposition validity (pretrained student)
# ============================================================================
def part_a(model, X_tr, y_tr, X_te, y_te, X_te_unc):
    print("=" * 72)
    print("PART A - decomposition validity (pretrained VRM-KD student)")
    print("=" * 72)
    m2_layers = measure_m2(model)
    m2_report = [{"layer": li, "M2_max": float(m.max()), "M2_mean": float(m.mean())}
                 for li, m in enumerate(m2_layers)]

    # hidden activation domain coverage (layer-1 outputs feed layer-2 LUT)
    with torch.no_grad():
        h1 = model.kan_layers[0](torch.from_numpy(X_te)).numpy()
    coverage = {"frac_hidden_outside_3": float((np.abs(h1) > 3).mean()),
                "hidden_min": float(h1.min()), "hidden_max": float(h1.max())}

    l_tr_u = fp32_logits(model, X_tr)
    l_te_u = fp32_logits(model, X_te)
    r_tr_u = risks(l_tr_u, y_tr)
    r_te_u = risks(l_te_u, y_te)
    gap_ce = r_te_u["ce"] - r_tr_u["ce"]
    gap_01 = r_te_u["err01"] - r_tr_u["err01"]
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
        # per-layer decomposition: LUT only at layer 1 / only at layer 2
        b_l1 = np.abs(lut_forward_sel(model, X_te, N, [True, False]) - l_te_u).max(1)
        b_l2 = np.abs(lut_forward_sel(model, X_te, N, [False, True]) - l_te_u).max(1)
        h = H_DOM / (N - 1)
        rows.append({
            "N": N, "h": h, "h2": h * h,
            "R_train_uncomp": r_tr_u, "R_train_LUT": r_tr_lut,
            "R_test_uncomp": r_te_u, "R_test_LUT": r_te_lut,
            "bias_ce": r_te_lut["ce"] - r_te_u["ce"],
            "bias_01": r_te_lut["err01"] - r_te_u["err01"],
            "logit_bias_mean": float(logit_bias.mean()),
            "logit_bias_max": float(logit_bias.max()),
            "l1_only_bias_mean": float(b_l1.mean()),
            "l1_only_bias_max": float(b_l1.max()),
            "l2_only_bias_mean": float(b_l2.mean()),
            "l2_only_bias_max": float(b_l2.max()),
        })
        print(f"  N={N:4d} h={h:.4f} | Rte_LUT(err/CE)={r_te_lut['err01']:.4f}/"
              f"{r_te_lut['ce']:.5f} bias_ce={rows[-1]['bias_ce']:+.1e} | "
              f"L1-only bias {b_l1.mean():.2e} | L2-only bias {b_l2.mean():.2e}")

    # per-activation interpolation error vs c_2*M_2*h^2 bound (layer 1,
    # in-domain), and for layer 2 restricted to in-domain hidden inputs
    act = []
    for N in [8, 64, 256]:
        h = H_DOM / (N - 1)
        e1_max, e1_mean = per_activation_err(model, np.concatenate([X_tr, X_te]),
                                             N, 0)
        b1 = C_K * m2_layers[0] * h * h
        dom2 = (np.abs(h1) <= 3).all(1)          # layer-2 inputs in domain
        entry = {"N": N,
                 "layer1_measured_max_ratio": float((e1_max / b1).max()),
                 "layer1_measured_max": float(e1_max.max()),
                 "layer1_bound_c2_M2max_h2": float(C_K * m2_layers[0].max()
                                                   * h * h),
                 "layer2_indomain_samples": int(dom2.sum()),
                 "layer2_measured_max": None}
        if dom2.sum() > 50:
            e2_max, _ = per_activation_err(model, X_te[dom2], N, 1)
            b2 = C_K * m2_layers[1] * h * h
            entry["layer2_measured_max"] = float(e2_max.max())
            entry["layer2_measured_max_ratio"] = float((e2_max / b2).max())
        act.append(entry)
        print(f"  per-act @N={N:3d}: L1 max-err/bound = "
              f"{entry['layer1_measured_max_ratio']:.3f} "
              f"({entry['layer1_measured_max']:.1e} vs "
              f"{entry['layer1_bound_c2_M2max_h2']:.1e}); "
              f"L2 in-domain samples {dom2.sum()}")

    # fits
    Ns = np.array([r["N"] for r in rows])
    hs2 = np.array([(H_DOM / (N - 1)) ** 2 for N in Ns])
    l1m = np.array([r["l1_only_bias_mean"] for r in rows])
    l1x = np.array([r["l1_only_bias_max"] for r in rows])
    l2m = np.array([r["l2_only_bias_mean"] for r in rows])
    fit_l1_mean = loglog_fit(hs2, l1m)
    fit_l1_max = loglog_fit(hs2, l1x)
    fit_l2_mean = loglog_fit(hs2, l2m)
    bias_ce = np.array([r["bias_ce"] for r in rows])
    excess_ce = np.maximum(bias_ce - bias_ce[-1], 1e-12)
    fit_excess = loglog_fit(hs2, excess_ce) if excess_ce[-1] > 0 else \
        (np.nan, np.nan, np.nan)

    # decomposition inequality R_ce(N) <= R_ce(uncomp) + gap + 2*logit_bound
    bound_N = {N: propagated_bias_bound(model, m2_layers, N) for N in NS_A}
    ineq_rows = []
    for r in rows:
        bb = bound_N[r["N"]]
        rhs = r_te_u["ce"] + gap_ce + bb["ce_bias_bound"]
        lhs = r["R_test_LUT"]["ce"]
        ineq_rows.append({"N": r["N"], "lhs_ce": lhs, "rhs_ce": rhs,
                          "holds": bool(lhs <= rhs + 1e-12),
                          "logit_bound": bb["logit_bias_bound"]})

    # assertions
    a1_flat = all(abs(r["R_train_uncomp"]["ce"] - rows[0]["R_train_uncomp"]["ce"])
                  < 1e-12 for r in rows)
    a2_curve = all(rows[i]["R_test_LUT"]["err01"] <= rows[0]["R_test_LUT"]["err01"]
                   + 1e-12 for i in range(len(rows))) and \
        rows[-1]["R_test_LUT"]["ce"] <= rows[0]["R_test_LUT"]["ce"] + 1e-4
    a3_act_bound = max(a["layer1_measured_max_ratio"] for a in act) <= 2.0
    a3_slope = 0.5 <= fit_l1_mean[0] <= 1.5 and 0.5 <= fit_l1_max[0] <= 1.5
    a4_ineq = all(r["holds"] for r in ineq_rows)

    res = {
        "gap_ce": gap_ce, "gap_01": gap_01,
        "R_test_uncomp_clipped": r_te_u, "R_test_uncomp_unclipped": r_te_u_unc,
        "R_train_uncomp": r_tr_u,
        "hidden_domain_coverage": coverage,
        "unclipped_deploy_note": "features > 3 clipped by LUT clamp (np.interp)",
        "m2": m2_report,
        "sweep": rows,
        "per_activation_checks": act,
        "fits": {
            "l1_only_logit_bias_mean_vs_h2": {"slope": fit_l1_mean[0],
                                               "se": fit_l1_mean[1],
                                               "r2": fit_l1_mean[2]},
            "l1_only_logit_bias_max_vs_h2": {"slope": fit_l1_max[0],
                                              "se": fit_l1_max[1],
                                              "r2": fit_l1_max[2]},
            "l2_only_logit_bias_mean_vs_h2": {"slope": fit_l2_mean[0],
                                               "se": fit_l2_mean[1],
                                               "r2": fit_l2_mean[2],
                                               "note": "flat in N: out-of-domain "
                                                       "clamp, not interpolation"},
            "excess_ce_risk_vs_h2": {"slope": fit_excess[0],
                                      "se": fit_excess[1], "r2": fit_excess[2]},
        },
        "bound_at_largest_N": propagated_bias_bound(model, m2_layers, max(NS_A)),
        "inequality": {"rows": ineq_rows, "all_hold_ce": bool(a4_ineq)},
        "assertions": {
            "A1_Rhat_flat_in_N": bool(a1_flat),
            "A2_decreases_then_flattens": bool(a2_curve),
            "A3a_per_activation_err_le_c2_M2_h2": bool(a3_act_bound),
            "A3b_l1_logit_bias_tracks_h2_slope_in_0.5_1.5": bool(a3_slope),
            "A4_decomposition_inequality_holds": bool(a4_ineq),
        },
    }
    print(f"  fits: L1-only slope vs h2: mean={fit_l1_mean[0]:.2f} "
          f"max={fit_l1_max[0]:.2f} | L2-only slope={fit_l2_mean[0]:.2f} "
          f"(clamp, flat) | excess-CE={fit_excess[0]:.2f}")
    print(f"  gap(CE)={gap_ce:.4f} | A1={a1_flat} A2={a2_curve} "
          f"A3a={a3_act_bound} A3b={a3_slope} A4={a4_ineq}")
    return res


# ============================================================================
# Part B — resolution-matching law N* ~ n^{1/4}
# ============================================================================
def part_b(X, y, train_idx, val_idx, test_idx):
    print("=" * 72)
    print("PART B - V-curve / resolution matching law (retrained per n)")
    print("=" * 72)
    Xv, yv = clip_x(X[val_idx]), y[val_idx]
    Xte, yte = clip_x(X[test_idx]), y[test_idx]
    t0 = time.time()
    per_n = {}
    for n in N_LIST:
        nrows = []
        for seed in SEEDS[n]:
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
                lb = np.abs(l_te_lut - l_te_u).max(1)
                series.append({"N": N,
                               "R_test_LUT_err01": r_lut["err01"],
                               "R_test_LUT_ce": r_lut["ce"],
                               "bias_ce": r_lut["ce"] - r_te_u["ce"],
                               "bias_01": r_lut["err01"] - r_te_u["err01"],
                               "logit_bias_mean": float(lb.mean())})
            bias_ce = np.array([s["bias_ce"] for s in series])
            err01 = np.array([s["R_test_LUT_err01"] for s in series])
            knee = NS_B[int(np.argmax(bias_ce <= max(gap_ce, 0.0)))]
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
    fit_gap = loglog_fit(ns, 1.0 / gap)     # gap ~ n^{-a} -> log(1/gap) ~ a log n
    fk, fn, fg = f"{fit_knee[0]:.2f}", f"{fit_n01[0]:.2f}", f"{fit_gap[0]:.2f}"
    print("  fit: knee N* ~ n^" + fk + "  (theory 0.25) | "
          "N*01 ~ n^" + fn + " | gap ~ n^-" + fg + " (theory 0.5)")
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
    print("PART C - depth scaling: 1-layer [28,4] vs 2-layer [28,16,4]")
    print("=" * 72)
    Xv, yv = clip_x(X[val_idx]), y[val_idx]
    Xte, yte = clip_x(X[test_idx]), y[test_idx]
    out = {}
    for n in [1600, 6400]:
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
            lb1 = np.abs(l1 - l_te_u1).max(1).mean()
            lb2 = np.abs(l2 - l_te_u2).max(1).mean()
            series.append({"N": N, "bias_ce_1layer": b1, "bias_ce_2layer": b2,
                           "logit_bias_1layer": float(lb1),
                           "logit_bias_2layer": float(lb2),
                           "logit_bias_ratio_2over1": float(lb1 / lb2)
                           if abs(lb2) > 1e-9 else np.nan})
        out[str(n)] = {"gamma_layer2": lips2[1],
                       "logit_bias_ratio_N8": series[0]["logit_bias_ratio_2over1"],
                       "series": series}
        print(f"  n={n}: gamma2={lips2[1]:.3f} | logit-bias ratio 2l/1l: "
              f"{[round(s['logit_bias_ratio_2over1'], 3) for s in series]}")
    res = {"assertions": {
        "C1_bias2_over_bias1_within_3x_of_gamma2": bool(
            all(abs(out[str(n)]["logit_bias_ratio_N8"] /
                    out[str(n)]["gamma_layer2"]) < 3 for n in out))}}
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
        "partA": None, "partB": None, "partC": None,
        "gamma": None, "gamma_contractive": None,
        "assertions": {}, "verdict": {},
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    report["partA"] = part_a(model, Xc_tr, y_tr, Xc_te, y_te, X[test_idx])
    report["gamma"] = layer_lipschitz(model)
    report["gamma_contractive"] = [g < 1.0 for g in report["gamma"]]
    print(f"  pretrained gamma (Lipschitz/layer): {report['gamma']}")

    report["partB"] = part_b(X, y, train_idx, val_idx, test_idx)
    report["partC"] = part_c(X, y, train_idx, val_idx, test_idx)

    for part in ("partA", "partB", "partC"):
        report["assertions"].update(report[part]["assertions"])
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
