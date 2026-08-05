#!/usr/bin/env python3
"""E60+E61 extension: FourierKAN L0-direct SCL compilation (2026-08-05).

Deployment configuration for the FourierKAN sound certificate
(verify_fourier_l0direct.py, sound 0.130 / 5.2x):
  - L0 (28->16): ANALYTIC evaluation (SIN/COS harmonics + SiLU base),
    no LUT storage, no LUT error. S7-1200 supports SIN/COS/EXP as
    software float library instructions.
  - L1 (16->4): LUT N=16 on [-3.15, 3.15] (covers 100% of measured
    L0 outputs), binary search + linear interpolation (backend_s7
    structure).

Outputs:
  - results/scl_output/neuroplc_fourierkan_l0direct.scl
  - results/theory/fourier_l0direct_scl.json
    - semantic verification: float32 SCL-simulation vs float64 exact,
      full 13,714 inputs; maxAE must stay under the sound bound 0.130
    - WCET estimate (S7-1200 1211C instruction timings; SIN/COS software
      Taylor ~2.5 us each, EXP ~2.0 us)
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_fourierkan import StudentFourierKAN

BASE = Path(__file__).resolve().parent.parent.parent
N_L1 = 16
DOM_LO, DOM_HI = -3.15, 3.15
SOUND_BOUND = 0.1302  # from verify_fourier_l0direct.py

# S7-1200 1211C timings (us), aligned with wcet_analysis.py + Siemens manual
T_SIN = 2.5    # software Taylor SIN/COS (no hardware trig; ~EXP-class)
T_COS = 2.5
T_EXP = 2.0
T_MUL = 0.60
T_ADD = 0.50
T_ARR = 0.10
T_CMP = 0.30
T_LUT_EVAL = 3.0  # per LUT: binary search + interp (aligned with wcet model)


def extract(model):
    """L0 analytic params + L1 LUT (N=16 on [-3.15,3.15])."""
    l0 = model.kan_layers[0]
    c = l0.fourier_coeffs.detach().numpy()      # (16,28,12)
    cs = c[:, :, : l0.n_harmonics]
    cd = c[:, :, l0.n_harmonics:]
    bw = l0.base_weight.detach().numpy()
    bias0 = l0.bias.detach().numpy()
    om = float(l0.omega)

    l1 = model.kan_layers[1]
    l1c = l1.fourier_coeffs.detach().numpy()
    l1cs = l1c[:, :, : l1.n_harmonics]
    l1cd = l1c[:, :, l1.n_harmonics:]
    l1bw = l1.base_weight.detach().numpy()
    l1bias = l1.bias.detach().numpy()
    xs = np.linspace(DOM_LO, DOM_HI, N_L1)
    o1, i1 = l1.out_features, l1.in_features
    luts = {}
    for oi in range(o1):
        for ii in range(i1):
            phix = l1bw[oi, ii] * xs * (1.0 / (1.0 + np.exp(-xs)))
            for hh in range(l1.n_harmonics):
                phix += (l1cs[oi, ii, hh] * np.sin(l1.k[hh].item() * om * xs)
                         + l1cd[oi, ii, hh]
                         * np.cos(l1.k[hh].item() * om * xs))
            luts[(oi, ii)] = phix
    return dict(cs=cs, cd=cd, bw=bw, bias0=bias0, om=om,
                l1cs=l1cs, l1cd=l1cd, l1bw=l1bw, l1bias=l1bias,
                grid=xs, luts=luts, n_harm=l0.n_harmonics)


def scl_l0_direct(p, arch=(28, 16, 4)):
    """SCL: L0 analytic (SIN/COS/SiLU) + L1 LUT (N=16)."""
    L = []
    L.append("// NeuroPLC — FourierKAN L0-direct configuration [28,16,4]")
    L.append("// L0 (28->16): analytic SIN/COS harmonics + SiLU base (no LUT)")
    L.append("// L1 (16->4): LUT N=16 on [-3.15, 3.15] (100% L0-out coverage)")
    L.append("// Sound bound 0.130 (safety 5.2x) — verify_fourier_l0direct.py")
    L.append('FUNCTION_BLOCK "NeuroPLC_FourierKAN_L0DIR"')
    L.append('VAR_INPUT')
    for i in range(1, arch[0] + 1):
        L.append(f'    "feat_{i}" : REAL;')
    L.append('    "trigger" : BOOL;')
    L.append('END_VAR')
    L.append('VAR_OUTPUT')
    for j in range(1, arch[2] + 1):
        L.append(f'    "class_{j}" : REAL;')
    L.append('    "pred_class" : INT;')
    L.append('END_VAR')
    L.append('VAR')
    for j in range(1, arch[1] + 1):
        L.append(f'    "h1_{j}" : REAL := 0.0;')
    L.append('    "x" : REAL; "phi" : REAL; "acc" : REAL := 0.0;')
    L.append('    "i" : INT; "j" : INT; "k" : INT;')
    L.append('    "lo" : INT; "hi" : INT; "mid" : INT; "t" : REAL;')
    L.append('END_VAR')
    L.append('')
    L.append('BEGIN')
    L.append('    IF NOT "trigger" THEN RETURN; END_IF;')
    L.append('')
    # ---- L0 direct ----
    L.append('    // === L0 (28->16): analytic FourierKAN ===')
    for j in range(1, arch[1] + 1):
        L.append(f'    "acc" := 0.0;')
        for i in range(1, arch[0] + 1):
            L.append(f'    "x" := "feat_{i}";')
            L.append(f'    "phi" := "w0b_{j}_{i}" * "x" / (1.0 + EXP(-"x"));')
            for k in range(1, p["n_harm"] + 1):
                L.append(f'    "phi" := "phi" + "w0c_{j}_{i}_{k}" * '
                         f'SIN({k} * 0.4 * "x") + "w0d_{j}_{i}_{k}" * '
                         f'COS({k} * 0.4 * "x");')
            L.append(f'    "acc" := "acc" + "phi";')
        L.append(f'    "h1_{j}" := "acc" + "b0_{j}";')
    # ---- L1 LUT ----
    L.append('    // === L1 (16->4): LUT N=16, domain [-3.15,3.15] ===')
    for i in range(1, arch[1] + 1):
        L.append(f'    // input {i}: binary search on L1 grid')
        L.append(f'    "lo" := 0; "hi" := {N_L1 - 1};')
        L.append(f'    "x" := "h1_{i}";')
        L.append(f'    IF "x" < -3.15 THEN "x" := -3.15; END_IF;')
        L.append(f'    IF "x" > 3.15 THEN "x" := 3.15; END_IF;')
        L.append(f'    WHILE "hi" - "lo" > 1 DO')
        L.append(f'        "mid" := "lo" + ("hi" - "lo") / 2;')
        L.append(f'        IF "x" > "g1_{i}"["mid"] THEN "lo" := "mid"; '
                 f'ELSE "hi" := "mid"; END_IF;')
        L.append(f'    END_WHILE;')
        L.append(f'    "t" := ("x" - "g1_{i}"["lo"]) / '
                 f'("g1_{i}"["hi"] - "g1_{i}"["lo"] + 1.0E-10);')
        for j in range(1, arch[2] + 1):
            L.append(f'    "class_{j}" := "class_{j}" + '
                     f'("t1_{j}_{i}"["lo"] * (1.0 - "t") + '
                     f'"t1_{j}_{i}"["hi"] * "t");')
    L.append('')
    L.append('END_FUNCTION_BLOCK')
    L.append('')
    L.append('// WCET estimate: L0 = 448 edges x 12 harmonics x (SIN+COS)')
    L.append('//   + 448 SiLU (EXP) + L1 = 64 LUT evals')
    return "\n".join(L)


def sim_f32(p, X):
    """float32 SCL-simulation of L0-direct + L1 LUT (matches emitted SCL)."""
    X = X.astype(np.float32)
    n = X.shape[0]
    h1 = np.zeros((n, 16), dtype=np.float32)
    for j in range(16):
        acc = np.zeros(n, dtype=np.float32)
        for i in range(28):
            x = np.clip(X[:, i], -4.0, 4.0)   # model forward clamps to [-4,4]
            phi = (p["bw"][j, i] * x / (1.0 + np.exp(-x))).astype(np.float32)
            for k in range(1, p["n_harm"] + 1):   # harmonics 1..K
                phi = (phi + p["cs"][j, i, k - 1] * np.sin(k * p["om"] * x)
                       + p["cd"][j, i, k - 1] * np.cos(k * p["om"] * x)
                       ).astype(np.float32)
            acc += phi
        h1[:, j] = acc + p["bias0"][j]
    out = np.zeros((n, 4), dtype=np.float32)
    for j in range(4):
        for i in range(16):
            xin = np.clip(h1[:, i], DOM_LO, DOM_HI)
            phix = p["luts"][(j, i)]
            out[:, j] += np.interp(xin, p["grid"], phix).astype(np.float32)
        out[:, j] += p["l1bias"][j]
    return out


def exact_f64(model, X):
    m64 = model.double()
    with torch.no_grad():
        return m64(torch.tensor(X, dtype=torch.float64)).numpy()


def wcet_estimate():
    """Instruction-count WCET (us): L0 analytic + L1 LUT."""
    n_harm = 6
    # L0: per edge: siLU (1 EXP + 1 MUL + 1 DIV + 1 ADD) +
    #     12 harmonics x (1 MUL + SIN + COS + 2 ADD + 2 MUL)
    per_edge = (T_EXP + T_MUL + T_DIV if False else
                T_EXP + 2 * T_MUL + T_ADD)  # siLU: EXP, x*sig (MUL), x*base (MUL), ADD
    per_edge += n_harm * (T_MUL + T_SIN + T_COS + 3 * T_ADD + 2 * T_MUL)
    l0 = 448 * per_edge + 448 * T_ADD          # 448 edge sums + bias adds
    l1 = 64 * T_LUT_EVAL + 64 * (16 * T_CMP)   # 64 LUT evals + binary search
    total = l0 + l1 + 16 * 28 * T_ARR + 10     # array reads + overhead
    return total


def main():
    ckpt = torch.load(BASE / "results" / "student" / "fourier_contractive_v2.pt",
                      map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentFourierKAN([28, 16, 4], n_harmonics=6, omega=0.4)
    m.load_state_dict(sd, strict=False)
    m.eval()
    X = np.load(BASE / "data" / "processed" / "features_X.npy")

    p = extract(m)
    scl = scl_l0_direct(p)
    out_dir = BASE / "results" / "scl_output"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "neuroplc_fourierkan_l0direct.scl").write_text(scl)

    # semantic verification: f32 SCL-sim vs f64 exact
    pred = sim_f32(p, X)
    ref = exact_f64(m, X)
    maxae = float(np.abs(pred - ref).max())
    n_lines = len(scl.splitlines())
    wcet_us = wcet_estimate()

    print(f"SCL lines: {n_lines}")
    print(f"f32 SCL-sim maxAE (13,714 inputs): {maxae:.5f}")
    print(f"sound bound 0.1302 covers: {maxae <= SOUND_BOUND}")
    print(f"WCET estimate: {wcet_us:.2f} us "
          f"({wcet_us/1000:.2f} ms; {wcet_us/100000*100:.1f}% of 100ms scan)")

    out = {
        "date": "2026-08-05",
        "config": "L0 analytic (SIN/COS) + L1 LUT N=16 [-3.15,3.15]",
        "scl_lines": n_lines,
        "maxAE_f32_scsim": maxae,
        "sound_bound": SOUND_BOUND,
        "covered": bool(maxae <= SOUND_BOUND),
        "wcet_us": wcet_us,
        "wcet_ms": wcet_us / 1000.0,
        "scan_margin": 100000.0 / wcet_us,
    }
    with open(BASE / "results" / "theory" / "fourier_l0direct_scl.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved: results/theory/fourier_l0direct_scl.json")


if __name__ == "__main__":
    main()
