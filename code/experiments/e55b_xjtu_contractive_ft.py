#!/usr/bin/env python3
"""E55b: XJTU-SY contractive fine-tuning (2026-08-05).

The released CWRU checkpoint is fine-tuned to XJTU-SY run-to-failure data
with the soft-contractive recipe (gamma projection, E68 semantics measured
on XJTU data). Goal: an XJTU-SY checkpoint with measurable certificate
status (gamma profile + DA bound) instead of the current "—" cells in
tab:cross_domain, and to check whether contractive FT can keep gamma < 1
(and hence a stronger certificate than the plain E55 FT).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.student_kan import StudentKAN

BASE = Path(__file__).resolve().parent.parent.parent
GAMMA_TARGET = 0.95
EPOCHS = 30
LR = 5e-4
BATCH = 128


def load_xjtu():
    sys.path.insert(0, str(BASE / "code" / "experiments"))
    import e55_xjtu_sy_cross_dataset as e55
    Xtr, ytr, Xva, yva, scaler = e55.load_xjtu_sy_data()
    # build a test split from the val portion (keep 80/20 of val)
    rng = np.random.RandomState(7)
    idx = rng.permutation(len(Xva))
    n_te = int(len(Xva) * 0.5)
    Xte, yte = Xva[idx[:n_te]], yva[idx[:n_te]]
    Xva2, yva2 = Xva[idx[n_te:]], yva[idx[n_te:]]
    return Xtr, ytr, Xte, yte


def measure_gamma(model, X):
    """E68 semantics: per-layer L-inf signal amplification on X."""
    Xt = torch.tensor(X, dtype=torch.float32)
    gammas = []
    with torch.no_grad():
        cur = Xt
        for layer in model.kan_layers:
            nxt = layer(cur)
            num = nxt.abs().max(dim=1).values + 1e-9
            den = cur.abs().max(dim=1).values + 1e-9
            gammas.append(float((num / den).max()))
            cur = nxt
    return gammas


def project_contractive(model, target, X):
    for _ in range(8):
        g = measure_gamma(model, X)
        if max(g) <= target:
            return g
        with torch.no_grad():
            for layer in model.kan_layers:
                sb = float(layer.scale_base)
                ss = float(layer.scale_spline)
                layer.base_weight.mul_(0.95)
                layer.spline_weight.mul_(0.95)
                layer.scale_base.mul_(0.95)
                layer.scale_spline.mul_(0.95)
    return measure_gamma(model, X)


def main():
    ckpt = torch.load(BASE / "results" / "student" / "kan_kd_vrmKD_best.pt",
                      map_location="cpu", weights_only=False)
    sd = ckpt["student_state_dict"] if "student_state_dict" in ckpt else ckpt
    m = StudentKAN([28, 16, 4])
    m.load_state_dict(sd, strict=False)
    Xtr, ytr, Xte, yte = load_xjtu()
    Xtr_t = torch.from_numpy(Xtr)
    ytr_t = torch.from_numpy(ytr)

    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=1e-5)
    rng = np.random.RandomState(42)
    n = len(Xtr)
    gamma_hist = []
    for ep in range(EPOCHS):
        m.train()
        perm = rng.permutation(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            logits = m(Xtr_t[idx])
            loss = F.cross_entropy(logits, ytr_t[idx])
            with torch.no_grad():
                g = measure_gamma(m, Xtr)
            pen = sum(max(0.0, gi - GAMMA_TARGET) ** 2 for gi in g)
            loss = loss + 10.0 * pen
            loss.backward()
            opt.step()
        if (ep + 1) % 5 == 0:
            project_contractive(m, GAMMA_TARGET, Xtr)
            m.eval()
            with torch.no_grad():
                va = float((m(torch.from_numpy(Xte)).argmax(1) ==
                            torch.from_numpy(yte)).float().mean())
            g = measure_gamma(m, Xte)
            gamma_hist.append(g)
            print(f"ep {ep+1}: test {va:.4f} gamma {[round(x,3) for x in g]}")

    m.eval()
    with torch.no_grad():
        acc = float((m(torch.from_numpy(Xte)).argmax(1) ==
                     torch.from_numpy(yte)).float().mean())
    g = measure_gamma(m, Xte)
    print(f"final: test_acc={acc:.4f} gamma={[round(x,3) for x in g]} "
          f"cond3={max(g) < 1.0}")
    torch.save({"student_state_dict": m.state_dict(), "test_acc": acc,
                "gamma": g}, BASE / "results" / "student" /
               "kan_xjtu_contractive_ft.pt")
    with open(BASE / "results" / "theory" / "xjtu_contractive_ft.json", "w") as f:
        json.dump({"date": "2026-08-05", "test_acc": acc, "gamma": g,
                   "gamma_target": GAMMA_TARGET, "epochs": EPOCHS,
                   "gamma_history": gamma_hist}, f, indent=2)
    print("Saved: results/theory/xjtu_contractive_ft.json")


if __name__ == "__main__":
    main()
