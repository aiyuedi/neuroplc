#!/usr/bin/env python3
"""Cond.3 (per-layer amplification gamma < 1) complement measurement (2026-08-05).

Fills the two remaining "—" cells in tab:cross_domain:
  - MNIST KAN (E42): model is not saved by e42 -> retrain, then measure gamma.
  - ChebyKAN (E54): chebykan_trained.pt exists -> measure gamma directly.

gamma = max over inputs of ||h_l(x)||_inf / ||h_{l-1}(x)||_inf  (E68 semantics)
Cond.3 satisfied iff all gamma < 1.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "experiments")))
from models.student_kan import StudentKAN

BASE = Path(__file__).resolve().parent.parent.parent


def measure_gamma(model, X, batch=1024):
    """Per-layer L-inf signal amplification (E68 semantics)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    gammas = []
    layers = getattr(model, "kan_layers", None) or getattr(model, "layers", None)
    with torch.no_grad():
        cur = Xt
        for layer in layers:
            nxt = layer(cur)
            num = nxt.abs().max(dim=1).values + 1e-9
            den = cur.abs().max(dim=1).values + 1e-9
            gammas.append(float((num / den).max()))
            cur = nxt
    return gammas


def mnist_cond3():
    """Retrain MNIST KAN (E42 settings) and measure gamma."""
    import e42_mnist_cross_domain as e42
    torch.manual_seed(e42.RANDOM_SEED)
    np.random.seed(e42.RANDOM_SEED)
    X_train, y_train, X_test, y_test, pca, scaler = e42.prepare_mnist_4class(
        digits=(0, 1, 2, 3), n_components=e42.ARCH[0])
    model = StudentKAN(e42.ARCH, grid_size=8, spline_order=3)
    res = e42.train_kan_mnist(model, X_train, y_train, X_test, y_test,
                              epochs=50)
    model.eval()
    acc = res["best_acc"]
    g = measure_gamma(model, X_test)
    torch.save({"student_state_dict": model.state_dict(), "test_acc": acc},
               BASE / "results" / "mnist_cross_domain" / "mnist_kan.pt")
    return {"test_acc": float(acc), "gamma": g,
            "cond3": bool(max(g) < 1.0)}


def chebykan_cond3():
    """Load chebykan_trained.pt, measure gamma on CWRU features."""
    from models.student_chebykan import StudentChebyKAN
    ck = torch.load(BASE / "results" / "chebykan_z3" / "chebykan_trained.pt",
                    map_location="cpu", weights_only=False)
    sd = ck.get("student_state_dict", ck.get("state_dict", ck))
    # arch probe: try [28,16,4] and [28,16,8,4]
    for arch in ([28, 16, 4], [28, 16, 8, 4]):
        try:
            m = StudentChebyKAN(arch)
            m.load_state_dict(sd, strict=False)
            print(f"  ChebyKAN arch={arch} loaded OK")
            break
        except Exception as ex:
            m = None
            print(f"  arch {arch} failed: {ex}")
    if m is None:
        return None
    m.eval()
    X = np.load(BASE / "data" / "processed" / "features_X.npy")
    g = measure_gamma(m, X)
    acc = float(ck.get("test_acc", ck.get("accuracy", float("nan"))))
    return {"test_acc": acc, "gamma": g, "cond3": bool(max(g) < 1.0),
            "arch": arch}


def main():
    out = {}
    print("== MNIST KAN (E42 retrain) ==")
    out["mnist"] = mnist_cond3()
    print("  ", out["mnist"])
    print("== ChebyKAN (E54) ==")
    out["chebykan"] = chebykan_cond3()
    print("  ", out["chebykan"])
    with open(BASE / "results" / "theory" / "cond3_complement.json", "w") as f:
        json.dump({"date": "2026-08-05",
                   "note": "Cond.3 (gamma<1) complement for tab:cross_domain",
                   "out": out}, f, indent=2)
    print("Saved: results/theory/cond3_complement.json")


if __name__ == "__main__":
    main()
