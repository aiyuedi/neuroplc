#!/usr/bin/env python3
"""
Task G: Z3 SMT Translation Validation — Full Experiment
=========================================================
Runs Z3-based translation validation on the CWRU-trained KAN [28,16,4] model,
verifying equivalence between PyTorch forward pass and compiled SCL code
for each of the 11 IR nodes.

Output:
  - Console summary (copy-paste into paper)
  - JSON report at results/smt_verify/
  - LaTeX table rows for the paper

Usage:
    python D:/neuroplc-paper/code/experiments/e22_smt_verify.py
"""

import sys, os, json, time
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # code/

from models.student_kan import StudentKAN
from neuroplc.ir import IRGraph, IROpType
from neuroplc.frontend import kan_to_ir
from neuroplc.smt_verify import verify_ir_graph, TranslationValidationReport

# Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CKPT_PATH = PROJECT_ROOT / "results" / "student" / "kan_kd_vrmKD_best.pt"
OUTPUT_DIR = PROJECT_ROOT / "results" / "smt_verify"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ARCHITECTURE = [28, 16, 4]
N_RANDOM_TESTS = 100


def main():
    print("=" * 72)
    print("Task G: Z3 SMT Translation Validation")
    print("=" * 72)

    # ── Load model ──
    print(f"\n[1] Loading model: {CKPT_PATH}")
    ckpt = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=True)
    model = StudentKAN(ARCHITECTURE)
    model.load_state_dict(ckpt["student_state_dict"], strict=False)
    model.eval()
    print(f"    Architecture: {ARCHITECTURE}")
    print(f"    CWRU val acc: {ckpt.get('val_acc', 'N/A')}")

    # ── Frontend: KAN -> IR ──
    print(f"\n[2] Frontend: KAN -> IR (lut_points=15)")
    ir_graph = kan_to_ir(model, name="kan_28_16_4", lut_points=15,
                         x_range=(-3.0, 3.0), adaptive=False)
    print(f"    IR nodes: {ir_graph.node_count}")
    print(f"    Operations: {ir_graph.op_counts}")

    warnings = ir_graph.validate()
    if warnings:
        print(f"    Warnings ({len(warnings)}):")
        for w in warnings:
            print(f"      - {w}")
    else:
        print(f"    Graph valid: YES")

    # ── Z3 Translation Validation ──
    print(f"\n[3] Z3 Translation Validation ({N_RANDOM_TESTS} tests per node)")
    t0 = time.perf_counter()
    report = verify_ir_graph(ir_graph, n_random_tests=N_RANDOM_TESTS,
                             do_z3_symbolic=True)
    elapsed = time.perf_counter() - t0

    print(report.summary())

    # ── Paper-ready summary ──
    print(f"\n{'=' * 72}")
    print("PAPER-READY SUMMARY")
    print(f"{'=' * 72}")

    # Count by op type
    by_op = {}
    for r in report.results:
        op = r.op_type
        if op not in by_op:
            by_op[op] = {"total": 0, "exact": 0, "bounded": 0, "failed": 0}
        by_op[op]["total"] += 1
        if r.status == "PASS":
            by_op[op]["exact"] += 1
        elif r.status == "PASS_BOUNDED":
            by_op[op]["bounded"] += 1
        else:
            by_op[op]["failed"] += 1

    print(f"\n  Per-op-type verification results:")
    print(f"  {'Op Type':<15s} {'Total':>5s} {'Exact':>6s} {'Bounded':>8s} {'Failed':>6s}")
    print(f"  {'-'*45}")
    for op, counts in sorted(by_op.items()):
        print(f"  {op:<15s} {counts['total']:>5d} {counts['exact']:>6d} "
              f"{counts['bounded']:>8d} {counts['failed']:>6d}")

    print(f"\n  Key metrics:")
    print(f"    Total nodes:            {report.total_nodes}")
    print(f"    Exact verified:         {report.exact_verified} "
          f"({100*report.exact_verified/report.total_nodes:.0f}%)")
    print(f"    Bounded-error verified: {report.bounded_verified} "
          f"({100*report.bounded_verified/report.total_nodes:.0f}%)")
    print(f"    Failed:                 {report.failed}")
    print(f"    Total Z3 time:          {report.total_z3_time_ms:.1f} ms")
    print(f"    Avg per node:           {report.total_z3_time_ms/report.total_nodes:.1f} ms")

    # BsplineLUT bound
    bspline_results = [r for r in report.results if r.op_type == "BsplineLUT"]
    if bspline_results:
        bounds = [r.bound_used for r in bspline_results if r.bound_used]
        print(f"    BsplineLUT error bound: {np.mean(bounds):.6f} "
              f"(M2*h^2/8, all {len(bounds)} functions)")

    # Verification coverage
    exact_types = set(r.op_type for r in report.results if r.status == "PASS")
    bounded_types = set(r.op_type for r in report.results if r.status == "PASS_BOUNDED")
    print(f"\n  Verification coverage:")
    print(f"    Exact (algebraic identity):  {sorted(exact_types)}")
    print(f"    Bounded-error (Theorem 1):   {sorted(bounded_types)}")

    # ── Save JSON report ──
    json_path = OUTPUT_DIR / "smt_verify_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"\n  Report saved: {json_path}")

    # ── LaTeX table rows ──
    print(f"\n  LaTeX table rows (for paper):")
    print(f"  % Auto-generated by e22_smt_verify.py")
    for r in report.results:
        status_map = {"PASS": "Exact", "PASS_BOUNDED": "$\\leq\\varepsilon$",
                      "FAIL": "FAIL", "SKIP": "---"}
        strat = r.strategy.replace("_", "\\_")
        details_short = r.details[:80].replace("_", "\\_") if r.details else "---"
        bound_str = f"{r.bound_used:.4f}" if r.bound_used else "---"
        print(f"  {r.node_name} & {r.op_type} & "
              f"{status_map.get(r.status, r.status)} & "
              f"{r.z3_time_ms:.0f}\\,ms & "
              f"{bound_str} \\\\")

    print(f"\n{'=' * 72}")
    print("DONE")
    print(f"{'=' * 72}")

    return report


if __name__ == "__main__":
    main()
