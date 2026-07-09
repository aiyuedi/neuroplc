#!/usr/bin/env python3
"""
NeuroPLC — E29: PLCSIM / TIA Portal Resource Analysis
======================================================
Combines TIA Portal real compilation data with WCET formal bounds
to produce a comprehensive resource analysis for the paper.

Data sources:
  1. TIA Portal V21 compilation (real block sizes from GetBlockInfo)
  2. WCET formal analysis (Z3-verified instruction-level bounds)
  3. S7-1200 CPU 1211C datasheet specs

This replaces the "estimated from instruction counts" language with
"measured from TIA Portal V21 compilation + Z3-verified WCET bounds."

Usage:
    python e29_plcsim_resource.py
    → Generates results/plcsim_resource/plcsim_resource_report.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Add parent to path for neuroplc imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np


# ============================================================================
# TIA Portal V21 Real Compilation Data
# ============================================================================
# Extracted from TIA Portal V21 via MCP GetBlockInfo on 2026-07-07
# Project: NeuroPLC_S7_1200_DB.ap21
# CPU: S7-1211C AC/DC/RLY (6ES7 211-1BE40-0XB0), Firmware V4.7

TIA_PORTAL_DATA = {
    "project": "NeuroPLC_S7_1200_DB.ap21",
    "cpu": {
        "model": "CPU 1211C AC/DC/Rly",
        "order_number": "6ES7 211-1BE40-0XB0",
        "firmware": "V4.7",
        "work_memory_kb": 50,       # Total work memory (code + data)
        "load_memory_kb": 1024,     # Internal load memory (1 MB)
        "retentive_memory_kb": 10,  # Retentive data
    },
    "blocks": {
        "DB1_NeuroPLC_KAN_Weights": {
            "type": "GlobalDB",
            "language": "DB",
            "load_memory_bytes": 94474,    # ~92.3 KB
            "work_memory_bytes": 32972,    # ~32.2 KB (data)
            "is_consistent": True,
            "description": "KAN [28,16,4] weights (grids + LUT tables + W + bias)",
        },
        "FB1_NeuroPLC_Inference": {
            "type": "FB",
            "language": "SCL",
            "load_memory_bytes": 169876,   # ~165.9 KB
            "work_memory_bytes": 13360,    # ~13.0 KB (code + temp vars)
            "is_consistent": True,
            "description": "Inference FB: 6-op forward pass (11 IR nodes)",
        },
        "OB1_Main": {
            "type": "OB",
            "language": "LAD",
            "load_memory_bytes": None,     # Not measured (trivial)
            "work_memory_bytes": None,
            "is_consistent": True,
        },
    },
    "compilation": {
        "errors": 0,
        "warnings": 0,
        "status": "Success",
        "timestamp": "2026-07-07T03:53:25Z",
    },
}


# ============================================================================
# S7-1200 Instruction Timing Model
# ============================================================================
# Source: Siemens S7-1200 System Manual, 05/2024, Appendix A
# CPU 1211C nominal timings (microseconds)

@dataclass
class S71200Timing:
    """S7-1200 CPU 1211C nominal instruction timings (μs)."""
    real_add:    float = 0.50   # REAL addition/subtraction
    real_mul:    float = 0.60   # REAL multiplication
    real_div:    float = 1.20   # REAL division
    real_cmp:    float = 0.30   # REAL comparison
    int_add:     float = 0.15   # INT addition
    int_div:     float = 0.50   # INT division
    array_idx:   float = 0.10   # Array indexing
    scalar_load: float = 0.08   # Scalar load
    branch:      float = 0.20   # Branch overhead
    loop_iter:   float = 0.20   # Loop per-iteration
    loop_setup:  float = 0.30   # Loop init
    exp_func:    float = 2.00   # EXP() call
    assign:      float = 0.10   # Assignment

    @property
    def tolerance(self) -> float:
        return 0.15  # ±15%


# ============================================================================
# WCET Analysis (from paper, Z3-verified)
# ============================================================================

WCET_NODES = [
    # (op_type, shape, wcet_us, ops_count, deterministic)
    ("StandardAct",  "28→28",    89.4,   140, True),
    ("BsplineLUT",   "28→16×28", 1866.0, 14644, True),
    ("MatMul",       "28→16",    413.8,  896, True),
    ("Add",          "16→16",    8.0,    16, True),
    ("StandardAct",  "16→16",    59.8,   80, True),
    ("BsplineLUT",   "16→4×16",  493.2,  4184, True),
    ("MatMul",       "16→4",     44.8,   128, True),
    ("Add",          "4→4",      2.0,    4, True),
    ("Softmax",      "4→4",      24.6,   24, True),
    ("Argmax",       "4→1",      5.6,    3, True),
]

TOTAL_WCET_US = sum(n[2] for n in WCET_NODES)  # ~3007.2 μs ≈ 3.01 ms
TOTAL_OPS = sum(n[3] for n in WCET_NODES)       # 20119 FLOPs


# ============================================================================
# Memory Analysis
# ============================================================================

@dataclass
class MemoryReport:
    """Comprehensive memory analysis."""
    # From TIA Portal
    db_load_bytes: int = 94474
    db_work_bytes: int = 32972
    fb_load_bytes: int = 169876
    fb_work_bytes: int = 13360

    # CPU specs
    cpu_work_total_kb: int = 50
    cpu_load_total_kb: int = 1024

    @property
    def total_work_bytes(self) -> int:
        return self.db_work_bytes + self.fb_work_bytes

    @property
    def total_load_bytes(self) -> int:
        return self.db_load_bytes + self.fb_load_bytes

    @property
    def total_work_kb(self) -> float:
        return self.total_work_bytes / 1024.0

    @property
    def total_load_kb(self) -> float:
        return self.total_load_bytes / 1024.0

    @property
    def work_utilization_pct(self) -> float:
        return self.total_work_bytes / (self.cpu_work_total_kb * 1024) * 100.0

    @property
    def load_utilization_pct(self) -> float:
        return self.total_load_bytes / (self.cpu_load_total_kb * 1024) * 100.0

    @property
    def work_remaining_kb(self) -> float:
        return self.cpu_work_total_kb - self.total_work_kb

    def breakdown(self) -> Dict:
        return {
            "DB_load_kb": round(self.db_load_bytes / 1024.0, 1),
            "DB_work_kb": round(self.db_work_bytes / 1024.0, 1),
            "FB_load_kb": round(self.fb_load_bytes / 1024.0, 1),
            "FB_work_kb": round(self.fb_work_bytes / 1024.0, 1),
            "Total_work_kb": round(self.total_work_kb, 1),
            "Total_load_kb": round(self.total_load_kb, 1),
            "CPU_work_kb": self.cpu_work_total_kb,
            "CPU_load_kb": self.cpu_load_total_kb,
            "Work_utilization_pct": round(self.work_utilization_pct, 1),
            "Load_utilization_pct": round(self.load_utilization_pct, 1),
            "Work_remaining_kb": round(self.work_remaining_kb, 1),
        }


# ============================================================================
# Comparison: Estimate vs Actual
# ============================================================================

def build_comparison_table() -> str:
    """Build paper-ready comparison table: Estimate vs TIA Portal Actual."""
    mem = MemoryReport()

    # Paper estimates (from static analysis)
    paper_estimates = {
        "db_load_kb": 40.3,     # From paper: "40.3 KB data block"
        "db_work_kb": 40.3,     # Same (paper didn't distinguish load/work)
        "fb_work_kb": 35.0,     # From paper: "~45 KB total work memory"
        "total_work_kb": 75.0,  # From paper
        "wcet_ms": 2.86,        # From paper: "≤ 2.86 ms"
    }

    tia_actuals = {
        "db_load_kb": mem.db_load_bytes / 1024.0,
        "db_work_kb": mem.db_work_bytes / 1024.0,
        "fb_load_kb": mem.fb_load_bytes / 1024.0,
        "fb_work_kb": mem.fb_work_bytes / 1024.0,
        "total_work_kb": mem.total_work_kb,
        "total_load_kb": mem.total_load_kb,
        "wcet_ms": TOTAL_WCET_US / 1000.0,
    }

    lines = [
        "=" * 75,
        "NeuroPLC Resource Analysis: Paper Estimate vs TIA Portal V21 Actual",
        "=" * 75,
        "",
        f"CPU: {TIA_PORTAL_DATA['cpu']['model']} "
        f"({TIA_PORTAL_DATA['cpu']['order_number']})",
        f"Firmware: {TIA_PORTAL_DATA['cpu']['firmware']}",
        f"Work Memory: {mem.cpu_work_total_kb} KB | "
        f"Load Memory: {mem.cpu_load_total_kb} KB",
        "",
        "─" * 75,
        f"{'Metric':<30} {'Paper Est.':>12} {'TIA Actual':>12} {'Diff':>10}",
        "─" * 75,
    ]

    comparisons = [
        ("DB Load Memory (KB)", paper_estimates["db_load_kb"], tia_actuals["db_load_kb"]),
        ("DB Work Memory (KB)", paper_estimates["db_work_kb"], tia_actuals["db_work_kb"]),
        ("FB Load Memory (KB)", "N/A", tia_actuals["fb_load_kb"]),
        ("FB Work Memory (KB)", paper_estimates["fb_work_kb"], tia_actuals["fb_work_kb"]),
        ("Total Work Memory (KB)", paper_estimates["total_work_kb"], tia_actuals["total_work_kb"]),
        ("Total Load Memory (KB)", "N/A", tia_actuals["total_load_kb"]),
        ("WCET (ms)", paper_estimates["wcet_ms"], tia_actuals["wcet_ms"]),
    ]

    for label, est, act in comparisons:
        if isinstance(est, str):
            diff_str = "NEW"
        else:
            diff = act - est
            diff_pct = (diff / est * 100) if est != 0 else float('inf')
            diff_str = f"{diff:+.1f} ({diff_pct:+.0f}%)"
        act_str = f"{act:.1f}" if isinstance(act, (int, float)) else act
        lines.append(f"{label:<30} {str(est):>12} {act_str:>12} {diff_str:>10}")

    lines.extend([
        "─" * 75,
        "",
        "KEY FINDINGS:",
        f"  1. Work memory: {mem.work_utilization_pct:.1f}% utilization "
        f"({mem.total_work_kb:.1f}/{mem.cpu_work_total_kb} KB)",
        f"  2. Remaining for user logic: {mem.work_remaining_kb:.1f} KB",
        f"  3. Code compiled with 0 errors, 0 warnings on TIA Portal V21",
        f"  4. WCET = {TOTAL_WCET_US/1000:.2f} ms → "
        f"{TOTAL_WCET_US/100000*100:.2f}% of 100ms cycle",
        f"  5. Memory is the binding constraint (not compute): "
        f"90.4% work memory vs 3.0% cycle time",
        "",
        f"Paper impact: 'estimated from instruction counts' → "
        f"'measured from TIA Portal V21 compilation'",
        "=" * 75,
    ])

    return "\n".join(lines)


def build_latex_table() -> str:
    """Build LaTeX table for the paper."""
    mem = MemoryReport()

    return r"""\begin{table}[t]
\centering
\caption{Resource Analysis: NeuroPLC-compiled KAN \arch{} on S7-1200 CPU 1211C.
Memory values measured from TIA Portal V21 compilation (block properties);
WCET is a \emph{formal upper bound} verified by Z3 SMT solver.
The compiled code occupies \textbf{%.1f\,KB} of the 50\,KB work memory
(\textbf{%.1f\%%} utilization), confirming that memory—not compute time—is the
binding constraint for neural inference on low-end PLCs.}
\label{tab:resource_analysis}
\small
\begin{tabular}{lrrr}
\toprule
\textbf{Resource} & \textbf{Static Estimate} & \textbf{TIA Portal V21} & \textbf{Unit} \\
\midrule
\multicolumn{4}{l}{\textit{Memory (TIA Portal V21 compilation)}} \\
\quad DB (weights) — Load  & 40.3 & %.1f & KB \\
\quad DB (weights) — Work  & 40.3 & %.1f & KB \\
\quad FB (inference) — Load & N/A & %.1f & KB \\
\quad FB (inference) — Work & 35.0 & %.1f & KB \\
\midrule
\quad \textbf{Total Work Memory} & \textbf{75.0} & \textbf{%.1f} & \textbf{KB} \\
\quad \textbf{Total Load Memory} & \textbf{N/A} & \textbf{%.1f} & \textbf{KB} \\
\quad Work Mem. Utilization   & 150\%% (est.) & \textbf{%.1f}\%% & of 50\,KB \\
\midrule
\multicolumn{4}{l}{\textit{Timing (Z3-verified WCET)}} \\
\quad Total Inference WCET & 2.86 & %.2f & ms \\
\quad Cycle Budget Usage    & 2.86 & %.2f & \%% of 100\,ms \\
\bottomrule
\end{tabular}

\vspace{4pt}
\footnotesize
TIA Portal data from project \texttt{NeuroPLC\_S7\_1200\_DB.ap21},
compiled 2026-07-07 with 0 errors, 0 warnings.
WCET bounds verified by Z3 SMT solver (per-instruction timing from
Siemens S7-1200 System Manual, Appendix~A).
The paper's original 75\,KB estimate over-counted work memory because
it did not separate load memory (stored in flash, not counted against
the work memory budget) from work memory (runtime RAM).
The actual work memory usage is \textbf{%.1f\,KB}, leaving %.1f\,KB
for user logic.
\end{table}""" % (
        mem.total_work_kb, mem.work_utilization_pct,
        mem.db_load_bytes / 1024.0,
        mem.db_work_bytes / 1024.0,
        mem.fb_load_bytes / 1024.0,
        mem.fb_work_bytes / 1024.0,
        mem.total_work_kb,
        mem.total_load_kb,
        mem.work_utilization_pct,
        TOTAL_WCET_US / 1000.0,
        TOTAL_WCET_US / 100000.0 * 100,
        mem.total_work_kb, mem.work_remaining_kb,
    )


# ============================================================================
# Main
# ============================================================================

def main():
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', '..', 'results', 'plcsim_resource'
    )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 75)
    print("NeuroPLC E29 — PLCSIM / TIA Portal Resource Analysis")
    print("=" * 75)
    print()

    # ── Memory analysis ──
    mem = MemoryReport()
    breakdown = mem.breakdown()
    print("Memory Breakdown (TIA Portal V21):")
    for k, v in breakdown.items():
        print(f"  {k}: {v}")
    print()

    # ── WCET analysis ──
    print(f"WCET Analysis (Z3-verified):")
    print(f"  {'Node':<16} {'Shape':<12} {'WCET(μs)':>10} {'Ops':>8} {'%':>6}")
    print(f"  {'─'*16} {'─'*12} {'─'*10} {'─'*8} {'─'*6}")
    for name, shape, wcet, ops, det in WCET_NODES:
        pct = wcet / TOTAL_WCET_US * 100
        print(f"  {name:<16} {shape:<12} {wcet:>10.1f} {ops:>8} {pct:>5.1f}%")
    print(f"  {'─'*16} {'─'*12} {'─'*10} {'─'*8} {'─'*6}")
    print(f"  {'TOTAL':<16} {'11 nodes':<12} {TOTAL_WCET_US:>10.1f} {TOTAL_OPS:>8}")
    print()

    # ── Comparison table ──
    comparison = build_comparison_table()
    print(comparison)
    print()

    # ── LaTeX table ──
    latex = build_latex_table()
    print("LaTeX Table:")
    print(latex)
    print()

    # ── Save results ──
    report = {
        "meta": {
            "experiment": "E29",
            "name": "PLCSIM / TIA Portal Resource Analysis",
            "timestamp": "2026-07-07T12:00:00+08:00",
            "description": "Combines TIA Portal V21 real compilation data with "
                           "Z3-verified WCET bounds for comprehensive resource analysis",
        },
        "tia_portal_data": TIA_PORTAL_DATA,
        "memory_breakdown": breakdown,
        "wcet": {
            "total_us": TOTAL_WCET_US,
            "total_ms": TOTAL_WCET_US / 1000.0,
            "total_ops": TOTAL_OPS,
            "cycle_budget_pct": TOTAL_WCET_US / 100000.0 * 100,
            "nodes": [
                {"op": n[0], "shape": n[1], "wcet_us": n[2], "ops": n[3],
                 "deterministic": n[4]}
                for n in WCET_NODES
            ],
        },
        "key_findings": [
            f"Work memory: {mem.work_utilization_pct:.1f}% utilization",
            f"Code compiled: 0 errors, 0 warnings on TIA Portal V21",
            f"WCET = {TOTAL_WCET_US/1000:.2f} ms (Z3-verified)",
            f"Memory is binding constraint (90.4% work mem vs 3.0% cycle)",
            "Paper upgrade: estimates → TIA Portal measured + Z3-verified",
        ],
    }

    report_path = os.path.join(output_dir, "plcsim_resource_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved: {report_path}")

    # ── Save LaTeX snippet ──
    latex_path = os.path.join(output_dir, "resource_table.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex)
    print(f"LaTeX table saved: {latex_path}")

    return report


if __name__ == "__main__":
    main()
