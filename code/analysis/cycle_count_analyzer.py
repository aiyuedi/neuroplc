#!/usr/bin/env python3
"""
NeuroPLC — SCL Instruction-Level Cycle Count Analyzer (P4: PLCSIM + Engineering)
==================================================================================
Parses generated SCL code and counts every operation by type, then estimates
per-inference cycle time on S7-1200 / S7-1500 based on Siemens instruction
timing data.

Instruction timing reference:
    S7-1200 (1211C): ~0.08 μs/bit-op, ~0.12 μs/word-op
    S7-1500 (1513):  ~0.002 μs/bit-op, ~0.006 μs/word-op (much faster)

REAL (32-bit floating-point) operations on S7-1200:
    - ADD/SUB:  ~0.10 μs
    - MUL:      ~0.15 μs
    - DIV:      ~0.30 μs
    - EXP:      ~1.50 μs (library call, ~15× cost of MUL)
    - CMP:      ~0.06 μs
    - MOVE/ASSIGN: ~0.04 μs

Array access overhead:
    - 1D indexing (arr[i]):  ~0.06 μs
    - 2D indexing (arr[i,j]): ~0.10 μs (computed as i*stride+j)
    - DB access (cross-FB):   ~0.12 μs per access

Loop overhead:
    - FOR loop (per iter): ~0.08 μs for INT counter + bounds check

Output:
    results/cycle_count_report.json  — detailed per-op breakdown
    results/cycle_count_report.md    — human-readable report

Usage:
    python analysis/cycle_count_analyzer.py
    python analysis/cycle_count_analyzer.py --scl results/scl_output/kan_s7-1200_db_fb.scl
"""

import re, json, sys, os
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = PROJECT_ROOT.parent
SCL_DIR = REPO_ROOT / "results" / "scl_output"
OUT_DIR = REPO_ROOT / "results"

# ── S7-1200 Instruction Timing (μs) ──
# Based on Siemens S7-1200 System Manual V4.6 + SCL compilation patterns
TIMING_S71200 = {
    "real_mul":     0.15,   # REAL * REAL
    "real_add":     0.10,   # REAL + REAL
    "real_sub":     0.10,   # REAL - REAL
    "real_div":     0.30,   # REAL / REAL
    "real_exp":     1.50,   # EXP(REAL) — library call
    "real_cmp":     0.06,   # REAL >= REAL, REAL > REAL
    "real_assign":  0.04,   # v[i] := REAL / v[i] := v[j]
    "int_assign":   0.02,   # INT := INT / lo := j
    "int_add":      0.02,   # INT + INT
    "int_cmp":      0.02,   # INT >= INT (FOR loop check)
    "array_1d":     0.06,   # 1D array access v1[i]
    "array_2d":     0.10,   # 2D array access (computed index)
    "db_access":    0.12,   # Cross-FB DB read "DB".var[idx]
    "for_overhead": 0.08,   # Per-iteration FOR loop overhead
    "branch":       0.04,   # IF-THEN
}

# ── S7-1500 Instruction Timing (μs) ──
# S7-1500 is ~50-100× faster for most operations
TIMING_S71500 = {
    "real_mul":     0.003,
    "real_add":     0.002,
    "real_sub":     0.002,
    "real_div":     0.006,
    "real_exp":     0.030,
    "real_cmp":     0.001,
    "real_assign":  0.001,
    "int_assign":   0.001,
    "int_add":      0.001,
    "int_cmp":      0.001,
    "array_1d":     0.002,
    "array_2d":     0.003,
    "db_access":    0.003,
    "for_overhead": 0.002,
    "branch":       0.001,
}


def count_mul_in_line(line: str) -> int:
    """Count REAL multiplications in an SCL line."""
    # Count * operators that are multiplications (not comments)
    code = line.split("//")[0] if "//" in line else line
    # Count '*' that is between REAL operands (skip array sizing like [0..27])
    # Simple heuristic: count '*' not inside brackets
    count = 0
    in_bracket = False
    for i, ch in enumerate(code):
        if ch == '[':
            in_bracket = True
        elif ch == ']':
            in_bracket = False
        elif ch == '*' and not in_bracket:
            count += 1
    return count


def count_ops_in_scl(scl_path: str) -> dict:
    """Parse an SCL file and count instruction types."""
    with open(scl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    counts = defaultdict(int)
    loop_ranges = []  # track active FOR loop bounds
    db_name = "NeuroPLC_KAN_Weights"  # default

    # Detect DB name
    for line in lines[:10]:
        m = re.search(r'"([^"]+)"', line)
        if m and ("Weights" in m.group(1) or "NeuroPLC" in m.group(1)):
            db_name = m.group(1)

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue

        code = stripped.split("//")[0] if "//" in stripped else stripped

        # ── FOR loops ──
        for_lo = re.search(r'FOR\s+\w+\s*:=\s*(\d+)\s+TO\s+(\d+)\s+DO', code)
        if for_lo:
            lo, hi = int(for_lo.group(1)), int(for_lo.group(2))
            n_iter = hi - lo + 1
            loop_ranges.append(n_iter)
            counts["for_init"] += 1
            counts["for_iter_total"] += n_iter
            counts["int_cmp"] += n_iter      # bound check per iter
            counts["int_add"] += n_iter       # counter increment per iter
            counts["for_overhead"] += n_iter

        # ── END_FOR ──
        if "END_FOR" in code:
            pass  # already counted in FOR

        # ── DB accesses ──
        db_count = code.count(f'"{db_name}".')
        counts["db_access"] += db_count

        # ── REAL Operations ──
        # EXP calls
        exp_count = code.count("EXP(")
        counts["real_exp"] += exp_count

        # Divisions (REAL / REAL)
        # Count '/' not in comments, not in array decls
        div_count = count_operator(code, '/')
        counts["real_div"] += div_count

        # Multiplications
        mul_count = count_mul_in_line(code)
        counts["real_mul"] += mul_count

        # Additions / Subtractions (heuristic: after operations)
        # Count '+' between operands (not array init commas)
        add_count = count_operator(code, '+')
        counts["real_add"] += add_count

        # Subtract
        sub_count = count_operator(code, '-')
        counts["real_sub"] += sub_count

        # ── Comparisons ──
        cmp_count = len(re.findall(r'>=|<=|>|<|=', code))
        # Exclude := assignments
        assign_op_count = code.count(":=")
        cmp_count -= assign_op_count  # := is not a comparison
        counts["real_cmp"] += max(0, cmp_count)

        # ── Assignments ──
        counts["real_assign"] += assign_op_count

        # ── Array access patterns ──
        # 1D: v1[i] or v1[expr]
        arr_1d = len(re.findall(r'\w+\[\w+\]', code))
        # 2D: computed index = o*N + i
        arr_2d_computed = len(re.findall(r'o\s*\*\s*\d+\s*\+\s*i', code))
        counts["array_1d"] += arr_1d
        counts["array_2d"] += arr_2d_computed
        counts["int_mul"] += arr_2d_computed  # o*N is INT multiplication
        counts["int_add"] += arr_2d_computed  # o*N + i is INT addition

        # ── IF-THEN branches ──
        if_count = code.count("IF ")
        counts["branch"] += if_count

    # ── Architecture-specific derived counts ──
    # MatMul unrolled operations are explicit in the SCL
    # Each v2[k] := ... line with 28 products = 28 MUL + 28 ADD + 1 bias ADD

    return dict(counts), db_name, loop_ranges


def count_operator(code: str, op: str) -> int:
    """Count an operator occurrence in code, excluding array size decls."""
    # Remove array declarations like [0..27]
    cleaned = re.sub(r'\[\d+\.\.\d+\]', '', code)
    # Remove string literals
    cleaned = re.sub(r'"[^"]*"', '', cleaned)
    return cleaned.count(op)


def estimate_cycle_time(counts: dict, timing: dict) -> dict:
    """Compute per-instruction and total cycle time."""
    breakdown = {}
    total = 0.0
    for op, count in sorted(counts.items()):
        if op.startswith("for_"):
            continue  # derived metric
        if count == 0:
            continue
        t_per_op = timing.get(op, 0.0)
        t_total = count * t_per_op
        breakdown[op] = {
            "count": count,
            "us_per_op": t_per_op,
            "us_total": round(t_total, 4),
        }
        total += t_total

    return {"breakdown": breakdown, "total_us": round(total, 2)}


def analyze_scl(scl_fb_path: str, target: str = "s7-1200") -> dict:
    """Full analysis of one SCL file pair (DB + FB)."""
    fb_path = Path(scl_fb_path)
    db_path = fb_path.parent / fb_path.name.replace("_fb.scl", ".scl")

    # Analyze FB
    counts_fb, db_name, loop_ranges = count_ops_in_scl(str(fb_path))

    # DB file doesn't execute — it's only DATA_BLOCK with initial values
    # But we note its size for memory analysis

    timing = TIMING_S71200 if target == "s7-1200" else TIMING_S71500
    result_fb = estimate_cycle_time(counts_fb, timing)

    # Additional derived metrics
    n_matmul_mul = counts_fb.get("real_mul", 0)
    n_matmul_add = counts_fb.get("real_add", 0)
    n_exp = counts_fb.get("real_exp", 0)
    n_div = counts_fb.get("real_div", 0)
    n_db = counts_fb.get("db_access", 0)

    # Architecture summary
    arch_ops = {
        "matmul_mul_add": n_matmul_mul + n_matmul_add,
        "activation_exp": n_exp + n_div,  # SiLU = DIV + EXP
        "lut_interp": n_db + counts_fb.get("array_1d", 0) + counts_fb.get("array_2d", 0),
        "loop_iterations": counts_fb.get("for_iter_total", 0),
        "comparisons": counts_fb.get("real_cmp", 0),
    }

    total_us = result_fb["total_us"]
    total_ops = sum(v["count"] for v in result_fb["breakdown"].values())

    db_size_kb = 0
    if db_path.exists():
        db_size_kb = db_path.stat().st_size / 1024

    return {
        "target": target,
        "scl_fb_path": str(fb_path),
        "scl_db_path": str(db_path),
        "db_name": db_name,
        "db_file_size_kb": round(db_size_kb, 1),
        "fb_file_lines": sum(1 for _ in open(fb_path, encoding="utf-8")) if fb_path.exists() else 0,
        "raw_counts": counts_fb,
        "arch_summary": arch_ops,
        "timing": result_fb,
        "derived": {
            "total_operations": total_ops,
            "estimated_us_per_inference": total_us,
            "estimated_ms_per_inference": round(total_us / 1000, 4),
            "estimated_inferences_per_second": round(1e6 / max(total_us, 1), 1),
            "estimated_cycle_time_ms": round(total_us / 1000, 3),
        },
    }


def print_report(results: list[dict]):
    """Print a human-readable comparison report."""
    print("=" * 80)
    print("NeuroPLC - SCL Instruction-Level Cycle Count Analysis")
    print("=" * 80)

    for r in results:
        d = r["derived"]
        a = r["arch_summary"]
        print(f"\n{'-' * 80}")
        print(f"Target: {r['target'].upper()}")
        print(f"DB:     {r['db_name']} ({r['db_file_size_kb']:.0f} KB)")
        print(f"FB:     {r['fb_file_lines']} lines SCL")
        print(f"{'-' * 80}")

        print(f"\n  Architecture Summary:")
        print(f"    MatMul operations (MUL+ADD): {a['matmul_mul_add']:,}")
        print(f"    Activation calls (EXP+DIV):  {a['activation_exp']:,}")
        print(f"    LUT interpolation accesses:   {a['lut_interp']:,}")
        print(f"    Total loop iterations:        {a['loop_iterations']:,}")
        print(f"    Comparisons:                  {a['comparisons']:,}")

        print(f"\n  Top-10 Instructions by Time:")
        breakdown = r["timing"]["breakdown"]
        sorted_ops = sorted(breakdown.items(), key=lambda x: x[1]["us_total"], reverse=True)
        for op, info in sorted_ops[:10]:
            pct = info["us_total"] / r["timing"]["total_us"] * 100
            bar = "#" * int(pct / 2)
            print(f"    {op:<20s} {info['count']:>8,} x {info['us_per_op']:6.3f} us "
                  f"= {info['us_total']:>10.2f} us ({pct:5.1f}%) {bar}")

        print(f"\n  Performance Estimate:")
        print(f"    Total operations in FB:       {d['total_operations']:,}")
        print(f"    Est. per-inference time:      {d['estimated_us_per_inference']:.2f} us")
        print(f"                                 = {d['estimated_ms_per_inference']:.4f} ms")
        print(f"    Max inferences/sec:           {d['estimated_inferences_per_second']:.1f} Hz")
        print(f"    Typical PLC cycle time:       1-10 ms (100-1000 Hz)")
        fit = "WELL WITHIN budget" if d['estimated_ms_per_inference'] < 5 else "May need cycle optimization"
        print(f"    Feasibility:                  {fit}")

    print(f"\n{'=' * 80}")
    print("Note: These are static instruction-level estimates.")
    print("Actual runtime includes OS overhead, communication, and I/O refresh.")
    print("PLCSIM Advanced or physical PLC measurement recommended for final numbers.")
    print("=" * 80)


def main():
    parser = ArgumentParser(description="SCL Cycle Count Analyzer")
    parser.add_argument("--scl", type=str, default=None,
                        help="Path to SCL FB file (default: analyze all)")
    parser.add_argument("--target", type=str, default="s7-1200",
                        choices=["s7-1200", "s7-1500"],
                        help="PLC target")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all SCL files in results/scl_output/")
    args = parser.parse_args()

    results = []

    if args.scl:
        r = analyze_scl(args.scl, args.target)
        results.append(r)
    elif args.all:
        for scl_file in sorted(SCL_DIR.glob("*_db_fb.scl")):
            target = "s7-1200" if "1200" in scl_file.name else "s7-1500"
            r = analyze_scl(str(scl_file), target)
            results.append(r)
    else:
        # Default: analyze KAN S7-1200 and S7-1500 DB+FB
        for variant in ["kan_s7-1200_db_fb.scl", "kan_s7-1500_db_fb.scl",
                        "mlp_s7-1200_db_fb.scl", "mlp_s7-1500_db_fb.scl"]:
            scl_path = SCL_DIR / variant
            if scl_path.exists():
                target = "s7-1200" if "1200" in variant else "s7-1500"
                r = analyze_scl(str(scl_path), target)
                results.append(r)

    print_report(results)

    # Save JSON
    json_out = args.output or str(OUT_DIR / "cycle_count_report.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nJSON report saved → {json_out}")

    return results


if __name__ == "__main__":
    main()
