#!/usr/bin/env python3
"""
NeuroPLC — Theorem 10: WCET (Worst-Case Execution Time) Analysis
=================================================================
Derives the theoretical WCET bound for compiled KAN SCL on Siemens S7-1200 CPU 1211C.

S7-1200 CPU 1211C instruction timing (from Siemens technical data):
  - Basic bit/logic/MOV: 0.075 us
  - Integer operations (ADD/SUB/MUL/DIV): 0.2 us
  - REAL (IEEE 754 soft-float) MUL: ~3.5 us
  - REAL (IEEE 754 soft-float) ADD: ~1.8 us
  - REAL (IEEE 754 soft-float) DIV: ~8 us
  - REAL (IEEE 754 soft-float) EXP: ~14 us
  - Array element access (indexed): ~2 us
  - Comparison (REAL): ~1.5 us
  - FB call overhead: ~30 us

For KAN [28,16,4]:
  - L0: 28→16 edges: 28×16 = 448 B-spline LUTs
  - L1: 16→4 edges: 16×4 = 64 B-spline LUTs
  - Total: 512 B-spline LUT evaluations
  - Per-LUT: 15-point array lookup + linear interpolation

Authors: NeuroPLC Qualitative Leap Plan — Day 2
"""

import json
from pathlib import Path

# ── S7-1200 CPU 1211C timing constants (us) ──
T = {
    'basic': 0.075,
    'int_op': 0.2,
    'real_mul': 3.5,
    'real_add': 1.8,
    'real_div': 8.0,
    'real_exp': 14.0,
    'array_access': 2.0,
    'real_cmp': 1.5,
    'fb_overhead': 30.0,
    'loop_overhead': 1.0,  # per iteration
}

# ── Architectural params ──
ARCH = [28, 16, 4]
N_LUT = 15  # LUT points per B-spline

def count_ops_per_edge(n_lut):
    """Count REAL operations per B-spline LUT edge evaluation.

    LUT linear interpolation: for input x in [grid[k], grid[k+1]):
        t = (x - grid[k]) / (grid[k+1] - grid[k])
        y = lut[k] + t * (lut[k+1] - lut[k])

    Operations:
        - Binary search for k: ~4 comparisons
        - grid access: 2 array reads
        - subtract: x - grid[k] → 1 REAL SUB
        - subtract: grid[k+1] - grid[k] → 1 REAL SUB
        - divide: t = ... → 1 REAL DIV
        - lut access: 2 array reads
        - subtract: lut[k+1] - lut[k] → 1 REAL SUB
        - multiply: t * diff → 1 REAL MUL
        - add: lut[k] + product → 1 REAL ADD
    """
    n_real_sub = 3
    n_real_div = 1
    n_real_mul = 1
    n_real_add = 1
    n_array_access = 4  # grid[k], grid[k+1], lut[k], lut[k+1]
    n_cmp = 4           # binary search
    n_loop = 4           # binary search iterations

    time_per_edge = (
        n_real_sub * T['real_add'] +   # SUB ~ ADD timing
        n_real_div * T['real_div'] +
        n_real_mul * T['real_mul'] +
        n_real_add * T['real_add'] +
        n_array_access * T['array_access'] +
        n_cmp * T['real_cmp'] +
        n_loop * T['loop_overhead']
    )
    return time_per_edge, {
        'real_sub': n_real_sub, 'real_div': n_real_div,
        'real_mul': n_real_mul, 'real_add': n_real_add,
        'array_access': n_array_access, 'cmp': n_cmp, 'loop_iters': n_loop,
    }


def count_ops_per_matmul(in_dim, out_dim):
    """Count operations for one MatMul layer.

    y_j = sum_i w_{j,i} * x_i  (for j=1..out_dim)

    Operations:
        - Multiply: out_dim * in_dim
        - Add (accumulate): out_dim * (in_dim - 1)
        - Array access: out_dim * in_dim (weights) + 1 (bias per output)
    """
    n_mul = out_dim * in_dim
    n_add = out_dim * (in_dim - 1)  # interior accumulations
    if n_add < 0:
        n_add = 0
    n_array = n_mul  # weight reads

    time = n_mul * T['real_mul'] + n_add * T['real_add'] + n_array * T['array_access']
    return time, {'mul': n_mul, 'add': n_add, 'array_reads': n_array}


def compute_wcet(arch, n_lut):
    """Compute total WCET for KAN architecture."""
    layers = len(arch) - 1

    total_time = 0.0
    per_layer = []

    for l in range(layers):
        in_d = arch[l]
        out_d = arch[l + 1]
        n_edges = in_d * out_d

        edge_time, edge_ops = count_ops_per_edge(n_lut)
        layer_edge_time = n_edges * edge_time

        matmul_time, matmul_ops = count_ops_per_matmul(in_d, out_d)

        layer_total = layer_edge_time + matmul_time + T['fb_overhead']
        total_time += layer_total

        per_layer.append({
            'layer': l,
            'in_dim': in_d, 'out_dim': out_d,
            'n_edges': n_edges,
            'edge_time_us': edge_time,
            'layer_edge_total_us': layer_edge_time,
            'matmul_total_us': matmul_time,
            'layer_total_us': layer_total,
            'edge_ops': edge_ops,
            'matmul_ops': matmul_ops,
        })

    # Add Softmax layer (last layer only)
    n_classes = arch[-1]
    softmax_ops = {
        'exp': n_classes,
        'div': n_classes,
        'add': n_classes - 1,
        'array': n_classes * 2,
    }
    softmax_time = (
        n_classes * T['real_exp'] +
        n_classes * T['real_div'] +
        (n_classes - 1) * T['real_add'] +
        n_classes * 2 * T['array_access']
    )
    total_time += softmax_time

    # Add Argmax
    argmax_time = (n_classes - 1) * T['real_cmp'] + n_classes * T['array_access']

    return total_time + argmax_time, per_layer, softmax_time, argmax_time


def main():
    print("=" * 70)
    print("Theorem 10: WCET Analysis — KAN-on-S7-1200 Execution Time")
    print("=" * 70)
    print()

    wcet_us, per_layer, softmax_us, argmax_us = compute_wcet(ARCH, N_LUT)

    print("Architecture: KAN", ARCH)
    print(f"LUT points: {N_LUT}")
    print()
    print("Per-LUT edge operations:")
    edge_time, edge_ops = count_ops_per_edge(N_LUT)
    print(f"  REAL SUB x{edge_ops['real_sub']}:  {edge_ops['real_sub'] * T['real_add']:.1f} us")
    print(f"  REAL DIV x{edge_ops['real_div']}:  {edge_ops['real_div'] * T['real_div']:.1f} us")
    print(f"  REAL MUL x{edge_ops['real_mul']}:  {edge_ops['real_mul'] * T['real_mul']:.1f} us")
    print(f"  REAL ADD x{edge_ops['real_add']}:  {edge_ops['real_add'] * T['real_add']:.1f} us")
    print(f"  Array access x{edge_ops['array_access']}: {edge_ops['array_access'] * T['array_access']:.1f} us")
    print(f"  Comparisons x{edge_ops['cmp']}:  {edge_ops['cmp'] * T['real_cmp']:.1f} us")
    print(f"  Total per edge: {edge_time:.1f} us")
    print()

    for pl in per_layer:
        print(f"Layer {pl['layer']} ({pl['in_dim']}->{pl['out_dim']}):")
        print(f"  Edges: {pl['n_edges']}")
        print(f"  Edge total: {pl['layer_edge_total_us']:.1f} us")
        print(f"  MatMul total: {pl['matmul_total_us']:.1f} us")
        print(f"  Layer total: {pl['layer_total_us']:.1f} us")
        print()

    print(f"Softmax ({ARCH[-1]}-class): {softmax_us:.1f} us")
    print(f"Argmax: {argmax_us:.1f} us")
    print()
    print(f"TOTAL WCET: {wcet_us:.1f} us = {wcet_us/1000:.2f} ms")
    print()

    # Safety analysis
    scan_cycle = 100_000  # 100ms typical
    margin = scan_cycle / wcet_us
    percent = wcet_us / scan_cycle * 100

    print(f"S7-1200 typical scan cycle: {scan_cycle/1000:.0f} ms")
    print(f"WCET margin: {margin:.1f}x")
    print(f"WCET usage: {percent:.1f}% of scan cycle")
    print()

    # General formula
    print("━" * 70)
    print("General WCET Formula (Theorem 10)")
    print("━" * 70)
    print()
    print("For SVNN network N with L layers, E edges, N_lut LUT points:")
    print()
    print(f"  WCET(N) = E * C_lut + E * C_matmul + C_softmax + C_overhead")
    print(f"  C_lut = {edge_time:.1f} us/edge")
    print(f"  C_matmul = per-MAC cost distributed per edge")
    print()
    print(f"For KAN[28,16,4] with N_lut=15:")
    print(f"  WCET = {wcet_us:.1f} us")
    print()

    # Parameterized form
    C_lut, _ = count_ops_per_edge(N_LUT)
    total_matmul_time = sum(pl['matmul_total_us'] for pl in per_layer)
    total_edges = sum(pl['n_edges'] for pl in per_layer)
    C_matmul = total_matmul_time / total_edges

    print(f"Parameterized: WCET(N,E,N_lut) = E * (C_lut({N_LUT}) + C_matmul) + C_softmax")
    print(f"  where C_lut({N_LUT}) = {C_lut:.1f} us, C_matmul = {C_matmul:.1f} us/edge")
    print()

    # Save results
    out = {
        'experiment': 'Theorem 10 (WCET)',
        'arch': ARCH,
        'n_lut': N_LUT,
        'wcet_us': round(wcet_us, 1),
        'wcet_ms': round(wcet_us / 1000, 3),
        'scan_cycle_ms': 100,
        'margin_x': round(margin, 1),
        'usage_pct': round(percent, 1),
        'c_lut_us': round(C_lut, 1),
        'c_matmul_per_edge_us': round(C_matmul, 1),
        'per_layer': per_layer,
        'safety_monitor_overhead_us': 66,  # from Algorithm 3
    }

    out_path = Path('D:/neuroplc-paper/results/theory/wcet_analysis.json')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)

    print(f"[DONE] Results saved to {out_path}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
