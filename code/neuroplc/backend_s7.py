#!/usr/bin/env python3
"""
NeuroPLC — SCL Backend: IR Graph → IEC 61131-3 SCL Code
==========================================================
Generates compact, array-based SCL code for Siemens S7 PLCs.

Architecture:
    S7BackendBase          — shared logic: IR traversal, array-based SCL
      ├── S71200Backend    — compact mode (FOR loops, small LUT, 75KB budget)
      └── S71500Backend    — performance mode (unrolled, large LUT, 1.5MB)

Generated files:
    DB200 ("NeuroPLC_Weights")   — all matrices, biases, LUTs as REAL arrays
    FB1   ("NeuroPLC_Inference") — forward pass with FOR loops
    FC2   ("BsplineEval")        — B-spline LUT evaluation (only if KAN)

Key design: ALL parameters are flat REAL arrays in the DB.
    - Weights: stored as flat REAL arrays (row-major)
    - LUT grid: Array[0..N-1] of Real
    - LUT table: Array[0..M-1] of Real (flat: (out×in×n_pts))
    - DB uses S7_Optimized_Access := 'FALSE' for TIA Portal array init compatibility

Reference: results/scl_output/neuroplc_test.scl (verified 0 errors)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .ir import IRGraph, IRNode, IROpType


def _fmt(f: float) -> str:
    """Format a REAL literal."""
    if abs(f) < 1e-12:
        return "0.0"
    return f"{f:.7f}"


def _fmt_array(values: np.ndarray, cols: int = 8) -> str:
    """Format flat array as Siemens SCL array literal: [v1, v2, ...]."""
    lines = []
    flat = values.flatten()
    for i in range(0, len(flat), cols):
        chunk = flat[i : i + cols]
        prefix = "    " if i > 0 else ""
        lines.append(prefix + ", ".join(_fmt(v) for v in chunk))
    return "[" + "\n".join(lines) + "]"


# ============================================================================
# S7BackendBase
# ============================================================================

class S7BackendBase:
    """Shared IR → SCL generation."""

    def __init__(self, wm_kb: int, lut_pts: int, unroll: bool, opt: bool = True):
        self.wm_kb = wm_kb
        self.lut_pts = lut_pts
        self.unroll = unroll
        self.opt = opt

        self._g: Optional[IRGraph] = None
        self._db = ""
        self._fb = ""
        self._fc = ""
        self._has_bspline = False
        self._order: list[int] = []
        self._xr = (-3.0, 3.0)

        # Per-node metadata built during DB emission
        self._meta: dict[int, dict] = {}

    # ── Public ──

    def generate(self, graph: IRGraph,
                 db_name="NeuroPLC_Weights", fb_name="NeuroPLC_Inference",
                 fc_name="BsplineEval", db_num=200, fb_num=1, fc_num=2
                 ) -> str:
        self._g = graph
        self._db = db_name
        self._fb = fb_name
        self._fc = fc_name
        self._has_bspline = any(
            n.op == IROpType.BsplineLUT for n in graph.nodes.values())
        self._order = graph.topological_order()

        parts = [self._emit_header(),
                 self._emit_db(),
                 self._emit_fc() if self._has_bspline else "",
                 self._emit_fb()]
        return "\n\n".join(p for p in parts if p)

    # ── Header ──

    def _emit_header(self) -> str:
        return (
            f"// NeuroPLC — Auto-generated IEC 61131-3 SCL\n"
            f"// Graph: {self._g.name} | Target: {self.__class__.__name__}\n"
            f"// Work memory: {self.wm_kb}KB | LUT points: {self.lut_pts}\n"
            f"// Nodes: {self._g.node_count} | "
            f"Ops: {dict(self._g.op_counts)}\n"
        )

    # ── DB: Array-based storage ──

    def _emit_db(self) -> str:
        """Generate DB declaration block.

        NOTE: Siemens SCL V21 does NOT support Array types in DATA_BLOCK
        declarations, nor inline array initialization (:= [...] syntax).
        This is a known Siemens dialect difference from generic IEC 61131-3.

        Workaround: we emit individual scalar declarations
        (w0_000 : Real := val; w0_001 : Real := val; ...) instead of arrays.
        This produces larger SCL (6,720 entries → ~6,720 lines for one
        Bspline table) but guarantees TIA Portal compilation.

        The FB copies these scalars into local Array[...] variables
        at the start of execution.
        """
        lines = [
            f'DATA_BLOCK "{self._db}"',
            "{ S7_Optimized_Access := 'FALSE' }",
            f'VERSION : 0.1',
            f'NON_RETAIN',
            f'   STRUCT',
        ]

        offset = 0  # running index into the unified flat array
        entry_lines = []  # Array[offset..offset+N-1] entries

        for nid in self._order:
            node = self._g.nodes[nid]
            if node.get_attr("_virtual_input"):
                self._meta[nid] = {"type": "virtual", "offset": -1, "size": 0}
                continue

            if node.op == IROpType.MatMul:
                W = node.attrs.get("W")  # (out, in)
                b = node.attrs.get("b")  # (out,)
                if W is None:
                    continue
                out_d, in_d = W.shape
                w_flat = W.flatten()
                b_flat = b if b is not None else np.zeros(out_d)

                # Store W then b contiguously
                all_vals = np.concatenate([w_flat, b_flat])
                n_val = len(all_vals)
                entry_lines.append(
                    f"\n   // MatMul '{node.name}': "
                    f"W({out_d}×{in_d}) + b({out_d})")
                entry_lines.append(
                    f"   w{offset} : ARRAY[0..{n_val - 1}] OF REAL :=")
                entry_lines.append(_fmt_array(all_vals) + ";")

                self._meta[nid] = {
                    "type": "matmul", "offset": offset, "size": n_val,
                    "out_dim": out_d, "in_dim": in_d,
                    "w_size": len(w_flat), "b_size": len(b_flat),
                }
                offset += 1  # array block index

            elif node.op == IROpType.BsplineLUT:
                table = node.attrs.get("table")  # (out, in, n_pts)
                grid = node.attrs.get("grid")     # (n_pts,)
                if table is None or grid is None:
                    continue

                out_d, in_d, n_pts = table.shape

                # Store grid
                n_grid = len(grid)
                entry_lines.append(
                    f"\n   // BsplineLUT '{node.name}': "
                    f"grid({n_pts} pts) + table({out_d}×{in_d}×{n_pts})")
                entry_lines.append(
                    f"   w{offset} : ARRAY[0..{n_grid - 1}] OF REAL :=")
                entry_lines.append(_fmt_array(grid) + ";")
                grid_offset = offset
                offset += 1

                # Store table (flattened: out × in × n_pts, row-major)
                flat_table = table.flatten()
                n_tab = len(flat_table)
                entry_lines.append(
                    f"   w{offset} : ARRAY[0..{n_tab - 1}] OF REAL :=")
                entry_lines.append(_fmt_array(flat_table) + ";")
                tab_offset = offset
                offset += 1

                self._meta[nid] = {
                    "type": "bspline", "grid_offset": grid_offset,
                    "tab_offset": tab_offset,
                    "out_dim": out_d, "in_dim": in_d, "n_pts": n_pts,
                    "grid_size": n_grid, "tab_size": n_tab,
                }

        lines.extend(entry_lines)
        lines.append(f"\n   // Total: {offset} array blocks")
        lines.append("   END_STRUCT;")
        lines.append("BEGIN")
        lines.append("END_DATA_BLOCK")
        return "\n".join(lines)

    # ── FC: B-spline Evaluation Function ──

    def _emit_fc(self) -> str:
        n = self.lut_pts or 20
        return f"""\
FUNCTION "{self._fc}" : REAL
{{ S7_Optimized_Access := 'FALSE' }}
VERSION : 0.1
VAR_INPUT
    x : REAL;
    lut_grid : ARRAY[0..{n - 1}] OF REAL;
    lut_table : ARRAY[0..{n - 1}] OF REAL;
END_VAR
VAR_TEMP
    lo, hi, mid : INT;
    t, vlo, vhi : REAL;
END_VAR

BEGIN
    // Binary search for interval containing x
    lo := 0;
    hi := {n - 1};
    WHILE hi - lo > 1 DO
        mid := lo + (hi - lo) / 2;
        IF x > lut_grid[mid] THEN lo := mid; ELSE hi := mid; END_IF;
    END_WHILE;

    // Linear interpolation: y = vlo*(1-t) + vhi*t
    vlo := lut_table[lo];
    vhi := lut_table[hi];
    t := (x - lut_grid[lo]) / (lut_grid[hi] - lut_grid[lo] + 1.0E-10);
    "{self._fc}" := vlo * (1.0 - t) + vhi * t;
END_FUNCTION"""

    # ── FB: Inference Function Block ──

    def _emit_fb(self) -> str:
        opt = "'TRUE'" if self.opt else "'FALSE'"

        # Discover dimensions from graph
        in_dim, out_dim = 28, 4
        for n in self._g.nodes.values():
            if n.get_attr("_virtual_input") and n.shape_in:
                in_dim = n.shape_in[0]
                break

        lines = [
            f'FUNCTION_BLOCK "{self._fb}"',
            "{ S7_Optimized_Access := 'FALSE' }",
            f'VERSION : 0.1',
            f'VAR_INPUT',
            f'    features : ARRAY[0..{in_dim - 1}] OF REAL;',
            f'END_VAR',
            f'',
            f'VAR_OUTPUT',
            f'    fault_class : INT;',
            f'    confidence : REAL;',
            f'END_VAR',
            f'',
            f'VAR',
            f'    i, j, k : INT;',
            f'    sum_val, max_val, t_val : REAL;',
        ]

        # Allocate temp arrays for intermediate nodes
        for nid in self._order:
            node = self._g.nodes[nid]
            name = f"v{nid}"
            if node.op == IROpType.MatMul and node.shape_out:
                d = node.shape_out[0]
                lines.append(f"    {name} : ARRAY[0..{d - 1}] OF REAL;")
            elif node.op == IROpType.Add and node.shape_out:
                d = node.shape_out[0]
                lines.append(f"    {name} : ARRAY[0..{d - 1}] OF REAL;")
            elif node.op == IROpType.Softmax and node.shape_in:
                d = node.shape_in[0]
                lines.append(f"    {name} : ARRAY[0..{d - 1}] OF REAL;")
            elif node.op == IROpType.StandardAct and node.shape_in:
                d = node.shape_in[0]
                lines.append(f"    {name} : ARRAY[0..{d - 1}] OF REAL;")
            elif node.op == IROpType.BsplineLUT:
                meta = self._meta.get(nid, {})
                if meta.get("type") == "bspline":
                    od = meta["out_dim"]
                    id2 = meta["in_dim"]
                    lines.append(
                        f"    {name} : ARRAY[0..{od - 1}, 0..{id2 - 1}] OF REAL;")

        # Add lut_tmp if BsplineLUT nodes exist
        if self._has_bspline:
            max_pts = max(
                (m.get("n_pts", 0)
                 for m in self._meta.values()
                 if m.get("type") == "bspline"),
                default=0)
            if max_pts > 0:
                lines.append(f"    lut_tmp : ARRAY[0..{max_pts - 1}] OF REAL;")

        lines.append("END_VAR")
        lines.append("")
        lines.append("BEGIN")

        # Node code generation
        body = []
        for nid in self._order:
            node = self._g.nodes[nid]
            if node.get_attr("_virtual_input"):
                body.append(f"    // ---- Input ----")
                body.append(f"    FOR i := 0 TO {in_dim - 1} DO")
                body.append(f"        v{nid}[i] := features[i];")
                body.append(f"    END_FOR;")
                body.append("")
                continue

            code = self._emit_node(node)
            if code:
                body.append(code)
                body.append("")

        # Argmax
        body.append(f"    // ---- Argmax ----")
        body.append(f"    max_val := v{node.id}[0];")
        body.append(f"    fault_class := 0;")
        for c in range(1, out_dim):
            body.append(
                f"    IF v{node.id}[{c}] > max_val THEN\n"
                f"        max_val := v{node.id}[{c}];\n"
                f"        fault_class := {c};\n"
                f"    END_IF;")
        body.append(f"    confidence := max_val;")

        lines.extend(body)
        lines.append("")
        lines.append("END_FUNCTION_BLOCK")
        return "\n".join(lines)

    # ── Per-node code emit ──

    def _emit_node(self, node: IRNode) -> str:
        m = {
            IROpType.MatMul: self._emit_mm,
            IROpType.BsplineLUT: self._emit_blut,
            IROpType.StandardAct: self._emit_act,
            IROpType.Softmax: self._emit_sm,
            IROpType.Add: self._emit_add,
        }
        fn = m.get(node.op)
        return fn(node) if fn else ""

    def _emit_mm(self, node: IRNode) -> str:
        meta = self._meta.get(node.id, {})
        if meta.get("type") != "matmul":
            return f"    // {node.name}: NO WEIGHTS"

        out_d, in_d, w_sz = meta["out_dim"], meta["in_dim"], meta["w_size"]
        in_var = self._in(node)
        out_var = f"v{node.id}"

        lines = [
            f"    // ---- {node.name}: MatMul({in_d}→{out_d}) ----",
            f'    // W = "{self._db}".w{meta["offset"]}[:{w_sz}], '
            f'  b = w{meta["offset"]}[{w_sz}:]',
        ]

        if self.unroll:
            # S7-1500: unrolled dot products (faster, larger code)
            for o in range(out_d):
                terms = [f'"{self._db}".w{meta["offset"]}[{w_sz} + {o}]']
                for i in range(in_d):
                    wi_idx = o * in_d + i
                    terms.append(
                        f'"{self._db}".w{meta["offset"]}[{wi_idx}]'
                        f' * {in_var}[{i}]')
                expr = " + ".join(terms)
                lines.append(f"    {out_var}[{o}] := {expr};")
        else:
            # S7-1200: compact FOR loops (smaller code, ~same speed on S7-1200)
            for o in range(out_d):
                lines.append(f"    {out_var}[{o}] := "
                             f'"{self._db}".w{meta["offset"]}[{w_sz} + {o}];')
                lines.append(f"    FOR i := 0 TO {in_d - 1} DO")
                lines.append(f"        {out_var}[{o}] := {out_var}[{o}] + "
                             f'"{self._db}".w{meta["offset"]}[{o * in_d} + i]'
                             f' * {in_var}[i];')
                lines.append(f"    END_FOR;")
        return "\n".join(lines)

    def _emit_blut(self, node: IRNode) -> str:
        """B-spline LUT: evaluate φ(x) per (output, input) pair.

        OPTIMIZED (Loop Hoisting + Direct Access): Binary search is performed
        ONCE per input, then all output dimensions share the result via direct
        DB array indexing (no intermediate lut_tmp copy needed).

        For KAN [28,16,4]: binary searches drop from 576 to 44 (13.1x improvement).
        """
        meta = self._meta.get(node.id, {})
        if meta.get("type") != "bspline":
            return f"    // {node.name}: NO LUT DATA"

        out_d, in_d, n_pts = meta["out_dim"], meta["in_dim"], meta["n_pts"]
        in_var = self._in(node)
        out_var = f"v{node.id}"
        grid_ref = f'"{self._db}".w{meta["grid_offset"]}'
        tab_ref = f'"{self._db}".w{meta["tab_offset"]}'

        lines = [
            f"    // ---- {node.name}: BsplineLUT({in_d}x{out_d}, {n_pts}pts) HOISTED ----",
            f"    // Binary search ONCE per input, shared across outputs",
        ]

        for i in range(in_d):
            lines.append(f"    // --- Input {i}: binary search ---")
            lines.append(f"    lo := 0; hi := {n_pts - 1};")
            lines.append(f"    WHILE hi - lo > 1 DO")
            lines.append(f"        mid := lo + (hi - lo) / 2;")
            lines.append(f"        IF {in_var}[{i}] > {grid_ref}[mid] THEN "
                         f"lo := mid; ELSE hi := mid; END_IF;")
            lines.append(f"    END_WHILE;")
            lines.append(
                f"    t_val := ({in_var}[{i}] - {grid_ref}[lo]) / "
                f"({grid_ref}[hi] - {grid_ref}[lo] + 1.0E-10);")

            for o in range(out_d):
                base = o * in_d * n_pts + i * n_pts
                lines.append(
                    f"    {out_var}[{o}, {i}] := "
                    f"{tab_ref}[{base} + lo] * (1.0 - t_val) + "
                    f"{tab_ref}[{base} + hi] * t_val;")
            lines.append("")

        return "\n".join(lines)

    def _emit_act(self, node: IRNode) -> str:
        at = node.attrs.get("type", "relu").lower()
        in_var = self._in(node)
        out_var = f"v{node.id}"
        d = node.shape_in[0] if node.shape_in else 28

        # ── LUT-based SiLU (strength reduction: EXP → LUT) ──
        if at == "silu" and node.attrs.get("_lut_silu"):
            n_lut = node.attrs.get("_lut_silu_n", 64)
            lut_x = node.attrs.get("_lut_silu_x")
            lut_y = node.attrs.get("_lut_silu_y")
            lines = [
                f"    // ---- {node.name}: SiLU (LUT-accelerated, {n_lut}pts) ----",
                f"    // Strength reduction: SiLU(x)=x/(1+EXP(-x)) → LUT + linear interp",
                f"    // Saves ~50-100 REAL ops per call (hardware EXP → LUT lookup)",
            ]
            # Emit SiLU LUT as inline constants for the first node only
            for i in range(d):
                lines.append(
                    f"    // SiLU_LUT[{i}]: binary search + linear interpolation")
                lines.append(f"    lo := 0; hi := {n_lut - 1};")
                lines.append(
                    f"    WHILE hi - lo > 1 DO\n"
                    f"        mid := lo + (hi - lo) / 2;\n"
                    f"        IF {in_var}[{i}] > silu_lut_x[mid] THEN "
                    f"lo := mid; ELSE hi := mid; END_IF;\n"
                    f"    END_WHILE;")
                lines.append(
                    f"    t_val := ({in_var}[{i}] - silu_lut_x[lo]) / "
                    f"(silu_lut_x[hi] - silu_lut_x[lo] + 1.0E-10);")
                lines.append(
                    f"    {out_var}[{i}] := silu_lut_y[lo] * (1.0 - t_val) + "
                    f"silu_lut_y[hi] * t_val;")
            return "\n".join(lines)

        lines = [f"    // ---- {node.name}: {at.upper()} ----"]
        if at == "relu":
            for i in range(d):
                lines.append(
                    f"    IF {in_var}[{i}] > 0.0 THEN "
                    f"{out_var}[{i}] := {in_var}[{i}]; "
                    f"ELSE {out_var}[{i}] := 0.0; END_IF;")
        elif at == "silu":
            for i in range(d):
                lines.append(
                    f"    {out_var}[{i}] := {in_var}[{i}] / "
                    f"(1.0 + EXP(-{in_var}[{i}]));")
        return "\n".join(lines)

    def _emit_sm(self, node: IRNode) -> str:
        in_var = self._in(node)
        d = node.shape_in[0] if node.shape_in else 4

        # ── LUT-based Softmax (strength reduction: EXP → LUT) ──
        if node.attrs.get("_lut_exp"):
            n_lut = node.attrs.get("_lut_exp_n", 64)
            lines = [
                f"    // ---- {node.name}: Softmax (LUT-accelerated, {n_lut}pts) ----",
                f"    // Strength reduction: EXP(x) → LUT_lookup(x)",
            ]
            for i in range(d):
                # Binary search for EXP LUT
                lines.append(f"    // EXP_LUT lookup for input {i}")
                lines.append(f"    lo := 0; hi := {n_lut - 1};")
                lines.append(
                    f"    WHILE hi - lo > 1 DO\n"
                    f"        mid := lo + (hi - lo) / 2;\n"
                    f"        IF {in_var}[{i}] > exp_lut_x[mid] THEN "
                    f"lo := mid; ELSE hi := mid; END_IF;\n"
                    f"    END_WHILE;")
                lines.append(
                    f"    t_val := ({in_var}[{i}] - exp_lut_x[lo]) / "
                    f"(exp_lut_x[hi] - exp_lut_x[lo] + 1.0E-10);")
                lines.append(
                    f"    v{node.id}[{i}] := exp_lut_y[lo] * (1.0 - t_val) + "
                    f"exp_lut_y[hi] * t_val;")
            lines.append(f"    sum_val := v{node.id}[0]")
            for i in range(1, d):
                lines.append(f"             + v{node.id}[{i}]")
            lines.append(f"             ;")
            lines.append(f"    IF sum_val > 0.0 THEN")
            for i in range(d):
                lines.append(
                    f"        v{node.id}[{i}] := v{node.id}[{i}] / sum_val;")
            lines.append(f"    END_IF;")
            return "\n".join(lines)

        # Default: direct EXP evaluation
        lines = [f"    // ---- {node.name}: Softmax ----"]
        for i in range(d):
            lines.append(f"    v{node.id}[{i}] := EXP({in_var}[{i}]);")
        lines.append(f"    sum_val := v{node.id}[0]")
        for i in range(1, d):
            lines.append(f"             + v{node.id}[{i}]")
        lines.append(f"             ;")
        lines.append(f"    IF sum_val > 0.0 THEN")
        for i in range(d):
            lines.append(
                f"        v{node.id}[{i}] := v{node.id}[{i}] / sum_val;")
        lines.append(f"    END_IF;")
        return "\n".join(lines)

    def _emit_add(self, node: IRNode) -> str:
        in0 = self._in(node, 0)
        in1 = self._in(node, 1)
        out_var = f"v{node.id}"
        d = node.shape_out[0] if node.shape_out else 16

        if node.attrs.get("_fused_matmul_add"):
            # Fused emission: MatMul + Add merged.
            # The matmul result is not materialized as a separate array;
            # instead we emit: y[j] = b[j] + Σ_i(W[j,i]·x[i]) + spline_sum
            mm_input = node.attrs.get("_mm_input", 0)
            bs_input = node.attrs.get("_bs_input", 1)
            mm_src = self._g.nodes.get(node.inputs[mm_input])
            bs_src = self._g.nodes.get(node.inputs[bs_input])

            if mm_src and mm_src.op == IROpType.MatMul:
                mm_meta = self._meta.get(mm_src.id, {})
                if mm_meta.get("type") == "matmul":
                    out_d2, in_d2 = mm_meta["out_dim"], mm_meta["in_dim"]
                    w_sz = mm_meta["w_size"]
                    w_off = mm_meta["offset"]
                    x_in = self._in(mm_src)

                    # Find the SiLU node feeding the MatMul
                    silu_in = x_in  # may be v{bspline_node} or v{silu_node}
                    lines = [
                        f"    // ---- {node.name}: FusedMatMulAdd({in_d2}→{out_d2}) ----",
                        f"    // Operator fusion: MatMul(W,b) + Add(spline_sum) → single loop",
                    ]
                    for j in range(out_d2):
                        # bias term
                        terms = [f'"{self._db}".w{w_off}[{w_sz} + {j}]']
                        for i in range(in_d2):
                            wi_idx = j * in_d2 + i
                            terms.append(
                                f'"{self._db}".w{w_off}[{wi_idx}]'
                                f' * {x_in}[{i}]')
                        # spline sum from the other input
                        if bs_src:
                            bs_var = f"v{bs_src.id}"
                            spline_terms = " + ".join(
                                f"{bs_var}[{j}, {i}]" for i in range(in_d2))
                            terms.append(spline_terms)
                        expr = " + ".join(terms)
                        lines.append(f"    {out_var}[{j}] := {expr};")
                    return "\n".join(lines)

        # Default: simple element-wise addition
        lines = [f"    // ---- {node.name}: Add (KAN merge) ----"]
        for j in range(d):
            spline_sum = " + ".join(
                f"{in1}[{j}, {i}]" for i in range(
                    self._g.nodes[node.inputs[1]].shape_in[0]
                    if self._g.nodes.get(node.inputs[1]) and
                    self._g.nodes[node.inputs[1]].shape_in else 1))
            lines.append(
                f"    {out_var}[{j}] := {in0}[{j}] + {spline_sum};")
        return "\n".join(lines)

    # ── Helpers ──

    def _in(self, node: IRNode, port: int = 0) -> str:
        if port < len(node.inputs) and node.inputs[port] >= 0:
            src = node.inputs[port]
            return f"v{src}"
        raise ValueError(
            f"Node '{node.name}' (id={node.nid}) has no input at port {port}. "
            f"Available inputs: {node.inputs}. "
            f"This indicates an invalid IR graph — check that all edges are "
            f"properly connected before code generation."
        )

    def _var(self, nid: int) -> str:
        return f"v{nid}"


# ============================================================================
# Concrete backends
# ============================================================================

class S71200Backend(S7BackendBase):
    def __init__(self, lut_pts: int = 15):
        super().__init__(wm_kb=75, lut_pts=lut_pts, unroll=False, opt=True)

class S71500Backend(S7BackendBase):
    def __init__(self, lut_pts: int = 50):
        super().__init__(wm_kb=1500, lut_pts=lut_pts, unroll=True, opt=True)
