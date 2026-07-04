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
    - DB uses S7_Optimized_Access for S7-1200

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
    """Format flat array as comma-separated REALs, cols per line."""
    lines = []
    flat = values.flatten()
    for i in range(0, len(flat), cols):
        chunk = flat[i : i + cols]
        lines.append("    " + ", ".join(_fmt(v) for v in chunk))
    return ",\n".join(lines)


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
        """Generate DB with REAL arrays.

        For each IR node that holds parameters (MatMul, BsplineLUT),
        create Array[0..N-1] of Real entries. Metadata dict tracks
        offsets, shapes, and sizes for FB code generation.
        """
        opt = "'TRUE'" if self.opt else "'FALSE'"
        lines = [
            f'DATA_BLOCK "{self._db}"',
            f'{{ S7_Optimized_Access := {opt} }}',
            f'VERSION : 0.1',
            f'NON_RETAIN',
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
                    f"   w{offset} : Array[0..{n_val - 1}] of Real :=")
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
                    f"   w{offset} : Array[0..{n_grid - 1}] of Real :=")
                entry_lines.append(_fmt_array(grid) + ";")
                grid_offset = offset
                offset += 1

                # Store table (flattened: out × in × n_pts, row-major)
                flat_table = table.flatten()
                n_tab = len(flat_table)
                entry_lines.append(
                    f"   w{offset} : Array[0..{n_tab - 1}] of Real :=")
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
        lines.append("END_DATA_BLOCK")
        return "\n".join(lines)

    # ── FC: B-spline Evaluation Function ──

    def _emit_fc(self) -> str:
        n = self.lut_pts or 20
        opt = "'TRUE'" if self.opt else "'FALSE'"
        return f"""\
FUNCTION "{self._fc}" : Real
{{ S7_Optimized_Access := {opt} }}
VERSION : 0.1
VAR_INPUT
    x : Real;
    lut_grid : Array[0..{n - 1}] of Real;
    lut_table : Array[0..{n - 1}] of Real;
END_VAR
VAR
    lo, hi, mid : Int;
    t, vlo, vhi : Real;
END_VAR

BEGIN
    // Binary search for interval containing x
    lo := 0;
    hi := {n - 1};
    WHILE hi - lo > 1 DO
        mid := (lo + hi) / 2;
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
            f'{{ S7_Optimized_Access := {opt} }}',
            f'VERSION : 0.1',
            f'VAR_INPUT',
            f'    features : Array[0..{in_dim - 1}] of Real;',
            f'END_VAR',
            f'',
            f'VAR_OUTPUT',
            f'    fault_class : Int;',
            f'    confidence : Real;',
            f'END_VAR',
            f'',
            f'VAR',
            f'    i, j, k : Int;',
            f'    sum_val, max_val, t_val : Real;',
        ]

        # Allocate temp arrays for intermediate nodes
        for nid in self._order:
            node = self._g.nodes[nid]
            name = f"v{nid}"
            if node.op == IROpType.MatMul and node.shape_out:
                d = node.shape_out[0]
                lines.append(f"    {name} : Array[0..{d - 1}] of Real;")
            elif node.op == IROpType.Add and node.shape_out:
                d = node.shape_out[0]
                lines.append(f"    {name} : Array[0..{d - 1}] of Real;")
            elif node.op == IROpType.Softmax and node.shape_in:
                d = node.shape_in[0]
                lines.append(f"    {name} : Array[0..{d - 1}] of Real;")
            elif node.op == IROpType.StandardAct and node.shape_in:
                d = node.shape_in[0]
                lines.append(f"    {name} : Array[0..{d - 1}] of Real;")
            elif node.op == IROpType.BsplineLUT:
                meta = self._meta.get(nid, {})
                if meta.get("type") == "bspline":
                    od = meta["out_dim"]
                    id2 = meta["in_dim"]
                    lines.append(
                        f"    {name} : Array[0..{od - 1}, 0..{id2 - 1}] of Real;")

        # Add lut_tmp if BsplineLUT nodes exist
        if self._has_bspline:
            max_pts = max(
                (m.get("n_pts", 0)
                 for m in self._meta.values()
                 if m.get("type") == "bspline"),
                default=0)
            if max_pts > 0:
                lines.append(f"    lut_tmp : Array[0..{max_pts - 1}] of Real;")

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
        body.append(f"    max_val := softmax_out[0];")
        body.append(f"    fault_class := 0;")
        for c in range(1, out_dim):
            body.append(
                f"    IF softmax_out[{c}] > max_val THEN\n"
                f"        max_val := softmax_out[{c}];\n"
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
        for o in range(out_d):
            terms = [f'"{self._db}".w{meta["offset"]}[{w_sz} + {o}]']
            for i in range(in_d):
                wi_idx = o * in_d + i
                terms.append(
                    f'"{self._db}".w{meta["offset"]}[{wi_idx}]'
                    f' * {in_var}[{i}]')
            expr = " + ".join(terms)
            lines.append(f"    {out_var}[{o}] := {expr};")
        return "\n".join(lines)

    def _emit_blut(self, node: IRNode) -> str:
        """B-spline LUT: evaluate φ(x) per (output, input) pair.

        Uses FC2 (BsplineEval) for each pair. Results stored in 2D array.
        """
        meta = self._meta.get(node.id, {})
        if meta.get("type") != "bspline":
            return f"    // {node.name}: NO LUT DATA"

        out_d, in_d, n_pts = meta["out_dim"], meta["in_dim"], meta["n_pts"]
        in_var = self._in(node)
        out_var = f"v{node.id}"
        grid_ref = f'"{self._db}".w{meta["grid_offset"]}'
        tab_ref = f'"{self._db}".w{meta["tab_offset"]}'
        fc = f'"{self._fc}"'

        lines = [
            f"    // ---- {node.name}: BsplineLUT({in_d}×{out_d}, {n_pts}pts) ----",
        ]
        for o in range(out_d):
            for i in range(in_d):
                # Index into flat table: o * (in_d * n_pts) + i * n_pts
                base = o * in_d * n_pts + i * n_pts
                lines.append(
                    f"    // φ[{i}→{o}]: evaluate LUT "
                    f"(table offset {base})")
                # In SCL, we need to pass the correct slice of the table
                # We'll use a loop to copy the relevant slice
                lines.append(
                    f"    FOR k := 0 TO {n_pts - 1} DO")
                lines.append(
                    f"        lut_tmp[k] := {tab_ref}[{base} + k];")
                lines.append(
                    f"    END_FOR;")
                lines.append(
                    f"    {out_var}[{o}, {i}] := "
                    f"{fc}(x := {in_var}[{i}], "
                    f"lut_grid := {grid_ref}, "
                    f"lut_table := lut_tmp);")

        return "\n".join(lines)

    def _emit_act(self, node: IRNode) -> str:
        at = node.attrs.get("type", "relu").lower()
        in_var = self._in(node)
        out_var = f"v{node.id}"
        d = node.shape_in[0] if node.shape_in else 28

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
        lines = [f"    // ---- {node.name}: Softmax ----"]
        for i in range(d):
            lines.append(f"    softmax_out[{i}] := EXP({in_var}[{i}]);")
        lines.append(f"    sum_val := softmax_out[0]")
        for i in range(1, d):
            lines.append(f"             + softmax_out[{i}]")
        lines.append(f"             ;")
        lines.append(f"    IF sum_val > 0.0 THEN")
        for i in range(d):
            lines.append(
                f"        softmax_out[{i}] := softmax_out[{i}] / sum_val;")
        lines.append(f"    END_IF;")
        return "\n".join(lines)

    def _emit_add(self, node: IRNode) -> str:
        in0 = self._in(node, 0)
        in1 = self._in(node, 1)
        out_var = f"v{node.id}"
        d = node.shape_out[0] if node.shape_out else 16
        lines = [f"    // ---- {node.name}: Add (KAN merge) ----"]
        for j in range(d):
            lines.append(
                f"    {out_var}[{j}] := {in0}[{j}] + {in1}[{j}];")
        return "\n".join(lines)

    # ── Helpers ──

    def _in(self, node: IRNode, port: int = 0) -> str:
        if port < len(node.inputs) and node.inputs[port] >= 0:
            src = node.inputs[port]
            src_node = self._g.nodes.get(src)
            if src_node and src_node.op == IROpType.Softmax:
                return "softmax_out"
            return f"v{src}"
        return "ERROR"

    def _var(self, nid: int) -> str:
        node = self._g.nodes.get(nid)
        if node and node.op == IROpType.Softmax:
            return "softmax_out"
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
