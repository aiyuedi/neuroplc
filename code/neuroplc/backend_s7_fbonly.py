#!/usr/bin/env python3
"""
NeuroPLC — FB-Only SCL Backend for Siemens TIA Portal V21
===========================================================
Workaround for Siemens SCL dialect: DATA_BLOCK does not support
Array type declarations or inline initialization (:= [...] syntax).

Design: All parameters (weights, LUT grids, tables) are embedded
inside the FUNCTION_BLOCK as local arrays, initialized on first scan
via assignment statements. No DB is generated.

This produces ~11,000 lines for KAN [28,16,4] — larger than the
generic IEC 61131-3 output, but guaranteed to compile in TIA Portal V21.

Usage:
    from neuroplc.backend_s7_fbonly import S71200FBOnlyBackend
    backend = S71200FBOnlyBackend(lut_pts=15)
    scl = backend.generate(ir_graph)
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from .ir import IRGraph, IROpType
from .backend_s7 import _fmt, _fmt_array


def _fmt_scalar_init(prefix: str, values: np.ndarray,
                     vals_per_line: int = 8) -> str:
    """Format scalar init assignments: w[0] := v0; w[1] := v1; ..."""
    lines = []
    flat = values.flatten()
    for i in range(0, len(flat), vals_per_line):
        chunk = flat[i : i + vals_per_line]
        parts = [f"{prefix}[{i+j}] := {_fmt(v)};" for j, v in enumerate(chunk)]
        lines.append("        " + " ".join(parts))
    return "\n".join(lines)


class S7FBOnlyBackendBase:
    """Base class for FB-only SCL generation (no DB)."""

    def __init__(self, wm_kb: int, lut_pts: int, unroll: bool):
        self.wm_kb = wm_kb
        self.lut_pts = lut_pts
        self.unroll = unroll
        self._g: Optional[IRGraph] = None
        self._has_bspline = False
        self._order: list[int] = []
        self._xr = (-3.0, 3.0)
        self._var_decls: list[str] = []       # VAR section declarations
        self._init_code: list[str] = []       # first-scan init assignment
        self._inference_code: list[str] = []  # main inference body
        self._meta: dict[int, dict] = {}
        self._var_idx = 0   # running index for local array names

    def _next_var(self, prefix: str) -> str:
        name = f"{prefix}{self._var_idx}"
        self._var_idx += 1
        return name

    # ── Public ──

    def generate(self, graph: IRGraph) -> str:
        self._g = graph
        self._has_bspline = any(
            n.op == IROpType.BsplineLUT for n in graph.nodes.values())
        self._order = graph.topological_order()

        # Phase 1: Collect parameter data (var decls + init code)
        self._collect_params()

        # Phase 2: Generate inference code per node
        self._collect_inference()

        # Phase 3: Assemble
        return self._assemble()

    # ── Phase 1: Collect parameters ──

    def _collect_params(self):
        for nid in self._order:
            node = self._g.nodes[nid]
            if node.get_attr("_virtual_input"):
                self._meta[nid] = {"type": "virtual", "var": None}
                continue

            if node.op == IROpType.MatMul:
                self._collect_matmul_params(nid, node)
            elif node.op == IROpType.BsplineLUT:
                self._collect_bspline_params(nid, node)

    def _collect_matmul_params(self, nid, node):
        W = node.attrs.get("W")
        b = node.attrs.get("b")
        if W is None:
            return
        out_d, in_d = W.shape

        w_var = self._next_var("w")
        b_var = self._next_var("b")

        w_flat = W.flatten()
        b_flat = b if b is not None else np.zeros(out_d, dtype=np.float32)

        self._var_decls.append(
            f"        {w_var} : ARRAY[0..{len(w_flat)-1}] OF REAL;"
            f"  // W({out_d}x{in_d})")
        self._var_decls.append(
            f"        {b_var} : ARRAY[0..{len(b_flat)-1}] OF REAL;"
            f"  // bias({out_d})")

        self._init_code.append(
            f"\n        // MatMul '{node.name}': W + bias")
        self._init_code.append(
            _fmt_scalar_init(w_var, w_flat))
        self._init_code.append(
            _fmt_scalar_init(b_var, b_flat))

        self._meta[nid] = {
            "type": "matmul", "w_var": w_var, "b_var": b_var,
            "out_dim": out_d, "in_dim": in_d,
            "w_size": len(w_flat), "b_size": len(b_flat),
        }

    def _collect_bspline_params(self, nid, node):
        table = node.attrs.get("table")
        grid = node.attrs.get("grid")
        if table is None or grid is None:
            return
        out_d, in_d, n_pts = table.shape

        grid_var = self._next_var("g")
        tab_var = self._next_var("t")

        self._var_decls.append(
            f"        {grid_var} : ARRAY[0..{len(grid)-1}] OF REAL;"
            f"  // LUT grid ({n_pts} pts)")
        self._var_decls.append(
            f"        {tab_var} : ARRAY[0..{table.size-1}] OF REAL;"
            f"  // LUT table ({out_d}x{in_d}x{n_pts})")

        self._init_code.append(
            f"\n        // BsplineLUT '{node.name}': grid + table")
        self._init_code.append(
            _fmt_scalar_init(grid_var, grid))
        self._init_code.append(
            _fmt_scalar_init(tab_var, table.flatten()))

        self._meta[nid] = {
            "type": "bspline", "grid_var": grid_var, "tab_var": tab_var,
            "out_dim": out_d, "in_dim": in_d, "n_pts": n_pts,
        }

    # ── Phase 2: Inference code ──

    def _collect_inference(self):
        self._inference_code.append("")
        for nid in self._order:
            node = self._g.nodes[nid]
            if node.get_attr("_virtual_input"):
                self._inference_code.append(
                    f"        // Input: features[0..{node.shape_in[0]-1 if node.shape_in else 27}]")
                continue

            # Build lookup: node ID → output variable name
            # (virtual input nodes use "features")
            varmap = {}
            for nid2 in self._order:
                node2 = self._g.nodes[nid2]
                if node2.get_attr("_virtual_input"):
                    varmap[nid2] = "features"
                else:
                    varmap[nid2] = f"v{nid2}"

            if node.op == IROpType.MatMul:
                self._emit_matmul_inference(nid, node, varmap)
            elif node.op == IROpType.BsplineLUT:
                self._emit_bspline_inference(nid, node, varmap)
            elif node.op == IROpType.StandardAct:
                self._emit_act_inference(nid, node, varmap)
            elif node.op == IROpType.Add:
                self._emit_add_inference(nid, node, varmap)
            elif node.op == IROpType.Softmax:
                self._emit_softmax_inference(nid, node, varmap)
            elif node.op == IROpType.Argmax:
                self._emit_argmax_inference(nid, node, varmap)

    def _emit_matmul_inference(self, nid, node, varmap):
        meta = self._meta.get(nid, {})
        w_var = meta.get("w_var", "?")
        b_var = meta.get("b_var", "?")
        out_d = meta.get("out_dim", 1)
        in_d = meta.get("in_dim", 1)
        out_name = f"v{nid}"
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"

        self._var_decls.append(
            f"        {out_name} : ARRAY[0..{out_d-1}] OF REAL;"
            f"  // matmul output")

        lines = [f"\n        // MatMul: {node.name}"]
        for o in range(out_d):
            terms = [f"{b_var}[{o}]"]
            for i in range(in_d):
                terms.append(
                    f"{w_var}[{o*in_d + i}] * {in_name}[{i}]")
            expr = " + ".join(terms)
            lines.append(f"        {out_name}[{o}] := {expr};")
        self._inference_code.extend(lines)

    def _emit_bspline_inference(self, nid, node, varmap):
        meta = self._meta.get(nid, {})
        grid_var = meta.get("grid_var", "?")
        tab_var = meta.get("tab_var", "?")
        out_d = meta.get("out_dim", 1)
        in_d = meta.get("in_dim", 1)
        n_pts = meta.get("n_pts", 15)
        out_name = f"v{nid}"
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"

        # Use 1D array: v[o*in_d + i] for index (o,i)
        self._var_decls.append(
            f"        {out_name} : ARRAY[0..{out_d*in_d-1}] OF REAL;"
            f"  // BsplineLUT output (1D: o*{in_d}+i)")

        lines = [f"\n        // BsplineLUT: {node.name} ({out_d}x{in_d}, {n_pts} pts)"]
        lines.append(f"        FOR i := 0 TO {in_d-1} DO")
        lines.append(f"            // Linear scan: find largest grid[j] <= input[i]")
        lines.append(f"            lo := 0;")
        lines.append(f"            FOR j := 1 TO {n_pts-2} DO")
        lines.append(f"                IF {in_name}[i] >= {grid_var}[j] THEN")
        lines.append(f"                    lo := j;")
        lines.append(f"                END_IF;")
        lines.append(f"            END_FOR;")
        lines.append(f"            hi := lo + 1;")
        lines.append(f"            t_val := ({in_name}[i] - {grid_var}[lo]) /")
        lines.append(f"                      ({grid_var}[hi] - {grid_var}[lo] + 1.0E-10);")
        lines.append(f"            FOR o := 0 TO {out_d-1} DO")
        lines.append(f"                base := (o * {in_d} + i) * {n_pts};")
        lines.append(f"                idx := o * {in_d} + i;")
        lines.append(f"                {out_name}[idx] :=")
        lines.append(f"                    {tab_var}[base + lo] * (1.0 - t_val) +")
        lines.append(f"                    {tab_var}[base + hi] * t_val;")
        lines.append(f"            END_FOR;")
        lines.append(f"        END_FOR;")

        self._inference_code.extend(lines)

    def _emit_act_inference(self, nid, node, varmap):
        act_type = node.attrs.get("type", "relu")
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"
        dim = node.shape_in[0] if node.shape_in else 16

        self._var_decls.append(
            f"        v{nid} : ARRAY[0..{dim-1}] OF REAL;"
            f"  // {act_type} output")
        lines = [f"\n        // StandardAct: {act_type}"]
        if act_type == "relu":
            lines.append(f"        FOR i := 0 TO {dim-1} DO")
            lines.append(f"            IF {in_name}[i] < 0.0 THEN")
            lines.append(f"                v{nid}[i] := 0.0;")
            lines.append(f"            ELSE")
            lines.append(f"                v{nid}[i] := {in_name}[i];")
            lines.append(f"            END_IF;")
            lines.append(f"        END_FOR;")
        elif act_type == "silu":
            lines.append(f"        FOR i := 0 TO {dim-1} DO")
            lines.append(f"            // SiLU = x / (1 + exp(-x))")
            lines.append(f"            v{nid}[i] := {in_name}[i] / (1.0 + EXP(-{in_name}[i]));")
            lines.append(f"        END_FOR;")
        self._inference_code.extend(lines)

    def _emit_add_inference(self, nid, node, varmap):
        in0 = node.inputs[0] if len(node.inputs) > 0 else 0
        in1 = node.inputs[1] if len(node.inputs) > 1 else 0
        v0_name = varmap.get(in0, "features")
        v1_name = varmap.get(in1, "features")
        dim = self._meta.get(in0, {}).get("out_dim",
              self._meta.get(in1, {}).get("out_dim", 4))

        self._var_decls.append(
            f"        v{nid} : ARRAY[0..{dim-1}] OF REAL;"
            f"  // add merge")
        lines = [f"\n        // Add: {node.name}"]
        bs_meta = self._meta.get(in1, {})
        if bs_meta.get("type") == "bspline":
            out_d = bs_meta.get("out_dim", dim)
            in_d = bs_meta.get("in_dim", 28)
            lines.append(f"        FOR o := 0 TO {out_d-1} DO")
            lines.append(f"            sum := {v0_name}[o];  // base path")
            lines.append(f"            FOR i := 0 TO {in_d-1} DO")
            lines.append(f"                idx := o * {in_d} + i;")
            lines.append(f"                sum := sum + {v1_name}[idx];  // spline path (1D)")
            lines.append(f"            END_FOR;")
            lines.append(f"            v{nid}[o] := sum;")
            lines.append(f"        END_FOR;")
        else:
            lines.append(f"        FOR i := 0 TO {dim-1} DO")
            lines.append(f"            v{nid}[i] := {v0_name}[i] + {v1_name}[i];")
            lines.append(f"        END_FOR;")
        self._inference_code.extend(lines)

    def _emit_softmax_inference(self, nid, node, varmap):
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"
        dim = 4

        self._var_decls.append(
            f"        v{nid} : ARRAY[0..{dim-1}] OF REAL;"
            f"  // softmax output")
        lines = [f"\n        // Softmax"]
        lines.append(f"        max_val := {in_name}[0];")
        lines.append(f"        FOR i := 1 TO {dim-1} DO")
        lines.append(f"            IF {in_name}[i] > max_val THEN")
        lines.append(f"                max_val := {in_name}[i];")
        lines.append(f"            END_IF;")
        lines.append(f"        END_FOR;")
        lines.append(f"        sum := 0.0;")
        lines.append(f"        FOR i := 0 TO {dim-1} DO")
        lines.append(f"            v{nid}[i] := EXP({in_name}[i] - max_val);")
        lines.append(f"            sum := sum + v{nid}[i];")
        lines.append(f"        END_FOR;")
        lines.append(f"        FOR i := 0 TO {dim-1} DO")
        lines.append(f"            v{nid}[i] := v{nid}[i] / sum;")
        lines.append(f"        END_FOR;")
        self._inference_code.extend(lines)

    def _emit_argmax_inference(self, nid, node, varmap):
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"
        dim = 4
        lines = [f"\n        // Argmax"]
        lines.append(f"        max_val := {in_name}[0];")
        lines.append(f"        fault_class := 0;")
        lines.append(f"        FOR i := 1 TO {dim-1} DO")
        lines.append(f"            IF {in_name}[i] > max_val THEN")
        lines.append(f"                max_val := {in_name}[i];")
        lines.append(f"                fault_class := i;")
        lines.append(f"            END_IF;")
        lines.append(f"        END_FOR;")
        lines.append(f"        confidence := max_val;")
        self._inference_code.extend(lines)

    # ── Phase 3: Assemble ──

    def _assemble(self) -> str:
        in_dim = self._g.input_nodes[0].shape_in[0] if self._g.input_nodes and self._g.input_nodes[0].shape_in else 28
        out_dim = 4  # fault classification: 4 classes

        lines = [
            f"// NeuroPLC — Auto-generated IEC 61131-3 SCL (FB-Only Mode)",
            f"// Graph: {self._g.name} | Target: {self.__class__.__name__}",
            f"// Work memory: {self.wm_kb}KB | LUT points: {self.lut_pts}",
            f"// Nodes: {self._g.node_count} | Ops: {dict(self._g.op_counts)}",
            f"// NOTE: FB-Only mode — all parameters embedded as local arrays.",
            f"// TIA Portal V21 compatible (no DB Array declarations).",
            f"",
            f'FUNCTION_BLOCK "NeuroPLC_Inference"',
            "{ S7_Optimized_Access := 'FALSE' }",
            f"VERSION : 0.1",
            f"",
            f"VAR_INPUT",
            f"    features : ARRAY[0..{in_dim-1}] OF REAL;  // {in_dim}-D feature vector",
            f"END_VAR",
            f"",
            f"VAR_OUTPUT",
            f"    fault_class : INT;  // 0=Normal, 1=InnerRace, 2=Ball, 3=OuterRace",
            f"    confidence : REAL;  // max softmax probability",
            f"END_VAR",
            f"",
            f"VAR",
            f"    // ── Model parameters ──",
        ]
        lines.extend(self._var_decls)

        lines.append(f"")
        lines.append(f"    // ── Temporary variables ──")
        lines.append(f"    init_done : BOOL := FALSE;  // first-scan guard")
        lines.append(f"    i, o, j, lo, hi, idx, base : INT;")
        lines.append(f"    sum, max_val, t_val : REAL;")
        lines.append(f"END_VAR")
        lines.append(f"")
        lines.append(f"BEGIN")
        lines.append(f"    // ═══ First-scan initialization ═══")
        lines.append(f"    IF NOT init_done THEN")
        lines.extend(self._init_code)
        lines.append(f"")
        lines.append(f"        init_done := TRUE;")
        lines.append(f"    END_IF;")
        lines.append(f"")
        lines.append(f"    // ═══ Inference forward pass ═══")
        lines.extend(self._inference_code)
        lines.append(f"")
        lines.append(f"END_FUNCTION_BLOCK")
        lines.append(f"")

        return "\n".join(lines)


class S71200FBOnlyBackend(S7FBOnlyBackendBase):
    """FB-Only backend for S7-1200 (50KB, 15-pt LUT, FOR loops)."""
    def __init__(self, lut_pts: int = 15):
        super().__init__(wm_kb=75, lut_pts=lut_pts, unroll=False)


class S71500FBOnlyBackend(S7FBOnlyBackendBase):
    """FB-Only backend for S7-1500 (1.5MB, 50-pt LUT, unrolled)."""
    def __init__(self, lut_pts: int = 50):
        super().__init__(wm_kb=1500, lut_pts=lut_pts, unroll=True)
