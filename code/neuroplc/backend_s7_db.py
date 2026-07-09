#!/usr/bin/env python3
"""
NeuroPLC — DB+FB SCL Backend for Siemens TIA Portal V21
=========================================================
Parameters go into a global DATA_BLOCK (start values as data, not code).
Inference logic stays in a compact FUNCTION_BLOCK referencing the DB.

Design rationale:
- FB-only mode puts ~8200 init assignments in code → exceeds S7-1200 64KB block limit
- DB BEGIN...END_DATA_BLOCK stores start values as data (in load memory)
- FB inference code is ~200 lines → well under 64KB limit
- FB references DB arrays via "DB_Name".array[index] syntax

Usage:
    from neuroplc.backend_s7_db import S71200DBBackend
    backend = S71200DBBackend(lut_pts=15)
    db_scl, fb_scl = backend.generate(ir_graph)
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple

from .ir import IRGraph, IROpType
from .backend_s7 import _fmt, _fmt_array
from .backend_s7_fbonly import S7FBOnlyBackendBase


class S7DBBackendBase(S7FBOnlyBackendBase):
    """DB + FB backend: parameters in global DB, inference in FB."""

    def __init__(self, wm_kb: int, lut_pts: int, unroll: bool,
                 db_name: str = "NeuroPLC_Weights", optimized_db: bool = False):
        super().__init__(wm_kb, lut_pts, unroll)
        self.DB_NAME = db_name
        self._optimized_db = optimized_db
        self._db_struct: list[str] = []     # DB STRUCT members
        self._db_init: list[str] = []       # DB BEGIN...END init values
        self._db_param_names: set[str] = set()  # param var names (for reference)

    def _q(self, var: str) -> str:
        """Qualify param reference: 'g0' -> '"DB".g0'"""
        if var in self._db_param_names:
            return f'"{self.DB_NAME}".{var}'
        return var

    def generate(self, graph: IRGraph) -> Tuple[str, str]:
        """Returns (db_scl, fb_scl)."""
        self._g = graph
        self._has_bspline = any(
            n.op == IROpType.BsplineLUT for n in graph.nodes.values())
        self._order = graph.topological_order()

        self._collect_params()
        self._collect_inference()
        return self._assemble_db(), self._assemble()

    # ── Phase 1: Collect params (for DB instead of FB VAR) ──

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

        # DB structure declarations
        self._db_struct.append(
            f"      {w_var} : ARRAY[0..{len(w_flat)-1}] OF REAL;"
            f"  // W({out_d}x{in_d})")
        self._db_struct.append(
            f"      {b_var} : ARRAY[0..{len(b_flat)-1}] OF REAL;"
            f"  // bias({out_d})")

        # DB start values
        self._db_init.append(f"\n      // MatMul '{node.name}': W + bias")
        self._db_init.append(self._fmt_db_init(w_var, w_flat))
        self._db_init.append(self._fmt_db_init(b_var, b_flat))

        self._db_param_names.add(w_var)
        self._db_param_names.add(b_var)

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

        self._db_struct.append(
            f"      {grid_var} : ARRAY[0..{len(grid)-1}] OF REAL;"
            f"  // LUT grid ({n_pts} pts)")
        self._db_struct.append(
            f"      {tab_var} : ARRAY[0..{table.size-1}] OF REAL;"
            f"  // LUT table ({out_d}x{in_d}x{n_pts})")

        self._db_init.append(
            f"\n      // BsplineLUT '{node.name}': grid + table")
        self._db_init.append(self._fmt_db_init(grid_var, grid))
        self._db_init.append(self._fmt_db_init(tab_var, table.flatten()))

        self._db_param_names.add(grid_var)
        self._db_param_names.add(tab_var)

        self._meta[nid] = {
            "type": "bspline", "grid_var": grid_var, "tab_var": tab_var,
            "out_dim": out_d, "in_dim": in_d, "n_pts": n_pts,
        }

    def _fmt_db_init(self, prefix: str, values: np.ndarray,
                     vals_per_line: int = 8) -> str:
        """Format DB start values: array[index] := value; ..."""
        lines = []
        flat = values.flatten()
        for i in range(0, len(flat), vals_per_line):
            chunk = flat[i : i + vals_per_line]
            parts = [f"{prefix}[{i+j}] := {_fmt(v)};" for j, v in enumerate(chunk)]
            lines.append("      " + " ".join(parts))
        return "\n".join(lines)

    # ── Override emit methods to qualify DB references ──

    def _emit_bspline_inference(self, nid, node, varmap):
        meta = self._meta.get(nid, {})
        grid_var = meta.get("grid_var", "?")
        tab_var = meta.get("tab_var", "?")
        out_d = meta.get("out_dim", 1)
        in_d = meta.get("in_dim", 1)
        n_pts = meta.get("n_pts", 15)
        out_name = f"v{nid}"
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"

        q_grid = self._q(grid_var)
        q_tab = self._q(tab_var)

        self._var_decls.append(
            f"        {out_name} : ARRAY[0..{out_d*in_d-1}] OF REAL;"
            f"  // BsplineLUT output (1D: o*{in_d}+i)")

        lines = [f"\n        // BsplineLUT: {node.name} ({out_d}x{in_d}, {n_pts} pts)"]
        lines.append(f"        FOR i := 0 TO {in_d-1} DO")
        lines.append(f"            // Linear scan: find largest grid[j] <= input[i]")
        lines.append(f"            lo := 0;")
        lines.append(f"            FOR j := 1 TO {n_pts-2} DO")
        lines.append(f"                IF {in_name}[i] >= {q_grid}[j] THEN")
        lines.append(f"                    lo := j;")
        lines.append(f"                END_IF;")
        lines.append(f"            END_FOR;")
        lines.append(f"            hi := lo + 1;")
        lines.append(f"            t_val := ({in_name}[i] - {q_grid}[lo]) /")
        lines.append(f"                      ({q_grid}[hi] - {q_grid}[lo] + 1.0E-10);")
        lines.append(f"            FOR o := 0 TO {out_d-1} DO")
        lines.append(f"                base := (o * {in_d} + i) * {n_pts};")
        lines.append(f"                idx := o * {in_d} + i;")
        lines.append(f"                {out_name}[idx] :=")
        lines.append(f"                    {q_tab}[base + lo] * (1.0 - t_val) +")
        lines.append(f"                    {q_tab}[base + hi] * t_val;")
        lines.append(f"            END_FOR;")
        lines.append(f"        END_FOR;")

        self._inference_code.extend(lines)

    def _emit_matmul_inference(self, nid, node, varmap):
        meta = self._meta.get(nid, {})
        w_var = meta.get("w_var", "?")
        b_var = meta.get("b_var", "?")
        out_d = meta.get("out_dim", 1)
        in_d = meta.get("in_dim", 1)
        out_name = f"v{nid}"
        in_name = varmap.get(node.inputs[0], "features") if node.inputs else "features"

        q_w = self._q(w_var)
        q_b = self._q(b_var)

        self._var_decls.append(
            f"        {out_name} : ARRAY[0..{out_d-1}] OF REAL;"
            f"  // matmul output")

        lines = [f"\n        // MatMul: {node.name}"]
        for o in range(out_d):
            terms = [f"{q_b}[{o}]"]
            for i in range(in_d):
                terms.append(
                    f"{q_w}[{o*in_d + i}] * {in_name}[{i}]")
            expr = " + ".join(terms)
            lines.append(f"        {out_name}[{o}] := {expr};")
        self._inference_code.extend(lines)

    # ── Assemble DB ──

    def _assemble_db(self) -> str:
        """Generate DATA_BLOCK with all parameter arrays and start values."""
        optimized_flag = "'TRUE'" if self._optimized_db else "'FALSE'"
        lines = [
            f'DATA_BLOCK "{self.DB_NAME}"',
            "{{ S7_Optimized_Access := {0} }}".format(optimized_flag),
            "VERSION : 0.1",
            "NON_RETAIN",
            "   STRUCT",
        ]
        lines.extend(self._db_struct)
        lines.append("   END_STRUCT;")
        lines.append("BEGIN")
        lines.extend(self._db_init)
        lines.append("")
        lines.append("END_DATA_BLOCK")
        lines.append("")
        return "\n".join(lines)

    # ── Assemble FB (overrides base: no param VARs, no init block) ──

    def _assemble(self) -> str:
        in_dim = self._g.input_nodes[0].shape_in[0] if self._g.input_nodes and self._g.input_nodes[0].shape_in else 28

        # Only inference output arrays go in FB VAR (not parameters)
        fb_var_decls = [d for d in self._var_decls
                       if not any(pn in d for pn in self._db_param_names)]

        lines = [
            f"// NeuroPLC — Auto-generated IEC 61131-3 SCL (DB+FB Mode)",
            f"// Graph: {self._g.name} | Target: {self.__class__.__name__}",
            f"// Work memory: {self.wm_kb}KB | LUT points: {self.lut_pts}",
            f"// Nodes: {self._g.node_count} | Ops: {dict(self._g.op_counts)}",
            f"// Parameters in DB \"{self.DB_NAME}\", inference in FB.",
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
            f"    // ── Inference temporaries ──",
        ]
        lines.extend(fb_var_decls)

        lines.append(f"")
        lines.append(f"    // ── Temporary variables ──")
        lines.append(f"    i, o, j, lo, hi, idx, base : INT;")
        lines.append(f"    sum, max_val, t_val : REAL;")
        lines.append(f"END_VAR")
        lines.append(f"")
        lines.append(f"BEGIN")
        lines.append(f"    // ═══ Inference forward pass ═══")
        lines.append(f"    // Parameters loaded from DB \"{self.DB_NAME}\" (start values = data, not code)")
        lines.extend(self._inference_code)
        lines.append(f"")
        lines.append(f"END_FUNCTION_BLOCK")
        lines.append(f"")

        return "\n".join(lines)


class S71200DBBackend(S7DBBackendBase):
    """DB+FB backend for S7-1200 (50KB, 15-pt LUT)."""
    def __init__(self, lut_pts: int = 15, db_name: str = "NeuroPLC_Weights"):
        super().__init__(wm_kb=75, lut_pts=lut_pts, unroll=False, db_name=db_name)


class S71500DBBackend(S7DBBackendBase):
    """DB+FB backend for S7-1500 (1.5MB, 50-pt LUT, optimized DB)."""
    def __init__(self, lut_pts: int = 50, db_name: str = "NeuroPLC_Weights"):
        super().__init__(wm_kb=1500, lut_pts=lut_pts, unroll=True,
                         db_name=db_name, optimized_db=True)
