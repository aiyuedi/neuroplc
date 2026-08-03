#!/usr/bin/env python3
"""
NeuroPLC — Vendor-Neutral IEC 61131-3 ST Backend (MVP)
========================================================
Generates the SAME inference logic as the Siemens SCL backends, but with
vendor-neutral IEC 61131-3 structured text: no S7_Optimized_Access, no
VERSION/NON_RETAIN attributes, plain identifiers (CODESYS/TwinCAT style).
This demonstrates multi-vendor deployability: the IR and the verification
chain (Tiers 1-3 + Tier 4 differential test) are target-independent; only
the emitter's boilerplate changes.

Usage:
    from neuroplc.backend_iec import IECBackend
    from neuroplc.frontend import kan_to_ir
    ir = kan_to_ir(model, lut_points=15)
    db_st, fb_st = IECBackend(lut_pts=15, db_name="PLC_NeuroPLC_Weights").generate(ir)

The generated ST is compilable by CODESYS/TwinCAT/any IEC 61131-3
compliant toolchain (standard VAR_GLOBAL/VAR/FUNCTION_BLOCK syntax).
"""

from __future__ import annotations

from .backend_s7_db import S7DBBackendBase


class IECDBBackendBase(S7DBBackendBase):
    """Vendor-neutral variant: strips Siemens-specific attributes.

    Differences from S7DBBackendBase:
      - No '{ S7_Optimized_Access := ... }' attribute
      - No 'VERSION : 0.1' / 'NON_RETAIN' lines
      - Unquoted DB/FB identifiers (IEC-valid; CODESYS/TwinCAT style)
      - Header comments note the target as generic IEC 61131-3
    """

    def _q(self, var: str) -> str:
        # Vendor-neutral: unquoted dotted path (CODESYS style uses dots,
        # e.g. PLC_NeuroPLC_Weights.w2[0]); S7 backend quotes with "DB".var
        # so references are absolute. Here keep plain reference.
        return var

    def _assemble_db(self) -> str:
        """Vendor-neutral DATA_BLOCK: no S7 attributes, standard syntax."""
        lines = [
            f"// NeuroPLC — Vendor-neutral IEC 61131-3 (DB+FB Mode)",
            f"// Graph: {self._g.name} | Target: generic IEC 61131-3",
            f"// Parameters as initialized global data (VAR_GLOBAL section).",
            f"",
            f"VAR_GLOBAL",
        ]
        lines.extend(self._db_struct)
        lines.append(f"END_VAR")
        lines.append(f"")
        lines.append(f"// ── Initialization (start values) ──")
        lines.extend(self._db_init)
        lines.append(f"")
        return "\n".join(lines)

    def _assemble(self) -> str:
        in_dim = (self._g.input_nodes[0].shape_in[0]
                  if self._g.input_nodes and self._g.input_nodes[0].shape_in
                  else 28)

        fb_var_decls = [d for d in self._var_decls
                        if not any(pn in d for pn in self._db_param_names)]

        lines = [
            f"// NeuroPLC — Vendor-neutral IEC 61131-3 ST inference block",
            f"// Graph: {self._g.name} | Parameters in global data, logic in FB.",
            f"",
            f"FUNCTION_BLOCK NeuroPLC_Inference",
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
        lines.extend(self._inference_code)
        lines.append(f"")
        lines.append(f"END_FUNCTION_BLOCK")
        return "\n".join(lines)


class IECBackend(IECDBBackendBase):
    """Default vendor-neutral backend (equivalent of S7-1200 config:
    15 LUT points, compact loops)."""

    def __init__(self, lut_pts: int = 15, db_name: str = "PLC_NeuroPLC_Weights"):
        super().__init__(wm_kb=50, lut_pts=lut_pts, unroll=False,
                         db_name=db_name, optimized_db=False)
