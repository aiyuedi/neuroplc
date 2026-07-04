#!/usr/bin/env python3
"""
NeuroPLC — SCL Code Templates
================================
Reusable SCL code snippets for the backend.

These are string templates with {placeholders} filled by backend_s7.py.
Kept in a separate module for readability and potential customization.

Reference: results/scl_output/neuroplc_test.scl (verified 0 errors in TIA Portal)

Templates:
    DB_DECL, FB_DECL, FC_DECL     — block headers
    MATMUL_ROW, MATMUL_FOR        — matrix multiplication
    RELU, SILU, SIGMOID           — activation functions
    SOFTMAX, ARGMAX               — output layer
    BSPLINE_BINSEARCH             — binary search in LUT
    BSPLINE_LERP                  — linear interpolation
"""

# ============================================================================
# Block headers
# ============================================================================

DB_HEADER = """\
DATA_BLOCK "{name}"
{{ S7_Optimized_Access := {optimized} }}
VERSION : 0.1
NON_RETAIN
"""

FB_HEADER = """\
FUNCTION_BLOCK "{name}"
{{ S7_Optimized_Access := {optimized} }}
VERSION : 0.1
VAR_INPUT
{inputs}
END_VAR
VAR_OUTPUT
{outputs}
END_VAR
VAR
{variables}
END_VAR
BEGIN
"""

FC_HEADER = """\
FUNCTION "{name}" : Real
{{ S7_Optimized_Access := {optimized} }}
VERSION : 0.1
VAR_INPUT
{inputs}
END_VAR
VAR
{variables}
END_VAR
BEGIN
"""

# ============================================================================
# Matrix multiplication
# ============================================================================

MATMUL_ROW = "    {out_var}[{o}] := {bias_term} + {dot_terms};"

MATMUL_FOR_COMPACT = """\
    FOR j := 0 TO {out_dim} - 1 DO
        {out_var}[j] := "{db}".b{l}[j];
        FOR i := 0 TO {in_dim} - 1 DO
            {out_var}[j] := {out_var}[j] + "{db}".w{l}[j * {in_dim} + i] * {in_var}[i];
        END_FOR;
    END_FOR;"""

# ============================================================================
# Activation functions
# ============================================================================

RELU = "    IF {in}[{i}] > 0.0 THEN {out}[{i}] := {in}[{i}]; ELSE {out}[{i}] := 0.0; END_IF;"

SILU = "    {out}[{i}] := {in}[{i}] / (1.0 + EXP(-{in}[{i}]));"

SIGMOID = "    {out}[{i}] := 1.0 / (1.0 + EXP(-{in}[{i}]));"

# ============================================================================
# Softmax + Argmax
# ============================================================================

SOFTMAX = """\
    // Softmax
    FOR j := 0 TO {dim} - 1 DO
        softmax_out[j] := EXP({in_var}[j]);
    END_FOR;
    sum_val := softmax_out[0];
    FOR j := 1 TO {dim} - 1 DO
        sum_val := sum_val + softmax_out[j];
    END_FOR;
    IF sum_val > 0.0 THEN
        FOR j := 0 TO {dim} - 1 DO
            softmax_out[j] := softmax_out[j] / sum_val;
        END_FOR;
    END_IF;"""

ARGMAX = """\
    // Argmax
    max_val := softmax_out[0];
    fault_class := 0;
    FOR j := 1 TO {dim} - 1 DO
        IF softmax_out[j] > max_val THEN
            max_val := softmax_out[j];
            fault_class := j;
        END_IF;
    END_FOR;
    confidence := max_val;"""

# ============================================================================
# B-spline LUT evaluation
# ============================================================================

BSPLINE_FUNCTION = """\
FUNCTION "{name}" : Real
{{ S7_Optimized_Access := 'TRUE' }}
VERSION : 0.1
VAR_INPUT
    x : Real;
    grid : Array[0..{n_pts_m1}] of Real;
    table : Array[0..{n_pts_m1}] of Real;
END_VAR
VAR
    lo, hi, mid : Int;
    t, vlo, vhi : Real;
END_VAR
BEGIN
    lo := 0;
    hi := {n_pts_m1};
    WHILE hi - lo > 1 DO
        mid := (lo + hi) / 2;
        IF x > grid[mid] THEN lo := mid; ELSE hi := mid; END_IF;
    END_WHILE;
    vlo := table[lo];
    vhi := table[hi];
    t := (x - grid[lo]) / (grid[hi] - grid[lo] + 1.0E-10);
    "{name}" := vlo * (1.0 - t) + vhi * t;
END_FUNCTION"""

# ============================================================================
# KAN-specific: Add merge
# ============================================================================

KAN_MERGE = """\
    {out_var}[{j}] := {scale_base} * {base_var}[{j}] + {scale_spline} * {spline_sum};"""
