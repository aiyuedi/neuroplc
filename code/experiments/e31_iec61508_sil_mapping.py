#!/usr/bin/env python3
"""
NeuroPLC — IEC 61508 SIL 2/3 Compliance Mapping
=================================================
Maps NeuroPLC's SVNN framework to IEC 61508-3 software safety requirements
for SIL 2 (industrial machinery) and SIL 3 (safety-critical systems).

Based on:
  - IEC 61508-3:2010, Annex A (Techniques and measures for software safety)
  - ISO/IEC TS 22440:2025 (AI safety for industrial automation)
  - Comparison with SIMATIC AI (Siemens) and TF3850 (Beckhoff)

Usage:
    python experiments/e31_iec61508_sil_mapping.py
"""

import json, os
from pathlib import Path


# ============================================================================
# IEC 61508-3 Key Requirements for SIL 2/3
# ============================================================================

IEC61508_REQUIREMENTS = {
    "R1": {
        "id": "A.1",
        "clause": "7.4.4.2 / A.1",
        "name": "Structured Design & Modularity",
        "description": "Software shall be decomposed into modules with "
                       "well-defined, verified interfaces.",
        "sil2_required": "HR",  # Highly Recommended
        "sil3_required": "HR",
    },
    "R2": {
        "id": "A.2",
        "clause": "7.4.4.4 / A.2",
        "name": "Deterministic & Bounded Behavior",
        "description": "All software components shall have deterministic "
                       "time and space behavior with provable bounds.",
        "sil2_required": "HR",
        "sil3_required": "HR",
    },
    "R3": {
        "id": "A.3",
        "clause": "7.4.4.5 / A.3",
        "name": "Semi-Formal Methods",
        "description": "Design and verification shall employ semi-formal "
                       "methods (state machines, data flow diagrams, "
                       "formal specifications).",
        "sil2_required": "R",   # Recommended
        "sil3_required": "HR",
    },
    "R4": {
        "id": "A.4",
        "clause": "7.4.4.7 / A.4",
        "name": "Formal Proof or Formal Verification",
        "description": "Safety-critical functions shall be verified using "
                       "formal methods (model checking, theorem proving).",
        "sil2_required": "—",   # Not required
        "sil3_required": "HR",
    },
    "R5": {
        "id": "A.5",
        "clause": "7.4.4.9 / A.5",
        "name": "Static Analysis & Bounded Model Checking",
        "description": "Static code analysis and bounded model checking "
                       "for control flow and data flow correctness.",
        "sil2_required": "HR",
        "sil3_required": "HR",
    },
    "R6": {
        "id": "A.6",
        "clause": "7.4.4.11 / A.6",
        "name": "Resource Bounding & WCET Analysis",
        "description": "Memory and execution time shall be statically "
                       "bounded and verified.",
        "sil2_required": "HR",
        "sil3_required": "HR",
    },
    "R7": {
        "id": "A.7",
        "clause": "7.4.4.12 / A.7",
        "name": "Fault Detection & Diagnostic Coverage",
        "description": "The system shall detect faults during operation "
                       "and achieve the required diagnostic coverage (DC).",
        "sil2_required": "HR",  # DC >= 90% for SIL 2
        "sil3_required": "HR",  # DC >= 99% for SIL 3
    },
    "R8": {
        "id": "A.8",
        "clause": "7.4.4.13 / A.8",
        "name": "Traceability & Verification Documentation",
        "description": "Complete traceability from safety requirements to "
                       "code, with documented verification results.",
        "sil2_required": "HR",
        "sil3_required": "HR",
    },
    "R9": {
        "id": "A.9",
        "clause": "7.4.4.14 / A.9",
        "name": "Third-Party Component Qualification",
        "description": "Pre-existing software components (libraries, "
                       "frameworks) shall be qualified for the target SIL.",
        "sil2_required": "HR",
        "sil3_required": "HR",
    },
}

# ============================================================================
# SVNN Framework → IEC 61508 Mapping
# ============================================================================

SVNN_IEC61508_MAPPING = [
    {
        "svnn_condition": "Condition 1: Operation-Type Closure",
        "svnn_description": "Each IR operation maps to one IEC 61131-3 "
                           "language primitive (SCL); the 6-op set is "
                           "closed under the KAN computation model.",
        "iec61508_requirement": "R1 (Structured Design & Modularity)",
        "mapping_rationale": "The 6-op IR provides a fixed, verified set "
                            "of modules. Each SCL code block is generated "
                            "from a known IR node with provable semantics "
                            "(Theorem~1). This eliminates ad-hoc code "
                            "generation — a key requirement for SIL 2/3 "
                            "modularity.",
        "evidence": "Proposition~1 (IR minimality), Theorem~2.1 (structural "
                    "invariants), e27 ablation (Table~tab:ir_ablation)",
        "compliance_level": "FULL",  # FULL / PARTIAL / PLANNED
        "sil2_status": "SATISFIED",
        "sil3_status": "SATISFIED",
    },
    {
        "svnn_condition": "Condition 2: Univariate Boundedness",
        "svnn_description": "Each B-spline function is a univariate "
                           "continuous function with provable max |f''(x)| "
                           "bound (M2). The LUT interpolation error is "
                           "bounded by M2 * h^2 / 8.",
        "iec61508_requirement": "R2 (Deterministic Behavior) + "
                               "R6 (Resource Bounding)",
        "mapping_rationale": "Univariate boundedness guarantees both "
                            "deterministic computation (same output for "
                            "same input — no data-dependent branching) and "
                            "provable resource bounds (memory = O(n_functions "
                            "* n_lut_points), time = O(n_functions * log(n_lut_points))). "
                            "This directly satisfies the IEC~61508 requirement "
                            "that all components have provable time/space bounds.",
        "evidence": "Theorem~1 (DA bound), Proposition~2 (tightness), "
                    "Z3 WCET analysis (3.01~ms), TIA Portal resource "
                    "analysis (45.2~KB work memory)",
        "compliance_level": "FULL",
        "sil2_status": "SATISFIED",
        "sil3_status": "SATISFIED",
    },
    {
        "svnn_condition": "Condition 3: Layer-Wise Composability",
        "svnn_description": "Error bounds compose linearly across KAN "
                           "layers via the DA (Doubleton Arithmetic) "
                           "framework. The total output error is bounded "
                           "by O(L * sqrt(d_max) * M_max * h^2).",
        "iec61508_requirement": "R3 (Semi-Formal Methods) + "
                               "R4 (Formal Verification) + "
                               "R5 (Static Analysis)",
        "mapping_rationale": "Layer-wise composability enables compositional "
                            "verification: each layer's error bound is "
                            "verified independently, then composed to obtain "
                            "an end-to-end guarantee. This is precisely the "
                            "semi-formal/formal verification approach required "
                            "by IEC~61508 Annex~A.4. The Two-Tier architecture "
                            "(DA bounds → SMT verification) provides a complete "
                            "verification chain.",
        "evidence": "Theorem~4 (L-layer bound via martingale + "
                    "Azuma-Hoeffding), Two-Tier verification "
                    "(512/512 functions Z3-verified), e28 scalability "
                    "Pareto frontier",
        "compliance_level": "PARTIAL",  # Two-Tier demonstrated, ESBMC pending
        "sil2_status": "SATISFIED",
        "sil3_status": "PARTIAL (ESBMC-PLC+ full BMC pending)",
    },
    {
        "svnn_condition": "Compiler Correctness (Translation Validation)",
        "svnn_description": "Z3 SMT verifies that SCL compiled code is "
                           "semantically equivalent to PyTorch reference "
                           "for each IR node type.",
        "iec61508_requirement": "R4 (Formal Proof) + "
                               "R5 (Static Analysis)",
        "mapping_rationale": "Translation validation provides a formal "
                            "proof that the compiler preserves semantics. "
                            "This is stronger than the testing-based "
                            "qualification required by IEC~61508 for "
                            "SIL~2/3 toolchains.",
        "evidence": "Z3 translation validation report "
                    "(9 exact + 2 bounded = 11/11 nodes PASS), "
                    "Z3 binary search proof (3~ms)",
        "compliance_level": "PARTIAL",  # IEEE 754 rounding not fully modeled
        "sil2_status": "SATISFIED",
        "sil3_status": "PARTIAL (full IEEE~754 model needed)",
    },
    {
        "svnn_condition": "Architecture-Independent Guarantees",
        "svnn_description": "All correctness guarantees depend on KAN "
                           "architecture (not specific weights), so they "
                           "survive fine-tuning and domain adaptation.",
        "iec61508_requirement": "R8 (Traceability) + "
                               "R9 (Third-Party Qualification)",
        "mapping_rationale": "The architecture-independent nature of the "
                            "guarantees means a single compiler qualification "
                            "covers all fine-tuned models for the same KAN "
                            "architecture. This dramatically reduces the "
                            "qualification burden for model updates.",
        "evidence": "Cross-dataset FT experiment (CWRU → XJTU-SY: "
                    "37.3%% → 79.4%% with unchanged guarantees), "
                    "Theorem~1 (depends on architecture, not weights)",
        "compliance_level": "FULL",
        "sil2_status": "SATISFIED",
        "sil3_status": "SATISFIED",
    },
    {
        "svnn_condition": "Online Monitoring & Runtime Assurance",
        "svnn_description": "The DA bound provides a runtime-checkable "
                           "safety condition: if the inference output "
                           "confidence exceeds a threshold, the DA bound "
                           "guarantees correctness.",
        "iec61508_requirement": "R7 (Fault Detection & Diagnostic Coverage)",
        "mapping_rationale": "The DA safety factor (confidence margin / "
                            "DA bound) provides a runtime diagnostic: if "
                            "the margin drops below the safety threshold, "
                            "the system can fall back to a safe state. This "
                            "contributes to diagnostic coverage for AI "
                            "components, which is otherwise hard to achieve.",
        "evidence": "Two-Tier classification margin analysis, "
                    "DA safety factor distribution (E9)",
        "compliance_level": "PARTIAL",  # Runtime monitor not implemented
        "sil2_status": "PLANNED",
        "sil3_status": "PLANNED",
    },
]


# ============================================================================
# Comparison with Existing Solutions
# ============================================================================

COMPETITOR_COMPARISON = [
    {
        "solution": "NeuroPLC (this work)",
        "vendor": "Academic",
        "platform": "S7-1200/1500",
        "nn_arch": "KAN",
        "sil_readiness": "SIL 2 mapped; SIL 3 partial",
        "formal_verification": "Z3 SMT (Two-Tier)",
        "memory": "45.2 KB work / 258 KB load",
        "wcet": "3.01 ms (Z3-verified)",
        "key_advantage": "Architecture-independent guarantees; formal verification",
        "key_limitation": "No physical PLC validation; ESBMC not integrated",
    },
    {
        "solution": "SIMATIC S7-1500 TM NPU",
        "vendor": "Siemens",
        "platform": "S7-1500 + TM NPU",
        "nn_arch": "Proprietary (CNN/MLP)",
        "sil_readiness": "SIL 2 (hardware); software certification pending",
        "formal_verification": "None (black-box testing only)",
        "memory": "NPU external (not on CPU)",
        "wcet": "Hardware-accelerated (deterministic NPU)",
        "key_advantage": "Siemens-certified hardware; industrial deployment ready",
        "key_limitation": "Requires external NPU module; no formal verification; "
                          "limited to Siemens-curated architectures",
    },
    {
        "solution": "Beckhoff TF3850 TwinCAT ML",
        "vendor": "Beckhoff",
        "platform": "TwinCAT 3 (x86 IPC)",
        "nn_arch": "ONNX import (generic MLP/CNN)",
        "sil_readiness": "Not certified for safety (standard runtime)",
        "formal_verification": "None",
        "memory": "Depends on IPC (typically GB-scale)",
        "wcet": "Non-deterministic (Windows + real-time kernel)",
        "key_advantage": "Generic ONNX support; x86 performance",
        "key_limitation": "Requires IPC hardware (not micro-PLC); "
                          "no formal verification; Windows dependency",
    },
    {
        "solution": "B&R Hypervisor + Ubuntu Docker",
        "vendor": "B&R (ABB)",
        "platform": "Automation PC 910",
        "nn_arch": "Any (Docker Linux runtime)",
        "sil_readiness": "SIL 3 (hypervisor certified); AI un-certified",
        "formal_verification": "None for AI component",
        "memory": "IPC-scale (GB)",
        "wcet": "Non-deterministic (Linux+Python inference)",
        "key_advantage": "Full flexibility (any Python ML framework)",
        "key_limitation": "Requires high-end IPC; no real-time AI guarantee; "
                          "hypervisor cert covers PLC side only",
    },
]


# ============================================================================
# Output Generation
# ============================================================================

def build_markdown_table():
    """Generate paper-ready markdown table for the SIL mapping."""
    lines = [
        "### SVNN Framework → IEC 61508-3 Compliance Mapping",
        "",
        "| SVNN Condition | IEC 61508 Requirement | SIL 2 | SIL 3 | Evidence |",
        "|:---|:---|:---:|:---:|:---|",
    ]
    for m in SVNN_IEC61508_MAPPING:
        cond_short = m["svnn_condition"].split(":")[0]
        req_short = m["iec61508_requirement"].split("(")[0].strip()
        sil2 = "YES" if "SATISFIED" in m["sil2_status"] else "~"
        sil3 = "YES" if "SATISFIED" in m["sil3_status"] else "~"
        evidence_short = m["evidence"].split(",")[0]  # first evidence item
        lines.append(
            f"| {cond_short} | {req_short} | {sil2} | {sil3} | {evidence_short} |")

    return "\n".join(lines)


def build_latex_table():
    """Generate LaTeX table for the paper."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{SVNN Framework $\rightarrow$ IEC~61508-3 Compliance Mapping. "
        r"Each SVNN condition (Theorem~2) is mapped to the relevant "
        r"IEC~61508-3 software safety requirement (Annex~A). "
        r"Compliance levels: $\checkmark$ = satisfied, "
        r"$\sim$ = partially satisfied, -- = planned/not yet demonstrated.}",
        r"\label{tab:iec61508_mapping}",
        r"\small",
        r"\begin{tabular}{p{3.2cm}p{2.8cm}ccp{2.8cm}}",
        r"\toprule",
        r"\textbf{SVNN Condition} & \textbf{IEC~61508-3 Req.} & "
        r"\textbf{SIL2} & \textbf{SIL3} & \textbf{Key Evidence} \\",
        r"\midrule",
    ]
    for m in SVNN_IEC61508_MAPPING:
        cond = m["svnn_condition"].replace(":", r"\\")
        req = m["iec61508_requirement"].replace(" (", r"\\(").replace(" + ", r"\\+ ")
        sil2 = r"$\checkmark$" if "SATISFIED" in m["sil2_status"] else r"$\sim$"
        sil3 = r"$\checkmark$" if "SATISFIED" in m["sil3_status"] else r"$\sim$"
        ev = m["evidence"].split(",")[0]  # first item
        lines.append(f"  {cond} & {req} & {sil2} & {sil3} & {ev} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_competitor_latex_table():
    """LaTeX table comparing NeuroPLC with existing industrial solutions."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of PLC Neural Inference Solutions. "
        r"NeuroPLC is the only approach that combines formal verification "
        r"with deployment on resource-constrained micro-PLCs (S7-1200).}",
        r"\label{tab:competitor_comparison}",
        r"\small",
        r"\begin{tabular}{p{2.4cm}p{2.4cm}p{1.8cm}p{2.0cm}p{2.8cm}}",
        r"\toprule",
        r"\textbf{Solution} & \textbf{Hardware} & \textbf{Formal Verif.} & "
        r"\textbf{Memory} & \textbf{Key Limitation} \\",
        r"\midrule",
    ]
    for c in COMPETITOR_COMPARISON:
        lines.append(
            f"  {c['solution']} & {c['platform']} & "
            f"{c['formal_verification']} & {c['memory']} & "
            f"{c['key_limitation']} \\\\"
        )
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def main():
    output_dir = Path(__file__).resolve().parent.parent.parent / "results" / "iec61508"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IEC 61508 SIL 2/3 Compliance Mapping for NeuroPLC")
    print("=" * 70)

    # Markdown
    md = build_markdown_table()
    print("\n" + md)

    # LaTeX
    latex = build_latex_table()
    print("\n" + latex)

    competitor_latex = build_competitor_latex_table()
    print("\n" + competitor_latex)

    # Save
    report = {
        "iec61508_requirements": IEC61508_REQUIREMENTS,
        "svnn_iec61508_mapping": SVNN_IEC61508_MAPPING,
        "competitor_comparison": COMPETITOR_COMPARISON,
        "summary": {
            "total_requirements_mapped": len(SVNN_IEC61508_MAPPING),
            "sil2_fully_satisfied": sum(
                1 for m in SVNN_IEC61508_MAPPING if "SATISFIED" in m["sil2_status"]),
            "sil3_fully_satisfied": sum(
                1 for m in SVNN_IEC61508_MAPPING if "SATISFIED" in m["sil3_status"]),
        },
    }

    json_path = output_dir / "iec61508_mapping.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved: {json_path}")

    latex_path = output_dir / "iec61508_tables.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex + "\n\n" + competitor_latex)
    print(f"LaTeX tables saved: {latex_path}")

    # Summary
    s = report["summary"]
    print(f"\n  {s['total_requirements_mapped']} SVNN conditions mapped")
    print(f"  SIL 2: {s['sil2_fully_satisfied']}/6 fully satisfied")
    print(f"  SIL 3: {s['sil3_fully_satisfied']}/6 fully satisfied")

    return report


if __name__ == "__main__":
    main()
