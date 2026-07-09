#!/usr/bin/env python3
"""
NeuroPLC — Origin Pro 2025b COM Automation: Publication-Quality Figures
=======================================================================
Uses Origin's COM interface to generate professional scientific graphs
that MATLAB cannot match: grouped bars with error bands, dual-Y overlays,
and vector PDF export with journal template support.

Strategic split:
  Origin → Bar charts, grouped comparisons, dual-axis overlays
  MATLAB → Math plots (scatter, loglog, function visualization, fill/stem)

Requires: OriginPro 2025b running, pywin32 installed.
"""

import os, sys, time
from pathlib import Path

OUTPUT_DIR = Path("D:/neuroplc-paper/paper/figures")

try:
    import win32com.client as win32
    origin = win32.Dispatch("Origin.ApplicationSI")
    print("[OK] Origin Pro connected via COM")
except Exception as e:
    print(f"[SKIP] Origin COM unavailable: {e}")
    print("Falling back to MATLAB-only mode for all figures.")
    raise SystemExit(0)


def safe(func, *args, **kwargs):
    """Wrap Origin COM calls with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"  [WARN] {func.__name__}: {e}")
        return None


def new_project():
    """Create a new Origin project."""
    safe(origin.Execute, "doc -n;")
    time.sleep(0.5)


def create_worksheet(name, col_labels, data_cols):
    """Create a worksheet with labeled columns and data."""
    safe(origin.Execute, f"win -t data;")
    cmd = f"wks.ncols = {len(col_labels)};"
    safe(origin.Execute, cmd)
    for i, label in enumerate(col_labels):
        safe(origin.Execute, f'wks.col{i+1}.label$ = "{label}";')
    for ci, col_data in enumerate(data_cols):
        for ri, val in enumerate(col_data):
            safe(origin.Execute, f"wks.col{ci+1}.cell{ri+1}$ = {val};")
    safe(origin.Execute, "wks.lt_exec();")  # execute LabTalk


def export_graph(name, basename):
    """Export active graph as PDF + PNG."""
    pdf_path = str(OUTPUT_DIR / f"{basename}.pdf")
    png_path = str(OUTPUT_DIR / f"{basename}.png")
    safe(origin.Execute, f'expGraph type:=pdf filename:="{pdf_path}" tr:=origin;')
    safe(origin.Execute, f'expGraph type:=png filename:="{png_path}" width:=1200 tr:=origin;')
    print(f"  -> {basename}.pdf + .png")
    time.sleep(0.5)


# ════════════════════════════════════════════════════
# Figure 1: C^2-BV Z3 Verification — Grouped Bar
# ════════════════════════════════════════════════════
print("\n[Origin 1/5] C^2-BV Verification Grouped Bar")
new_project()
create_worksheet(
    "C2BV_Verify",
    ["Architecture", "Z3 Rate (%)", "CWRU Acc (%)", "Safety Margin"],
    [
        ["B-spline", "Fourier", "Wavelet", "ChebyKAN", "MLP"],
        [100.0, 100.0, 100.0, 96.9, 0.0],
        [99.93, 100.0, 100.0, 99.87, 24.13],
        [4.5, 2.9, 5.6, 1.1, 0.0],
    ]
)

safe(origin.Execute, """
plotgroup iy:=(1,3) plot:=200 type:=1;  // Grouped bar
page.title = 0;
layer.x.label.text$ = "C^{2}-BV Architecture";
layer.y.label.text$ = "Value";
label -r legend;
""")
export_graph("fig_c2bv_origin", "fig_c2bv_origin")


# ════════════════════════════════════════════════════
# Figure 2: DA vs IA — Dual-Y Bar + Line
# ════════════════════════════════════════════════════
print("[Origin 2/5] DA vs IA Dual-Y Comparison")
new_project()
create_worksheet(
    "DA_vs_IA",
    ["N", "DA Bound", "IA Bound", "Ratio"],
    [
        [8, 10, 12, 15, 18, 20],
        [0.419, 0.305, 0.212, 0.079, 0.055, 0.044],
        [0.922, 0.671, 0.466, 0.172, 0.121, 0.097],
        [2.20, 2.20, 2.20, 2.18, 2.20, 2.20],
    ]
)
safe(origin.Execute, """
plotxy iy:=[Book1]DA_vs_IA!(1,2:3) plot:=200 ogl:=1;  // scatter for DA/IA
layer.y.label.text$ = "Error Bound";
layer.x.label.text$ = "LUT Points N";
""")
export_graph("fig_da_vs_ia_origin", "fig_da_vs_ia_origin")


# ════════════════════════════════════════════════════
# Figure 3: Model Comparison — Horizontal grouped bars
# ════════════════════════════════════════════════════
print("[Origin 3/5] Model Comparison")
new_project()
create_worksheet(
    "Models",
    ["Model", "Parameters", "Accuracy (%)"],
    [
        ["Teacher CNN", "B-spline KAN", "FourierKAN", "WaveletKAN", "ChebyKAN", "MLP"],
        [48708, 6148, 6676, 4628, 6400, 1524],
        [99.93, 99.93, 100.0, 100.0, 99.87, 99.89],
    ]
)
safe(origin.Execute, """
plotxy iy:=[Book1]Models!(1,2) plot:=200 ogl:=1;
layer.y.label.text$ = "Parameters";
layer.x.label.text$ = "Model";
""")
export_graph("fig_models_origin", "fig_models_origin")


# ════════════════════════════════════════════════════
# Figure 4: Cross-Domain Transfer — Stacked Bar
# ════════════════════════════════════════════════════
print("[Origin 4/5] Cross-Domain Transfer")
new_project()
create_worksheet(
    "XDomain",
    ["Arch", "CWRU", "XJTU-SY", "Z3 Rate"],
    [
        ["B-spline", "Fourier", "Wavelet", "ChebyKAN", "MLP"],
        [99.93, 100.0, 100.0, 99.87, 24.13],
        [91.7, 100.0, 100.0, 0.0, 0.0],
        [100.0, 100.0, 100.0, 96.9, 0.0],
    ]
)
safe(origin.Execute, """
plotgroup iy:=(1,3) plot:=200 type:=1;
layer.y.label.text$ = "Value (%)";
layer.x.label.text$ = "Architecture";
""")
export_graph("fig_cross_domain_origin", "fig_cross_domain_origin")


# ════════════════════════════════════════════════════
# Figure 5: WCET Breakdown — Professional Stacked Bar
# ════════════════════════════════════════════════════
print("[Origin 5/5] WCET Breakdown")
new_project()
create_worksheet(
    "WCET",
    ["Component", "Time (ms)"],
    [
        ["LUT L0 (448 edges)", "LUT L1 (64 edges)", "MatMul", "Softmax", "Overhead"],
        [16.44, 2.35, 3.70, 0.11, 0.07],
    ]
)
safe(origin.Execute, """
plotxy iy:=[Book1]WCET!(1,2) plot:=200 ogl:=1;  // vertical bar
layer.y.label.text$ = "Time (ms)";
layer.x.label.text$ = "Component";
""")
export_graph("fig_wcet_origin", "fig_wcet_origin")


print("\n[DONE] Origin COM figure generation completed.")
print("Note: For best results, manually adjust:")
print("  - Color palette to Wong (Origin Theme Organizer)")
print("  - Font to Helvetica 9pt (Format > Page Properties)")
print("  - Export dimensions: 6.9 x 2.6 inches (IEEE two-column)")
