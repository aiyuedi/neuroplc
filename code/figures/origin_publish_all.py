#!/usr/bin/env python3
"""NeuroPLC — 5 Origin Pro figures via origin_auto Demo Pattern.
Pattern from memory/demo: create_graph + add_plot + export."""

import os, sys, time
sys.path.insert(0, r"D:\dev-tools\research\origin_auto")
from origin_auto import OriginSession

OUT = r"D:\neuroplc-paper\paper\figures"
os.makedirs(OUT, exist_ok=True)


def fill_text(o, col, *values):
    for i, v in enumerate(values, 1):
        o.execute(f'wks.col{col}.cell{i}$ = {v};')


def publish(wb, plot_y_cols, title, xlab, ylab, basename, is_bar=True):
    """Plot multiple Y columns as grouped bars, export."""
    o = wb._s
    o.execute(f'page.title=0; layer.x.label.text$ = "{xlab}"; layer.y.label.text$ = "{ylab}";')
    g = o.create_graph(basename, wb, x_col=1, y_col=plot_y_cols,
                       plot_type="column" if is_bar else "line+symbol")
    g.set_labels(xlab, ylab, title)
    r1 = g.export(os.path.join(OUT, basename), fmt="pdf", width=2400)
    r2 = g.export(os.path.join(OUT, basename), fmt="png", width=1600)
    sz = os.path.getsize(r1) if isinstance(r1, str) and os.path.exists(r1) else 0
    print(f"   [OK] {basename}.pdf ({sz/1024:.0f} KB)")


with OriginSession(visible=True) as o:
    print("Origin connected.\n")

    # ═══════════ 1. C2-BV Architecture Z3 ═══════════
    print("[1/5] C2-BV Architecture Z3 Verification")
    o.new_project()
    wb = o.new_workbook("C2BV")
    wb.set_column_names(["Arch", "Z3_Rate", "CWRU_Acc", "Safety_Margin"])
    fill_text(o, 1, "B-spline", "Fourier", "Wavelet", "ChebyKAN", "MLP")
    o.execute("wks.col2.SetData([100,100,100,96.9,0],0); wks.col3.SetData([99.93,100,100,99.87,24.13],0); wks.col4.SetData([4.5,2.9,5.6,1.1,0],0);")
    publish(wb, 2, "C2-BV Architecture Verification",
            r"C\+ (2)-BV Architecture", "Value", "fig_c2bv_origin")

    # ═══════════ 2. DA vs IA ═══════════
    print("[2/5] DA vs IA Comparison")
    o.new_project()
    wb2 = o.new_workbook("DA_IA")
    wb2.set_column_names(["N", "DA", "IA"])
    o.execute("wks.col1.SetData([8,10,12,15,18,20],0); wks.col2.SetData([0.419,0.305,0.212,0.079,0.055,0.044],0); wks.col3.SetData([0.922,0.671,0.466,0.172,0.121,0.097],0);")
    publish(wb2, 2, "DA vs IA Bound Comparison",
            "LUT Points N", "Error Bound", "fig_da_vs_ia_origin")

    # ═══════════ 3. Model Comparison ═══════════
    print("[3/5] Model Comparison")
    o.new_project()
    wb3 = o.new_workbook("Models")
    wb3.set_column_names(["Model", "Params", "Acc"])
    fill_text(o, 1, "Teacher", "B-KAN", "F-KAN", "W-KAN", "C-KAN", "MLP")
    o.execute("wks.col2.SetData([48708,6148,6676,4628,6400,1524],0); wks.col3.SetData([99.93,99.93,100,100,99.87,99.89],0);")
    publish(wb3, 2, "Model Parameters & Accuracy",
            "Model", "Value", "fig_models_origin")

    # ═══════════ 4. Cross-Domain ═══════════
    print("[4/5] Cross-Domain Transfer")
    o.new_project()
    wb4 = o.new_workbook("XDomain")
    wb4.set_column_names(["Arch", "CWRU", "XJTU", "Z3"])
    fill_text(o, 1, "B-KAN", "F-KAN", "W-KAN", "C-KAN", "MLP")
    o.execute("wks.col2.SetData([99.93,100,100,99.87,24.13],0); wks.col3.SetData([91.7,100,100,0,0],0); wks.col4.SetData([100,100,100,96.9,0],0);")
    publish(wb4, 2, "Cross-Dataset Performance",
            "Architecture", "Value (%)", "fig_cross_domain_origin")

    # ═══════════ 5. WCET ═══════════
    print("[5/5] WCET Breakdown")
    o.new_project()
    wb5 = o.new_workbook("WCET")
    wb5.set_column_names(["Component", "Time_ms"])
    o.execute('wks.col1.text$ = "LUT L0";')
    fill_text(o, 1, "LUT L0", "LUT L1", "MatMul", "Softmax", "Overhead")
    o.execute("wks.col2.SetData([16.44,2.35,3.70,0.11,0.07],0);")
    publish(wb5, 2, "WCET Breakdown (S7-1200)",
            "Component", "Time (ms)", "fig_wcet_origin")

    print(f"\n=== DONE -> {OUT} ===")
