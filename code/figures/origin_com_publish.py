#!/usr/bin/env python3
"""NeuroPLC — Origin Pro 2025b COM: 5 publication figures (runs as .py file)."""
import win32com.client, os, time

OUT = r"D:\neuroplc-paper\paper\figures"
os.makedirs(OUT, exist_ok=True)

o = win32com.client.Dispatch("Origin.ApplicationSI")
print("Origin COM connected")

def export(name):
    """Export active graph to PDF+PNG."""
    pdf = os.path.join(OUT, name + ".pdf")
    png = os.path.join(OUT, name + ".png")
    # Key: expGraph with separate path (dir) and filename
    d = OUT.replace("\\", "\\\\")
    f = name
    o.Execute(f'expGraph type:=pdf filename:="{f}" path:="{d}" tr:=origin;')
    time.sleep(0.5)
    o.Execute(f'expGraph type:=png filename:="{f}" path:="{d}" width:=1600 tr:=origin;')
    time.sleep(0.5)
    print(f"  -> {name}.pdf + .png")

# ═══════════════════════════════════
# Figure 1: C^2-BV Z3 Verification
# ═══════════════════════════════════
print("\n[1/5] C2-BV Architecture Z3 Verification")
o.Execute("doc -n;")  # new project
o.Execute("wks.ncols = 4;")
o.Execute(r'wks.col1.label$ = "Arch";')
o.Execute(r'wks.col2.label$ = "Z3 Rate";')
o.Execute(r'wks.col3.label$ = "CWRU Acc";')
o.Execute(r'wks.col4.label$ = "Safety Margin";')
o.Execute("wks.col1.text$ = B-spline; wks.col1.cell2$ = Fourier; wks.col1.cell3$ = Wavelet; wks.col1.cell4$ = ChebyKAN; wks.col1.cell5$ = MLP;")
o.Execute("wks.col2.SetData([100,100,100,96.9,0],0);")
o.Execute("wks.col3.SetData([99.93,100,100,99.87,24.13],0);")
o.Execute("wks.col4.SetData([4.5,2.9,5.6,1.1,0],0);")
o.Execute("plotgroup iy:=(1,3) plot:=200 type:=1; page.title=0;")
o.Execute(r'layer.x.label.text$ = "C\+ (2)-BV Architecture";')
o.Execute(r'layer.y.label.text$ = "Value";')
export("fig_c2bv_origin")

# ═══════════════════════════════════
# Figure 2: DA vs IA
# ═══════════════════════════════════
print("[2/5] DA vs IA Grouped Bar")
o.Execute("doc -n;")
o.Execute("wks.ncols = 3;")
o.Execute(r'wks.col1.label$ = "N";')
o.Execute(r'wks.col2.label$ = "DA";')
o.Execute(r'wks.col3.label$ = "IA";')
o.Execute("wks.col1.SetData([8,10,12,15,18,20],0);")
o.Execute("wks.col2.SetData([0.419,0.305,0.212,0.079,0.055,0.044],0);")
o.Execute("wks.col3.SetData([0.922,0.671,0.466,0.172,0.121,0.097],0);")
o.Execute("plotgroup iy:=(1,2) plot:=200 type:=1; page.title=0;")
o.Execute(r'layer.x.label.text$ = "LUT Points N";')
o.Execute(r'layer.y.label.text$ = "Error Bound";')
export("fig_da_vs_ia_origin")

# ═══════════════════════════════════
# Figure 3: Model Comparison
# ═══════════════════════════════════
print("[3/5] Model Comparison")
o.Execute("doc -n;")
o.Execute("wks.ncols = 3;")
o.Execute(r'wks.col1.label$ = "Model";')
o.Execute(r'wks.col2.label$ = "Params";')
o.Execute(r'wks.col3.label$ = "Acc";')
o.Execute("wks.col1.text$ = Teacher; wks.col1.cell2$ = B-KAN; wks.col1.cell3$ = F-KAN; wks.col1.cell4$ = W-KAN; wks.col1.cell5$ = C-KAN; wks.col1.cell6$ = MLP;")
o.Execute("wks.col2.SetData([48708,6148,6676,4628,6400,1524],0);")
o.Execute("wks.col3.SetData([99.93,99.93,100,100,99.87,99.89],0);")
o.Execute("plotgroup iy:=(1,2) plot:=200 type:=1; page.title=0;")
o.Execute(r'layer.x.label.text$ = "Model";')
o.Execute(r'layer.y.label.text$ = "Value";')
export("fig_models_origin")

# ═══════════════════════════════════
# Figure 4: Cross-Domain Transfer
# ═══════════════════════════════════
print("[4/5] Cross-Domain Transfer")
o.Execute("doc -n;")
o.Execute("wks.ncols = 4;")
o.Execute(r'wks.col1.label$ = "Arch";')
o.Execute(r'wks.col2.label$ = "CWRU";')
o.Execute(r'wks.col3.label$ = "XJTU-SY";')
o.Execute(r'wks.col4.label$ = "Z3 Rate";')
o.Execute("wks.col1.text$ = B-KAN; wks.col1.cell2$ = F-KAN; wks.col1.cell3$ = W-KAN; wks.col1.cell4$ = C-KAN; wks.col1.cell5$ = MLP;")
o.Execute("wks.col2.SetData([99.93,100,100,99.87,24.13],0);")
o.Execute("wks.col3.SetData([91.7,100,100,0,0],0);")
o.Execute("wks.col4.SetData([100,100,100,96.9,0],0);")
o.Execute("plotgroup iy:=(1,3) plot:=200 type:=1; page.title=0;")
o.Execute(r'layer.x.label.text$ = "Architecture";')
o.Execute(r'layer.y.label.text$ = "Value (%)";')
export("fig_cross_domain_origin")

# ═══════════════════════════════════
# Figure 5: WCET Breakdown
# ═══════════════════════════════════
print("[5/5] WCET Breakdown")
o.Execute("doc -n;")
o.Execute("wks.ncols = 2;")
o.Execute(r'wks.col1.label$ = "Component";')
o.Execute(r'wks.col2.label$ = "Time (ms)";')
o.Execute("wks.col1.text$ = LUT L0; wks.col1.cell2$ = LUT L1; wks.col1.cell3$ = MatMul; wks.col1.cell4$ = Softmax; wks.col1.cell5$ = Overhead;")
o.Execute("wks.col2.SetData([16.44,2.35,3.70,0.11,0.07],0);")
o.Execute("plotxy iy:=[%H]1!(1,2) plot:=200 ogl:=1; page.title=0;")
o.Execute(r'layer.y.label.text$ = "Time (ms)";')
o.Execute(r'layer.x.label.text$ = "Component";')
export("fig_wcet_origin")

print(f"\nALL 5 DONE -> {OUT}")
