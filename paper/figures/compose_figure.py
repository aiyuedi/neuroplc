"""
NeuroPLC Paper - Compose Multi-Panel Figure using svgutils
使用 svgutils 组合 SVG 面板为顶刊级多面板图表
"""
from svgutils.compose import Figure, SVG, Panel, Text
from pathlib import Path

# ==================== 配置 ====================
PANELS_DIR = Path("D:/neuroplc-paper/paper/figures/panels")
OUTPUT_DIR = Path("D:/neuroplc-paper/paper/figures/final")

# IEEE 双栏尺寸
WIDTH_MM = 183  # 7.16 inches = 183mm
HEIGHT_MM = 120  # 约 4.7 inches

# ==================== 组合图表 ====================
def compose_2x2_figure():
    """
    组合 4 个面板为 2x2 图表
    """
    print("Composing 2x2 figure...")

    # 创建面板配置
    panels_config = {
        "width_mm": WIDTH_MM,
        "height_mm": HEIGHT_MM,
        "journal": "ieee",
        "panels": [
            # Panel A: DA vs IA (左上)
            {
                "id": "A",
                "src": str(PANELS_DIR / "panel_a_da_vs_ia.svg"),
                "x_mm": 0,
                "y_mm": 0,
                "scale": 1.0,
                "label": "A"
            },
            # Panel B: Adaptive LUT (右上)
            {
                "id": "B",
                "src": str(PANELS_DIR / "panel_b_adaptive_lut.svg"),
                "x_mm": 92,
                "y_mm": 0,
                "scale": 1.0,
                "label": "B"
            },
            # Panel C: DA Scaling (左下)
            {
                "id": "C",
                "src": str(PANELS_DIR / "panel_c_da_scaling.svg"),
                "x_mm": 0,
                "y_mm": 62,
                "scale": 1.0,
                "label": "C"
            },
            # Panel D: Segment Bounds (右下)
            {
                "id": "D",
                "src": str(PANELS_DIR / "panel_d_segment_bounds.svg"),
                "x_mm": 92,
                "y_mm": 62,
                "scale": 1.0,
                "label": "D"
            }
        ]
    }

    # 保存配置文件
    import json
    config_path = PANELS_DIR / "panels-config.json"
    with open(config_path, 'w') as f:
        json.dump(panels_config, f, indent=2)
    print(f"  Config saved: {config_path}")

    # 使用 compose.py 组合
    compose_script = Path("C:/Users/ASUS/.claude/skills/scientific-figure/scripts/compose.py")
    if compose_script.exists():
        print(f"  Using compose.py: {compose_script}")
        # 这里需要调用 compose.py，但需要 uv run
        # 暂时跳过，直接用 svgutils
    else:
        print("  compose.py not found, using direct svgutils")

    # 直接用 svgutils 组合
    fig = Figure(
        f"{WIDTH_MM}mm", f"{HEIGHT_MM}mm",
        Panel(
            SVG(str(PANELS_DIR / "panel_a_da_vs_ia.svg")).scale(1.0),
            Text("A", 5, 15, size=12, weight="bold"),
        ).move(0, 0),
        Panel(
            SVG(str(PANELS_DIR / "panel_b_adaptive_lut.svg")).scale(1.0),
            Text("B", 5, 15, size=12, weight="bold"),
        ).move(92, 0),
        Panel(
            SVG(str(PANELS_DIR / "panel_c_da_scaling.svg")).scale(1.0),
            Text("C", 5, 15, size=12, weight="bold"),
        ).move(0, 62),
        Panel(
            SVG(str(PANELS_DIR / "panel_d_segment_bounds.svg")).scale(1.0),
            Text("D", 5, 15, size=12, weight="bold"),
        ).move(92, 62),
    )

    # 保存为 SVG
    output_svg = OUTPUT_DIR / "figure_2x2_composed.svg"
    fig.save(str(output_svg))
    print(f"  Composed SVG: {output_svg}")

    return output_svg

# ==================== 导出 PDF/PNG ====================
def export_figure(svg_path):
    """
    导出 SVG 为 PDF 和 PNG
    """
    print("Exporting to PDF and PNG...")

    # 检查 Inkscape
    import subprocess
    try:
        result = subprocess.run(['inkscape', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("  Inkscape found, using for high-quality export")
            # 使用 Inkscape 导出 PDF
            pdf_path = OUTPUT_DIR / "figure_2x2_composed.pdf"
            subprocess.run(['inkscape', str(svg_path), '--export-pdf=' + str(pdf_path),
                          '--export-dpi=300'], check=True)
            print(f"  PDF: {pdf_path}")

            # 使用 Inkscape 导出 PNG
            png_path = OUTPUT_DIR / "figure_2x2_composed.png"
            subprocess.run(['inkscape', str(svg_path), '--export-png=' + str(png_path),
                          '--export-dpi=300'], check=True)
            print(f"  PNG: {png_path}")
        else:
            raise Exception("Inkscape not working")
    except:
        print("  Inkscape not found, using cairosvg")
        # 使用 cairosvg
        try:
            import cairosvg
            pdf_path = OUTPUT_DIR / "figure_2x2_composed.pdf"
            cairosvg.svg2pdf(url=str(svg_path), write_to=str(pdf_path), dpi=300)
            print(f"  PDF: {pdf_path}")

            png_path = OUTPUT_DIR / "figure_2x2_composed.png"
            cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=300)
            print(f"  PNG: {png_path}")
        except ImportError:
            print("  cairosvg not installed, skipping export")
            print("  Install with: pip install cairosvg")

# ==================== Run ====================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Composing Multi-Panel Scientific Figure")
    print("=" * 60 + "\n")

    svg_path = compose_2x2_figure()
    export_figure(svg_path)

    print("\n" + "=" * 60)
    print("Figure composition complete!")
    print("=" * 60)
