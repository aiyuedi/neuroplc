#!/usr/bin/env python3
"""
NeuroPLC SCL Code Generator — Unified entry point
===================================================
Generates IEC 61131-3 SCL from trained PyTorch KAN/MLP models.

Supports two output formats:
  single  — Single-file SCL via full compiler pipeline (backend_s7)
            Best for: paper figures, line counts, quick inspection
  db_fb   — DB+FB split SCL, TIA Portal compatible (backend_s7_db)
            Best for: TIA Portal import + compile verification

Usage:
  python generate.py                          # all models, both targets, both formats
  python generate.py --model kan --target s7-1200 --format db_fb
  python generate.py --model mlp --target s7-1500 --format single
"""

import sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))

import torch
from models.student_kan import StudentKAN
from models.student_mlp import StudentMLP

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "scl_output")
BASE = os.path.join(os.path.dirname(__file__), "..")


def generate_single(model, model_type, target, lut_pts):
    """Generate single-file SCL via full compiler pipeline (backend_s7)."""
    from neuroplc.compiler import NeuroPLCCompiler

    compiler = NeuroPLCCompiler(target=target, verbose=False)
    output_path = os.path.join(OUTPUT_DIR, f"{model_type}_{target}.scl")
    result = compiler.compile(model, output=output_path, model_type=model_type)
    lines = result.scl_code.count("\n")
    mem = result.analyzer_report.get("memory", {}).get("total_kb", "?")
    return output_path, lines, mem


def generate_db_fb(model, model_type, target, lut_pts):
    """Generate DB+FB split SCL (TIA Portal compatible, backend_s7_db)."""
    from neuroplc.frontend import kan_to_ir, mlp_to_ir
    from neuroplc.backend_s7_db import S71200DBBackend, S71500DBBackend

    model.eval()
    if model_type == "kan":
        ir_graph = kan_to_ir(model, lut_points=lut_pts, x_range=(-3.0, 3.0), adaptive=True)
    else:
        ir_graph = mlp_to_ir(model)

    tag = f"{model_type}_{target}"
    db_name = f"NeuroPLC_{model_type.upper()}_Weights"

    if "1200" in target:
        backend = S71200DBBackend(lut_pts=lut_pts, db_name=db_name)
    else:
        backend = S71500DBBackend(lut_pts=lut_pts, db_name=db_name)

    db_scl, fb_scl = backend.generate(ir_graph)

    db_path = os.path.join(OUTPUT_DIR, f"{tag}_db.scl")
    fb_path = os.path.join(OUTPUT_DIR, f"{tag}_db_fb.scl")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(db_path, "w", encoding="utf-8") as f:
        f.write(db_scl)
    with open(fb_path, "w", encoding="utf-8") as f:
        f.write(fb_scl)

    return (db_path, fb_path), db_scl.count("\n") + fb_scl.count("\n"), "DB+FB"


def load_models():
    """Load trained KAN and MLP checkpoints."""
    kan = StudentKAN([28, 16, 4])
    ckpt = torch.load(os.path.join(BASE, "results", "student", "kan_kd_vrmKD_best.pt"),
                      map_location="cpu", weights_only=True)
    kan.load_state_dict(ckpt["student_state_dict"])
    kan.eval()

    mlp = StudentMLP(input_dim=28, hidden_dims=[32, 16], num_classes=4)
    ckpt = torch.load(os.path.join(BASE, "results", "student", "mlp_kd_vrmKD_best.pt"),
                      map_location="cpu", weights_only=True)
    mlp.load_state_dict(ckpt["student_state_dict"])
    mlp.eval()

    return kan, mlp


def main():
    parser = argparse.ArgumentParser(description="NeuroPLC SCL Code Generator")
    parser.add_argument("--model", choices=["kan", "mlp", "all"], default="all")
    parser.add_argument("--target", choices=["s7-1200", "s7-1500", "all"], default="all")
    parser.add_argument("--format", choices=["single", "db_fb", "all"], default="all",
                        help="single=full compiler pipeline, db_fb=TIA Portal compatible")
    args = parser.parse_args()

    print("=" * 60)
    print("  NeuroPLC SCL Code Generator")
    print("=" * 60)

    kan, mlp = load_models()
    print(f"  KAN [28,16,4] loaded OK")
    print(f"  MLP [28,32,16,4] loaded OK\n")

    models = []
    if args.model in ("kan", "all"):
        models.append(("kan", kan))
    if args.model in ("mlp", "all"):
        models.append(("mlp", mlp))

    targets = []
    if args.target in ("s7-1200", "all"):
        targets.append("s7-1200")
    if args.target in ("s7-1500", "all"):
        targets.append("s7-1500")

    formats = []
    if args.format in ("single", "all"):
        formats.append("single")
    if args.format in ("db_fb", "all"):
        formats.append("db_fb")

    for model_type, model in models:
        for target in targets:
            lut_pts = 15 if "1200" in target else 50
            if model_type == "mlp":
                lut_pts = 0  # MLP has no B-spline LUTs

            for fmt in formats:
                label = f"{model_type.upper()} → {target} [{fmt}]"
                print(f"  {label}: ", end="", flush=True)

                if fmt == "single":
                    path, lines, mem = generate_single(model, model_type, target, lut_pts)
                    print(f"{lines} lines, {mem}KB → {os.path.basename(path)}")
                else:
                    (db_path, fb_path), lines, _ = generate_db_fb(model, model_type, target, lut_pts)
                    print(f"{lines} lines (DB+FB) → {os.path.basename(db_path)} + {os.path.basename(fb_path)}")

    print(f"\n  Output: {OUTPUT_DIR}")
    print("  Done.")


if __name__ == "__main__":
    main()
