#!/usr/bin/env python3
"""
NeuroPLC — ModelScope GPU Training Entry Point
================================================
Upload this project to ModelScope (魔搭社区) for free GPU training.

Setup (one-time):
    1. Go to https://modelscope.cn/my/mynotebook
    2. Create a new notebook → GPU instance (24GB VRAM, CUDA 12.8)
    3. Upload this project (zip or git clone)
    4. Run: python code/train_on_modelscope.py --all

Workflow:
    Local (板板)                           ModelScope (GPU)
    ─────────────────                   ──────────────────
    1. 写代码                            2. 上传项目
    3. 下载结果 ←────────────────────── 4. 跑训练 (GPU)
    5. 本地验证 + 论文

Usage:
    # Train everything
    python train_on_modelscope.py --all

    # Train teacher only
    python train_on_modelscope.py --teacher

    # Train student via KD only
    python train_on_modelscope.py --student

    # Quick test: 5 epochs to verify pipeline
    python train_on_modelscope.py --all --test-mode
"""

import sys
import os
import argparse
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def check_gpu():
    """Verify GPU availability on ModelScope."""
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        return True
    else:
        print("WARNING: GPU not available. Training will be slow on CPU.")
        return False


def run_preprocess():
    """Run preprocessing pipeline."""
    print("\n" + "=" * 60)
    print("Step 1: Data Preprocessing")
    print("=" * 60)
    os.chdir(PROJECT_ROOT / "code")
    subprocess.run(
        [sys.executable, "data_pipeline/preprocess.py", "--mode", "both", "--cross-load"],
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def train_teacher(test_mode: bool = False):
    """Train Teacher 1D-CNN."""
    print("\n" + "=" * 60)
    print("Step 2: Teacher CNN Training")
    print("=" * 60)
    os.chdir(PROJECT_ROOT / "code")
    cmd = [sys.executable, "training/train_teacher.py"]
    if test_mode:
        cmd += ["--epochs", "5", "--tag", "test"]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def train_student_kd(test_mode: bool = False):
    """Train Student KAN via VRM-KD."""
    print("\n" + "=" * 60)
    print("Step 3: Student KAN via VRM-KD")
    print("=" * 60)
    os.chdir(PROJECT_ROOT / "code")
    cmd = [sys.executable, "training/train_student_kd.py"]
    if test_mode:
        cmd += ["--epochs", "5", "--tag", "test"]
    subprocess.run(cmd, check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"})


def run_evaluation():
    """Run all 7 experiments."""
    print("\n" + "=" * 60)
    print("Step 4: Evaluation (E1-E7)")
    print("=" * 60)
    os.chdir(PROJECT_ROOT / "code")
    subprocess.run(
        [sys.executable, "evaluate.py", "--all"],
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def run_visualization():
    """Generate all figures."""
    print("\n" + "=" * 60)
    print("Step 5: Visualization")
    print("=" * 60)
    os.chdir(PROJECT_ROOT / "code")
    subprocess.run(
        [sys.executable, "analysis/visualize.py", "--all"],
        check=True,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def main():
    parser = argparse.ArgumentParser(description="NeuroPLC GPU Training on ModelScope")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")
    parser.add_argument("--teacher", action="store_true", help="Train teacher only")
    parser.add_argument("--student", action="store_true", help="Train student only")
    parser.add_argument("--test-mode", action="store_true", help="Quick test (5 epochs)")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip preprocessing")
    parser.add_argument("--skip-viz", action="store_true", help="Skip visualization")
    args = parser.parse_args()

    # ── Header ──
    print("=" * 60)
    print("  NeuroPLC — ModelScope GPU Training")
    print("=" * 60)
    print(f"  Project root: {PROJECT_ROOT}")
    print(f"  Python:        {sys.executable}")
    print(f"  Test mode:     {args.test_mode}")
    print("=" * 60)

    has_gpu = check_gpu()

    if args.all:
        args.teacher = True
        args.student = True

    if not args.teacher and not args.student:
        print("\nNo training target specified. Use --all, --teacher, or --student.")
        return

    # ── Pipeline ──
    if not args.skip_preprocess:
        run_preprocess()

    if args.teacher:
        train_teacher(args.test_mode)

    if args.student:
        train_student_kd(args.test_mode)

    run_evaluation()

    if not args.skip_viz:
        run_visualization()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"\n  Results saved to: {PROJECT_ROOT / 'results'}")
    print(f"  Download this entire directory back to local.\n")


if __name__ == "__main__":
    main()
