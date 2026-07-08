#!/usr/bin/env python3
"""
NeuroPLC — Multi-Seed Training Orchestrator (P2: Statistical Rigor)
=====================================================================
Trains teacher CNN + student KAN (VRM-KD) across 5 random seeds,
producing independent checkpoints for statistical analysis.

Seeds: [42, 123, 456, 789, 1024]

Output structure:
    results/teacher/teacher_seed{seed}_best.pt
    results/student/kan_kd_vrmKD_seed{seed}_best.pt
    results/multiseed/manifest.json

Usage:
    python training/train_multiseed.py              # Full 5-seed training
    python training/train_multiseed.py --seeds 42,123  # Specific seeds
    python training/train_multiseed.py --skip-teacher   # Only train students
    python training/train_multiseed.py --dry-run        # Check what would run
"""

import os, sys, json, time, subprocess
from pathlib import Path
from argparse import ArgumentParser

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
RESULTS_DIR = PROJECT_ROOT / "results"
MANIFEST_PATH = RESULTS_DIR / "multiseed" / "manifest.json"


def check_checkpoint_exists(kind, seed):
    """Check if a specific seed checkpoint already exists."""
    if kind == "teacher":
        path = RESULTS_DIR / "teacher" / f"teacher_seed{seed}_best.pt"
    else:
        path = RESULTS_DIR / "student" / f"kan_kd_vrmKD_seed{seed}_best.pt"
    return path.exists()


def train_teacher(seed, dry_run=False):
    """Train teacher CNN for one seed."""
    tag = f"seed{seed}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_teacher.py"),
        "--seed", str(seed),
        "--tag", tag,
        "--no-mlflow",
    ]
    print(f"  [TEACHER seed={seed}] {cmd[0]} {' '.join(cmd[2:])}")
    if dry_run:
        return True, "DRY_RUN"
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "code"),
                                capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            # Rename checkpoint to include seed
            default_ckpt = RESULTS_DIR / "teacher" / "teacher_best.pt"
            seed_ckpt = RESULTS_DIR / "teacher" / f"teacher_seed{seed}_best.pt"
            if default_ckpt.exists() and not seed_ckpt.exists():
                import shutil
                shutil.copy2(default_ckpt, seed_ckpt)
            return True, "OK"
        else:
            return False, result.stderr[-200:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def train_student(seed, teacher_seed, dry_run=False):
    """Train student KAN (VRM-KD) for one seed."""
    teacher_ckpt = RESULTS_DIR / "teacher" / f"teacher_seed{teacher_seed}_best.pt"
    if not teacher_ckpt.exists():
        teacher_ckpt = RESULTS_DIR / "teacher" / "teacher_best.pt"
    if not teacher_ckpt.exists():
        return False, f"Teacher checkpoint not found: {teacher_ckpt}"

    tag = f"seed{seed}"
    cmd = [
        sys.executable, str(PROJECT_ROOT / "training" / "train_student_kd.py"),
        "--teacher", str(teacher_ckpt),
        "--seed", str(seed),
        "--tag", tag,
        "--no-mlflow",
    ]
    print(f"  [STUDENT seed={seed}] {cmd[0]} {' '.join(cmd[2:])}")
    if dry_run:
        return True, "DRY_RUN"
    try:
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT / "code"),
                                capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            default_ckpt = RESULTS_DIR / "student" / "kan_kd_vrmKD_best.pt"
            seed_ckpt = RESULTS_DIR / "student" / f"kan_kd_vrmKD_seed{seed}_best.pt"
            if default_ckpt.exists() and not seed_ckpt.exists():
                import shutil
                shutil.copy2(default_ckpt, seed_ckpt)
            return True, "OK"
        else:
            return False, result.stderr[-200:]
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    except Exception as e:
        return False, str(e)


def main():
    parser = ArgumentParser(description="Multi-Seed Training for Statistical Rigor")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,1024",
                        help="Comma-separated seeds (default: 5 seeds)")
    parser.add_argument("--skip-teacher", action="store_true",
                        help="Skip teacher training (use existing checkpoints)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without running")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    print("=" * 60)
    print(f"Multi-Seed Training: {len(seeds)} seeds → {seeds}")
    print(f"Dry run: {args.dry_run}")
    print("=" * 60)

    manifest = {"seeds": seeds, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                "teacher": {}, "student": {}}

    # ── Phase 1: Teacher ──
    if not args.skip_teacher:
        print("\n[Phase 1] Training teacher CNN...")
        for seed in seeds:
            ok, msg = train_teacher(seed, dry_run=args.dry_run)
            manifest["teacher"][str(seed)] = {"ok": ok, "msg": msg}
            if not ok and not args.dry_run:
                print(f"  ❌ Teacher seed={seed} FAILED: {msg}")
    else:
        print("\n[Phase 1] Skipping teacher (--skip-teacher)")

    # ── Phase 2: Student KAN (VRM-KD) ──
    print("\n[Phase 2] Training student KAN (VRM-KD)...")
    for seed in seeds:
        ok, msg = train_student(seed, seed, dry_run=args.dry_run)
        manifest["student"][str(seed)] = {"ok": ok, "msg": msg}
        if not ok and not args.dry_run:
            print(f"  ❌ Student seed={seed} FAILED: {msg}")

    # ── Save manifest ──
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["n_seeds"] = len(seeds)
    manifest["teacher_ok"] = sum(1 for v in manifest["teacher"].values() if v["ok"])
    manifest["student_ok"] = sum(1 for v in manifest["student"].values() if v["ok"])

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print(f"COMPLETE: Teacher={manifest['teacher_ok']}/{len(seeds)}, "
          f"Student={manifest['student_ok']}/{len(seeds)}")
    print(f"Manifest → {MANIFEST_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
