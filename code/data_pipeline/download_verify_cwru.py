#!/usr/bin/env python3
"""
CWRU Bearing Data Center — GFW-Compatible Downloader & Verifier
=================================================================
Supports importing CWRU .mat files from local directories (manual download
via 百度网盘 / 123云盘 / OneDrive) and verifying dataset integrity.

Background:
    GitHub (raw.githubusercontent.com) is blocked by GFW in mainland China.
    All 4 download URLs in the original download_cwru.py are unreachable.
    This script is the replacement — designed for the China network environment.

Modes:
    python download_verify_cwru.py --source local --input <dir>
        Import & organize .mat files from a manually downloaded directory.
        Works with flat dumps (all .mat in one folder) or pre-organized dirs.

    python download_verify_cwru.py --verify
        Check that all 52 expected files exist and are valid .mat files.

    python download_verify_cwru.py --manifest
        Print a JSON manifest of expected files (useful for manual checks).

    python download_verify_cwru.py --source auto
        Try to download (uses original strategies, will likely fail in China).

Expected input structure (flat dump):
    some_folder/
    ├── 97.mat
    ├── 98.mat
    ├── ...
    └── 3012.mat

Output structure (organized):
    data/raw/12k_DE/
    ├── Normal/      # 97.mat – 100.mat
    ├── IR007/       # 105.mat – 108.mat
    ├── IR014/       # 169.mat – 172.mat
    ├── IR021/       # 209.mat – 212.mat
    ├── IR028/       # 3001.mat – 3004.mat
    ├── B007/        # 118.mat – 121.mat
    ├── B014/        # 185.mat – 188.mat
    ├── B021/        # 222.mat – 225.mat
    ├── B028/        # 3005.mat – 3008.mat
    ├── OR007@6/     # 130.mat – 133.mat
    ├── OR014@6/     # 197.mat – 200.mat
    ├── OR021@6/     # 234.mat – 237.mat
    └── OR028@6/     # 3009.mat – 3012.mat

Total: 52 files, ~120 MB.

Usage:
    python download_verify_cwru.py --verify              # Quick check
    python download_verify_cwru.py --source local --input D:\Downloads\cwru
    python download_verify_cwru.py --manifest > manifest.json
"""

import os
import sys
import json
import shutil
import argparse
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np

# Fix GBK encoding in Windows terminal
if sys.platform == "win32":
    sys.stdout = open(sys.stdout.fileno(), mode="w", encoding="utf-8",
                      buffering=1, errors="replace")

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.json"

# Minimum valid .mat file size for CWRU 12k DE (some 0.028" files ~970KB)
MIN_FILE_SIZE = 500_000

# ============================================================================
# Complete CWRU 12k DE File Manifest (52 files)
# ============================================================================
# (fault_type, diameter_inch, load_hp) → file_number
CWRU_MANIFEST: dict = {
    # ── Normal baseline (4 files: 97-100) ──
    ("Normal", None,  0): 97,
    ("Normal", None,  1): 98,
    ("Normal", None,  2): 99,
    ("Normal", None,  3): 100,

    # ── Inner Race / 内圈故障 (16 files) ──
    ("IR", 0.007, 0): 105,   ("IR", 0.007, 1): 106,
    ("IR", 0.007, 2): 107,   ("IR", 0.007, 3): 108,
    ("IR", 0.014, 0): 169,   ("IR", 0.014, 1): 170,
    ("IR", 0.014, 2): 171,   ("IR", 0.014, 3): 172,
    ("IR", 0.021, 0): 209,   ("IR", 0.021, 1): 210,
    ("IR", 0.021, 2): 211,   ("IR", 0.021, 3): 212,
    ("IR", 0.028, 0): 3001,  ("IR", 0.028, 1): 3002,
    ("IR", 0.028, 2): 3003,  ("IR", 0.028, 3): 3004,

    # ── Ball / 滚珠故障 (16 files) ──
    ("Ball", 0.007, 0): 118,   ("Ball", 0.007, 1): 119,
    ("Ball", 0.007, 2): 120,   ("Ball", 0.007, 3): 121,
    ("Ball", 0.014, 0): 185,   ("Ball", 0.014, 1): 186,
    ("Ball", 0.014, 2): 187,   ("Ball", 0.014, 3): 188,
    ("Ball", 0.021, 0): 222,   ("Ball", 0.021, 1): 223,
    ("Ball", 0.021, 2): 224,   ("Ball", 0.021, 3): 225,
    ("Ball", 0.028, 0): 3005,  ("Ball", 0.028, 1): 3006,
    ("Ball", 0.028, 2): 3007,  ("Ball", 0.028, 3): 3008,

    # ── Outer Race @6:00 / 外圈故障 (16 files) ──
    ("OR", 0.007, 0): 130,   ("OR", 0.007, 1): 131,
    ("OR", 0.007, 2): 132,   ("OR", 0.007, 3): 133,
    ("OR", 0.014, 0): 197,   ("OR", 0.014, 1): 198,
    ("OR", 0.014, 2): 199,   ("OR", 0.014, 3): 200,
    ("OR", 0.021, 0): 234,   ("OR", 0.021, 1): 235,
    ("OR", 0.021, 2): 236,   ("OR", 0.021, 3): 237,
    ("OR", 0.028, 0): 3009,  ("OR", 0.028, 1): 3010,
    ("OR", 0.028, 2): 3011,  ("OR", 0.028, 3): 3012,
}

LABEL_MAP = {"Normal": 0, "IR": 1, "Ball": 2, "OR": 3}


# ============================================================================
# Subdirectory naming
# ============================================================================

def _subdir_name(fault_type: str, diameter: Optional[float]) -> str:
    """Map (fault_type, diameter) → organized subdirectory name."""
    if fault_type == "Normal":
        return "Normal"
    diam_str = f"{int(diameter * 1000):03d}"  # 0.007 → 007
    return f"{fault_type}{diam_str}"


def _build_file_map() -> dict[int, tuple[str, Optional[float], int, str]]:
    """
    Build lookup: file_number → (fault_type, diameter, load_hp, subdir).
    """
    mapping = {}
    for (fault_type, diameter, load_hp), file_num in CWRU_MANIFEST.items():
        subdir = _subdir_name(fault_type, diameter)
        mapping[file_num] = (fault_type, diameter, load_hp, subdir)
    return mapping


# ============================================================================
# File verification
# ============================================================================

def verify_mat_file(path: Path) -> tuple[bool, str, int]:
    """
    Verify a single .mat file.

    Returns:
        (is_valid, error_message, file_size_bytes)
    """
    if not path.exists():
        return False, "file not found", 0

    size = path.stat().st_size
    if size < MIN_FILE_SIZE:
        return False, f"file too small ({size:,} bytes, expected ≥ {MIN_FILE_SIZE:,})", size

    # Try to load as .mat to verify it's valid MATLAB format
    try:
        import scipy.io
        mat = scipy.io.loadmat(str(path))
        # Check that at least one DE_time variable exists
        has_de = any("DE_time" in k for k in mat if not k.startswith("_"))
        if not has_de:
            return False, "valid .mat but no DE_time variable found", size
        return True, "ok", size
    except Exception as e:
        return False, f"invalid .mat: {str(e)[:80]}", size


def compute_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file (for integrity tracking)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================
# Verification mode
# ============================================================================

def verify_dataset(base_dir: Path, verbose: bool = True) -> dict:
    """
    Check all 52 expected files in the organized directory structure.

    Returns:
        {
            "total": 52,
            "ok": 45,
            "missing": [...],
            "corrupt": [...],
            "total_size_mb": 120.5,
            "details": {97: {...}, 98: {...}, ...}
        }
    """
    file_map = _build_file_map()
    ok, missing, corrupt = [], [], []
    total_bytes = 0
    details = {}

    for file_num in sorted(file_map.keys()):
        fault_type, diameter, load_hp, subdir = file_map[file_num]
        fname = f"{file_num}.mat"
        dest = base_dir / "12k_DE" / subdir / fname
        is_valid, err_msg, size = verify_mat_file(dest)

        details[file_num] = {
            "path": str(dest),
            "subdir": subdir,
            "fault_type": fault_type,
            "diameter": diameter,
            "load_hp": load_hp,
            "size_bytes": size,
            "status": "ok" if is_valid else err_msg,
        }

        if is_valid:
            ok.append(file_num)
            total_bytes += size
        elif dest.exists():
            corrupt.append((file_num, err_msg))
        else:
            missing.append((file_num, err_msg))

    status = "complete" if len(ok) == 52 else \
             "partial" if len(ok) > 0 else "empty"

    # Also scan for unexpected .mat files
    base_12k = base_dir / "12k_DE"
    unexpected = []
    if base_12k.exists():
        for mat_file in sorted(base_12k.rglob("*.mat")):
            try:
                fn = int(mat_file.stem)
                if fn not in file_map:
                    unexpected.append(str(mat_file))
            except ValueError:
                unexpected.append(str(mat_file))

    result = {
        "status": status,
        "total_expected": 52,
        "ok": len(ok),
        "missing": missing,
        "corrupt": corrupt,
        "unexpected_files": unexpected,
        "total_size_mb": round(total_bytes / 1_048_576, 1),
        "details": details,
    }

    if verbose:
        _print_verification_report(result)

    return result


def _print_verification_report(result: dict):
    """Pretty-print the verification report."""
    print()
    print("=" * 68)
    print("  NeuroPLC — CWRU Dataset Integrity Check")
    print("=" * 68)
    print(f"  Status:         {result['status'].upper()}")
    print(f"  Files OK:       {result['ok']} / {result['total_expected']}")
    print(f"  Missing:        {len(result['missing'])}")
    print(f"  Corrupt:        {len(result['corrupt'])}")
    print(f"  Total on disk:  {result['total_size_mb']:.1f} MB")
    print("=" * 68)

    if result["missing"]:
        print(f"\n  ✗ Missing files ({len(result['missing'])}):")
        for file_num, reason in result["missing"]:
            detail = result["details"][file_num]
            print(f"    {file_num:>5d}.mat  ({detail['fault_type']} "
                  f"load={detail['load_hp']}hp)  ← {detail['subdir']}/")

    if result["corrupt"]:
        print(f"\n  ⚠ Corrupt files ({len(result['corrupt'])}):")
        for file_num, reason in result["corrupt"]:
            print(f"    {file_num:>5d}.mat  — {reason}")

    if result["unexpected_files"]:
        print(f"\n  ⚡ Unexpected .mat files ({len(result['unexpected'])}):")
        for p in result["unexpected_files"][:10]:
            print(f"    {p}")
        if len(result["unexpected_files"]) > 10:
            print(f"    ... and {len(result['unexpected_files']) - 10} more")

    if result["ok"] == 52:
        print(f"\n  ✅ All {52} files present and valid!")
        print(f"  → Next: python preprocess.py")
    elif result["ok"] > 0:
        print(f"\n  ⚠ Partial dataset: {result['ok']}/{result['total_expected']} files.")
        print(f"  → Preprocessing will work with available data.")
        print(f"  → To get missing files, download from:")
        print(f"      123云盘: https://www.123pan.com/s/xBwHjv-WIzk.html (提取码 EXLF)")
        print(f"      百度网盘: https://pan.baidu.com/s/1k9xkejB-3YRqDunKA9AUsw (提取码 htgw)")
    else:
        print(f"\n  🔴 No CWRU data found!")
        print(f"  → Download from one of these sources first:")
        print(f"      123云盘: https://www.123pan.com/s/xBwHjv-WIzk.html (提取码 EXLF)")
        print(f"      百度网盘: https://pan.baidu.com/s/1k9xkejB-3YRqDunKA9AUsw (提取码 htgw)")
        print(f"      官网: https://engineering.case.edu/bearingdatacenter")
        print(f"  → Then run: python download_verify_cwru.py --source local --input <folder>")


# ============================================================================
# Local import mode (= GFW-compatible replacement for download)
# ============================================================================

def import_from_local(source_dir: Path, output_root: Path,
                      copy: bool = True, verbose: bool = True) -> dict:
    """
    Import .mat files from a manually-downloaded source directory.

    Handles two common layouts:
    1. Flat dump: all .mat files in one folder (typical of 百度网盘/123云盘)
       → auto-organize into subdirectories by file number
    2. Pre-organized: files already in <FaultType>/<num>.mat structure
       → copy/verify with same structure

    Args:
        source_dir: where the user downloaded/extracted .mat files
        output_root: usually DATA_RAW (data/raw/)
        copy: True=copy files, False=symlink (Windows may need admin for symlink)

    Returns:
        Same structure as verify_dataset()
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    file_map = _build_file_map()
    source_dir = source_dir.resolve()
    output_root = output_root.resolve()

    # ── Discover all .mat files in source ──
    all_mats: list[Path] = list(source_dir.rglob("*.mat"))
    if not all_mats:
        raise FileNotFoundError(f"No .mat files found in {source_dir}")

    # ── Classify each .mat by file number ──
    matched: dict[int, Path] = {}    # file_num → source_path
    unmatched: list[Path] = []        # can't parse file number
    duplicates: dict[int, list[Path]] = {}  # multiple sources for same file_num

    for mat_path in all_mats:
        try:
            fn = int(mat_path.stem)
        except ValueError:
            unmatched.append(mat_path)
            continue

        if fn in matched:
            if fn not in duplicates:
                duplicates[fn] = [matched[fn]]
            duplicates[fn].append(mat_path)
        else:
            matched[fn] = mat_path

    # ── Report discovery ──
    if verbose:
        print()
        print("=" * 68)
        print("  NeuroPLC — Local Data Import")
        print("=" * 68)
        print(f"  Source:         {source_dir}")
        print(f"  .mat found:     {len(all_mats)}")
        print(f"  Recognized:     {len(matched)}")
        print(f"  Unmatched:      {len(unmatched)}")
        if duplicates:
            print(f"  Duplicates:     {len(duplicates)} files have multiple copies")
        print("=" * 68)

    # ── Warn about unmatched ──
    if unmatched and verbose:
        print(f"\n  ⚡ {len(unmatched)} file(s) could not be parsed:")
        for p in unmatched[:10]:
            print(f"    {p.name}")
        if len(unmatched) > 10:
            print(f"    ... and {len(unmatched) - 10} more")

    # ── Handle duplicates: pick the largest ──
    if duplicates:
        if verbose:
            print(f"\n  ⚡ Duplicate file numbers detected — using largest copy:")
        for fn, paths in duplicates.items():
            best = max(paths, key=lambda p: p.stat().st_size)
            for p in paths:
                if p != best:
                    if verbose:
                        print(f"    {fn}.mat: skipping {p} ({p.stat().st_size:,}B)")
            matched[fn] = best

    # ── Copy/organize ──
    imported, failed_verify, skipped_existing = 0, 0, 0
    import_log: list[dict] = []

    for file_num in sorted(file_map.keys()):
        fault_type, diameter, load_hp, subdir = file_map[file_num]
        fname = f"{file_num}.mat"
        dest_dir = output_root / "12k_DE" / subdir
        dest = dest_dir / fname

        if file_num not in matched:
            continue  # not in source

        src = matched[file_num]

        # Check if destination already exists and is valid
        dest_valid, _, dest_size = verify_mat_file(dest)
        if dest_valid:
            skipped_existing += 1
            import_log.append({
                "file_num": file_num, "status": "skipped",
                "reason": "already exists and valid",
                "subdir": subdir, "size_bytes": dest_size,
            })
            continue

        # Verify source before copying
        src_valid, src_err, src_size = verify_mat_file(src)
        if not src_valid:
            failed_verify += 1
            import_log.append({
                "file_num": file_num, "status": "failed",
                "reason": f"source invalid: {src_err}",
                "subdir": subdir, "size_bytes": src_size,
            })
            if verbose:
                print(f"  ✗ {fname}: source invalid — {src_err}")
            continue

        # Copy
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            if copy:
                shutil.copy2(src, dest)
            else:
                if dest.exists() or dest.is_symlink():
                    dest.unlink()
                os.symlink(src, dest)
        except OSError as e:
            failed_verify += 1
            import_log.append({
                "file_num": file_num, "status": "failed",
                "reason": f"copy error: {e}",
                "subdir": subdir, "size_bytes": src_size,
            })
            if verbose:
                print(f"  ✗ {fname}: copy failed — {e}")
            continue

        # Verify after copy
        dest_valid, dest_err, dest_size = verify_mat_file(dest)
        if dest_valid:
            imported += 1
            import_log.append({
                "file_num": file_num, "status": "imported",
                "subdir": subdir, "size_bytes": dest_size,
                "source": str(src),
            })
            if verbose:
                print(f"  ✓ {fname:>10s} → {subdir}/{fname}  "
                      f"({dest_size / 1_048_576:.1f} MB)")
        else:
            failed_verify += 1
            import_log.append({
                "file_num": file_num, "status": "failed",
                "reason": f"post-copy verification failed: {dest_err}",
                "subdir": subdir, "size_bytes": dest_size,
            })
            if verbose:
                print(f"  ✗ {fname}: copy verification failed — {dest_err}")

    # ── Summary ──
    if verbose:
        print()
        print("=" * 68)
        print("  Import Summary")
        print("=" * 68)
        print(f"  Imported:          {imported}")
        print(f"  Already present:   {skipped_existing}")
        print(f"  Failed:            {failed_verify}")
        print(f"  Not in source:     {52 - imported - skipped_existing - failed_verify}")
        print("=" * 68)

    # ── Run full verification ──
    verify_result = verify_dataset(output_root, verbose=True)

    # Attach import log
    verify_result["import_log"] = import_log

    # Save manifest
    manifest = _generate_manifest(verify_result)
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    if verbose:
        print(f"\n  ✓ Manifest saved: {MANIFEST_PATH}")

    return verify_result


# ============================================================================
# Manifest generation
# ============================================================================

def _generate_manifest(verify_result: dict) -> dict:
    """Generate a JSON manifest from verification results."""
    entries = []
    for file_num, detail in sorted(verify_result["details"].items()):
        entries.append({
            "file": f"{file_num}.mat",
            "file_number": file_num,
            "fault_type": detail["fault_type"],
            "diameter_inch": detail["diameter"],
            "load_hp": detail["load_hp"],
            "label": LABEL_MAP[detail["fault_type"]],
            "subdir": detail["subdir"],
            "status": detail["status"],
            "size_bytes": detail["size_bytes"],
        })
    return {
        "dataset": "CWRU Bearing Data Center — 12k Drive End",
        "total_files": 52,
        "ok_files": verify_result["ok"],
        "generated": __import__("datetime").datetime.now().isoformat(),
        "files": entries,
    }


def export_manifest(output_path: Optional[Path] = None):
    """Generate and print/save a JSON manifest of all expected files."""
    verify_result = verify_dataset(DATA_RAW, verbose=False)
    manifest = _generate_manifest(verify_result)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Manifest saved: {output_path}")
    else:
        print(json.dumps(manifest, indent=2, ensure_ascii=False))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CWRU Dataset Downloader & Verifier — NeuroPLC (GFW-Compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Verify existing data
  python download_verify_cwru.py --verify

  # Import from local folder (123云盘 / 百度网盘 download)
  python download_verify_cwru.py --source local --input D:\\Downloads\\cwru_data

  # Import with move instead of copy (faster on same drive)
  python download_verify_cwru.py --source local --input D:\\Downloads\\cwru --move

  # Generate manifest of expected files
  python download_verify_cwru.py --manifest

  # Export manifest to file
  python download_verify_cwru.py --manifest --output data/manifest.json

Download sources (China-friendly):
  123云盘: https://www.123pan.com/s/xBwHjv-WIzk.html  提取码: EXLF
  百度网盘: https://pan.baidu.com/s/1k9xkejB-3YRqDunKA9AUsw  提取码: htgw
  OneDrive: https://1drv.ms/u/s!At5AiOeueyrEgyUY038Ln_SQ8SRo
  CSDN: https://download.csdn.net/download/weixin_45780075/89031833
        """
    )

    parser.add_argument(
        "--source", choices=["local", "auto"],
        default="local",
        help="Data source mode. 'local' imports from a local folder (default). "
             "'auto' tries online download (will likely fail in China)."
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="Source directory for --source local (e.g. D:\\Downloads\\cwru_data)"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Verify existing dataset integrity (implied after import)."
    )
    parser.add_argument(
        "--manifest", action="store_true",
        help="Print JSON manifest of expected/actual files."
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for --manifest (default: print to stdout)."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_RAW),
        help=f"Root output directory (default: {DATA_RAW})."
    )
    parser.add_argument(
        "--move", action="store_true",
        help="Move files instead of copying (faster, but removes from source)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan source and report what would happen, no actual import."
    )

    args = parser.parse_args()
    output_root = Path(args.output_dir)

    # ── Manifest mode ──
    if args.manifest:
        out_path = Path(args.output) if args.output else None
        export_manifest(out_path)
        return

    # ── Verify-only mode ──
    if args.verify:
        verify_dataset(output_root, verbose=True)
        return

    # ── Local import mode ──
    if args.source == "local":
        if not args.input:
            parser.error("--source local requires --input <directory>\n"
                         "  Use --verify to check existing data instead.")
        source_dir = Path(args.input)

        if args.dry_run:
            _dry_run_import(source_dir, output_root)
            return

        result = import_from_local(
            source_dir, output_root,
            copy=not args.move,
            verbose=True,
        )

        # Exit code based on completeness
        if result["ok"] == 52:
            print(f"\n  ✅ Full dataset ready ({result['total_size_mb']:.1f} MB)")
            print(f"  → Next: python preprocess.py")
            sys.exit(0)
        elif result["ok"] > 0:
            print(f"\n  ⚠ Partial: {result['ok']}/52 files. "
                  f"Preprocessing will work with available data.")
            sys.exit(0)
        else:
            print(f"\n  🔴 Import failed: no valid files imported.")
            sys.exit(1)

    # ── Auto-download mode (legacy, likely blocked by GFW) ──
    if args.source == "auto":
        print()
        print("  ⚠ --source auto is deprecated for users in mainland China.")
        print("  GitHub (raw.githubusercontent.com) is blocked by GFW.")
        print()
        print("  Use --source local instead:")
        print("    1. Download from 123云盘 or 百度网盘 (see --help)")
        print("    2. python download_verify_cwru.py --source local --input <folder>")
        print()
        choice = input("  Try online download anyway? (y/N): ").strip().lower()
        if choice != 'y':
            sys.exit(0)

        # Fall through to legacy download
        _legacy_download(output_root)


# ============================================================================
# Dry-run for local import
# ============================================================================

def _dry_run_import(source_dir: Path, output_root: Path):
    """Preview what import_from_local would do."""
    file_map = _build_file_map()
    all_mats = list(source_dir.rglob("*.mat"))
    matched: dict[int, Path] = {}
    unmatched = []

    for mat_path in all_mats:
        try:
            fn = int(mat_path.stem)
            if fn in file_map:
                matched[fn] = mat_path
            # ignore non-CWRU file numbers
        except ValueError:
            unmatched.append(mat_path)

    print()
    print("=" * 68)
    print("  Dry Run — Local Import Preview")
    print("=" * 68)
    print(f"  Source:        {source_dir}")
    print(f"  .mat found:    {len(all_mats)}")
    print(f"  CWRU-matched:  {len(matched)}")
    print(f"  Unmatched:     {len(unmatched)}")
    print("=" * 68)

    to_import, already_ok, missing = [], [], []

    for file_num in sorted(file_map.keys()):
        fault_type, diameter, load_hp, subdir = file_map[file_num]
        fname = f"{file_num}.mat"
        dest = output_root / "12k_DE" / subdir / fname

        if file_num in matched:
            dest_valid, _, _ = verify_mat_file(dest)
            if dest_valid:
                already_ok.append(file_num)
            else:
                to_import.append((file_num, matched[file_num], subdir))
        else:
            missing.append(file_num)

    if to_import:
        print(f"\n  To import ({len(to_import)}):")
        for fn, src, subdir in to_import:
            print(f"    {fn}.mat → {subdir}/  ({src.stat().st_size / 1_048_576:.1f} MB)")

    if already_ok:
        print(f"\n  Already OK ({len(already_ok)}):")
        # Just show first/last few
        for fn in already_ok[:5]:
            print(f"    {fn}.mat  ✓")
        if len(already_ok) > 10:
            print(f"    ... and {len(already_ok) - 5} more")

    if missing:
        print(f"\n  Missing from source ({len(missing)}):")
        for fn in missing[:10]:
            detail = file_map[fn]
            print(f"    {fn}.mat  ({detail[0]} load={detail[2]}hp)")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    print()
    print(f"  Summary: {len(to_import)} to import, "
          f"{len(already_ok)} already OK, "
          f"{len(missing)} missing from source")
    print(f"  Run without --dry-run to execute.")


# ============================================================================
# Legacy online download (deprecated — blocked by GFW)
# ============================================================================

def _legacy_download(output_root: Path):
    """Original download logic. Kept for completeness, likely blocked by GFW."""
    import urllib.request
    import urllib.error
    import ssl
    import time

    URLS = [
        "https://engineering.case.edu/bearingdatacenter/download-data-file/"
        "?filename={num}.mat",
        "https://raw.githubusercontent.com/abhivasani95/CWRU_Bearing_Dataset/"
        "refs/heads/master/12k%20Drive%20End%20Bearing%20Fault%20Data/{num}.mat",
        "https://raw.githubusercontent.com/Jerry-1996/CWRU-Bearing-dataset/"
        "master/12k_Drive_End/{num}.mat",
    ]

    file_map = _build_file_map()
    total = len(file_map)
    ok, failed = 0, 0
    total_bytes = 0
    t0 = time.time()

    ctx = ssl.create_default_context()
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "*/*",
    }

    print()
    print("=" * 68)
    print("  Legacy Online Download (⚠ likely blocked by GFW in China)")
    print("=" * 68)
    print(f"  Total files: {total}")
    print("=" * 68)

    for i, file_num in enumerate(sorted(file_map.keys())):
        fault_type, diameter, load_hp, subdir = file_map[file_num]
        fname = f"{file_num}.mat"
        dest_dir = output_root / "12k_DE" / subdir
        dest = dest_dir / fname

        # Skip if exists
        dest_valid, _, dest_size = verify_mat_file(dest)
        if dest_valid:
            total_bytes += dest_size
            print(f"  [{i+1:2d}/{total}] {fname:>10s}  ✓ already exists "
                  f"({dest_size / 1_048_576:.1f} MB)")
            continue

        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded = False

        for url_template in URLS:
            url = url_template.replace("{num}", str(file_num))
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                    data = resp.read()
                if len(data) >= MIN_FILE_SIZE:
                    tmp = dest.with_suffix(".tmp")
                    tmp.write_bytes(data)
                    tmp.replace(dest)
                    if verify_mat_file(dest)[0]:
                        sz = dest.stat().st_size
                        total_bytes += sz
                        ok += 1
                        print(f"  [{i+1:2d}/{total}] {fname:>10s}  ✓ "
                              f"({sz / 1_048_576:.1f} MB)")
                        downloaded = True
                        break
            except Exception:
                continue

        if not downloaded:
            failed += 1
            print(f"  [{i+1:2d}/{total}] {fname:>10s}  ✗ all sources failed")

    elapsed = time.time() - t0
    print()
    print("=" * 68)
    print(f"  Downloaded: {ok}  |  Failed: {failed}  |  "
          f"Total: {total_bytes / 1_048_576:.1f} MB  |  {elapsed:.0f}s")
    print("=" * 68)


if __name__ == "__main__":
    main()
