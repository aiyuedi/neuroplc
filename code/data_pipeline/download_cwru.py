#!/usr/bin/env python3
"""
CWRU Bearing Data Center — Automatic Downloader
================================================
Downloads 12k Drive End bearing fault data for the NeuroPLC project.

Dataset: Case Western Reserve University Bearing Data Center
         https://engineering.case.edu/bearingdatacenter

Output structure:
    data/raw/12k_DE/
    ├── Normal/        # 97–100  (baseline, all loads)
    ├── IR007/         # 105–108 (Inner Race 0.007")
    ├── IR014/         # 169–172
    ├── IR021/         # 209–212
    ├── IR028/         # 3001–3004
    ├── B007/          # 118–121 (Ball 0.007")
    ├── B014/          # 185–188
    ├── B021/          # 222–225
    ├── B028/          # 3005–3008
    ├── OR007@6/       # 130–133 (Outer Race 0.007" @6:00)
    ├── OR014@6/       # 197–200
    ├── OR021@6/       # 234–237
    └── OR028@6/       # 3009–3012

Total: 52 files, ~120 MB.

Usage:
    python download_cwru.py              # Download 12k DE (default)
    python download_cwru.py --dry-run    # List files, don't download
    python download_cwru.py --force      # Re-download even if exists
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error
import ssl
from pathlib import Path
from typing import Optional

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"

# ============================================================================
# CWRU 12k Drive End — Complete File Manifest
# ============================================================================
# Source: https://engineering.case.edu/bearingdatacenter
#
# File variable naming inside .mat:
#   X{num}_DE_time  — Drive End accelerometer (12 kHz)
#   X{num}_FE_time  — Fan End accelerometer (12 kHz)
#   X{num}_BA_time  — Base accelerometer
#   X{num}RPM        — Shaft speed (RPM)

# (fault_type, diameter_inch, load_hp) → file_number
CWRU_12K_DE: dict = {
    # ── Normal baseline ──
    ("Normal", None,  0): 97,
    ("Normal", None,  1): 98,
    ("Normal", None,  2): 99,
    ("Normal", None,  3): 100,

    # ── Inner Race (内圈) ──
    ("IR",     0.007, 0): 105,   ("IR", 0.007, 1): 106,
    ("IR",     0.007, 2): 107,   ("IR", 0.007, 3): 108,
    ("IR",     0.014, 0): 169,   ("IR", 0.014, 1): 170,
    ("IR",     0.014, 2): 171,   ("IR", 0.014, 3): 172,
    ("IR",     0.021, 0): 209,   ("IR", 0.021, 1): 210,
    ("IR",     0.021, 2): 211,   ("IR", 0.021, 3): 212,
    ("IR",     0.028, 0): 3001,  ("IR", 0.028, 1): 3002,
    ("IR",     0.028, 2): 3003,  ("IR", 0.028, 3): 3004,

    # ── Ball (滚珠) ──
    ("Ball",   0.007, 0): 118,   ("Ball", 0.007, 1): 119,
    ("Ball",   0.007, 2): 120,   ("Ball", 0.007, 3): 121,
    ("Ball",   0.014, 0): 185,   ("Ball", 0.014, 1): 186,
    ("Ball",   0.014, 2): 187,   ("Ball", 0.014, 3): 188,
    ("Ball",   0.021, 0): 222,   ("Ball", 0.021, 1): 223,
    ("Ball",   0.021, 2): 224,   ("Ball", 0.021, 3): 225,
    ("Ball",   0.028, 0): 3005,  ("Ball", 0.028, 1): 3006,
    ("Ball",   0.028, 2): 3007,  ("Ball", 0.028, 3): 3008,

    # ── Outer Race @6:00 (外圈) ──
    ("OR",     0.007, 0): 130,   ("OR", 0.007, 1): 131,
    ("OR",     0.007, 2): 132,   ("OR", 0.007, 3): 133,
    ("OR",     0.014, 0): 197,   ("OR", 0.014, 1): 198,
    ("OR",     0.014, 2): 199,   ("OR", 0.014, 3): 200,
    ("OR",     0.021, 0): 234,   ("OR", 0.021, 1): 235,
    ("OR",     0.021, 2): 236,   ("OR", 0.021, 3): 237,
    ("OR",     0.028, 0): 3009,  ("OR", 0.028, 1): 3010,
    ("OR",     0.028, 2): 3011,  ("OR", 0.028, 3): 3012,
}

# Label encoding
LABEL_MAP = {"Normal": 0, "IR": 1, "Ball": 2, "OR": 3}

# Minimum valid file size (bytes) — 12k DE ≈ 2.6–3.3 MB
MIN_FILE_SIZE = 2_000_000

# ============================================================================
# Download URL Strategies
# ============================================================================

def _build_all_urls(file_num: int) -> list[tuple[str, str]]:
    """
    Return [(url, label), ...] for all download strategies.
    Ordered by preference.
    """
    strategies: list[tuple[str, str]] = []

    # Strategy 1: Official CWRU site (PHP download handler)
    strategies.append((
        f"https://engineering.case.edu/bearingdatacenter/download-data-file/"
        f"?filename={file_num}.mat",
        "CWRU official (PHP)"
    ))

    # Strategy 2: GitHub mirror — widely used, well-maintained
    strategies.append((
        f"https://raw.githubusercontent.com/abhivasani95/CWRU_Bearing_Dataset/"
        f"refs/heads/master/12k%20Drive%20End%20Bearing%20Fault%20Data/{file_num}.mat",
        "GitHub mirror (abhivasani95)"
    ))

    # Strategy 3: Alternative GitHub mirror
    strategies.append((
        f"https://raw.githubusercontent.com/Jerry-1996/CWRU-Bearing-dataset/"
        f"master/12k_Drive_End/{file_num}.mat",
        "GitHub mirror (Jerry-1996)"
    ))

    # Strategy 4: Kaggle-style mirror
    strategies.append((
        f"https://raw.githubusercontent.com/brunoklein99/deep-learning-para-"
        f"manutencao-preditiva/master/dados/12k_Drive_End_Bearing_Fault_Data/"
        f"{file_num}.mat",
        "GitHub mirror (brunoklein99)"
    ))

    return strategies


# ============================================================================
# Download Engine
# ============================================================================

def _make_request(url: str, timeout: int = 45) -> bytes:
    """Make an HTTP GET request with browser-like headers."""
    # Create an unverified SSL context for sites with cert issues
    ctx = ssl.create_default_context()

    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/131.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return resp.read()


def download_one(
    file_num: int,
    dest: Path,
    *,
    force: bool = False,
) -> tuple[bool, str]:
    """
    Download a single .mat file to dest.
    Tries all URL strategies in order. Skips if dest exists and is valid.

    Returns:
        (success, source_label)
        success=True, label="already exists" if file is already present
    """
    # ── Skip if already downloaded and valid ──
    if not force and dest.exists():
        size = dest.stat().st_size
        if size >= MIN_FILE_SIZE:
            return True, "already exists"
        else:
            print(f"      ⚠ Corrupt file ({size} bytes), re-downloading...")
            dest.unlink()

    # ── Ensure parent dir exists ──
    dest.parent.mkdir(parents=True, exist_ok=True)

    # ── Try each strategy ──
    urls = _build_all_urls(file_num)
    last_error = None

    for url, label in urls:
        try:
            data = _make_request(url)
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, TimeoutError, OSError) as e:
            last_error = str(e)
            continue

        # Verify we got actual .mat data (not an HTML error page)
        if len(data) < MIN_FILE_SIZE:
            last_error = f"Response too small ({len(data)} bytes, likely HTML error page)"
            continue

        # Write atomically
        tmp = dest.with_suffix(".tmp")
        tmp.write_bytes(data)
        tmp.replace(dest)

        actual_size = dest.stat().st_size
        if actual_size >= MIN_FILE_SIZE:
            return True, label
        else:
            dest.unlink(missing_ok=True)
            last_error = f"File too small after write ({actual_size} bytes)"

    # All strategies failed
    err_msg = f"All {len(urls)} sources failed. Last error: {last_error}"
    return False, err_msg


# ============================================================================
# Helpers
# ============================================================================

def _subdir_name(fault_type: str, diameter: Optional[float]) -> str:
    """Map (fault_type, diameter) → subdirectory name."""
    if fault_type == "Normal":
        return "Normal"
    diam_str = f"{diameter:.3f}".replace(".", "")  # 0.007 → 0007
    return f"{fault_type}{diam_str}"


def _format_size(n_bytes: int) -> str:
    """Human-readable file size."""
    if n_bytes < 1024:
        return f"{n_bytes} B"
    elif n_bytes < 1024 * 1024:
        return f"{n_bytes / 1024:.0f} KB"
    else:
        return f"{n_bytes / 1024 / 1024:.1f} MB"


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CWRU Bearing Dataset Downloader — NeuroPLC Project"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List all files that would be downloaded, then exit."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files already exist."
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DATA_RAW),
        help=f"Root output directory (default: {DATA_RAW})"
    )
    parser.add_argument(
        "--skip", type=str, nargs="*", default=[],
        choices=["Normal", "IR", "Ball", "OR"],
        help="Skip specific fault types."
    )
    args = parser.parse_args()

    output_root = Path(args.output_dir)

    # ── Build download list ──
    downloads: list[dict] = []
    for (fault_type, diameter, load_hp), file_num in CWRU_12K_DE.items():
        if fault_type in args.skip:
            continue
        subdir = _subdir_name(fault_type, diameter)
        label = LABEL_MAP[fault_type]
        downloads.append({
            "file_num": file_num,
            "fault_type": fault_type,
            "diameter": diameter,
            "load_hp": load_hp,
            "label": label,
            "subdir": subdir,
            "fname": f"{file_num}.mat",
            "dest": output_root / "12k_DE" / subdir / f"{file_num}.mat",
        })

    total = len(downloads)

    # ── Header ──
    print()
    print("=" * 68)
    print("  NeuroPLC — CWRU Bearing Dataset Downloader")
    print("=" * 68)
    print(f"  Category:     12k Drive End")
    print(f"  Fault types:  Normal, IR, Ball, OR @6:00")
    print(f"  Fault sizes:  0.007 / 0.014 / 0.021 / 0.028 inch")
    print(f"  Loads:        0 / 1 / 2 / 3 hp")
    print(f"  Total files:  {total}")
    print(f"  Output:       {output_root / '12k_DE'}")
    print("=" * 68)

    if args.dry_run:
        print("\n[Dry Run] Would download:\n")
        for d in downloads:
            diam_str = f' {d["diameter"]:.3f}"' if d["diameter"] else ""
            print(f"  {d['subdir']:>10s}/{d['fname']:<10s}  "
                  f"label={d['label']}  load={d['load_hp']}hp  "
                  f"({d['fault_type']}{diam_str})")
        print(f"\n  Total: {total} files")
        return

    # ── Download loop ──
    ok = 0
    skipped = 0
    failed: list[str] = []
    total_bytes = 0
    t0 = time.time()

    for i, d in enumerate(downloads):
        prefix = f"[{i+1:2d}/{total}]"
        rel_path = f"12k_DE/{d['subdir']}/{d['fname']}"

        success, source = download_one(d["file_num"], d["dest"], force=args.force)

        if success:
            size = d["dest"].stat().st_size if d["dest"].exists() else 0
            total_bytes += size
            if source == "already exists":
                skipped += 1
                print(f"  {prefix} {rel_path:<45s} ✓ skipped ({_format_size(size)})")
            else:
                ok += 1
                print(f"  {prefix} {rel_path:<45s} ✓ {_format_size(size)}  ← {source}")
        else:
            failed.append(f"{rel_path}: {source}")
            print(f"  {prefix} {rel_path:<45s} ✗ {source[:60]}")

    elapsed = time.time() - t0

    # ── Summary ──
    print()
    print("=" * 68)
    print("  Download Report")
    print("=" * 68)
    print(f"  Downloaded:  {ok} new")
    print(f"  Skipped:     {skipped} (already present)")
    print(f"  Failed:      {len(failed)}")
    print(f"  Data on disk:{_format_size(total_bytes)}")
    print(f"  Elapsed:     {elapsed:.0f}s")
    print("=" * 68)

    if failed:
        print(f"\n  ⚠ {len(failed)} file(s) failed:")
        for f in failed[:10]:
            print(f"    - {f}")
        if len(failed) > 10:
            print(f"    ... and {len(failed) - 10} more")
        print("\n  → Re-run: python download_cwru.py   (skips already-downloaded)")
        print("  → Or download manually from:")
        print("    https://engineering.case.edu/bearingdatacenter")
        print("    Place .mat files in data/raw/12k_DE/<FaultType>/")
        print()
        print("  Manual download guide:")
        print("    1. Go to the URL above")
        print("    2. Click '12k Drive End Bearing Fault Data'")
        print("    3. Download each .mat file listed above")
        print("    4. Put them in the correct subdirectory")
        sys.exit(1)
    else:
        print(f"\n  ✓ All {total} files ready.")
        print(f"  → Next: python preprocess.py")
        print()


if __name__ == "__main__":
    main()
