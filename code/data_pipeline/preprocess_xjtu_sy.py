#!/usr/bin/env python3
"""
XJTU-SY Bearing Dataset — Preprocessing for Cross-Dataset Validation
======================================================================
Loads the raw XJTU-SY CSV files, extracts the same 28-D features used in
CWRU preprocessing, and evaluates the CWRU-trained KAN model in zero-shot
cross-dataset transfer mode.

XJTU-SY data structure:
    3 operating conditions: 35Hz/12kN, 37.5Hz/11kN, 40Hz/10kN
    5 bearings per condition (BearingX_1 to BearingX_5)
    Each bearing: N CSV files (1.csv to N.csv), 1.28s snapshot per minute
    Each CSV: Horizontal_vibration_signals, Vertical_vibration_signals
    Sampling rate: 25.6 kHz → 32,768 points per file

Label mapping (from Wang et al. 2020, IEEE Trans. Reliability):
    Condition 1 (35Hz/12kN):
        Bearing1_1: Outer race         Bearing1_2: Outer race
        Bearing1_3: Outer race         Bearing1_4: Cage
        Bearing1_5: Inner race + Outer race (compound)
    Condition 2 (37.5Hz/11kN):
        Bearing2_1: Inner race         Bearing2_2: Outer race
        Bearing2_3: Cage               Bearing2_4: Outer race
        Bearing2_5: Outer race
    Condition 3 (40Hz/10kN):
        Bearing3_1: Outer race         Bearing3_2: Outer race (cage+inner+ball)
        Bearing3_3: Inner race         Bearing3_4: Inner race
        Bearing3_5: Outer race

For cross-dataset transfer: we take the LAST 300 windows per bearing as
"degraded" samples (fault has developed) and FIRST 300 windows from
Bearing1_1 as "healthy" baseline (early in life, before significant wear).

Usage:
    python preprocess_xjtu_sy.py
    python preprocess_xjtu_sy.py --windows-per-bearing 200 --overlap 0.5
"""

import os, sys, json, argparse
from pathlib import Path
import numpy as np
from scipy import signal
from scipy.fft import fft
from scipy.stats import skew, kurtosis

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
XJTU_DIR = PROJECT_ROOT / "xjtu-sy-data" / "extracted" / "XJTU-SY_Bearing_Datasets"
XJTU_PROCESSED = PROJECT_ROOT / "data" / "xjtu_sy"
XJTU_PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Fault mode labels (Wang et al. 2020, verified from dataset introduction) ──
# Class mapping: 0=Normal, 1=InnerRace, 2=OuterRace, 3=Ball/Cage
FAULT_LABELS = {
    "35Hz12kN": {
        "Bearing1_1": 2,  # Outer race
        "Bearing1_2": 2,  # Outer race
        "Bearing1_3": 2,  # Outer race
        "Bearing1_4": 3,  # Cage (mapped to Ball class)
        "Bearing1_5": 2,  # Inner+Outer → Outer dominant
    },
    "37.5Hz11kN": {
        "Bearing2_1": 1,  # Inner race
        "Bearing2_2": 2,  # Outer race
        "Bearing2_3": 3,  # Cage
        "Bearing2_4": 2,  # Outer race
        "Bearing2_5": 2,  # Outer race
    },
    "40Hz10kN": {
        "Bearing3_1": 2,  # Outer race
        "Bearing3_2": 2,  # Outer race (compound, outer dominant)
        "Bearing3_3": 1,  # Inner race
        "Bearing3_4": 1,  # Inner race
        "Bearing3_5": 2,  # Outer race
    },
}

# ── CWRU feature extraction (reuse the same 28-D feature pipeline) ──

def extract_28d_features(signal_1d: np.ndarray, fs: int = 25600) -> np.ndarray:
    """Extract 28-D features from 1-D vibration signal (matches CWRU preprocessing).

    Features:
        10 time-domain: RMS, peak, peak-to-peak, crest factor, skewness, kurtosis,
                        shape factor, impulse factor, MAV, variance
        10 freq-domain: spectral centroid, spread, skew, kurtosis, entropy,
                        5 sub-band energy ratios (6kHz bands)
        8 dispersion entropy: 4 RCMDE + 4 RCHFDE (simplified: use
           multi-scale Shannon entropy as proxy for PLC compatibility)
    """
    N = len(signal_1d)
    if N < 64:
        return np.zeros(28, dtype=np.float32)

    # ── Time-domain features (10) ──
    x = signal_1d.astype(np.float64)
    rms = np.sqrt(np.mean(x ** 2))
    peak_val = np.max(np.abs(x))
    pp_val = np.max(x) - np.min(x)
    crest = peak_val / (rms + 1e-10)
    sk = float(skew(x))
    ku = float(kurtosis(x))
    mav = np.mean(np.abs(x))
    var_val = np.var(x)
    shape_factor = rms / (mav + 1e-10)
    impulse_factor = peak_val / (mav + 1e-10)

    time_features = np.array([rms, peak_val, pp_val, crest, sk, ku,
                               shape_factor, impulse_factor, mav, var_val])

    # ── Frequency-domain features (10) ──
    X = np.abs(fft(x))[:N//2]
    freqs = np.linspace(0, fs/2, N//2)

    X_norm = X / (X.sum() + 1e-10)
    centroid = np.sum(freqs * X_norm)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * X_norm))

    # Spectral skewness and kurtosis
    freq_sk = float(skew(X))
    freq_ku = float(kurtosis(X))

    # Spectral entropy
    X_prob = X / (X.sum() + 1e-10) + 1e-12
    spec_entropy = -np.sum(X_prob * np.log(X_prob))

    # 5 sub-band energy ratios (0-6kHz, 1.2kHz bands)
    band_edges = np.linspace(0, 6000, 6)
    band_energies = []
    for j in range(5):
        mask = (freqs >= band_edges[j]) & (freqs < band_edges[j+1])
        band_energy = X[mask].sum() / (X.sum() + 1e-10)
        band_energies.append(band_energy)

    freq_features = np.array([centroid, spread, freq_sk, freq_ku, spec_entropy]
                              + band_energies)

    # ── Multi-scale entropy features (8) ──
    # Simplified: coarse-graining + Shannon entropy at multiple scales
    # This approximates RCMDE/RCHFDE for PLC-compatible feature extraction
    ent_features = []
    for scale in [1, 2, 3, 4]:
        # Coarse-graining
        if N >= scale * 100:
            coarse = np.array([x[i::scale].mean() for i in range(scale)])
            coarse = coarse - coarse.mean()
            coarse /= (coarse.std() + 1e-10)
            # Shannon entropy of distribution
            hist, _ = np.histogram(coarse, bins=20)
            hist = hist / (hist.sum() + 1e-10) + 1e-12
            ent = -np.sum(hist * np.log(hist)) / np.log(20)  # normalized
            ent_features.append(ent)
        else:
            ent_features.append(0.0)

        # Fluctuation-based variant
        if N >= scale * 100:
            diff = np.diff(x)[:N//scale]
            diff = diff - diff.mean()
            diff /= (diff.std() + 1e-10)
            hist, _ = np.histogram(diff, bins=20)
            hist = hist / (hist.sum() + 1e-10) + 1e-12
            ent = -np.sum(hist * np.log(hist)) / np.log(20)
            ent_features.append(ent)
        else:
            ent_features.append(0.0)

    features = np.concatenate([time_features, freq_features,
                                np.array(ent_features)])
    return features.astype(np.float32)


def process_bearing(csv_dir: Path, fs: int = 25600,
                    window_size: int = 1024,
                    stride: int = 512,
                    n_windows: int = 300,
                    take_last: bool = True) -> np.ndarray:
    """Extract 28-D features from the last (or first) N windows of a bearing.

    Each CSV contains 32,768 points (1.28s at 25.6kHz).
    We segment into 1024-point sliding windows with 50% overlap.
    """
    csv_files = sorted(csv_dir.glob("*.csv"),
                        key=lambda p: int(p.stem))
    if not csv_files:
        return np.zeros((0, 28), dtype=np.float32)

    if take_last:
        csv_files = csv_files[-n_windows // 10:]  # take last ~30 CSVs

    features_list = []
    for csv_path in csv_files:
        try:
            data = np.loadtxt(csv_path, delimiter=",", skiprows=1, max_rows=32768)
        except (ValueError, OSError):
            continue

        if data.ndim == 1:
            signal_1d = data  # single column
        else:
            signal_1d = data[:, 0]  # horizontal vibration

        # Sliding window segmentation
        N = len(signal_1d)
        n_win = (N - window_size) // stride + 1
        for w in range(min(n_win, 40)):  # max 40 windows per CSV
            start = w * stride
            segment = signal_1d[start:start + window_size]
            feat = extract_28d_features(segment, fs=fs)
            features_list.append(feat)

    if not features_list:
        return np.zeros((0, 28), dtype=np.float32)

    features = np.stack(features_list)
    if take_last:
        features = features[-n_windows:]  # last N windows
    else:
        features = features[:n_windows]   # first N windows

    return features


def main():
    parser = argparse.ArgumentParser(description="XJTU-SY Preprocessing")
    parser.add_argument("--windows-per-bearing", type=int, default=300,
                        help="Windows to extract per bearing (default: 300)")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="Sliding window overlap fraction (default: 0.5)")
    args = parser.parse_args()

    window_size = 1024
    stride = int(window_size * (1 - args.overlap))
    n_windows = args.windows_per_bearing

    print(f"XJTU-SY Preprocessing: {n_windows} windows/bearing, "
          f"W={window_size}, stride={stride}")

    all_features = []
    all_labels = []
    all_bearings = []

    # ── Process degraded bearings ──
    for condition_dir in sorted(XJTU_DIR.glob("*kN")):
        cond_name = condition_dir.name
        if cond_name not in FAULT_LABELS:
            continue
        print(f"\n  Condition: {cond_name}")

        for bearing_dir in sorted(condition_dir.glob("Bearing*")):
            bearing_name = bearing_dir.name
            label = FAULT_LABELS[cond_name].get(bearing_name, -1)
            if label < 0:
                continue

            print(f"    {bearing_name}: extracting {n_windows} windows (label={label})...")
            feats = process_bearing(bearing_dir,
                                     window_size=window_size,
                                     stride=stride,
                                     n_windows=n_windows,
                                     take_last=True)

            if len(feats) > 0:
                all_features.append(feats)
                all_labels.extend([label] * len(feats))
                all_bearings.extend([f"{cond_name}/{bearing_name}"] * len(feats))
                print(f"      {len(feats)} windows extracted")
            else:
                print(f"      SKIP (no valid data)")

    # ── Process healthy baseline (Bearing1_1, first windows) ──
    print(f"\n  Healthy baseline: Bearing1_1 (first windows)")
    healthy_dir = XJTU_DIR / "35Hz12kN" / "Bearing1_1"
    healthy_feats = process_bearing(healthy_dir,
                                     window_size=window_size,
                                     stride=stride,
                                     n_windows=n_windows,
                                     take_last=False)
    if len(healthy_feats) > 0:
        all_features.append(healthy_feats)
        all_labels.extend([0] * len(healthy_feats))  # class 0 = Normal
        all_bearings.extend(["healthy"] * len(healthy_feats))
        print(f"      {len(healthy_feats)} healthy windows extracted")

    # ── Stack and apply CWRU scaler ──
    X_xjtu = np.concatenate(all_features, axis=0)
    y_xjtu = np.array(all_labels, dtype=np.int32)

    # Apply CWRU scaler
    cwru_scaler = np.load(PROJECT_ROOT / "data" / "processed" / "features_scaler.npz")
    mean = cwru_scaler["mean"]
    scale = cwru_scaler["scale"]
    X_xjtu_scaled = (X_xjtu - mean) * scale

    print(f"\n  Total XJTU-SY samples: {len(X_xjtu)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y_xjtu, return_counts=True)))}")
    print(f"  Shape: {X_xjtu_scaled.shape}")

    # ── Save ──
    np.save(XJTU_PROCESSED / "features_X.npy", X_xjtu_scaled)
    np.save(XJTU_PROCESSED / "features_y.npy", y_xjtu)
    np.save(XJTU_PROCESSED / "bearings.npy",
            np.array(all_bearings, dtype=object))
    np.savez(XJTU_PROCESSED / "features_meta.npz",
             bearings=np.array(all_bearings),
             labels=y_xjtu)

    # ── Stats ──
    stats = {
        "total_samples": int(len(X_xjtu)),
        "n_features": 28,
        "n_classes": 4,
        "class_names": ["Normal", "InnerRace", "OuterRace", "Ball/Cage"],
        "class_counts": {str(k): int(v) for k, v in
                         zip(*np.unique(y_xjtu, return_counts=True))},
        "conditions": list(FAULT_LABELS.keys()),
        "windows_per_bearing": n_windows,
        "window_size": window_size,
        "stride": stride,
        "source": "XJTU-SY run-to-failure dataset",
        "label_note": ("Fault modes mapped to CWRU 4-class schema: "
                       "Normal(early-life B1_1) / Inner race / Outer race / "
                       "Cage+Ball(mapped to class 3)"),
    }
    with open(XJTU_PROCESSED / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n  Saved to: {XJTU_PROCESSED}")
    print("  Done.")


if __name__ == "__main__":
    main()
