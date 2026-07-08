#!/usr/bin/env python3
"""
CWRU Bearing Data — Preprocessing Pipeline
============================================
Sliding window segmentation → statistical feature extraction → dataset splits.

Two modes:
    1. Waveform mode (for Teacher CNN):
       - Sliding window on raw vibration signal → [N, 1024] arrays
       - No feature engineering, let CNN learn representations

    2. Feature mode (for Student KAN/MLP):
       - 28-D features per window (10 time + 10 freq + 8 dispersion entropy)
       - Compact, interpretable, suitable for PLC inference

Usage:
    python preprocess.py                      # Both modes, all loads
    python preprocess.py --mode features       # Features only
    python preprocess.py --mode waveform       # Waveform only
    python preprocess.py --cross-load           # Cross-load split (1hp train)
    python preprocess.py --dry-run              # Print summary without writing

Outputs:
    data/processed/
    ├── waveform_X.npy          # Shape (N, 1024)
    ├── waveform_y.npy          # Shape (N,)    — labels
    ├── waveform_load.npy       # Shape (N,)    — load marker
    ├── features_X.npy          # Shape (N, 28) — [v2] 20统计 + 8离散熵
    ├── features_y.npy          # Shape (N,)
    ├── features_load.npy       # Shape (N,)
    └── stats.json              # Per-fault-type per-load statistics

    data/splits/
    ├── standard/
    │   ├── train_idx.npy, val_idx.npy, test_idx.npy
    │   └── train_X.npy ... (if --export-splits)
    └── cross_load/
        ├── source_1hp/
        └── target_0hp/, target_2hp/, target_3hp/
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.io
from scipy.fft import rfft, rfftfreq
from scipy.stats import norm
from scipy.special import erf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Paths
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_SPLITS = PROJECT_ROOT / "data" / "splits"

# ============================================================================
# Constants
# ============================================================================
SAMPLE_RATE = 12000               # Hz
WINDOW_SIZE = 1024                # points ≈ 0.085 s
STRIDE = 512                      # 50% overlap

# Label encoding
LABEL_MAP = {
    "Normal": 0,
    "IR": 1,      # Inner Race (内圈)
    "Ball": 2,    # Ball (滚珠)
    "OR": 3,      # Outer Race (外圈) — all @6:00 position
}

LABEL_NAMES = {0: "Normal", 1: "InnerRace", 2: "Ball", 3: "OuterRace"}

# ============================================================================
# File Discovery
# ============================================================================

def _infer_fault_type(subdir_name: str) -> Optional[str]:
    """Map subdirectory name → fault type code."""
    if subdir_name == "Normal":
        return "Normal"
    for code in ["IR", "Ball", "OR"]:
        if subdir_name.startswith(code):
            return code
    return None


def discover_files(raw_dir: Path) -> list[dict]:
    """
    Scan data/raw/12k_DE/ and return metadata for each .mat file.

    Returns list of dicts with keys:
        path, file_number, fault_type, diameter, load_hp, label_name, subdir
    """
    base = raw_dir / "12k_DE"
    if not base.exists():
        raise FileNotFoundError(
            f"CWRU data not found at {base}\n"
            f"Run: python download_cwru.py  first"
        )

    files = []
    for subdir in sorted(base.iterdir()):
        if not subdir.is_dir():
            continue
        fault_type = _infer_fault_type(subdir.name)
        if fault_type is None:
            print(f"  ⚠ Skipping unknown directory: {subdir.name}")
            continue

        for mat_file in sorted(subdir.glob("*.mat")):
            # Parse file number from filename: "105.mat" → 105
            try:
                file_num = int(mat_file.stem)
            except ValueError:
                continue

            # Infer diameter from subdir name
            if fault_type == "Normal":
                diameter = None
            else:
                # IR007 → 0.007
                diam_str = subdir.name[len(fault_type):]
                diameter = int(diam_str) / 1000.0

            # Infer load from CWRU convention
            load_hp = _infer_load(file_num)

            files.append({
                "path": mat_file,
                "file_number": file_num,
                "fault_type": fault_type,
                "diameter": diameter,
                "load_hp": load_hp,
                "label": LABEL_MAP[fault_type],
                "subdir": subdir.name,
            })

    return sorted(files, key=lambda f: (f["label"], f["load_hp"], f["file_number"]))


def _infer_load(file_num: int) -> int:
    """
    Infer load (hp) from file number using the CWRU convention.

    Normal files:
        97→0, 98→1, 99→2, 100→3

    Fault files follow a pattern within each diameter group:
        0→0hp, 1→1hp, 2→2hp, 3→3hp

    The 4 consecutive files in a diameter group correspond to loads 0→3.
    """
    # Normal baseline: explicit mapping
    normal_map = {97: 0, 98: 1, 99: 2, 100: 3}
    if file_num in normal_map:
        return normal_map[file_num]

    # Fault files: grouped by 4 consecutive numbers
    # IR007: 105→0, 106→1, 107→2, 108→3
    # IR014: 169→0, 170→1, 171→2, 172→3
    # IR021: 209→0, 210→1, 211→2, 212→3
    # IR028: 3001→0, 3002→1, 3003→2, 3004→3
    # ... same pattern for Ball and OR
    grouped_starts = {
        (105, 108): 0,   # IR007
        (169, 172): 0,   # IR014
        (209, 212): 0,   # IR021
        (3001, 3004): 0,  # IR028
        (118, 121): 0,   # B007
        (185, 188): 0,   # B014
        (222, 225): 0,   # B021
        (3005, 3008): 0,  # B028
        (130, 133): 0,   # OR007
        (197, 200): 0,   # OR014
        (234, 237): 0,   # OR021
        (3009, 3012): 0,  # OR028
    }
    for (start, end), base_load in grouped_starts.items():
        if start <= file_num <= end:
            return base_load + (file_num - start)

    # Fallback
    return -1


# ============================================================================
# Signal Loading
# ============================================================================

def load_signal(mat_path: Path, channel: str = "DE") -> np.ndarray:
    """
    Load vibration signal from .mat file.

    The .mat variable name format:
        X{file_number}_{channel}_time

    Example: X105_DE_time → 12k Drive End, file #105
    """
    mat = scipy.io.loadmat(str(mat_path))

    # Extract file number from path
    file_num = int(mat_path.stem)

    # CWRU variable naming: X<num>_<channel>_time
    var_name = f"X{file_num}_{channel}_time"

    if var_name in mat:
        return mat[var_name].flatten().astype(np.float64)

    # Sometimes stored without the file number prefix
    alt_name = f"{channel}_time"
    if alt_name in mat:
        return mat[alt_name].flatten().astype(np.float64)

    # Last resort: find any key matching the pattern
    for key in mat:
        if channel in key and "time" in key.lower() and not key.startswith("_"):
            return mat[key].flatten().astype(np.float64)

    raise KeyError(
        f"Cannot find signal for channel '{channel}' in {mat_path.name}\n"
        f"  Available keys: {[k for k in mat if not k.startswith('_')]}"
    )


# ============================================================================
# Sliding Window
# ============================================================================

def sliding_window(signal_1d: np.ndarray,
                   window_size: int = WINDOW_SIZE,
                   stride: int = STRIDE) -> np.ndarray:
    """
    Extract overlapping windows from 1-D time series.

    Args:
        signal_1d:  1-D numpy array (N_points,)
        window_size: samples per window
        stride:      samples between consecutive window starts

    Returns:
        windows: shape (num_windows, window_size)
    """
    n = len(signal_1d)
    if n < window_size:
        raise ValueError(
            f"Signal too short: {n} pts < window_size {window_size}"
        )
    num_windows = (n - window_size) // stride + 1
    # Use as_strided for zero-copy: much faster than a Python loop
    shape = (num_windows, window_size)
    strides_bytes = (signal_1d.strides[0] * stride, signal_1d.strides[0])
    return np.lib.stride_tricks.as_strided(signal_1d, shape=shape,
                                           strides=strides_bytes)


# ============================================================================
# Dispersion Entropy Features (8-D) — [v2 新增]
# ============================================================================
# 参考文献:
#   Rostaghi & Azami, "Dispersion Entropy: A Measure for Time-Series Analysis",
#     IEEE Signal Processing Letters, 2016.
#   Azami et al., "Refined Composite Multiscale Dispersion Entropy and its
#     Application to Biomedical Signals", IEEE Trans. Biomedical Eng., 2017.
#   Chen et al., "HRCGMFDE: Hierarchical Refined Composite Generalized
#     Multiscale Fluctuation Dispersion Entropy", Shock & Vibration, 2024.
#   Ding et al., "RCMREDE 2D", EAAI, 2025.
#
# 配置: m=4 (embedding), c=6 (classes), tau=1 (delay), scales=[1,2,3,4]
# 输出: RCMDE(4维) + RCHFDE(4维) = 8维


def _ncdf_mapping(x: np.ndarray, c: int = 6) -> np.ndarray:
    """
    Map a 1-D signal to integer classes 1..c via Normal Cumulative Distribution.

    x → y = round(c * Φ((x-μ)/σ) + 0.5), clipped to [1, c]

    Args:
        x: 1-D signal array
        c: number of classes (default 6)

    Returns:
        Integer array of same shape, values in [1, c]
    """
    mu, sigma = np.mean(x), np.std(x)
    if sigma < 1e-12:
        return np.ones_like(x, dtype=np.int32)
    z = (x - mu) / sigma
    # Φ(z) via erf: Φ(z) = 0.5 * (1 + erf(z / sqrt(2)))
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    y = np.round(c * phi + 0.5)
    return np.clip(y, 1, c).astype(np.int32)


def _dispersion_entropy(x: np.ndarray, m: int = 4, c: int = 6,
                        tau: int = 1) -> float:
    """
    Compute Dispersion Entropy for a 1-D signal.

    DE = -Σ p(π_i) · log(p(π_i))

    where p(π_i) is the probability of dispersion pattern π_i.

    Args:
        x:   1-D signal
        m:   embedding dimension
        c:   number of classes
        tau: time delay

    Returns:
        DE value (normalized by log(c^m))
    """
    n = len(x)
    if n < m * tau:
        return 0.0

    # Map to class symbols
    y = _ncdf_mapping(x, c)

    # Build embedding vectors via stride tricks: shape (N_emb, m)
    n_emb = n - (m - 1) * tau
    if n_emb < 1:
        return 0.0
    emb_shape = (n_emb, m)
    emb_strides = (y.strides[0] * tau, y.strides[0])
    embeddings = np.lib.stride_tricks.as_strided(y, shape=emb_shape,
                                                  strides=emb_strides)

    # Each embedding vector is a "dispersion pattern"
    # Encode as integer: π = Σ_{j=0}^{m-1} (y_j - 1) * c^(m-1-j)
    # This gives a unique integer in [0, c^m - 1] for each pattern
    powers = c ** np.arange(m - 1, -1, -1, dtype=np.int64)
    pattern_codes = (embeddings.astype(np.int64) - 1).dot(powers)

    # Count pattern frequencies
    _, counts = np.unique(pattern_codes, return_counts=True)
    probs = counts / len(pattern_codes)

    # Shannon entropy, normalized
    de = -np.sum(probs * np.log(probs + 1e-12))
    de_norm = de / np.log(float(c ** m))
    return float(de_norm)


def _coarse_graining(x: np.ndarray, scale: int, offset: int = 0) -> np.ndarray:
    """
    Coarse-grain a 1-D signal at given scale and starting offset.

    y_j = (1/scale) * Σ_{i=(j-1)*scale+offset+1}^{j*scale+offset} x_i

    For RCMDE: average DE over offset = 0, 1, ..., scale-1.
    """
    n = len(x)
    if offset >= scale:
        raise ValueError(f"offset {offset} must be < scale {scale}")
    available = n - offset
    n_coarse = available // scale
    if n_coarse < 1:
        return np.array([np.mean(x)])
    # Extract the segment aligned to offset, then reshape and average
    seg = x[offset:offset + n_coarse * scale]
    return np.mean(seg.reshape(n_coarse, scale), axis=1)


def rcmde(x: np.ndarray, scales: list = [1, 2, 3, 4],
          m: int = 4, c: int = 6, tau: int = 1) -> np.ndarray:
    """
    Refined Composite Multiscale Dispersion Entropy.

    For each scale s:
      1. Create s coarse-grained series (offset = 0..s-1)
      2. Compute DE for each coarse-grained series
      3. Average the DE values

    Args:
        x:      1-D signal (a single window)
        scales: list of scale factors
        m:      embedding dimension
        c:      number of classes
        tau:    time delay

    Returns:
        Array of shape (len(scales),) — one RCMDE value per scale
    """
    result = np.zeros(len(scales))
    for idx, s in enumerate(scales):
        if s == 1:
            result[idx] = _dispersion_entropy(x, m=m, c=c, tau=tau)
        else:
            de_values = np.zeros(s)
            for offset in range(s):
                coarse = _coarse_graining(x, s, offset)
                if len(coarse) >= m * tau:
                    de_values[offset] = _dispersion_entropy(
                        coarse, m=m, c=c, tau=tau)
                else:
                    de_values[offset] = 0.0
            result[idx] = np.mean(de_values)
    return result


def _fuzzy_membership(y: np.ndarray, c: int = 6) -> np.ndarray:
    """
    Compute fuzzy membership matrix for dispersion entropy (RCHFDE component).

    Instead of crisp class assignment, each sample gets membership degrees
    to all c classes via Gaussian membership functions centered at each class.

    Returns:
        shape (len(y), c) — fuzzy membership matrix
    """
    n = len(y)
    y_norm = _ncdf_mapping(y, c)  # integer class assignments as centers
    # Gaussian membership: μ_k(y) = exp(-(y_class - k)² / (2σ²))
    # where y_class is the "continuous" class value
    z = (y - np.mean(y)) / (np.std(y) + 1e-12)
    phi = 0.5 * (1.0 + erf(z / np.sqrt(2.0)))
    y_continuous = c * phi + 0.5  # continuous class value in [0.5, c+0.5]

    centers = np.arange(1, c + 1, dtype=np.float64)
    sigma = 0.8  # width of membership function
    diff = y_continuous[:, np.newaxis] - centers[np.newaxis, :]  # (n, c)
    membership = np.exp(-0.5 * (diff / sigma) ** 2)
    # Normalize so each row sums to 1
    membership /= membership.sum(axis=1, keepdims=True) + 1e-12
    return membership


def _fuzzy_dispersion_entropy(x: np.ndarray, m: int = 4, c: int = 6,
                               tau: int = 1) -> float:
    """
    Fuzzy Dispersion Entropy — uses fuzzy membership instead of crisp symbols.

    Reference: Chen et al., "HRCGMFDE", Shock & Vibration, 2024.
    """
    n = len(x)
    if n < m * tau:
        return 0.0

    fuzzy_mat = _fuzzy_membership(x, c)  # (n, c)

    # Build fuzzy embedding: for each time step, take membership vector
    n_emb = n - (m - 1) * tau
    if n_emb < 1:
        return 0.0

    # For each embedding position j (0..m-1), we have membership vectors
    # The "fuzzy pattern" probability uses the product of memberships
    # For efficiency, we estimate via sampling/averaging
    # Simplified approach: compute DE on the argmax (crisp assignment) but
    # weight counts by the membership confidence

    # More principled approach: compute the fuzzy entropy directly
    # p(π) ≈ (1/N) * Σ_i Π_{j=0}^{m-1} μ_{π_j}(y_{i+jτ})
    # But enumerating all c^m patterns is expensive for c=6, m=4 (1296 patterns)

    # Alternative: use the "fluctuation-based" dispersion entropy
    # Focus on the differences between consecutive class assignments
    y_crisp = _ncdf_mapping(x, c)
    n_emb = n - (m - 1) * tau
    emb_shape = (n_emb, m)
    emb_strides = (y_crisp.strides[0] * tau, y_crisp.strides[0])
    embeddings = np.lib.stride_tricks.as_strided(
        y_crisp.astype(np.int32), shape=emb_shape, strides=emb_strides)

    # Weight each pattern by the product of its fuzzy memberships
    powers = c ** np.arange(m - 1, -1, -1, dtype=np.int64)
    pattern_codes = (embeddings.astype(np.int64) - 1).dot(powers)

    # Compute weight for each embedding as product of memberships
    weights = np.ones(n_emb)
    for j in range(m):
        idx_j = np.arange(n_emb) + j * tau
        class_j = embeddings[:, j] - 1  # 0-indexed class
        weights *= fuzzy_mat[idx_j, class_j]

    # Weighted histogram
    unique_codes, inverse = np.unique(pattern_codes, return_inverse=True)
    weighted_counts = np.bincount(inverse, weights=weights, minlength=len(unique_codes))
    probs = weighted_counts / (weighted_counts.sum() + 1e-12)

    fde = -np.sum(probs * np.log(probs + 1e-12))
    fde_norm = fde / np.log(float(c ** m))
    return float(fde_norm)


def _hierarchical_decomposition(x: np.ndarray, level: int) -> list[np.ndarray]:
    """
    Hierarchical decomposition of a signal.

    At each level, apply:
      - Low-pass (Q₀): moving average of 2 points →  y_{2j} + y_{2j+1}
      - High-pass (Q₁): moving difference of 2 points → y_{2j} - y_{2j+1}

    Returns list of 2^level component signals.

    Reference: Chen et al., Shock & Vibration, 2024.
    """
    if level == 0:
        return [x.copy()]

    components = [x.copy()]
    for _ in range(level):
        new_components = []
        for comp in components:
            n = len(comp) // 2
            if n < 2:
                new_components.append(comp)
                continue
            # Low-pass: average adjacent pairs
            low = (comp[0:2*n:2] + comp[1:2*n:2]) / 2.0
            # High-pass: difference of adjacent pairs
            high = (comp[0:2*n:2] - comp[1:2*n:2]) / 2.0
            new_components.append(low)
            new_components.append(high)
        components = new_components
    return components


def rchfde(x: np.ndarray, scales: list = [1, 2, 3, 4],
           m: int = 4, c: int = 6, tau: int = 1,
           hier_level: int = 2) -> np.ndarray:
    """
    Refined Composite Hierarchical Fuzzy Dispersion Entropy.

    For each scale s:
      1. Coarse-grain to scale s (refined composite over s offsets)
      2. Apply hierarchical decomposition (level = min(2, floor(log2(len))))
      3. Compute Fuzzy DE on each component
      4. Average across components and offsets

    Args:
        x:          1-D signal
        scales:     scale factors
        m, c, tau:  DE parameters
        hier_level: hierarchical decomposition depth

    Returns:
        Array of shape (len(scales),)
    """
    result = np.zeros(len(scales))
    # Adapt hierarchy level to signal length
    n = len(x)

    for idx, s in enumerate(scales):
        de_values = []
        # Refined composite: average over s starting offsets
        for offset in range(min(s, max(1, n // (m * tau * 2)))):
            coarse = _coarse_graining(x, s, offset)
            if len(coarse) < m * tau * 2:
                # Signal too short for hierarchical decomp — use plain fuzzy DE
                if len(coarse) >= m * tau:
                    de_values.append(_fuzzy_dispersion_entropy(
                        coarse, m=m, c=c, tau=tau))
                continue

            # Hierarchical decomposition
            actual_level = min(hier_level, int(np.log2(len(coarse))) - 2)
            actual_level = max(1, actual_level)
            try:
                hierarchy = _hierarchical_decomposition(coarse, actual_level)
            except (ValueError, IndexError):
                de_values.append(_fuzzy_dispersion_entropy(
                    coarse, m=m, c=c, tau=tau))
                continue

            # Compute FDE for each hierarchical component
            for comp in hierarchy:
                if len(comp) >= m * tau:
                    de_values.append(_fuzzy_dispersion_entropy(
                        comp, m=m, c=c, tau=tau))

        result[idx] = np.mean(de_values) if de_values else 0.0

    return result


def extract_dispersion_entropy_features(windows: np.ndarray,
                                         scales: list = [1, 2, 3, 4],
                                         m: int = 4, c: int = 6,
                                         tau: int = 1) -> np.ndarray:
    """
    Extract 8-D dispersion entropy features from all windows.

    Args:
        windows: shape (N, window_size) — sliding window output
        scales:  scale factors for multi-scale analysis
        m, c, tau: dispersion entropy parameters

    Returns:
        features: shape (N, 8) — [RCMDE_s1, RCMDE_s2, RCMDE_s3, RCMDE_s4,
                                   RCHFDE_s1, RCHFDE_s2, RCHFDE_s3, RCHFDE_s4]
    """
    N = windows.shape[0]
    n_scales = len(scales)
    de_feats = np.zeros((N, 2 * n_scales), dtype=np.float64)

    # Progress tracking
    for i in range(N):
        if i % 2000 == 0 and i > 0:
            pass  # caller handles progress

        signal = windows[i]
        # RCMDE: refined composite multi-scale
        de_feats[i, :n_scales] = rcmde(signal, scales=scales,
                                        m=m, c=c, tau=tau)
        # RCHFDE: refined composite hierarchical fuzzy
        de_feats[i, n_scales:] = rchfde(signal, scales=scales,
                                         m=m, c=c, tau=tau)

    return de_feats


# ============================================================================
# Statistical Feature Extraction (28-D = 20统计 + 8离散熵) [v2 upgraded]
# ============================================================================

def extract_features(windows: np.ndarray, sample_rate: int = SAMPLE_RATE,
                     dispersion_entropy: bool = True,
                     de_scales: list = [1, 2, 3, 4],
                     de_m: int = 4, de_c: int = 6, de_tau: int = 1) -> np.ndarray:
    """
    Extract statistical + dispersion entropy features per window.

    Time-domain (10):
        0.  RMS               (均方根)
        1.  Peak              (峰值)
        2.  Peak-to-Peak      (峰峰值)
        3.  Crest Factor      (波峰因子 = peak / rms)
        4.  Skewness          (偏度)
        5.  Kurtosis          (峭度)
        6.  Shape Factor      (波形因子 = rms / mean_abs)
        7.  Impulse Factor    (脉冲因子 = peak / mean_abs)
        8.  Mean Abs          (平均绝对值)
        9.  Variance          (方差)

    Frequency-domain (10):
        10. Spectral Centroid  (频谱质心)
        11. Spectral Spread    (频谱展宽)
        12. Spectral Skewness  (频谱偏度)
        13. Spectral Kurtosis  (频谱峭度)
        14. Spectral Entropy   (频谱熵)
        15. Band Energy 0-1200 Hz
        16. Band Energy 1200-2400 Hz
        17. Band Energy 2400-3600 Hz
        18. Band Energy 3600-4800 Hz
        19. Band Energy 4800-6000 Hz

    [v2] Dispersion Entropy (8):
        20. RCMDE scale 1     (精炼复合多尺度色散熵)
        21. RCMDE scale 2
        22. RCMDE scale 3
        23. RCMDE scale 4
        24. RCHFDE scale 1    (精炼复合层次模糊色散熵)
        25. RCHFDE scale 2
        26. RCHFDE scale 3
        27. RCHFDE scale 4

    Total: 28-D

    Args:
        windows:           shape (N, window_size)
        sample_rate:       Hz
        dispersion_entropy: [v2] whether to compute DE features
        de_scales:         [v2] scale factors
        de_m, de_c, de_tau: [v2] DE parameters

    Returns:
        features:          shape (N, 20) or (N, 28) with dispersion_entropy
    """
    N = windows.shape[0]
    feat_dim = 28 if dispersion_entropy else 20
    feats = np.zeros((N, feat_dim), dtype=np.float64)

    # ── Time domain ──
    rms_val = np.sqrt(np.mean(windows ** 2, axis=1))
    peak_val = np.max(np.abs(windows), axis=1)
    p2p_val = np.max(windows, axis=1) - np.min(windows, axis=1)
    mean_abs = np.mean(np.abs(windows), axis=1)
    var_val = np.var(windows, axis=1, ddof=1)

    # Higher moments — subtract mean before computing
    centered = windows - np.mean(windows, axis=1, keepdims=True)
    skew_val = np.mean(centered ** 3, axis=1) / (np.std(windows, axis=1, ddof=1) ** 3 + 1e-12)
    kurt_val = np.mean(centered ** 4, axis=1) / (np.std(windows, axis=1, ddof=1) ** 4 + 1e-12)

    feats[:, 0] = rms_val
    feats[:, 1] = peak_val
    feats[:, 2] = p2p_val
    feats[:, 3] = peak_val / (rms_val + 1e-12)                       # crest factor
    feats[:, 4] = skew_val
    feats[:, 5] = kurt_val
    feats[:, 6] = rms_val / (mean_abs + 1e-12)                       # shape factor
    feats[:, 7] = peak_val / (mean_abs + 1e-12)                      # impulse factor
    feats[:, 8] = mean_abs
    feats[:, 9] = var_val

    # ── Frequency domain ──
    n_fft = windows.shape[1]
    # rfft returns n_fft//2 + 1 bins
    spectrum = np.abs(rfft(windows, n=n_fft, axis=1))  # (N, n_bins)
    freqs = rfftfreq(n_fft, d=1.0 / sample_rate)       # (n_bins,)

    # Power spectrum
    power = spectrum ** 2

    # Spectral centroid (weighted mean frequency)
    total_power = np.sum(power, axis=1, keepdims=True) + 1e-12
    centroid = np.sum(power * freqs[np.newaxis, :], axis=1) / total_power.squeeze()
    feats[:, 10] = centroid

    # Spectral spread (std dev of frequencies around centroid)
    freq_diff = freqs[np.newaxis, :] - centroid[:, np.newaxis]  # (N, n_bins)
    spread = np.sqrt(np.sum(power * freq_diff ** 2, axis=1) / total_power.squeeze())
    feats[:, 11] = spread

    # Spectral skewness
    s_skew = np.sum(power * freq_diff ** 3, axis=1) / \
             (total_power.squeeze() * (spread ** 3 + 1e-12))
    feats[:, 12] = s_skew

    # Spectral kurtosis
    s_kurt = np.sum(power * freq_diff ** 4, axis=1) / \
             (total_power.squeeze() * (spread ** 4 + 1e-12))
    feats[:, 13] = s_kurt

    # Spectral entropy
    psd_norm = power / total_power  # normalize to sum=1
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12), axis=1)
    feats[:, 14] = entropy

    # Sub-band energies (5 bands, 0–6000 Hz)
    band_edges = [0, 1200, 2400, 3600, 4800, 6000]
    for b in range(5):
        lo, hi = band_edges[b], band_edges[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        band_power = np.sum(power[:, mask], axis=1)
        feats[:, 15 + b] = band_power / total_power.squeeze()

    # ── [v2] Dispersion Entropy Features (8-D) ──
    if dispersion_entropy:
        # Note: extract_dispersion_entropy_features is slow (~1-2 min for 12K windows)
        # Progress is printed by the caller in main()
        de_feats = extract_dispersion_entropy_features(
            windows, scales=de_scales, m=de_m, c=de_c, tau=de_tau)
        feats[:, 20:28] = de_feats

    return feats


# ============================================================================
# Normalization
# ============================================================================

def normalize(X: np.ndarray, scaler: Optional[StandardScaler] = None,
              method: str = "z-score"):
    """
    Normalize feature matrix. Returns (X_norm, scaler).
    If scaler is None, fit a new one. Otherwise transform with the provided one.
    """
    if method == "z-score":
        if scaler is None:
            scaler = StandardScaler()
            X_norm = scaler.fit_transform(X)
        else:
            X_norm = scaler.transform(X)
    elif method == "minmax":
        X_min = X.min(axis=0, keepdims=True)
        X_max = X.max(axis=0, keepdims=True)
        X_norm = (X - X_min) / (X_max - X_min + 1e-12)
    else:
        X_norm = X
    return X_norm, scaler


# ============================================================================
# Split Generation
# ============================================================================

def create_splits(y: np.ndarray, load_markers: np.ndarray,
                  test_size: float = 0.20, val_size: float = 0.10,
                  random_state: int = 42) -> dict:
    """
    Create stratified train/val/test splits.

    Standard split: random stratified (all loads mixed).
    Cross-load split: train on specified source load, test on target loads.

    Returns dict with keys: train_idx, val_idx, test_idx
      — each is a boolean mask over the original N samples.
    """
    N = len(y)
    indices = np.arange(N)

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=random_state
    )
    # Second split: train vs val
    val_frac_of_train = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_frac_of_train,
        stratify=y[train_val_idx], random_state=random_state
    )

    masks = {}
    for name, subset_idx in [("train", train_idx), ("val", val_idx),
                              ("test", test_idx)]:
        mask = np.zeros(N, dtype=bool)
        mask[subset_idx] = True
        masks[name] = mask

    return masks


def create_cross_load_splits(y: np.ndarray, load_markers: np.ndarray,
                              source_load: int = 1,
                              target_loads: list[int] = [0, 2, 3],
                              val_size: float = 0.10,
                              random_state: int = 42) -> dict:
    """
    Cross-load split: train on source_load, test on target loads.

    Returns dict with keys:
        train_idx, val_idx                — source load
        test_idx                          — all target loads combined
        test_{load}hp_idx                 — per-target-load test masks
    """
    N = len(y)
    source_mask = load_markers == source_load

    if not np.any(source_mask):
        raise ValueError(f"No samples found for source load {source_load}hp")

    # Split source into train/val
    # Note: unlike create_splits, there's no prior test-split from source —
    # test IS the target loads. So val_size is used directly.
    source_indices = np.where(source_mask)[0]
    train_idx_raw, val_idx_raw = train_test_split(
        source_indices, test_size=val_size,
        stratify=y[source_indices], random_state=random_state
    )

    masks = {}
    for name, idx_subset in [("train", train_idx_raw), ("val", val_idx_raw)]:
        m = np.zeros(N, dtype=bool)
        m[idx_subset] = True
        masks[name] = m

    # Test: all target loads combined
    target_mask = np.isin(load_markers, target_loads)
    masks["test"] = target_mask

    # Per-target-load test masks
    for tgt_load in target_loads:
        tgt = load_markers == tgt_load
        masks[f"test_{tgt_load}hp"] = tgt

    return masks


# ============================================================================
# Statistics Computation
# ============================================================================

def compute_stats(features: np.ndarray, labels: np.ndarray,
                  loads: np.ndarray) -> dict:
    """
    Compute per-class per-load statistics for reporting.
    """
    stats = {}
    for lbl in np.unique(labels):
        lbl_name = LABEL_NAMES.get(int(lbl), f"class_{int(lbl)}")
        stats[lbl_name] = {}
        for ldp in np.unique(loads):
            mask = (labels == lbl) & (loads == ldp)
            if not np.any(mask):
                continue
            subset = features[mask]
            stats[lbl_name][f"{int(ldp)}hp"] = {
                "count": int(np.sum(mask)),
                "mean": subset.mean(axis=0).tolist()[:5],    # first 5 dims
                "std": subset.std(axis=0).tolist()[:5],
            }
    return stats


# ============================================================================
# Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="CWRU Preprocessing Pipeline — NeuroPLC"
    )
    parser.add_argument("--mode", choices=["waveform", "features", "both"],
                        default="both",
                        help="Which data format to produce (default: both)")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--channel", default="DE",
                        help="Signal channel: DE, FE, BA")
    parser.add_argument("--normalize", choices=["z-score", "minmax", "none"],
                        default="z-score")
    parser.add_argument("--cross-load", action="store_true",
                        help="Also create cross-load splits (E6 experiment)")
    parser.add_argument("--export-splits", action="store_true",
                        help="Export split arrays alongside index masks")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover files and print summary, no processing")
    parser.add_argument("--no-dispersion-entropy", action="store_true",
                        help="[v2] Skip dispersion entropy features (28-D → 20-D)")
    parser.add_argument("--output-dir", type=str, default=str(DATA_PROCESSED))
    parser.add_argument("--splits-dir", type=str, default=str(DATA_SPLITS))
    args = parser.parse_args()

    # ── Discover files ──
    try:
        file_list = discover_files(DATA_RAW)
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    if not file_list:
        print(f"No .mat files found in {DATA_RAW / '12k_DE'}")
        print("Run: python download_cwru.py  first")
        sys.exit(1)

    print()
    print("=" * 68)
    print("  NeuroPLC — Data Preprocessing Pipeline")
    print("=" * 68)
    print(f"  Input:        {DATA_RAW / '12k_DE'}")
    print(f"  Channel:      {args.channel}")
    print(f"  Files found:  {len(file_list)}")
    print(f"  Window size:  {args.window_size}")
    print(f"  Stride:       {args.stride}")
    print(f"  Mode:         {args.mode}")
    print(f"  Normalize:    {args.normalize}")
    print("=" * 68)

    if args.dry_run:
        print("\n  Files discovered:\n")
        for f in file_list:
            diam_str = f' {f["diameter"]:.3f}"' if f["diameter"] else ""
            print(f"    {f['subdir']:>10s}/{f['path'].name:<10s}  "
                  f"label={f['label']}({f['fault_type']}{diam_str})  "
                  f"load={f['load_hp']}hp")
        print(f"\n  Total: {len(file_list)} files")
        return

    # ── Process each file ──
    t0 = time.time()
    waveform_list: list[np.ndarray] = []
    features_list: list[np.ndarray] = []
    label_list: list[np.ndarray] = []
    load_list: list[np.ndarray] = []

    total_windows = 0

    for i, finfo in enumerate(file_list):
        path = finfo["path"]
        lbl = finfo["label"]
        ldp = finfo["load_hp"]

        try:
            sig = load_signal(path, channel=args.channel)
        except Exception as e:
            print(f"\n  ✗ Error loading {path.name}: {e}")
            continue

        windows = sliding_window(sig, args.window_size, args.stride)
        n_win = len(windows)
        total_windows += n_win

        # Track which windows belong to which file (for load markers)
        load_arr = np.full(n_win, ldp, dtype=np.int8)
        lbl_arr = np.full(n_win, lbl, dtype=np.int8)

        # Accumulate waveform
        waveform_list.append(windows.astype(np.float32))
        label_list.append(lbl_arr)
        load_list.append(load_arr)

        # Extract features if needed
        if args.mode in ("features", "both"):
            feats = extract_features(
                windows,
                dispersion_entropy=not args.no_dispersion_entropy)
            features_list.append(feats.astype(np.float32))

        print(f"  [{i+1:2d}/{len(file_list)}] {finfo['subdir']}/{path.name}  "
              f"→ {n_win:5d} windows  label={lbl} load={ldp}hp")

    print(f"\n  Total windows: {total_windows}")

    # ── Concatenate ──
    waveform_X = np.concatenate(waveform_list, axis=0)  # (N, 1024)
    y = np.concatenate(label_list, axis=0)               # (N,)
    load_markers = np.concatenate(load_list, axis=0)      # (N,)

    print(f"  Waveform X shape: {waveform_X.shape}")
    print(f"  Labels:          0={np.sum(y==0)}, 1={np.sum(y==1)}, "
          f"2={np.sum(y==2)}, 3={np.sum(y==3)}")

    if features_list:
        features_X = np.concatenate(features_list, axis=0)  # (N, 28) or (N, 20)
        feat_dim = features_X.shape[1]
        print(f"  Features X shape: {features_X.shape} "
              f"({'28-D (20 statistical + 8 dispersion entropy)' if feat_dim == 28 else f'{feat_dim}-D (no dispersion entropy)'})")

    # ── Normalize ──
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("waveform", "both"):
        wav_X_norm, wav_scaler = normalize(waveform_X, method=args.normalize)
        np.save(output_dir / "waveform_X.npy", wav_X_norm.astype(np.float32))
        np.save(output_dir / "waveform_y.npy", y)
        np.save(output_dir / "waveform_load.npy", load_markers)
        print(f"\n  ✓ Saved: waveform_X.npy ({wav_X_norm.shape})")
        print(f"  ✓ Saved: waveform_y.npy, waveform_load.npy")

    if args.mode in ("features", "both"):
        feat_X_norm, feat_scaler = normalize(features_X, method=args.normalize)
        np.save(output_dir / "features_X.npy", feat_X_norm.astype(np.float32))
        np.save(output_dir / "features_y.npy", y)
        np.save(output_dir / "features_load.npy", load_markers)
        # Also save scaler for deployment
        if feat_scaler is not None:
            np.savez(output_dir / "features_scaler.npz",
                     mean=feat_scaler.mean_, scale=feat_scaler.scale_)
        print(f"  ✓ Saved: features_X.npy ({feat_X_norm.shape})")
        print(f"  ✓ Saved: features_y.npy, features_load.npy, features_scaler.npz")

    # ── Statistics ──
    X_for_stats = features_X if features_list else waveform_X
    stats = compute_stats(X_for_stats, y, load_markers)
    stats["meta"] = {
        "total_windows": int(total_windows),
        "window_size": args.window_size,
        "stride": args.stride,
        "channel": args.channel,
        "normalize": args.normalize,
        "feature_dim": X_for_stats.shape[1] if features_list else args.window_size,
        "class_distribution": {
            LABEL_NAMES[int(k)] if int(k) in LABEL_NAMES else str(k): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        }
    }
    stats_path = output_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  ✓ Saved: stats.json")

    # ── Splits ──
    splits_dir = Path(args.splits_dir)
    splits_dir.mkdir(parents=True, exist_ok=True)

    # Standard stratified split
    standard_dir = splits_dir / "standard"
    standard_dir.mkdir(exist_ok=True)
    masks = create_splits(y, load_markers)
    for name in ["train", "val", "test"]:
        np.save(standard_dir / f"{name}_idx.npy", masks[name])
        print(f"  Split {name}: {np.sum(masks[name]):6d} samples "
              f"({np.sum(masks[name])/len(y)*100:.1f}%)")

    if args.export_splits:
        for name, mask in masks.items():
            if args.mode in ("waveform", "both"):
                sub = waveform_X[mask]
                sub_y = y[mask]
                np.save(standard_dir / f"waveform_{name}_X.npy", sub.astype(np.float32))
                np.save(standard_dir / f"waveform_{name}_y.npy", sub_y)
            if args.mode in ("features", "both"):
                sub = features_X[mask]
                sub_y = y[mask]
                np.save(standard_dir / f"features_{name}_X.npy", sub.astype(np.float32))
                np.save(standard_dir / f"features_{name}_y.npy", sub_y)

    # Cross-load split (E6 experiment)
    if args.cross_load:
        cl_dir = splits_dir / "cross_load" / "source_1hp"
        cl_dir.mkdir(parents=True, exist_ok=True)
        cl_masks = create_cross_load_splits(y, load_markers)
        for name, mask in cl_masks.items():
            np.save(cl_dir / f"{name}_idx.npy", mask)
            n_positive = np.sum(mask)
            print(f"  Cross-load {name}: {n_positive:6d} samples "
                  f"({n_positive/len(y)*100:.1f}%)")

    elapsed = time.time() - t0

    # ── Final summary ──
    print()
    print("=" * 68)
    print("  Preprocessing Complete")
    print("=" * 68)
    print(f"  Total samples:        {total_windows}")
    feat_dim = features_X.shape[1] if features_list else args.window_size
    print(f"  Feature dimension:    {feat_dim}"
          f"{' (28-D with dispersion entropy)' if feat_dim == 28 else ' (no dispersion entropy)' if features_list else ''}")
    print(f"  Mode:                 {args.mode}")
    print(f"  Cross-load splits:    {'yes' if args.cross_load else 'no'}")
    print(f"  Elapsed:              {elapsed:.1f}s")
    print(f"  Output:               {output_dir}")
    print(f"  Splits:               {splits_dir}")
    print("=" * 68)
    print(f"\n  → Next: python train_teacher.py   (Teacher CNN)")
    print(f"  →   or: python train_student_kd.py (Student KAN via VRM-KD)")
    print()


if __name__ == "__main__":
    main()
