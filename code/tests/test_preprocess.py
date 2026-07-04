#!/usr/bin/env python3
"""
NeuroPLC — Preprocessing Correctness Tests
============================================
Verify sliding window, feature extraction, normalization, and splits.

Usage:
    pytest code/tests/test_preprocess.py -v
    pytest code/tests/test_preprocess.py -v -k "test_features"
"""

import pytest
import numpy as np
from pathlib import Path


# ============================================================================
# Test: Sliding Window
# ============================================================================

class TestSlidingWindow:
    """Verify sliding_window function."""

    def test_output_shape(self, test_signal):
        from preprocess import sliding_window
        ws, st = 1024, 512
        windows = sliding_window(test_signal, ws, st)
        expected_n = (len(test_signal) - ws) // st + 1
        assert windows.shape == (expected_n, ws), \
            f"Expected {(expected_n, ws)}, got {windows.shape}"

    def test_deterministic(self, test_signal):
        from preprocess import sliding_window
        w1 = sliding_window(test_signal, 1024, 512)
        w2 = sliding_window(test_signal, 1024, 512)
        assert np.allclose(w1, w2), "sliding_window is not deterministic"

    def test_signal_too_short(self):
        from preprocess import sliding_window
        short = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sliding_window(short, window_size=1024, stride=512)

    def test_no_overlap(self, test_signal):
        from preprocess import sliding_window
        ws = 1024
        windows = sliding_window(test_signal, ws, stride=ws)
        # Adjacent non-overlapping windows
        n_expected = len(test_signal) // ws
        assert windows.shape[0] == n_expected

    def test_zero_copy_view(self):
        """sliding_window should return a view (as_strided), not a copy."""
        from preprocess import sliding_window
        x = np.arange(100, dtype=np.float64)
        windows = sliding_window(x, 10, 5)
        x[0] = 999.0  # Modify original — view should reflect
        # First window, first element should now be 999
        assert windows[0, 0] == 999.0, \
            "sliding_window returned a copy, expected view"


# ============================================================================
# Test: Statistical Features
# ============================================================================

class TestStatisticalFeatures:
    """Verify 20-D statistical feature extraction."""

    def test_output_shape(self, test_windows):
        from preprocess import extract_features
        feats = extract_features(test_windows, dispersion_entropy=False)
        assert feats.shape == (len(test_windows), 20), \
            f"Expected ({len(test_windows)}, 20), got {feats.shape}"

    def test_output_shape_with_de(self, test_windows):
        from preprocess import extract_features
        feats = extract_features(test_windows, dispersion_entropy=True)
        assert feats.shape == (len(test_windows), 28), \
            f"Expected ({len(test_windows)}, 28), got {feats.shape}"

    def test_no_nan(self, test_windows):
        from preprocess import extract_features
        feats = extract_features(test_windows, dispersion_entropy=False)
        assert not np.any(np.isnan(feats)), "NaN found in features"
        assert not np.any(np.isinf(feats)), "Inf found in features"

    def test_rms_positive(self, test_windows):
        from preprocess import extract_features
        feats = extract_features(test_windows, dispersion_entropy=False)
        assert np.all(feats[:, 0] >= 0), "RMS should be non-negative"

    def test_crest_factor(self, test_windows):
        from preprocess import extract_features
        feats = extract_features(test_windows, dispersion_entropy=False)
        # Crest factor = peak / rms >= 1
        assert np.all(feats[:, 3] >= 1.0 - 1e-6), \
            "Crest factor should be >= 1"

    def test_constant_signal(self):
        """Features for constant signal should be well-defined."""
        from preprocess import extract_features
        x = np.ones((1, 1024))
        feats = extract_features(x, dispersion_entropy=False)
        # For constant signal: RMS = 1, peak = 1, variance = 0
        assert abs(feats[0, 0] - 1.0) < 1e-6, f"RMS should be 1, got {feats[0, 0]}"
        assert abs(feats[0, 9]) < 1e-6, f"Variance should be 0, got {feats[0, 9]}"

    def test_deterministic(self, test_windows):
        from preprocess import extract_features
        f1 = extract_features(test_windows, dispersion_entropy=False)
        f2 = extract_features(test_windows, dispersion_entropy=False)
        assert np.allclose(f1, f2), "Feature extraction is not deterministic"


# ============================================================================
# Test: Dispersion Entropy
# ============================================================================

class TestDispersionEntropy:
    """Verify dispersion entropy computation."""

    def test_ncdf_mapping_range(self, test_signal):
        from preprocess import _ncdf_mapping
        y = _ncdf_mapping(test_signal, c=6)
        assert y.min() >= 1, f"Min class should be >= 1, got {y.min()}"
        assert y.max() <= 6, f"Max class should be <= 6, got {y.max()}"

    def test_ncdf_mapping_constant(self):
        from preprocess import _ncdf_mapping
        x = np.ones(100)
        y = _ncdf_mapping(x, c=6)
        # Constant signal → all mapped to same class
        assert len(np.unique(y)) == 1, \
            "Constant signal should map to single class"

    def test_dispersion_entropy_range(self, test_signal):
        from preprocess import _dispersion_entropy
        de = _dispersion_entropy(test_signal, m=4, c=6, tau=1)
        # Normalized DE should be in [0, 1]
        assert 0.0 <= de <= 1.0, f"DE {de} outside [0, 1]"

    def test_dispersion_entropy_constant(self):
        from preprocess import _dispersion_entropy
        x = np.ones(200)
        de = _dispersion_entropy(x, m=4, c=6, tau=1)
        # Constant signal → only one pattern → zero entropy
        assert abs(de) < 1e-6, f"DE for constant signal should be ~0, got {de}"

    def test_dispersion_entropy_white_noise(self):
        from preprocess import _dispersion_entropy
        rng = np.random.RandomState(123)
        x = rng.randn(1000)
        de = _dispersion_entropy(x, m=4, c=6, tau=1)
        # White noise should have high entropy (>0.7)
        assert de > 0.7, f"White noise DE should be >0.7, got {de:.3f}"

    def test_rcmde_shape(self, test_signal):
        from preprocess import rcmde
        result = rcmde(test_signal[:5000], scales=[1, 2, 3, 4], m=4, c=6, tau=1)
        assert result.shape == (4,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_rchfde_shape(self, test_signal):
        from preprocess import rchfde
        result = rchfde(test_signal[:5000], scales=[1, 2, 3, 4],
                        m=4, c=6, tau=1, hier_level=2)
        assert result.shape == (4,)
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_de_features_output(self, test_windows):
        from preprocess import extract_dispersion_entropy_features
        de = extract_dispersion_entropy_features(
            test_windows[:5], scales=[1, 2, 3, 4], m=4, c=6, tau=1)
        assert de.shape == (5, 8), f"Expected (5, 8), got {de.shape}"
        # RCMDE and RCHFDE should differ (different methods)
        rcmde_vals = de[:, :4]
        rchfde_vals = de[:, 4:]
        assert not np.allclose(rcmde_vals, rchfde_vals), \
            "RCMDE and RCHFDE returned identical values"

    def test_de_features_no_nan(self, test_windows):
        from preprocess import extract_dispersion_entropy_features
        de = extract_dispersion_entropy_features(
            test_windows[:5], scales=[1, 2, 3, 4])
        assert not np.any(np.isnan(de)), "NaN in dispersion entropy"
        assert not np.any(np.isinf(de)), "Inf in dispersion entropy"


# ============================================================================
# Test: Normalization
# ============================================================================

class TestNormalization:
    """Verify feature normalization."""

    def test_zscore_zero_mean_unit_var(self, test_windows):
        from preprocess import extract_features, normalize
        feats = extract_features(test_windows, dispersion_entropy=False)
        X_norm, scaler = normalize(feats, method="z-score")
        # Mean should be ~0, std ~1
        assert np.allclose(X_norm.mean(axis=0), 0, atol=1e-6)
        assert np.allclose(X_norm.std(axis=0, ddof=0), 1, atol=1e-6)

    def test_zscore_transform_consistent(self, test_windows):
        from preprocess import extract_features, normalize
        feats = extract_features(test_windows, dispersion_entropy=False)
        X_norm, scaler = normalize(feats, method="z-score")
        X_norm2, _ = normalize(feats, scaler=scaler, method="z-score")
        assert np.allclose(X_norm, X_norm2)

    def test_minmax_range(self, test_windows):
        from preprocess import extract_features, normalize
        feats = extract_features(test_windows, dispersion_entropy=False)
        X_norm, _ = normalize(feats, method="minmax")
        assert np.all(X_norm >= 0) and np.all(X_norm <= 1 + 1e-6)


# ============================================================================
# Test: Splits
# ============================================================================

class TestSplits:
    """Verify train/val/test split creation."""

    @pytest.fixture
    def split_data(self):
        """Generate data large enough for stratified splits."""
        N = 500
        y = np.tile([0, 1, 2, 3], N // 4 + 1)[:N]
        load_markers = np.zeros(N, dtype=np.int8)
        return N, y, load_markers

    def test_standard_split_sizes(self, split_data):
        from preprocess import create_splits
        N, y, load_markers = split_data
        masks = create_splits(y, load_markers)

        assert np.sum(masks["test"]) / N == pytest.approx(0.20, abs=0.05)
        assert np.sum(masks["val"]) / N == pytest.approx(0.10, abs=0.05)
        assert np.sum(masks["train"]) / N == pytest.approx(0.70, abs=0.05)

    def test_no_overlap(self, split_data):
        from preprocess import create_splits
        N, y, load_markers = split_data
        masks = create_splits(y, load_markers)

        train_idx = np.where(masks["train"])[0]
        val_idx = np.where(masks["val"])[0]
        test_idx = np.where(masks["test"])[0]

        # No overlap between splits
        assert len(set(train_idx) & set(val_idx)) == 0
        assert len(set(train_idx) & set(test_idx)) == 0
        assert len(set(val_idx) & set(test_idx)) == 0

    def test_stratified(self, split_data):
        from preprocess import create_splits
        N, y, load_markers = split_data
        masks = create_splits(y, load_markers)

        # Class distribution should be similar across splits
        for split_name in ["train", "val", "test"]:
            split_y = y[masks[split_name]]
            _, counts = np.unique(split_y, return_counts=True)
            # Each class should have roughly equal representation
            ratios = counts / counts.sum()
            assert np.all(ratios > 0.10), \
                f"{split_name} split has class ratio < 0.10: {ratios}"


# ============================================================================
# Test: Cross-Load Splits
# ============================================================================

class TestCrossLoadSplits:
    """Verify cross-load split creation."""

    def test_cross_load_basic(self):
        from preprocess import create_cross_load_splits
        N = 1000
        y = np.tile([0, 1, 2, 3], N // 4 + 1)[:N]
        load_markers = np.array([0] * 500 + [1] * 200 + [2] * 150 + [3] * 150, dtype=np.int8)
        masks = create_cross_load_splits(y, load_markers, source_load=1,
                                          target_loads=[0, 2, 3])

        # Train + val should only contain source load
        train_mask = masks["train"]
        val_mask = masks["val"]
        assert np.all(load_markers[train_mask] == 1), "Train has non-source loads"
        assert np.all(load_markers[val_mask] == 1), "Val has non-source loads"

        # Test should only contain target loads
        test_mask = masks["test"]
        assert not np.any(load_markers[test_mask] == 1), "Test has source load"

    def test_cross_load_no_source_in_test(self):
        from preprocess import create_cross_load_splits
        N = 500
        y = np.zeros(N, dtype=np.int8)
        load_markers = np.array([0] * 200 + [1] * 150 + [2] * 100 + [3] * 50, dtype=np.int8)
        masks = create_cross_load_splits(y, load_markers, source_load=1,
                                          target_loads=[0, 2, 3])
        test_mask = masks["test"]
        assert not np.any(load_markers[test_mask] == 1), \
            "Source load found in test set"
