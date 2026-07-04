#!/usr/bin/env python3
"""
NeuroPLC — Pytest Configuration
=================================
Shared fixtures and configuration for all tests.

Usage:
    cd D:/neuroplc-paper
    python -m pytest code/tests/ -v
    python -m pytest code/tests/ -v -k "test_data"  # data tests only
    python -m pytest code/tests/ -v --cov=code       # with coverage
"""

import sys
from pathlib import Path

import pytest
import numpy as np

# ── Path setup ──
CODE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.insert(0, str(CODE_DIR))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def project_root():
    """Root directory of the NeuroPLC project."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def code_dir():
    """Code directory."""
    return CODE_DIR


@pytest.fixture(scope="session")
def data_raw():
    """Path to raw CWRU data."""
    return PROJECT_ROOT / "data" / "raw"


@pytest.fixture(scope="session")
def data_processed():
    """Path to processed data."""
    return PROJECT_ROOT / "data" / "processed"


@pytest.fixture(scope="session")
def has_cwru_data(data_raw):
    """Check if CWRU data is available. Skip data-dependent tests if not."""
    base = data_raw / "12k_DE"
    if not base.exists():
        return False
    mat_files = list(base.rglob("*.mat"))
    return len(mat_files) > 0


@pytest.fixture(scope="session")
def test_signal():
    """Generate a synthetic vibration-like signal for testing."""
    rng = np.random.RandomState(42)
    t = np.linspace(0, 1, 12000, endpoint=False)
    # Multi-component: fundamental + harmonics + noise
    sig = (
        np.sin(2 * np.pi * 30 * t)           # shaft speed ~1800 RPM
        + 0.5 * np.sin(2 * np.pi * 120 * t)  # bearing fault frequency
        + 0.2 * np.sin(2 * np.pi * 300 * t)  # harmonic
        + 0.05 * rng.randn(12000)            # noise
    )
    return sig.astype(np.float64)


@pytest.fixture(scope="session")
def test_windows(test_signal):
    """Generate sliding windows from test signal."""
    from preprocess import sliding_window
    return sliding_window(test_signal, window_size=1024, stride=512)


# ── Markers ──
pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers",
        "requires_data: marks tests that need CWRU data on disk"
    )
    config.addinivalue_line(
        "markers",
        "requires_tia: marks tests that need TIA Portal running"
    )


def pytest_collection_modifyitems(config, items):
    """Auto-skip data-dependent tests if CWRU data is missing."""
    data_raw = PROJECT_ROOT / "data" / "raw" / "12k_DE"
    has_data = data_raw.exists() and len(list(data_raw.rglob("*.mat"))) > 0

    if not has_data:
        skip_data = pytest.mark.skip(
            reason="CWRU data not available. Run: python download_verify_cwru.py --source local --input <folder>"
        )
        for item in items:
            if "requires_data" in item.keywords:
                item.add_marker(skip_data)
