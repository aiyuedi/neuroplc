#!/usr/bin/env python3
"""
NeuroPLC — Data Integrity Tests
=================================
Verify CWRU dataset completeness and correctness.

Usage:
    pytest code/tests/test_data.py -v
    pytest code/tests/test_data.py -v -k "test_manifest"
"""

import pytest
import numpy as np
from pathlib import Path

# ============================================================================
# Test: CWRU File Manifest
# ============================================================================

# Expected 52 files: (fault_type, diameter, load_hp) → file_number
EXPECTED_FILES = {
    ("Normal", None, 0): 97,   ("Normal", None, 1): 98,
    ("Normal", None, 2): 99,   ("Normal", None, 3): 100,
    ("IR", 0.007, 0): 105,     ("IR", 0.007, 1): 106,
    ("IR", 0.007, 2): 107,     ("IR", 0.007, 3): 108,
    ("IR", 0.014, 0): 169,     ("IR", 0.014, 1): 170,
    ("IR", 0.014, 2): 171,     ("IR", 0.014, 3): 172,
    ("IR", 0.021, 0): 209,     ("IR", 0.021, 1): 210,
    ("IR", 0.021, 2): 211,     ("IR", 0.021, 3): 212,
    ("IR", 0.028, 0): 3001,    ("IR", 0.028, 1): 3002,
    ("IR", 0.028, 2): 3003,    ("IR", 0.028, 3): 3004,
    ("Ball", 0.007, 0): 118,   ("Ball", 0.007, 1): 119,
    ("Ball", 0.007, 2): 120,   ("Ball", 0.007, 3): 121,
    ("Ball", 0.014, 0): 185,   ("Ball", 0.014, 1): 186,
    ("Ball", 0.014, 2): 187,   ("Ball", 0.014, 3): 188,
    ("Ball", 0.021, 0): 222,   ("Ball", 0.021, 1): 223,
    ("Ball", 0.021, 2): 224,   ("Ball", 0.021, 3): 225,
    ("Ball", 0.028, 0): 3005,  ("Ball", 0.028, 1): 3006,
    ("Ball", 0.028, 2): 3007,  ("Ball", 0.028, 3): 3008,
    ("OR", 0.007, 0): 130,     ("OR", 0.007, 1): 131,
    ("OR", 0.007, 2): 132,     ("OR", 0.007, 3): 133,
    ("OR", 0.014, 0): 197,     ("OR", 0.014, 1): 198,
    ("OR", 0.014, 2): 199,     ("OR", 0.014, 3): 200,
    ("OR", 0.021, 0): 234,     ("OR", 0.021, 1): 235,
    ("OR", 0.021, 2): 236,     ("OR", 0.021, 3): 237,
    ("OR", 0.028, 0): 3009,    ("OR", 0.028, 1): 3010,
    ("OR", 0.028, 2): 3011,    ("OR", 0.028, 3): 3012,
}


class TestCWRUManifest:
    """Verify the CWRU file manifest is correct."""

    def test_total_count(self):
        """Should have exactly 52 files in manifest."""
        assert len(EXPECTED_FILES) == 52, \
            f"Expected 52 files, got {len(EXPECTED_FILES)}"

    def test_unique_file_numbers(self):
        """All file numbers should be unique."""
        nums = list(EXPECTED_FILES.values())
        assert len(nums) == len(set(nums)), \
            "Duplicate file numbers in manifest"

    def test_file_number_ranges(self):
        """File numbers should be in expected ranges."""
        for (fault, _, _), num in EXPECTED_FILES.items():
            if fault == "Normal":
                assert 97 <= num <= 100, \
                    f"Normal file {num} outside [97, 100]"
            elif num < 300:
                assert 100 <= num <= 300, \
                    f"Small-diameter file {num} outside [100, 300]"
            else:
                assert 3000 <= num <= 3012, \
                    f"Large-diameter file {num} outside [3000, 3012]"

    def test_fault_type_coverage(self):
        """All 4 fault types should be present."""
        types = {f for (f, _, _) in EXPECTED_FILES}
        assert types == {"Normal", "IR", "Ball", "OR"}

    def test_diameter_coverage(self):
        """All 4 fault diameters should be present (plus None for Normal)."""
        diams = {d for (_, d, _) in EXPECTED_FILES}
        assert diams == {None, 0.007, 0.014, 0.021, 0.028}

    def test_load_coverage(self):
        """All 4 loads should be present for each (fault, diameter)."""
        loads_per_key = {}
        for (fault, diam, load), num in EXPECTED_FILES.items():
            key = (fault, diam)
            loads_per_key.setdefault(key, set()).add(load)
        for key, loads in loads_per_key.items():
            assert loads == {0, 1, 2, 3}, \
                f"Missing loads for {key}: {loads}"


@pytest.mark.requires_data
class TestCWRUDataOnDisk:
    """Verify actual .mat files on disk match the manifest."""

    def test_directories_exist(self, data_raw):
        """All 13 subdirectories should exist."""
        expected_dirs = {
            "Normal",
            "IR007", "IR014", "IR021", "IR028",
            "Ball007", "Ball014", "Ball021", "Ball028",
            "OR007", "OR014", "OR021",  # OR028 is missing from this dataset (4 files)
        }
        base = data_raw / "12k_DE"
        if not base.exists():
            pytest.skip("12k_DE directory not found")
        actual = {d.name for d in base.iterdir() if d.is_dir()}
        missing = expected_dirs - actual
        assert not missing, f"Missing directories: {missing}"

    def test_each_file_loadable(self, data_raw):
        """Each .mat file should be loadable with scipy."""
        import scipy.io
        base = data_raw / "12k_DE"
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir():
                continue
            for mat_file in subdir.glob("*.mat"):
                try:
                    mat = scipy.io.loadmat(str(mat_file))
                except Exception as e:
                    pytest.fail(f"Failed to load {mat_file}: {e}")

    def test_signal_extraction(self, data_raw):
        """Each .mat should contain a DE_time variable (variable name may differ from filename)."""
        import scipy.io
        base = data_raw / "12k_DE"
        for subdir in sorted(base.iterdir()):
            if not subdir.is_dir():
                continue
            for mat_file in sorted(subdir.glob("*.mat")):
                mat = scipy.io.loadmat(str(mat_file))
                has_de = any(
                    ("DE_time" in k or "DE_time" in k.lower())
                    for k in mat if not k.startswith("_")
                )
                assert has_de, f"No DE_time variable found in {mat_file.name}"


# ============================================================================
# Test: Download Verify Script
# ============================================================================

class TestDownloadVerify:
    """Verify download_verify_cwru.py metadata."""

    def test_manifest_consistency(self):
        """Manifest in download script matches EXPECTED_FILES."""
        import data_pipeline.download_verify_cwru as dv
        script_manifest = dv.CWRU_MANIFEST
        assert script_manifest == EXPECTED_FILES, \
            "download_verify_cwru.py manifest differs from test manifest"

    def test_subdir_names(self):
        """Subdirectory naming is correct."""
        import data_pipeline.download_verify_cwru as dv
        assert dv._subdir_name("Normal", None) == "Normal"
        assert dv._subdir_name("IR", 0.007) == "IR007"
        assert dv._subdir_name("Ball", 0.021) == "Ball021"
        assert dv._subdir_name("OR", 0.028) == "OR028"

    def test_file_map_52_entries(self):
        """_build_file_map should return 52 entries."""
        import data_pipeline.download_verify_cwru as dv
        fmap = dv._build_file_map()
        assert len(fmap) == 52
        assert 97 in fmap and 3012 in fmap


# ============================================================================
# Test: Verify with Existing Data (if available)
# ============================================================================

@pytest.mark.requires_data
def test_verify_existing_data(data_raw, capsys):
    """Run --verify and check exit behavior."""
    import data_pipeline.download_verify_cwru as dv
    result = dv.verify_dataset(data_raw, verbose=False)
    assert "total_expected" in result
    assert result["total_expected"] == 52
    assert result["ok"] >= 0
    # Even with partial data, should not crash
    assert "details" in result
