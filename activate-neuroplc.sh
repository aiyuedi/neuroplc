#!/usr/bin/env bash
# =============================================================================
# NeuroPLC — Convenience Scripts
# =============================================================================
# Usage:  source activate-neuroplc.sh     (once per terminal)
#         ./run-tests.sh                   (run all tests)
#         ./run-preprocess.sh              (preprocess data)
# =============================================================================

export PYTHONIOENCODING=utf-8
export PYTHONPATH="D:/neuroplc-paper/code:$PYTHONPATH"
export NEUROPLC_VENV="/d/dev-tools/research/venv"
export NEUROPLC_PYTHON="$NEUROPLC_VENV/Scripts/python"

# Alias for convenience
alias npp="$NEUROPLC_PYTHON"
alias npptest="cd D:/neuroplc-paper && PYTHONIOENCODING=utf-8 $NEUROPLC_PYTHON -m pytest code/tests/ -q -s"
alias nppverify="$NEUROPLC_PYTHON D:/neuroplc-paper/code/download_verify_cwru.py --verify"

echo "NeuroPLC environment activated."
echo "  Python: $NEUROPLC_PYTHON"
echo "  npp     → run Python"
echo "  npptest → run tests"
echo "  nppverify → verify dataset"
