#!/usr/bin/env bash
# =============================================================================
# NeuroPLC — One-Command Full Pipeline
# =============================================================================
# Usage:
#   bash run_all.sh           # Full pipeline (preprocess → train → eval → viz → paper)
#   bash run_all.sh --quick   # Quick mode: 5 epochs per stage (for testing)
#   bash run_all.sh --gpu     # For ModelScope GPU (already exported env vars)
#
# For ModelScope GPU:
#   1. Upload project to ModelScope notebook
#   2. bash run_all.sh --gpu
#   3. Download results/ back to local
# =============================================================================

set -e  # Stop on first error

export PYTHONIOENCODING=utf-8
export PYTHONPATH="D:/neuroplc-paper/code:$PYTHONPATH"
PYTHON="/d/dev-tools/research/venv/Scripts/python"
PROJECT="D:/neuroplc-paper"

MODE="${1:-full}"

echo "============================================"
echo "  NeuroPLC — Full Pipeline"
echo "  Mode: $MODE"
echo "  Python: $PYTHON"
echo "  Project: $PROJECT"
echo "============================================"

# ── Step 1: Preprocessing ──
echo ""
echo "[1/8] Preprocessing CWRU data..."
cd "$PROJECT"
$PYTHON code/data_pipeline/preprocess.py --mode both --cross-load

# ── Step 2: Train Teacher ──
echo ""
echo "[2/8] Training Teacher CNN..."
if [ "$MODE" = "--gpu" ] || [ "$MODE" = "--quick" ]; then
    EPOCHS="--epochs 5"
else
    EPOCHS="--epochs 80"
fi
$PYTHON code/training/train_teacher.py $EPOCHS

# ── Step 3: Train Student KAN (VRM-KD) ──
echo ""
echo "[3/8] Training Student KAN via VRM-KD..."
if [ "$MODE" = "--gpu" ] || [ "$MODE" = "--quick" ]; then
    KAN_EPOCHS="--epochs 5"
else
    KAN_EPOCHS="--epochs 100"
fi
$PYTHON code/training/train_student_kd.py --student-type kan $KAN_EPOCHS --tag vrmKD

# ── Step 4: KD Ablation — Hinton-only ──
echo ""
echo "[4/8] KD Ablation: Hinton-only..."
$PYTHON code/training/train_student_kd.py --student-type kan $KAN_EPOCHS --no-vrm --tag hintonKD

# ── Step 5: KD Ablation — No-KD (from scratch) ──
echo ""
echo "[5/8] KD Ablation: No-KD (scratch)..."
$PYTHON code/training/train_student_kd.py --student-type kan $KAN_EPOCHS --no-kd --tag noKD

# ── Step 6: MLP Baseline ──
echo ""
echo "[6/8] Training MLP baseline (VRM-KD)..."
$PYTHON code/training/train_student_kd.py --student-type mlp $KAN_EPOCHS --tag vrmKD

# ── Step 7: Evaluation (E1-E7) ──
echo ""
echo "[7/8] Running evaluation (E1-E7)..."
$PYTHON code/evaluate.py --all

# ── Step 8: Visualization ──
echo ""
echo "[8/8] Generating figures..."
$PYTHON code/analysis/visualize.py --all

# ── Done ──
echo ""
echo "============================================"
echo "  Pipeline Complete!"
echo "============================================"
echo "  Results: $PROJECT/results/"
echo "  Figures: $PROJECT/results/figures/"
echo "  Checkpoints: $PROJECT/results/teacher/"
echo "                $PROJECT/results/student/"
echo ""
echo "  Next: cd paper && pdflatex main.tex"
echo "============================================"
