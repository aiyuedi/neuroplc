#!/bin/bash
# ============================================================
# NeuroPLC GPU Pipeline
# Paste this ENTIRE block into your ModelScope GPU terminal
# and press Enter. Wait ~30-40 minutes.
# ============================================================

set -e
export PYTHONIOENCODING=utf-8
WORK=/mnt/workspace/neuroplc-paper
EP=60

echo "============================================"
echo "  NeuroPLC GPU Pipeline"
echo "============================================"

# 1. Clone from Gitee (public repo, no auth needed in China)
echo "[1/6] Cloning from Gitee..."
if [ -d "$WORK" ]; then
    cd $WORK && git pull
else
    git clone https://gitee.com/aiyue-emperor/neuroplc.git $WORK
    cd $WORK
fi

# 2. Extract preprocessed data
echo "[2/6] Extracting data..."
if [ -f "data/processed/waveform_X.npy" ]; then
    echo "  Data already extracted"
elif [ -f "neuroplc-gpu.zip" ]; then
    unzip -o neuroplc-gpu.zip
    echo "  Extracted neuroplc-gpu.zip"
elif [ -f "neuroplc-modelscope.zip" ]; then
    unzip -o neuroplc-modelscope.zip
    echo "  Extracted neuroplc-modelscope.zip"
fi

# 3. Install dependencies
echo "[3/6] Installing dependencies..."
pip install -q numpy scipy scikit-learn matplotlib tqdm pyyaml mlflow

# 4. Preprocess (if data needs preprocessing)
echo "[4/6] Checking preprocessing..."
cd $WORK/code
if [ ! -f "$WORK/data/processed/waveform_X.npy" ]; then
    echo "  Running preprocessing..."
    python preprocess.py --mode both --cross-load
else
    echo "  Preprocessed data found, skipping"
fi

# 5. Train all models
echo "[5/6] Training pipeline..."
export PYTHONPATH=$WORK/code:$PYTHONPATH

echo "--- [1/5] Teacher CNN ($EP epochs) ---"
python train_teacher.py --epochs $EP

echo "--- [2/5] KAN VRM-KD ($EP epochs) ---"
python train_student_kd.py --student-type kan --epochs $EP --tag vrmKD

echo "--- [3/5] KAN Hinton-KD ($EP epochs) ---"
python train_student_kd.py --student-type kan --epochs $EP --no-vrm --tag hintonKD

echo "--- [4/5] KAN No-KD ($EP epochs) ---"
python train_student_kd.py --student-type kan --epochs $EP --no-kd --tag noKD

echo "--- [5/5] MLP VRM-KD ($EP epochs) ---"
python train_student_kd.py --student-type mlp --epochs $EP --tag vrmKD

# 6. Evaluate + Figures
echo "[6/6] Evaluation & Figures..."
python evaluate.py --all
python visualize.py --all

echo ""
echo "============================================"
echo "  PIPELINE COMPLETE!"
echo "============================================"
echo "  Results: $WORK/results/"
echo "  Figures: $WORK/results/figures/"
echo "  Next: download results/ back to local"
echo "============================================"
