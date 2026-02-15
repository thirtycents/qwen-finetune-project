#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  🚀 Qwen-0.6-ft All-in-One Runner"
echo "============================================"
echo ""

# 1. Environment Setup
echo "[*] Step 1: Setting up environment..."
bash "${PROJECT_DIR}/scripts/setup_env.sh"

# 2. Data Preparation
echo ""
echo "[*] Step 2: Preparing data..."
source /home/lymanth/miniforge3/etc/profile.d/conda.sh
conda activate qwen-fc
python "${PROJECT_DIR}/scripts/prepare_data.py"

# 3. Training (Background)
echo ""
echo "[*] Step 3: Starting training in background..."
nohup bash "${PROJECT_DIR}/scripts/train.sh" > "${PROJECT_DIR}/training.log" 2>&1 &
TRAIN_PID=$!
echo "    Training PID: ${TRAIN_PID}"
echo "    Logs: ${PROJECT_DIR}/training.log"

# 4. Dashboard
echo ""
echo "[*] Step 4: Launching Dashboard..."
echo "    The dashboard will monitor the training progress."
bash "${PROJECT_DIR}/scripts/start_dashboard.sh"
