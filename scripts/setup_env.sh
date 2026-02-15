#!/bin/bash
# ============================================================
# setup_env.sh - 一键环境配置脚本
# ============================================================
# 功能：创建 conda 虚拟环境并安装所有依赖
# 使用方式：bash scripts/setup_env.sh
#
# 原理说明：
# - conda 虚拟环境：将项目的 Python 依赖隔离在独立环境中，
#   避免不同项目之间的包版本冲突。
# - 为什么用 conda 而不是 venv？因为 conda 更适合管理
#   包含 CUDA/PyTorch 等复杂依赖的深度学习项目。
# ============================================================

set -e  # 遇到错误立即停止（防止后续命令在错误状态下继续执行）

# ---- 配置 ----
ENV_NAME="qwen-fc"          # 虚拟环境名称（fc = function calling）
PYTHON_VERSION="3.10"       # Python 版本（3.10 兼容性最好）
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"  # 自动获取项目根目录

echo "============================================"
echo "  Qwen3-0.6B Function Calling 环境配置"
echo "============================================"
echo ""
echo "项目目录: ${PROJECT_DIR}"
echo "环境名称: ${ENV_NAME}"
echo "Python:   ${PYTHON_VERSION}"
echo ""

# ---- Step 1: 检查 conda 是否安装 ----
if ! command -v conda &> /dev/null; then
    echo "[错误] 未检测到 conda！"
    echo "请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "安装命令（Linux）："
    echo "  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "  bash Miniconda3-latest-Linux-x86_64.sh"
    exit 1
fi
echo "[✓] conda 已安装"

# ---- Step 2: 创建虚拟环境 ----
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[!] 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo "[*] 正在创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "[✓] 环境创建完成"
fi

# ---- Step 3: 激活环境 ----
echo "[*] 激活环境..."
# 注意：在脚本中激活 conda 需要 source
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[✓] 当前环境: $(python --version)"

# ---- Step 4: 检查 PyTorch ----
echo ""
echo "[*] 检查 PyTorch..."
if python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "[✓] PyTorch 已安装"
    python -c "
import torch
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
else:
    print('    [警告] CUDA 不可用！训练将使用 CPU（会非常慢）')
    print('    请确保已安装 CUDA 版本的 PyTorch')
    print('    安装命令: pip install torch --index-url https://download.pytorch.org/whl/cu124')
"
else
    echo "[!] PyTorch 未安装，正在安装 CUDA 12.4 版本..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo "[✓] PyTorch 安装完成"
fi

# ---- Step 5: 安装 LLaMA-Factory（自动安装兼容的依赖）----
echo ""
echo "[*] 安装 LLaMA-Factory（会自动安装兼容的 transformers, peft, trl 等）..."
pip install llamafactory
echo "[✓] LLaMA-Factory 安装完成"

# ---- Step 6: 安装其他依赖 ----
echo ""
echo "[*] 安装其他依赖..."
pip install -r "${PROJECT_DIR}/requirements.txt"
echo "[✓] 依赖安装完成"

# ---- Step 7: 验证安装 ----
echo ""
echo "============================================"
echo "  安装验证"
echo "============================================"
python -c "
packages = {
    'torch': 'PyTorch（深度学习框架）',
    'transformers': 'HuggingFace Transformers（模型加载）',
    'peft': 'PEFT（参数高效微调，LoRA）',
    'datasets': 'HuggingFace Datasets（数据集加载）',
    'trl': 'TRL（强化学习对齐训练）',
    'accelerate': 'Accelerate（训练加速）',
    'sklearn': 'scikit-learn（评测指标计算）',
    'matplotlib': 'Matplotlib（绘图）',
    'pandas': 'Pandas（数据处理）',
}

all_ok = True
for pkg, desc in packages.items():
    try:
        __import__(pkg)
        print(f'  [✓] {desc}')
    except ImportError:
        print(f'  [✗] {desc} - 未安装！')
        all_ok = False

# 单独检查 vLLM（可能安装失败因为 CUDA 版本）
try:
    import vllm
    print(f'  [✓] vLLM（高性能推理引擎）')
except ImportError:
    print(f'  [!] vLLM 未安装（可能需要特定 CUDA 版本）')
    print(f'      推理部署时再安装也可以：pip install vllm')

print()
if all_ok:
    print('  所有核心依赖安装成功！')
else:
    print('  [警告] 部分依赖安装失败，请检查上面的错误信息')
"

echo ""
echo "============================================"
echo "  配置完成！"
echo "============================================"
echo ""
echo "使用方式："
echo "  1. 激活环境:  conda activate ${ENV_NAME}"
echo "  2. 准备数据:  python scripts/prepare_data.py"
echo "  3. 开始训练:  bash scripts/train.sh"
echo "  4. 运行评测:  python eval/evaluate.py"
echo ""
