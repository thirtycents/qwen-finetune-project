#!/bin/bash
set -e

ENV_NAME="qwen-fc"
PYTHON_VERSION="3.11"
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  Qwen3-0.6B Function Calling 环境配置"
echo "============================================"
echo ""
echo "项目目录: ${PROJECT_DIR}"
echo "环境名称: ${ENV_NAME}"
echo "Python:   ${PYTHON_VERSION}"
echo ""

if ! command -v conda &> /dev/null; then
    echo "[错误] 未检测到 conda！"
    echo "请先安装 Miniconda: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
echo "[✓] conda 已安装"

if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[!] 环境 '${ENV_NAME}' 已存在，跳过创建"
else
    echo "[*] 正在创建 conda 环境: ${ENV_NAME} (Python ${PYTHON_VERSION})..."
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    echo "[✓] 环境创建完成"
fi

echo "[*] 激活环境..."
eval "$(conda shell.bash hook)"
conda activate "${ENV_NAME}"
echo "[✓] 当前环境: $(python --version)"

echo ""
echo "[*] 检查/安装 PyTorch (CUDA 12.4)..."
if python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "[✓] PyTorch 已安装"
    python -c "
import torch
if torch.cuda.is_available():
    print(f'    GPU: {torch.cuda.get_device_name(0)}')
    print(f'    显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
else
    echo "[*] 安装 PyTorch Nightly (CUDA 12.8+ for Blackwell/RTX 50-series)..."
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
fi

echo ""
echo "[*] 克隆并安装 LLaMA-Factory (官方推荐方式)..."
cd /tmp
if [ -d "LLaMA-Factory" ]; then
    rm -rf LLaMA-Factory
fi
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install hatchling editables
pip install -e ".[torch,metrics]" --no-build-isolation
echo "[✓] LLaMA-Factory 安装完成"

echo ""
echo "[*] 安装 vLLM (推理引擎)..."
pip install vllm
echo "[✓] vLLM 安装完成"

echo ""
echo "[*] 安装其他依赖..."
pip install pandas scikit-learn matplotlib seaborn
pip install ruff pytest openai aiohttp tqdm
echo "[✓] 其他依赖安装完成"

cd "${PROJECT_DIR}"

echo ""
echo "============================================"
echo "  安装验证"
echo "============================================"
python -c "
packages = {
    'torch': 'PyTorch',
    'transformers': 'Transformers',
    'peft': 'PEFT (LoRA)',
    'trl': 'TRL (RL training)',
    'accelerate': 'Accelerate',
    'llamafactory': 'LLaMA-Factory',
    'vllm': 'vLLM',
}

for pkg, desc in packages.items():
    try:
        __import__(pkg.replace('-', '_'))
        print(f'  [✓] {desc}')
    except ImportError:
        print(f'  [✗] {desc} - 未安装')
"

echo ""
echo "============================================"
echo "  配置完成！"
echo "============================================"
echo ""
echo "下一步："
echo "  1. 激活环境: conda activate ${ENV_NAME}"
echo "  2. 准备数据: python scripts/prepare_data.py"
echo "  3. 开始训练: bash scripts/train.sh"
echo ""
