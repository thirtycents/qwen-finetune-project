#!/bin/bash
# start_dashboard.sh - 启动 Streamlit 仪表板
# 用法: bash scripts/start_dashboard.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DASHBOARD_DIR="${PROJECT_DIR}/dashboard"

# 激活 conda 环境（如果存在）
if command -v conda &> /dev/null; then
    if [ -n "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "qwen-fc" ]; then
        echo "⚠️  当前 conda 环境: $CONDA_DEFAULT_ENV"
        echo "正在切换到 qwen-fc..."
        eval "$(conda shell.bash hook)"
        conda activate qwen-fc
    elif [ -z "$CONDA_DEFAULT_ENV" ]; then
        echo "正在激活 conda 环境 qwen-fc..."
        eval "$(conda shell.bash hook)"
        conda activate qwen-fc
    fi
fi

echo "📦 检查并安装 dashboard 依赖..."
pip install -q -r "${DASHBOARD_DIR}/requirements.txt"

echo ""
echo "🚀 启动 Qwen3-0.6B FC Dashboard..."
echo "    访问地址: http://localhost:8501"
echo ""
echo "提示："
echo "  - 训练监控需要先运行: bash scripts/train.sh"
echo "  - 推理测试需要先运行: bash scripts/serve.sh --model outputs/qwen3-0.6b-fc-merged"
echo ""

cd "${DASHBOARD_DIR}"
streamlit run app.py --server.port 8501 --server.address localhost
