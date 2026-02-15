#!/bin/bash
# ============================================================================
# serve.sh — 一键启动 vLLM 推理服务（本地开发版）
# ============================================================================
#
# 这个脚本在本地启动 vLLM 推理服务器（不需要 Docker 或 Kubernetes）。
# 适用于开发调试和快速测试。
#
# 使用方法：
#   bash scripts/serve.sh                                          # 使用默认模型
#   bash scripts/serve.sh --model outputs/qwen3-0.6b-fc-merged     # 指定模型路径
#   bash scripts/serve.sh --port 8080                               # 指定端口
#
# 启动后可以这样测试：
#   curl http://localhost:8000/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -d '{
#       "model": "qwen3-fc",
#       "messages": [{"role": "user", "content": "What is the weather in Beijing?"}],
#       "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}}}]
#     }'
# ============================================================================

# ---- 严格模式 ----
# set -e: 任何命令失败就立即退出（不会继续执行后面的命令）
# set -u: 使用未定义的变量时报错（避免拼写错误）
# set -o pipefail: 管道中任何命令失败都算失败
set -euo pipefail

# ---- 颜色定义 ----
# ANSI 转义码让终端输出彩色文字，方便区分不同级别的信息
RED='\033[0;31m'      # 红色：错误
GREEN='\033[0;32m'    # 绿色：成功
YELLOW='\033[1;33m'   # 黄色：警告
BLUE='\033[0;34m'     # 蓝色：信息
NC='\033[0m'          # 重置颜色

# ---- 默认参数 ----
MODEL_PATH="outputs/qwen3-0.6b-fc-merged"
PORT=8000
HOST="0.0.0.0"
GPU_MEMORY_UTILIZATION=0.90
MAX_MODEL_LEN=4096
DTYPE="bfloat16"

# ---- 参数解析 ----
# 解析命令行参数，允许用户覆盖默认值
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2    # shift 2 表示跳过当前参数和它的值
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --gpu-memory)
            GPU_MEMORY_UTILIZATION="$2"
            shift 2
            ;;
        --max-len)
            MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: bash scripts/serve.sh [选项]"
            echo ""
            echo "选项:"
            echo "  --model <路径>       模型路径 (默认: outputs/qwen3-0.6b-fc-merged)"
            echo "  --port <端口>        服务端口 (默认: 8000)"
            echo "  --host <地址>        监听地址 (默认: 0.0.0.0)"
            echo "  --gpu-memory <比例>  GPU 显存利用率 (默认: 0.90)"
            echo "  --max-len <长度>     最大上下文长度 (默认: 4096)"
            echo "  --help, -h           显示帮助信息"
            exit 0
            ;;
        *)
            echo -e "${RED}[错误] 未知参数: $1${NC}"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# ---- 环境检查 ----
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Qwen3-FC vLLM 推理服务启动器${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查 Python 是否可用
if ! command -v python &> /dev/null; then
    echo -e "${RED}[错误] 找不到 Python，请先安装 Python 3.8+${NC}"
    exit 1
fi

# 检查 vLLM 是否安装
if ! python -c "import vllm" 2>/dev/null; then
    echo -e "${RED}[错误] vLLM 未安装。请运行:${NC}"
    echo -e "${YELLOW}  pip install vllm${NC}"
    exit 1
fi

# 检查 GPU 是否可用
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}[✓] GPU 检测:${NC}"
    # nvidia-smi 显示 GPU 信息，--query-gpu 可以指定显示哪些字段
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}[警告] 未检测到 nvidia-smi，可能没有 GPU 或驱动未安装${NC}"
fi

# 检查模型路径
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}[错误] 模型路径不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}提示：请先运行以下命令:${NC}"
    echo "  1. bash scripts/train.sh        # 训练模型"
    echo "  2. python scripts/merge_lora.py  # 合并 LoRA"
    exit 1
fi

echo -e "${GREEN}[✓] 模型路径: $MODEL_PATH${NC}"
echo -e "${GREEN}[✓] 服务地址: http://$HOST:$PORT${NC}"
echo -e "${GREEN}[✓] 显存利用率: $GPU_MEMORY_UTILIZATION${NC}"
echo -e "${GREEN}[✓] 最大上下文: $MAX_MODEL_LEN tokens${NC}"
echo ""
echo -e "${YELLOW}启动服务中...（首次加载模型可能需要 1-2 分钟）${NC}"
echo -e "${YELLOW}启动后可以用 Ctrl+C 停止服务${NC}"
echo -e "${BLUE}========================================${NC}"

# ---- 启动 vLLM 服务器 ----
# python -m vllm.entrypoints.openai.api_server
# -m: 以模块方式运行（等同于 from vllm.entrypoints.openai import api_server）
#
# 参数说明：
#   --model              : 模型路径
#   --served-model-name  : API 中使用的模型名称（客户端指定 model 参数时用这个名字）
#   --host               : 监听地址，0.0.0.0 表示接受来自所有网络的请求
#   --port               : 监听端口
#   --gpu-memory-utilization : GPU 显存使用比例
#   --max-model-len      : 最大序列长度（输入 + 输出的总 token 数）
#   --trust-remote-code  : 允许执行模型仓库的自定义代码
#   --dtype              : 模型精度
#   --enforce-eager      : 禁用 CUDA Graph（兼容性更好）
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --served-model-name "qwen3-fc" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --trust-remote-code \
    --dtype "$DTYPE" \
    --enforce-eager
