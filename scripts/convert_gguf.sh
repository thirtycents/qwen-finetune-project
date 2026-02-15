#!/bin/bash
set -euo pipefail

# ============================================================================
# GGUF 转换脚本 - 将 Hugging Face 模型转换为 GGUF 格式用于边缘设备部署
# ============================================================================
#
# 什么是 GGUF？
#   GGUF (GPT-Generated Unified Format) 是一种高效的模型存储格式，特别优化
#   了本地推理性能。相比原生 Hugging Face 格式，GGUF 具有：
#   - 更小的文件体积（通过量化）
#   - 更快的加载速度（单文件格式）
#   - 更低的内存占用（适合边缘设备）
#   - 更好的推理速度（CPU/GPU 优化）
#
# 什么是量化？
#   量化是将浮点数精度降低的过程，例如：
#   - 原始模型：32位浮点 (FP32)，精度最高但文件最大
#   - 量化后：16位 (FP16)、8位 (Q8)、4位 (Q4) 等
#   - 更低精度 = 更小文件 + 更快推理，但质量会有轻微下降
#
# 为什么边缘部署很重要？
#   边缘部署（在本地 GPU/CPU 上运行模型）的优势：
#   - 数据隐私：模型和数据不离开设备
#   - 低延迟：无网络延迟，实时响应
#   - 离线可用：无需网络连接即可运行
#   - 成本低：无需昂贵的云 API 调用
#
# 使用示例：
#   ./convert_gguf.sh                                    # 使用默认参数
#   ./convert_gguf.sh /path/to/model /path/to/output    # 指定输入输出路径
#
# ============================================================================

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 默认参数
INPUT_MODEL="${1:-${PROJECT_ROOT}/outputs/qwen3-0.6b-fc-merged}"
OUTPUT_DIR="${2:-${PROJECT_ROOT}/outputs/gguf}"

# GGUF 转换相关路径
LLAMA_CPP_DIR="${PROJECT_ROOT}/llama.cpp"
CONVERT_SCRIPT="${LLAMA_CPP_DIR}/convert_hf_to_gguf.py"
QUANTIZE_BIN="${LLAMA_CPP_DIR}/build/bin/llama-quantize"

# 颜色定义（用于终端输出）
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================================
# 工具函数
# ============================================================================

# 打印信息消息
info() {
    echo -e "${BLUE}[信息]${NC} $1"
}

# 打印成功消息
success() {
    echo -e "${GREEN}[成功]${NC} $1"
}

# 打印警告消息
warn() {
    echo -e "${YELLOW}[警告]${NC} $1"
}

# 打印错误消息并退出
error() {
    echo -e "${RED}[错误]${NC} $1" >&2
    exit 1
}

# ============================================================================
# 主逻辑
# ============================================================================

info "========================================"
info "Qwen 0.6B 模型 GGUF 转换工具"
info "========================================"
info ""

# 验证输入模型存在
if [ ! -d "$INPUT_MODEL" ]; then
    error "输入模型目录不存在: $INPUT_MODEL"
fi
info "✓ 输入模型: $INPUT_MODEL"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"
info "✓ 输出目录: $OUTPUT_DIR"

# ============================================================================
# 步骤 1: 检查并初始化 llama.cpp
# ============================================================================

info ""
info "步骤 1: 检查 llama.cpp 环境..."

if [ ! -d "$LLAMA_CPP_DIR" ]; then
    info "  llama.cpp 未找到，正在克隆仓库..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" || error "克隆 llama.cpp 失败"
    success "✓ 克隆完成"
else
    info "✓ llama.cpp 已存在"
fi

# 检查是否已编译
if [ ! -f "$QUANTIZE_BIN" ]; then
    info "  正在编译 llama.cpp..."
    cd "$LLAMA_CPP_DIR"
    cmake -B build || error "CMake 配置失败"
    cmake --build build --config Release -j || error "编译失败"
    cd "$PROJECT_ROOT"
    success "✓ 编译完成"
else
    success "✓ llama.cpp 已编译"
fi

# ============================================================================
# 步骤 2: 转换 HF 模型到 GGUF (FP16 格式)
# ============================================================================

info ""
info "步骤 2: 转换 Hugging Face 模型到 GGUF (FP16)..."

HF_GGUF="${OUTPUT_DIR}/qwen-0.6b-fc-f16.gguf"

if [ -f "$HF_GGUF" ]; then
    warn "  FP16 GGUF 已存在，跳过转换"
else
    cd "$LLAMA_CPP_DIR"
    python3 convert_hf_to_gguf.py "$INPUT_MODEL" --outfile "$HF_GGUF" || error "GGUF 转换失败"
    cd "$PROJECT_ROOT"
    success "✓ 已生成: $HF_GGUF"
fi

# ============================================================================
# 步骤 3: 量化到 Q4_K_M (推荐用于边缘设备)
# ============================================================================

info ""
info "步骤 3: 量化到 Q4_K_M (4位，最小体积，最快推理)..."

Q4_GGUF="${OUTPUT_DIR}/qwen-0.6b-fc-q4_k_m.gguf"

if [ -f "$Q4_GGUF" ]; then
    warn "  Q4_K_M 量化版本已存在，跳过"
else
    "$QUANTIZE_BIN" "$HF_GGUF" "$Q4_GGUF" q4_k_m || error "Q4_K_M 量化失败"
    success "✓ 已生成: $Q4_GGUF"
fi

# ============================================================================
# 步骤 4: 量化到 Q8_0 (更高质量)
# ============================================================================

info ""
info "步骤 4: 量化到 Q8_0 (8位，更高质量)..."

Q8_GGUF="${OUTPUT_DIR}/qwen-0.6b-fc-q8_0.gguf"

if [ -f "$Q8_GGUF" ]; then
    warn "  Q8_0 量化版本已存在，跳过"
else
    "$QUANTIZE_BIN" "$HF_GGUF" "$Q8_GGUF" q8_0 || error "Q8_0 量化失败"
    success "✓ 已生成: $Q8_GGUF"
fi

# ============================================================================
# 步骤 5: 显示文件大小对比表
# ============================================================================

info ""
info "步骤 5: 文件大小对比"
info "========================================"

# 计算文件大小（MB）
get_size_mb() {
    if [ -f "$1" ]; then
        echo "$(( $(stat -f%z "$1" 2>/dev/null || stat -c%s "$1") / 1024 / 1024 ))"
    else
        echo "N/A"
    fi
}

SIZE_F16=$(get_size_mb "$HF_GGUF")
SIZE_Q4=$(get_size_mb "$Q4_GGUF")
SIZE_Q8=$(get_size_mb "$Q8_GGUF")

printf "%-20s | %10s | %12s\n" "模型版本" "文件大小 (MB)" "说明"
printf "%-20s | %10s | %12s\n" "-------------------" "----------" "--------------------"
printf "%-20s | %10s | %12s\n" "FP16 (原始)" "$SIZE_F16" "精度最高"
printf "%-20s | %10s | %12s\n" "Q4_K_M (推荐)" "$SIZE_Q4" "推荐用于边缘设备"
printf "%-20s | %10s | %12s\n" "Q8_0 (高质量)" "$SIZE_Q8" "质量和体积平衡"

info "========================================"

# ============================================================================
# 步骤 6: 打印使用说明
# ============================================================================

info ""
info "步骤 6: 使用说明"
info "========================================"

info ""
info "使用 llama.cpp 运行模型："
info "  ./llama.cpp/build/bin/llama-cli -m $Q4_GGUF -p '提问内容' -n 256"
info ""

info "使用 Ollama 运行模型："
info "  1. 创建 Modelfile："
info "     cat > Modelfile << EOF"
info "     FROM $Q4_GGUF"
info "     PARAMETER num_ctx 2048"
info "     PARAMETER temperature 0.7"
info "     EOF"
info ""
info "  2. 创建 Ollama 模型："
info "     ollama create qwen-0.6b -f Modelfile"
info ""
info "  3. 运行模型："
info "     ollama run qwen-0.6b"
info ""

info "Python 推理（使用 llama-cpp-python）："
info "  pip install llama-cpp-python"
info "  python3 << 'EOF'"
info "from llama_cpp import Llama"
info "llm = Llama(model_path='$Q4_GGUF', n_gpu_layers=-1, verbose=False)"
info "response = llm('你好，请介绍一下自己', max_tokens=256)"
info "print(response['choices'][0]['text'])"
info "EOF"
info ""

info "========================================"
success "✓ GGUF 转换完成！"
info "========================================"
