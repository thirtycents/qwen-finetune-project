#!/bin/bash

# ============================================================================
# Qwen-0.6 微调基准测试脚本
# ============================================================================
# 功能: 包装 eval/benchmark.py，支持快速模式和完整模式
# 使用: ./benchmark.sh [--quick|--full|--help]
#
# 模式说明:
#   --quick (默认)  : 快速测试，小参数范围，快速验证环境是否正常
#   --full          : 完整矩阵测试，全面评估性能，生成对比图表
#   --help          : 显示使用说明
# ============================================================================

set -e  # 任何命令失败都中止执行

# 获取脚本所在目录（脚本的父目录是 scripts 目录）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 获取项目根目录（scripts 的父目录）
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# vLLM 服务器地址，支持环境变量覆盖
VLLM_URL="${VLLM_URL:-http://localhost:8000}"

# 基准测试输出文件路径
OUTPUT_FILE="${PROJECT_ROOT}/eval/benchmark_results.json"

# ============================================================================
# 函数: 检查 vLLM 服务器是否在线
# ============================================================================
check_vllm_server() {
    echo "正在检查 vLLM 服务器: $VLLM_URL"
    
    # 尝试连接到 vLLM 的 /health 端点
    # -s: 静默模式（不显示进度条）
    # -o /dev/null: 丢弃响应内容（只关心状态码）
    # -w "%{http_code}": 输出 HTTP 状态码
    http_code=$(curl -s -o /dev/null -w "%{http_code}" "$VLLM_URL/health" 2>/dev/null || echo "000")
    
    if [ "$http_code" = "200" ]; then
        echo "✓ vLLM 服务器在线，健康状态良好"
        return 0
    else
        echo "✗ 无法连接到 vLLM 服务器 (HTTP 状态码: $http_code)"
        echo "  请确保 vLLM 服务器正在运行: vllm serve <model_id>"
        return 1
    fi
}

# ============================================================================
# 函数: 显示使用说明
# ============================================================================
show_help() {
    cat << 'EOF'
用法: ./benchmark.sh [MODE]

模式:
  --quick             快速测试模式（默认）
                      并发度: 1, 4
                      上下文长度: 256, 1024
                      请求数: 10 个
                      用途: 快速验证基准测试环境是否正常工作

  --full              完整矩阵测试模式
                      并发度: 1, 2, 4, 8, 16, 32
                      上下文长度: 256, 1024, 2048, 4096
                      请求数: 20 个
                      生成性能对比图表
                      用途: 全面评估模型在不同条件下的性能

  --help              显示此帮助信息

环境变量:
  VLLM_URL            vLLM 服务器地址（默认: http://localhost:8000）

示例:
  ./benchmark.sh              # 运行快速测试
  ./benchmark.sh --quick      # 显式运行快速测试
  ./benchmark.sh --full       # 运行完整矩阵测试
  VLLM_URL=http://192.168.1.100:8000 ./benchmark.sh --full

输出:
  基准测试结果保存到: eval/benchmark_results.json
EOF
}

# ============================================================================
# 函数: 打印横幅标题
# ============================================================================
print_banner() {
    local mode="$1"
    local concurrency="$2"
    local context_length="$3"
    local num_requests="$4"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║                    Qwen-0.6 基准测试启动                              ║"
    echo "╠═══════════════════════════════════════════════════════════════════════╣"
    echo "║ 模式:             $mode"
    echo "║ 并发度:           $concurrency"
    echo "║ 上下文长度(tokens): $context_length"
    echo "║ 每组请求数:       $num_requests"
    echo "║ vLLM 服务器:      $VLLM_URL"
    echo "║ 输出文件:         $OUTPUT_FILE"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo ""
}

# ============================================================================
# 函数: 执行快速测试
# ============================================================================
run_quick_mode() {
    print_banner "快速测试" "1, 4" "256, 1024" "10"
    
    python "${PROJECT_ROOT}/eval/benchmark.py" \
        --base-url "$VLLM_URL" \
        --concurrency 1 4 \
        --context-length 256 1024 \
        --num-requests 10 \
        --output "$OUTPUT_FILE" \
        --timeout 300
    
    echo ""
    echo "✓ 快速测试完成！结果已保存到: $OUTPUT_FILE"
}

# ============================================================================
# 函数: 执行完整测试
# ============================================================================
run_full_mode() {
    print_banner "完整矩阵测试" "1, 2, 4, 8, 16, 32" "256, 1024, 2048, 4096" "20"
    
    python "${PROJECT_ROOT}/eval/benchmark.py" \
        --base-url "$VLLM_URL" \
        --concurrency 1 2 4 8 16 32 \
        --context-length 256 1024 2048 4096 \
        --num-requests 20 \
        --output "$OUTPUT_FILE" \
        --full-matrix \
        --chart \
        --timeout 600
    
    echo ""
    echo "✓ 完整测试完成！结果已保存到: $OUTPUT_FILE"
}

# ============================================================================
# 主程序入口
# ============================================================================
main() {
    # 确保 vLLM 服务器在线
    if ! check_vllm_server; then
        exit 1
    fi
    
    # 根据第一个参数选择执行模式
    case "${1:-}" in
        --quick)
            run_quick_mode
            ;;
        --full)
            run_full_mode
            ;;
        --help|-h)
            show_help
            ;;
        "")
            # 默认运行快速测试
            run_quick_mode
            ;;
        *)
            echo "错误: 未知的模式 '$1'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# 执行主程序
main "$@"
