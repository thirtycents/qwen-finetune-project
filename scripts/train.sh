#!/bin/bash
# ============================================================
# train.sh - 一键训练脚本
# ============================================================
# 功能：完整训练流程 — SFT → LoRA 合并 → GRPO 强化学习
# 使用方式：bash scripts/train.sh
#
# 原理说明：
# -----------
# LoRA SFT 训练的完整流程：
# 1. LLaMA-Factory 读取 YAML 配置文件
# 2. 下载/加载 Qwen3-0.6B 基础模型
# 3. 在模型的每一层旁边添加 LoRA 旁路（小矩阵 A 和 B）
# 4. 冻结原始模型参数，只训练 LoRA 的小矩阵
# 5. 用 xlam-60k 数据集教模型学会函数调用格式
# 6. 训练完成后，LoRA 参数保存到 outputs/ 目录
#
# 什么是 llamafactory-cli？
# -----------
# LLaMA-Factory 提供的命令行工具，它封装了：
# - 模型加载（自动从 HuggingFace 下载）
# - 数据加载（根据 dataset_info.json 找到数据文件）
# - 训练循环（前向传播→计算损失→反向传播→更新参数）
# - 检查点保存（定期保存模型进度）
# - 验证评估（定期在验证集上测试效果）
#
# 预计训练时间（12GB 显存 GPU）：
# -----------
# - 约 60k 条数据 × 3 轮 = ~180k 步
# - 有效 batch_size = 2 × 8 = 16
# - 约 180000 / 16 ≈ 11250 步
# - 每步约 1-2 秒
# - 预计总时间：3-6 小时
#
# 前置条件：
# -----------
# 1. 已运行 bash scripts/setup_env.sh（安装依赖）
# 2. 已运行 python scripts/prepare_data.py（准备数据）
# 3. 已激活 conda 环境：conda activate qwen-fc
# ============================================================

set -e  # 遇到错误立即停止

# ---- 获取项目根目录 ----
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================"
echo "  Qwen3-0.6B Function Calling 完整训练流程"
echo "============================================"
echo ""
echo "项目目录: ${PROJECT_DIR}"
echo "配置文件: configs/qwen3_lora_sft.yaml"
echo ""

# ---- 检查数据是否已准备 ----
if [ ! -f "${PROJECT_DIR}/data/processed/train.json" ]; then
    echo "[错误] 训练数据未找到！"
    echo "请先运行: python scripts/prepare_data.py"
    exit 1
fi

echo "[✓] 训练数据已就绪"
echo ""

# ---- 检查 GPU 状态 ----
echo "[*] GPU 信息："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
else
    echo "    [警告] nvidia-smi 未找到，无法确认 GPU 状态"
    echo ""
fi

# ---- 开始训练 ----
echo "[*] 开始 LoRA SFT 训练..."
echo "    （训练日志会实时打印在下面）"
echo "    （可以用 Ctrl+C 中断，下次重新运行会从最近的检查点继续）"
echo ""
echo "============================================"
echo ""

# 核心命令：调用 LLaMA-Factory 的 CLI 工具启动训练
# train 子命令 = 执行训练任务
# 后面跟配置文件路径，LLaMA-Factory 会读取其中的所有参数
llamafactory-cli train "${PROJECT_DIR}/configs/qwen3_lora_sft.yaml"

echo ""
echo "============================================"
echo "  训练完成！"
echo "============================================"
echo ""
echo "训练产物保存在: outputs/qwen3-0.6b-fc-lora/"
echo ""

echo ""
echo "============================================"
echo "  第二步：合并 LoRA 权重"
echo "============================================"
echo ""

python "${PROJECT_DIR}/scripts/merge_lora.py"

echo ""
echo "============================================"
echo "  第三步：GRPO 强化学习对齐训练"
echo "============================================"
echo ""
echo "[*] 开始 GRPO 训练..."
echo "    （GRPO 在 SFT 基础上进一步优化函数调用质量）"
echo ""

llamafactory-cli train "${PROJECT_DIR}/configs/qwen3_grpo.yaml"

echo ""
echo "============================================"
echo "  全部训练完成！(SFT + LoRA 合并 + GRPO)"
echo "============================================"
echo ""
echo "训练产物："
echo "  SFT LoRA:   outputs/qwen3-0.6b-fc-lora/"
echo "  合并模型:    outputs/qwen3-0.6b-fc-merged/"
echo "  GRPO 模型:   outputs/qwen3-0.6b-fc-grpo/"
echo ""
echo "下一步操作："
echo "  1. 运行推理:     python eval/run_inference.py"
echo "  2. 评测效果:     python eval/evaluate.py --predictions eval/predictions.jsonl"
echo "  3. 启动服务:     bash scripts/serve.sh"
echo ""
