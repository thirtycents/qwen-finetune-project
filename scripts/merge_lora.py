#!/usr/bin/env python3
"""
merge_lora.py — 将 LoRA 适配器合并回基座模型
==============================================

什么是 LoRA 合并？
-----------------
LoRA (Low-Rank Adaptation) 训练时，我们并不修改原始模型的权重，而是在每一层旁边
添加一对低秩矩阵 A 和 B。推理时输出 = 原始权重 × 输入 + B × A × 输入。

合并(merge)就是把这两个矩阵相乘后，直接加到原始权重上：
    新权重 = 原始权重 + scaling × B × A
    其中 scaling = alpha / rank （LoRA 的缩放因子）

合并后的模型和原始模型结构完全一样，但权重已经包含了微调学到的知识。
好处：推理时不再需要额外加载 LoRA 模块，速度更快，部署更简单。

使用方法：
---------
python scripts/merge_lora.py \
    --base-model Qwen/Qwen3-0.6B \
    --adapter-path outputs/qwen3-0.6b-fc-lora \
    --output-path outputs/qwen3-0.6b-fc-merged

参数说明：
    --base-model    : 基座模型的路径或 HuggingFace 模型名称
    --adapter-path  : LoRA 训练输出的适配器目录（包含 adapter_config.json 和 adapter_model.safetensors）
    --output-path   : 合并后的完整模型保存路径
    --device-map    : 模型加载设备映射，默认 "auto"（自动分配到可用 GPU/CPU）
    --push-to-hub   : 是否推送到 HuggingFace Hub（可选）
    --hub-repo-id   : HuggingFace Hub 仓库 ID（配合 --push-to-hub 使用）
"""

import argparse    # 命令行参数解析库，Python 标准库
import os          # 操作系统接口，用于文件路径操作
import sys         # 系统相关功能，用于退出程序
import shutil      # 高级文件操作，用于复制文件

import torch       # PyTorch 深度学习框架，用于张量操作和模型加载

# transformers 是 HuggingFace 的核心库，提供预训练模型和分词器
# AutoModelForCausalLM: 自动加载因果语言模型（用于文本生成的模型）
# AutoTokenizer: 自动加载对应的分词器（将文本转换为模型能理解的数字序列）
from transformers import AutoModelForCausalLM, AutoTokenizer

# peft (Parameter-Efficient Fine-Tuning) 库专门处理 LoRA 等高效微调方法
# PeftModel: 加载带有 LoRA 适配器的模型
from peft import PeftModel


def parse_args():
    """
    解析命令行参数。

    argparse 的工作原理：
    1. 创建一个 ArgumentParser 对象
    2. 用 add_argument() 定义每个参数的名称、类型、默认值和帮助信息
    3. 调用 parse_args() 解析用户输入的命令行参数
    """
    parser = argparse.ArgumentParser(
        description="将 LoRA 适配器合并回基座模型",
        # formatter_class 让帮助信息保留原始格式（换行、缩进等）
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="基座模型路径或 HuggingFace 模型 ID（默认：Qwen/Qwen3-0.6B）",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="outputs/qwen3-0.6b-fc-lora",
        help="LoRA 适配器目录路径（默认：outputs/qwen3-0.6b-fc-lora）",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/qwen3-0.6b-fc-merged",
        help="合并后模型的保存路径（默认：outputs/qwen3-0.6b-fc-merged）",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help=(
            "设备映射策略（默认：auto）。\n"
            "'auto' 会自动将模型分配到可用的 GPU 上；\n"
            "'cpu' 强制使用 CPU（适合显存不足的情况，但速度较慢）"
        ),
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",  # 这是一个开关参数，出现即为 True，不出现为 False
        help="合并后是否推送到 HuggingFace Hub",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default=None,
        help="HuggingFace Hub 仓库 ID（例如：your-name/qwen3-0.6b-fc）",
    )

    return parser.parse_args()


def validate_paths(args):
    """
    验证输入路径是否有效。

    在执行合并之前，先检查：
    1. 适配器目录是否存在
    2. 适配器目录中是否包含必要的配置文件
    这样可以在早期发现问题，避免浪费时间加载大模型后才报错。
    """
    # ---- 检查适配器目录 ----
    if not os.path.isdir(args.adapter_path):
        print(f"[错误] 适配器目录不存在: {args.adapter_path}")
        print("请先运行训练脚本生成 LoRA 适配器，或检查路径是否正确。")
        sys.exit(1)

    # adapter_config.json 是 PEFT 库保存 LoRA 配置的文件
    # 它记录了 LoRA 的 rank、alpha、target_modules 等超参数
    config_file = os.path.join(args.adapter_path, "adapter_config.json")
    if not os.path.isfile(config_file):
        print(f"[错误] 在 {args.adapter_path} 中找不到 adapter_config.json")
        print("这个文件是 LoRA 适配器的配置文件，训练完成后应该自动生成。")
        print("请检查训练是否成功完成。")
        sys.exit(1)

    # ---- 检查输出目录 ----
    if os.path.exists(args.output_path):
        print(f"[警告] 输出目录已存在: {args.output_path}")
        print("合并后的模型将覆盖该目录中的内容。")


def merge_lora(args):
    """
    执行 LoRA 合并的核心函数。

    整体流程：
    1. 加载基座模型（原始的 Qwen3-0.6B）
    2. 加载 LoRA 适配器（训练产生的低秩矩阵）
    3. 调用 merge_and_unload() 将低秩矩阵合并到原始权重
    4. 保存合并后的完整模型

    merge_and_unload() 内部做的事情（简化版）：
        for layer in model.layers:
            # A 和 B 是 LoRA 训练的两个小矩阵
            # A 的形状: (rank, hidden_size)  例如 (32, 1024)
            # B 的形状: (hidden_size, rank)  例如 (1024, 32)
            delta_weight = B @ A * (alpha / rank)
            layer.weight.data += delta_weight
            # 然后移除 LoRA 模块，恢复原始结构
    """
    print("=" * 60)
    print("  LoRA 适配器合并工具")
    print("=" * 60)
    print(f"  基座模型  : {args.base_model}")
    print(f"  适配器路径: {args.adapter_path}")
    print(f"  输出路径  : {args.output_path}")
    print(f"  设备映射  : {args.device_map}")
    print("=" * 60)

    # ---- 第一步：加载分词器 ----
    # 分词器(tokenizer)负责将文本转换为数字(token IDs)，以及将数字转回文本
    # 例如："你好世界" → [12345, 67890]  →  模型处理  → [11111, 22222] → "我很好"
    # trust_remote_code=True: 允许执行模型仓库中的自定义代码（Qwen 模型需要这个）
    print("\n[1/5] 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
    )
    print(f"  分词器词汇表大小: {len(tokenizer)}")

    # ---- 第二步：加载基座模型 ----
    # torch_dtype=torch.bfloat16: 使用 BF16 半精度浮点数，节省一半显存
    #   - FP32（32位）：精度最高，但占用显存最大
    #   - BF16（16位）：精度略低，但显存减半，训练/推理速度更快
    #   - BF16 比 FP16 数值更稳定（指数位更多），是目前主流选择
    print("\n[2/5] 加载基座模型（这可能需要一些时间）...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=True,
    )
    # 获取模型参数量，方便确认加载是否正确
    total_params = sum(p.numel() for p in base_model.parameters())
    print(f"  基座模型参数量: {total_params / 1e6:.1f}M")

    # ---- 第三步：加载 LoRA 适配器 ----
    # PeftModel.from_pretrained() 会：
    # 1. 读取 adapter_config.json 获取 LoRA 配置
    # 2. 在基座模型的指定层上添加 LoRA 模块（A 和 B 矩阵）
    # 3. 加载训练好的 LoRA 权重到这些模块中
    print("\n[3/5] 加载 LoRA 适配器...")
    model_with_lora = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        torch_dtype=torch.bfloat16,
    )
    # 计算 LoRA 新增的参数量
    lora_params = sum(
        p.numel() for n, p in model_with_lora.named_parameters()
        if "lora_" in n  # LoRA 参数的名称中都包含 "lora_"
    )
    print(f"  LoRA 适配器参数量: {lora_params / 1e6:.1f}M")
    print(f"  LoRA 参数占比: {lora_params / total_params * 100:.2f}%")

    # ---- 第四步：合并权重 ----
    # merge_and_unload() 是 PEFT 库提供的核心方法：
    # - merge: 将 LoRA 的 delta 权重加到原始权重上
    # - unload: 移除 LoRA 模块，恢复模型的原始结构
    # 合并后，模型就是一个标准的 transformers 模型，不再依赖 PEFT 库
    print("\n[4/5] 合并 LoRA 权重到基座模型...")
    merged_model = model_with_lora.merge_and_unload()
    print("  合并完成！")

    # 验证合并后的参数量应该等于基座模型
    merged_params = sum(p.numel() for p in merged_model.parameters())
    print(f"  合并后模型参数量: {merged_params / 1e6:.1f}M")
    assert merged_params == total_params, (
        f"参数量不匹配！基座: {total_params}, 合并后: {merged_params}"
    )

    # ---- 第五步：保存合并后的模型 ----
    # save_pretrained() 会保存：
    # - model.safetensors (或 pytorch_model.bin)：模型权重
    # - config.json：模型架构配置
    # - generation_config.json：生成参数配置
    print(f"\n[5/5] 保存合并后的模型到 {args.output_path}...")
    os.makedirs(args.output_path, exist_ok=True)

    merged_model.save_pretrained(
        args.output_path,
        safe_serialization=True,  # 使用 safetensors 格式（比 pickle 更安全）
    )
    tokenizer.save_pretrained(args.output_path)

    # 复制一些可能有用的额外文件（如果存在的话）
    # 例如 tokenizer_config.json、special_tokens_map.json 等
    # save_pretrained 通常已经处理了，这里做个兜底
    extra_files = [
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]
    for fname in extra_files:
        src = os.path.join(args.base_model, fname) if os.path.isdir(args.base_model) else None
        dst = os.path.join(args.output_path, fname)
        if src and os.path.isfile(src) and not os.path.isfile(dst):
            shutil.copy2(src, dst)
            print(f"  额外复制: {fname}")

    print(f"\n  模型已保存到: {args.output_path}")

    # ---- 可选：推送到 HuggingFace Hub ----
    if args.push_to_hub:
        if not args.hub_repo_id:
            print("\n[警告] 指定了 --push-to-hub 但未提供 --hub-repo-id，跳过推送。")
        else:
            print(f"\n[额外] 推送模型到 HuggingFace Hub: {args.hub_repo_id}")
            merged_model.push_to_hub(args.hub_repo_id)
            tokenizer.push_to_hub(args.hub_repo_id)
            print("  推送完成！")

    # ---- 打印输出目录内容 ----
    print("\n输出目录内容:")
    for f in sorted(os.listdir(args.output_path)):
        fpath = os.path.join(args.output_path, f)
        size = os.path.getsize(fpath)
        if size > 1024 * 1024:
            print(f"  {f:40s} {size / 1024 / 1024:.1f} MB")
        elif size > 1024:
            print(f"  {f:40s} {size / 1024:.1f} KB")
        else:
            print(f"  {f:40s} {size} B")

    print("\n" + "=" * 60)
    print("  合并完成！")
    print("  接下来你可以：")
    print("  1. 用合并后的模型直接推理（不需要 PEFT 库）")
    print("  2. 用 vLLM 部署合并后的模型")
    print("  3. 转换为 GGUF 格式用于边缘部署")
    print("=" * 60)


def main():
    """主函数：解析参数 → 验证路径 → 执行合并"""
    args = parse_args()
    validate_paths(args)
    merge_lora(args)


# Python 的入口点惯例：
# 当直接运行这个脚本时（python merge_lora.py），__name__ == "__main__"
# 当被其他脚本 import 时，__name__ == "merge_lora"，不会自动执行 main()
if __name__ == "__main__":
    main()
