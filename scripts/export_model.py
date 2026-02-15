#!/usr/bin/env python3
"""
export_model.py — 模型导出与格式转换工具
=========================================

这个脚本提供了模型导出的多种功能：
1. 验证合并后的模型是否可以正常加载和推理
2. 生成模型卡片（Model Card）用于 HuggingFace Hub 发布
3. 统计模型信息（参数量、层数、词表大小等）

使用方法：
---------
# 验证模型
python scripts/export_model.py --model-path outputs/qwen3-0.6b-fc-merged --action verify

# 生成模型信息报告
python scripts/export_model.py --model-path outputs/qwen3-0.6b-fc-merged --action info

# 生成模型卡片
python scripts/export_model.py --model-path outputs/qwen3-0.6b-fc-merged --action card

# 执行所有操作
python scripts/export_model.py --model-path outputs/qwen3-0.6b-fc-merged --action all
"""

import argparse     # 命令行参数解析
import json         # JSON 格式处理
import os           # 文件路径操作
import sys          # 系统功能
import time         # 计时

import torch        # PyTorch 深度学习框架

# transformers 库的核心组件
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="模型导出与格式转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="outputs/qwen3-0.6b-fc-merged",
        help="模型路径（合并后的模型目录）",
    )
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["verify", "info", "card", "all"],
        help="执行的操作：verify=验证推理, info=模型信息, card=生成模型卡片, all=全部",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="运行设备：auto, cpu, cuda（默认：auto）",
    )
    return parser.parse_args()


def print_model_info(model, tokenizer, model_path: str):
    """
    打印模型的详细信息。

    这个函数会遍历模型的所有参数和配置，输出一个清晰的信息表。
    这些信息在以下场景中很有用：
    - 确认模型是否正确加载
    - 写论文或技术报告时引用模型规格
    - 部署前评估显存和计算需求
    """
    print("\n" + "=" * 60)
    print("  模型信息报告")
    print("=" * 60)

    # ---- 基本信息 ----
    # model.config 包含了模型架构的所有配置参数
    config = model.config
    print(f"\n  模型路径        : {model_path}")
    print(f"  模型类型        : {config.model_type}")

    # 参数量统计
    # numel() = number of elements，返回张量中元素的总数
    total_params = sum(p.numel() for p in model.parameters())
    # 可训练参数：requires_grad=True 的参数才会在反向传播中更新
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数量        : {total_params:,} ({total_params / 1e6:.1f}M)")
    print(f"  可训练参数量    : {trainable_params:,} ({trainable_params / 1e6:.1f}M)")

    # ---- 架构信息 ----
    # hidden_size: 每一层的隐藏状态维度（Qwen3-0.6B 通常是 1024）
    # num_hidden_layers: Transformer 的层数
    # num_attention_heads: 注意力头的数量
    # 多头注意力的原理：将 hidden_size 分成多个 head，每个 head 独立计算注意力
    # 这样模型可以同时关注输入的不同方面（语法、语义、位置等）
    print(f"\n  隐藏层维度      : {getattr(config, 'hidden_size', 'N/A')}")
    print(f"  Transformer 层数: {getattr(config, 'num_hidden_layers', 'N/A')}")
    print(f"  注意力头数      : {getattr(config, 'num_attention_heads', 'N/A')}")
    # KV heads: Grouped-Query Attention (GQA) 中的 KV 头数
    # GQA 是一种优化技术：多个 Query 头共享同一组 Key-Value 头
    # 好处：大幅减少 KV Cache 的显存占用，推理速度更快
    print(f"  KV 注意力头数   : {getattr(config, 'num_key_value_heads', 'N/A')}")
    print(f"  中间层维度      : {getattr(config, 'intermediate_size', 'N/A')}")
    print(f"  最大位置编码    : {getattr(config, 'max_position_embeddings', 'N/A')}")
    print(f"  词汇表大小      : {getattr(config, 'vocab_size', 'N/A')}")

    # ---- 分词器信息 ----
    print(f"\n  分词器词汇量    : {len(tokenizer)}")
    # 特殊 token 是模型用来标记特殊位置的符号
    # BOS (Begin of Sequence): 序列开始标记
    # EOS (End of Sequence): 序列结束标记
    # PAD: 填充标记，用于让不同长度的输入对齐
    print(f"  BOS Token ID    : {tokenizer.bos_token_id}")
    print(f"  EOS Token ID    : {tokenizer.eos_token_id}")
    print(f"  PAD Token ID    : {tokenizer.pad_token_id}")

    # ---- 模型文件大小 ----
    if os.path.isdir(model_path):
        total_size = 0
        file_count = 0
        print(f"\n  模型文件:")
        for f in sorted(os.listdir(model_path)):
            fpath = os.path.join(model_path, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                total_size += size
                file_count += 1
                if size > 1024 * 1024:
                    print(f"    {f:40s} {size / 1024 / 1024:.1f} MB")
                elif size > 1024:
                    print(f"    {f:40s} {size / 1024:.1f} KB")
                else:
                    print(f"    {f:40s} {size} B")
        print(f"\n  文件总数        : {file_count}")
        print(f"  磁盘总占用      : {total_size / 1024 / 1024:.1f} MB")

    # ---- 显存估计 ----
    # 估算模型在不同精度下的显存需求
    # FP32: 每个参数 4 字节
    # FP16/BF16: 每个参数 2 字节
    # INT8: 每个参数 1 字节
    # INT4: 每个参数 0.5 字节
    fp32_gb = total_params * 4 / 1024**3
    fp16_gb = total_params * 2 / 1024**3
    int8_gb = total_params * 1 / 1024**3
    int4_gb = total_params * 0.5 / 1024**3
    print(f"\n  显存需求估计（仅模型权重，不含 KV Cache）:")
    print(f"    FP32  : {fp32_gb:.2f} GB")
    print(f"    BF16  : {fp16_gb:.2f} GB")
    print(f"    INT8  : {int8_gb:.2f} GB")
    print(f"    INT4  : {int4_gb:.2f} GB")

    print("\n" + "=" * 60)


def verify_model(model, tokenizer, device: str):
    """
    验证模型是否能正常进行推理。

    通过一个简单的 Function Calling 测试用例来验证：
    1. 模型可以正常生成文本
    2. 输出格式符合预期（JSON 格式的函数调用）

    这一步很重要：确保合并操作没有破坏模型的能力。
    """
    print("\n" + "=" * 60)
    print("  模型推理验证")
    print("=" * 60)

    # 构造一个 Function Calling 的测试输入
    # 这是项目的核心场景：用户提问 + 可用工具 → 模型生成函数调用
    test_prompt = (
        "You are a helpful assistant with access to tools. "
        "When you need to call a function, output a JSON object with 'name' and 'arguments' fields.\n\n"
        "Available functions:\n"
        '- get_weather(city: str, unit: str = "celsius") -> str: Get current weather\n\n'
        "User: What's the weather in Beijing?\n"
        "Assistant:"
    )

    print(f"\n  测试输入:\n  {test_prompt[:100]}...")

    # tokenizer() 将文本转换为模型输入格式
    # return_tensors="pt": 返回 PyTorch 张量（pt = PyTorch）
    # 返回的 inputs 包含：
    # - input_ids: token ID 序列 [1, 12345, 67890, ...]
    # - attention_mask: 注意力掩码 [1, 1, 1, ...]（1 = 有效 token，0 = padding）
    inputs = tokenizer(test_prompt, return_tensors="pt")

    # 确定设备：优先使用模型已经所在的设备
    if device == "auto":
        target_device = next(model.parameters()).device
    else:
        target_device = torch.device(device)

    # .to(device): 将张量移动到指定设备（GPU 或 CPU）
    inputs = {k: v.to(target_device) for k, v in inputs.items()}

    print(f"  输入 token 数: {inputs['input_ids'].shape[1]}")
    print(f"  运行设备: {target_device}")

    # 计时推理过程
    start_time = time.time()

    # torch.no_grad(): 禁用梯度计算
    # 推理时不需要计算梯度（那是训练时才需要的），禁用可以：
    # 1. 节省显存（不需要保存中间结果用于反向传播）
    # 2. 加快计算速度
    with torch.no_grad():
        # model.generate() 是 HuggingFace 的文本生成方法
        # max_new_tokens: 最多生成多少个新 token
        # do_sample=False: 使用贪心解码（每步选概率最高的 token）
        #   - do_sample=True 时会随机采样，生成更多样但不确定的结果
        #   - 验证时用贪心解码，保证结果可复现
        # temperature=1.0: 温度参数，控制输出的随机性
        #   - 温度越低，输出越确定（趋向贪心）
        #   - 温度越高，输出越随机（更有创意但可能胡说）
        # pad_token_id: 告诉模型用哪个 token 做填充
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    elapsed = time.time() - start_time

    # 解码生成的 token 回文本
    # outputs[0] 取第一个（也是唯一一个）样本
    # [inputs['input_ids'].shape[1]:] 只取新生成的部分，跳过输入
    # skip_special_tokens=True: 跳过特殊 token（如 <|endoftext|>）
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    new_tokens = len(generated_ids)
    tokens_per_sec = new_tokens / elapsed if elapsed > 0 else 0

    print(f"\n  生成结果:")
    print(f"  {generated_text[:300]}")
    print(f"\n  生成 token 数  : {new_tokens}")
    print(f"  耗时           : {elapsed:.2f} 秒")
    print(f"  生成速度       : {tokens_per_sec:.1f} tokens/s")

    # 简单检查输出是否包含函数调用的特征
    has_json = "{" in generated_text and "}" in generated_text
    has_func_name = any(
        kw in generated_text.lower()
        for kw in ["get_weather", "name", "function"]
    )

    print(f"\n  验证结果:")
    print(f"    包含 JSON 结构 : {'✅ 是' if has_json else '❌ 否'}")
    print(f"    包含函数名     : {'✅ 是' if has_func_name else '❌ 否'}")

    if has_json and has_func_name:
        print("\n  ✅ 模型验证通过！Function Calling 能力正常。")
    else:
        print("\n  ⚠️ 模型输出可能不符合预期。")
        print("  这可能是因为：")
        print("  1. 模型还没有经过 Function Calling 微调（如果用的是原始基座模型）")
        print("  2. 合并过程出现了问题")
        print("  3. 测试 prompt 需要调整")
        print("  建议运行完整的评估脚本进行详细检查。")

    print("\n" + "=" * 60)


def generate_model_card(model_path: str):
    """
    生成 Model Card（模型卡片）。

    Model Card 是 HuggingFace 社区的标准文档格式，用于描述模型的：
    - 训练方法和数据
    - 使用方式
    - 局限性和注意事项

    这个文件会保存为 README.md 放在模型目录中。
    发布到 HuggingFace Hub 时，这个文件会显示在模型页面上。
    """
    card_content = """---
license: apache-2.0
language:
- en
- zh
pipeline_tag: text-generation
tags:
- function-calling
- tool-use
- qwen3
- lora
- fine-tuned
base_model: Qwen/Qwen3-0.6B
datasets:
- Salesforce/xlam-function-calling-60k
library_name: transformers
---

# Qwen3-0.6B Function Calling (Fine-tuned)

## 模型简介 / Model Description

本模型基于 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 使用 LoRA 方法微调，
专门用于 **Function Calling（函数调用）** 任务。

This model is fine-tuned from [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) using LoRA,
specialized for **Function Calling** tasks.

## 训练详情 / Training Details

- **基座模型**: Qwen/Qwen3-0.6B
- **微调方法**: LoRA (Low-Rank Adaptation)
  - Rank: 32
  - Alpha: 64
  - Target: all linear layers
- **训练数据**: [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- **训练框架**: [LLaMA-Factory](https://github.com/hiyouga/LLamaFactory)
- **精度**: BF16
- **硬件**: NVIDIA GPU (12GB VRAM)

## 使用方法 / Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "your-name/qwen3-0.6b-fc"  # 替换为你的模型路径
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "You are a helpful assistant with access to tools..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## 使用 vLLM 部署 / Deploy with vLLM

```bash
python -m vllm.entrypoints.openai.api_server \\
    --model your-name/qwen3-0.6b-fc \\
    --port 8000
```

## 评估结果 / Evaluation Results

| 指标 / Metric | 数值 / Score |
|---|---|
| Parse Rate | TBD |
| Schema Hit Rate | TBD |
| Function Name Accuracy | TBD |
| Parameter F1 | TBD |
| Execution Rate | TBD |

## 局限性 / Limitations

- 模型参数量较小(0.6B)，复杂多轮对话能力有限
- 主要针对英文 Function Calling 场景优化
- 对于未见过的 API schema，泛化能力可能不足

## 引用 / Citation

如果本模型对你有帮助，欢迎引用：

```bibtex
@misc{qwen3-fc-finetuned,
  title={Qwen3-0.6B Function Calling Fine-tuned},
  year={2025},
  publisher={HuggingFace},
  howpublished={\\url{https://huggingface.co/your-name/qwen3-0.6b-fc}}
}
```
"""

    card_path = os.path.join(model_path, "MODEL_CARD.md")
    with open(card_path, "w", encoding="utf-8") as f:
        f.write(card_content.strip())

    print(f"\n  ✅ 模型卡片已保存到: {card_path}")
    print("  发布到 HuggingFace Hub 时，将此文件重命名为 README.md 即可。")


def main():
    """主函数"""
    args = parse_args()

    # 验证模型路径
    if not os.path.isdir(args.model_path):
        # 如果不是本地目录，可能是 HuggingFace 模型 ID
        print(f"[提示] {args.model_path} 不是本地目录，将尝试从 HuggingFace 下载。")

    actions = (
        ["verify", "info", "card"] if args.action == "all" else [args.action]
    )

    # 只在需要时加载模型（info 和 verify 需要，card 不需要）
    model = None
    tokenizer = None

    if "info" in actions or "verify" in actions:
        print("\n加载模型中...")
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map=args.device,
            trust_remote_code=True,
        )
        print("模型加载完成！")

    # 按顺序执行操作
    for action in actions:
        if action == "info" and model is not None and tokenizer is not None:
            print_model_info(model, tokenizer, args.model_path)
        elif action == "verify" and model is not None and tokenizer is not None:
            verify_model(model, tokenizer, args.device)
        elif action == "card":
            generate_model_card(args.model_path)

    print("\n所有操作完成！")


if __name__ == "__main__":
    main()
