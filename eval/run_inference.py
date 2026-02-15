#!/usr/bin/env python3
"""
============================================================
run_inference.py - 模型推理脚本
============================================================
功能：用训练好的模型对测试数据进行推理，生成预测结果。

背景知识：
-----------
"推理"（Inference）= 用训练好的模型来生成输出。
训练是"教模型"，推理是"让模型答题"。

本脚本支持两种推理方式：
1. 本地推理：直接加载模型到 GPU（适合单机测试）
2. API 推理：通过 vLLM 的 OpenAI 兼容接口（适合已部署的服务）

使用方式：
-----------
    # 方式 1：使用合并后的模型（无 LoRA）
    python eval/run_inference.py --model-path outputs/qwen3-0.6b-fc-merged

    # 方式 2：使用基础模型 + LoRA adapter
    python eval/run_inference.py --model-path Qwen/Qwen3-0.6B --adapter-path outputs/qwen3-0.6b-fc-lora

    # 方式 3：使用 vLLM API
    python eval/run_inference.py --api-base http://localhost:8000/v1

输出：
-----------
    eval/predictions.jsonl（每行一个预测结果）
============================================================
"""

import argparse
import json
import sys
from pathlib import Path

# tqdm 提供进度条
from tqdm import tqdm


# ============================================================
# 系统提示词（必须与训练时一致！）
# ============================================================
SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "When you need to call a function, output a JSON object with 'name' and 'arguments' fields."
)


def load_test_data(filepath: str, max_samples: int | None = None) -> list[dict]:
    """
    加载测试数据。

    从验证集文件中读取样本，提取用户问题和工具列表。

    Args:
        filepath: 验证集 JSON 文件路径
        max_samples: 最多加载多少条（None = 全部）

    Returns:
        测试样本列表，每个元素包含 query 和 tools
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if max_samples is not None:
        data = data[:max_samples]

    samples = []
    for item in data:
        conversations = item.get("conversations", [])
        # 提取用户问题（第一个 human 角色）
        query = ""
        for conv in conversations:
            if conv.get("from") == "human":
                query = conv.get("value", "")
                break

        tools_str = item.get("tools", "[]")
        system = item.get("system", SYSTEM_PROMPT)

        samples.append({
            "query": query,
            "tools": tools_str,
            "system": system,
        })

    return samples


def run_local_inference(
    model_path: str,
    adapter_path: str | None,
    samples: list[dict],
    max_new_tokens: int = 512,
) -> list[str]:
    """
    使用本地模型进行推理。

    原理：
        1. 加载预训练模型到 GPU
        2. 如果有 LoRA adapter，加载并合并
        3. 对每个样本，构建输入提示词
        4. 用模型生成输出文本

    什么是 LoRA adapter？
        LoRA 训练后保存的不是完整模型，而是一组小矩阵（adapter）。
        推理时需要先加载基础模型，再叠加 adapter 的参数。

    Args:
        model_path: 模型路径（HuggingFace 模型名或本地路径）
        adapter_path: LoRA adapter 路径（可选）
        samples: 测试样本列表
        max_new_tokens: 最大生成 token 数

    Returns:
        预测文本列表
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[错误] 请安装 transformers 和 torch")
        sys.exit(1)

    print(f"[*] 加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,  # 使用 bf16 节省显存
        device_map="auto",           # 自动选择 GPU
        trust_remote_code=True,
    )

    # 加载 LoRA adapter（如果指定了的话）
    if adapter_path:
        print(f"[*] 加载 LoRA adapter: {adapter_path}")
        try:
            from peft import PeftModel
        except ImportError:
            print("[错误] 请安装 peft: pip install peft")
            sys.exit(1)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # 合并 LoRA 参数
        print("[✓] LoRA adapter 已合并")

    model.eval()  # 切换到评估模式（关闭 dropout 等）

    predictions = []

    for sample in tqdm(samples, desc="推理进度"):
        # 构建对话消息
        messages = [
            {"role": "system", "content": sample["system"]},
            {"role": "user", "content": sample["query"]},
        ]

        # 使用 tokenizer 的 chat template 格式化输入
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():  # 推理时不需要计算梯度
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,    # 贪心解码（确定性输出）
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )

        # 只取新生成的部分（去掉输入）
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        prediction = tokenizer.decode(generated, skip_special_tokens=True)
        predictions.append(prediction.strip())

    return predictions


def run_api_inference(
    api_base: str,
    model_name: str,
    samples: list[dict],
    max_tokens: int = 512,
) -> list[str]:
    """
    通过 vLLM 的 OpenAI 兼容 API 进行推理。

    原理：
        vLLM 启动后会提供一个与 OpenAI API 格式兼容的接口。
        我们可以像调用 GPT API 一样调用自己部署的模型。

    优势：
        - 不需要在本地加载模型（适合模型已部署到服务器的情况）
        - vLLM 的推理速度比 transformers 快很多

    Args:
        api_base: API 基地址（如 http://localhost:8000/v1）
        model_name: 模型名称（vLLM 中注册的名字）
        samples: 测试样本列表
        max_tokens: 最大生成 token 数

    Returns:
        预测文本列表
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("[错误] 请安装 openai: pip install openai")
        sys.exit(1)

    # 创建 OpenAI 客户端，指向我们的 vLLM 服务
    client = OpenAI(
        base_url=api_base,
        api_key="not-needed",  # vLLM 默认不需要 API key
    )

    predictions = []

    for sample in tqdm(samples, desc="API 推理进度"):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sample["system"]},
                    {"role": "user", "content": sample["query"]},
                ],
                max_tokens=max_tokens,
                temperature=0,  # 贪心解码
            )
            prediction = response.choices[0].message.content or ""
            predictions.append(prediction.strip())
        except Exception as e:
            print(f"  [警告] API 调用失败: {e}")
            predictions.append("")

    return predictions


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description="用模型对测试数据进行推理，生成预测结果",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 本地模型推理
  python eval/run_inference.py --model-path outputs/qwen3-0.6b-fc-merged

  # 本地模型 + LoRA adapter
  python eval/run_inference.py --model-path Qwen/Qwen3-0.6B --adapter-path outputs/qwen3-0.6b-fc-lora

  # vLLM API 推理
  python eval/run_inference.py --api-base http://localhost:8000/v1 --model-name Qwen/Qwen3-0.6B
        """,
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="本地模型路径或 HuggingFace 模型名",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="LoRA adapter 路径（可选，仅本地推理时使用）",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=None,
        help="vLLM API 基地址（如 http://localhost:8000/v1）",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="API 推理时的模型名称（默认: Qwen/Qwen3-0.6B）",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/processed/val.json",
        help="测试数据路径（默认: data/processed/val.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/predictions.jsonl",
        help="预测结果输出路径（默认: eval/predictions.jsonl）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最多推理多少条样本（默认: 全部）",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="最大生成 token 数（默认: 512）",
    )

    args = parser.parse_args()

    # 检查推理方式
    if args.api_base is None and args.model_path is None:
        print("[错误] 请指定 --model-path（本地推理）或 --api-base（API 推理）")
        sys.exit(1)

    # 加载测试数据
    if not Path(args.test_data).exists():
        print(f"[错误] 测试数据不存在: {args.test_data}")
        print("请先运行: python scripts/prepare_data.py")
        sys.exit(1)

    print("[*] 加载测试数据...")
    samples = load_test_data(args.test_data, args.max_samples)
    print(f"    共 {len(samples)} 条样本")
    print()

    # 执行推理
    if args.api_base:
        print(f"[*] 使用 API 推理: {args.api_base}")
        predictions = run_api_inference(
            args.api_base, args.model_name, samples, args.max_new_tokens
        )
    else:
        print(f"[*] 使用本地推理: {args.model_path}")
        predictions = run_local_inference(
            args.model_path, args.adapter_path, samples, args.max_new_tokens
        )

    # 保存预测结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, pred in enumerate(predictions):
            json_line = json.dumps(
                {"index": i, "prediction": pred}, ensure_ascii=False
            )
            f.write(json_line + "\n")

    print()
    print(f"[✓] 预测结果已保存到: {output_path}")
    print(f"    共 {len(predictions)} 条预测")
    print()
    print("下一步：")
    print(f"  python eval/evaluate.py --predictions {output_path}")
    print()


if __name__ == "__main__":
    main()
