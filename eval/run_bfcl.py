#!/usr/bin/env python3
"""
============================================================
run_bfcl.py - BFCL 基准评测集成脚本
============================================================
功能：在 Berkeley Function Calling Leaderboard (BFCL) 基准上评测模型。

背景知识：
-----------
BFCL 是什么？
    Berkeley Function Calling Leaderboard 是加州大学伯克利分校
    Gorilla 团队发布的函数调用能力评测基准。它包含多种真实世界
    的函数调用场景，用于全面评估模型的工具使用能力。

为什么要用 BFCL？
    1. 标准化评测：与其他模型的结果可以直接对比
    2. 覆盖全面：包含简单函数调用、并行调用、多步调用等场景
    3. 工业认可：BFCL 是业界广泛使用的函数调用评测标准

BFCL 评测类型：
    - Simple Function Call：单个函数调用
    - Multiple Function Call：同时调用多个函数
    - Parallel Function Call：并行调用相同函数
    - Function Relevance Detection：判断是否需要调用函数

注意：
    本脚本是一个简化版的 BFCL 集成，不是完整的 BFCL 评测流程。
    完整评测请参考：https://github.com/ShishirPatil/gorilla

使用方式：
-----------
    # 使用本地模型
    python eval/run_bfcl.py --model-path outputs/qwen3-0.6b-fc-merged

    # 使用 vLLM API
    python eval/run_bfcl.py --api-base http://localhost:8000/v1
============================================================
"""

import argparse
import json
import os
import sys
from pathlib import Path

from tqdm import tqdm

# BFCL 测试数据 URL（从 Gorilla 项目获取）
BFCL_DATA_URL = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/"
    "main/berkeley-function-call-leaderboard/data/"
)

# 我们评测的类别（简化版，只测最核心的）
BFCL_CATEGORIES = [
    "gorilla_openfunctions_v1_test_simple",   # 简单函数调用
    "gorilla_openfunctions_v1_test_multiple",  # 多函数调用
]


def download_bfcl_data(output_dir: str) -> dict[str, list[dict]]:
    """
    下载 BFCL 测试数据。

    原理：
        BFCL 的测试数据存放在 GitHub 上的 Gorilla 仓库中。
        我们下载 JSON 格式的测试集，每个类别一个文件。

    Args:
        output_dir: 数据保存目录

    Returns:
        {类别名: [测试样本列表]}
    """
    try:
        import urllib.request
    except ImportError:
        print("[错误] urllib 不可用")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    all_data = {}

    for category in BFCL_CATEGORIES:
        filepath = os.path.join(output_dir, f"{category}.json")

        if os.path.exists(filepath):
            print(f"  [✓] 已缓存: {category}")
            with open(filepath, "r", encoding="utf-8") as f:
                all_data[category] = json.load(f)
            continue

        url = f"{BFCL_DATA_URL}{category}.json"
        print(f"  [*] 下载: {category}...")

        try:
            urllib.request.urlretrieve(url, filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                all_data[category] = json.load(f)
            print(f"  [✓] 已下载: {len(all_data[category])} 条样本")
        except Exception as e:
            print(f"  [警告] 下载失败: {e}")
            print(f"  你可以手动下载: {url}")
            all_data[category] = []

    return all_data


def format_bfcl_prompt(sample: dict) -> tuple[str, str]:
    """
    将 BFCL 样本格式化为模型输入。

    BFCL 的样本格式可能与我们的训练格式不同，
    这里做一个适配转换。

    Args:
        sample: BFCL 测试样本

    Returns:
        (system_prompt, user_query)
    """
    # BFCL 样本通常包含：question 和 function（可用函数列表）
    question = sample.get("question", sample.get("prompt", ""))
    functions = sample.get("function", sample.get("functions", []))

    # 构建系统提示词（与训练时一致的格式）
    system = (
        "You are a helpful assistant with access to tools. "
        "When you need to call a function, output a JSON object with 'name' and 'arguments' fields.\n\n"
        "Available tools:\n"
        + json.dumps(functions, indent=2, ensure_ascii=False)
    )

    return system, question if isinstance(question, str) else json.dumps(question)


def run_bfcl_inference(
    samples: list[dict],
    model_path: str | None = None,
    api_base: str | None = None,
    model_name: str = "Qwen/Qwen3-0.6B",
    max_tokens: int = 512,
) -> list[str]:
    """
    对 BFCL 测试样本运行推理。

    Args:
        samples: BFCL 测试样本列表
        model_path: 本地模型路径（二选一）
        api_base: vLLM API 地址（二选一）
        model_name: 模型名称
        max_tokens: 最大生成 token 数

    Returns:
        预测文本列表
    """
    predictions = []

    if api_base:
        # API 推理
        try:
            from openai import OpenAI
        except ImportError:
            print("[错误] 请安装 openai: pip install openai")
            sys.exit(1)

        client = OpenAI(base_url=api_base, api_key="not-needed")

        for sample in tqdm(samples, desc="BFCL API 推理"):
            system, query = format_bfcl_prompt(sample)
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": query},
                    ],
                    max_tokens=max_tokens,
                    temperature=0,
                )
                pred = response.choices[0].message.content or ""
                predictions.append(pred.strip())
            except Exception as e:
                print(f"  [警告] API 调用失败: {e}")
                predictions.append("")

    else:
        # 本地推理
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
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        for sample in tqdm(samples, desc="BFCL 本地推理"):
            system, query = format_bfcl_prompt(sample)
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            pred = tokenizer.decode(generated, skip_special_tokens=True)
            predictions.append(pred.strip())

    return predictions


def evaluate_bfcl_results(
    predictions: list[str], samples: list[dict]
) -> dict:
    """
    评估 BFCL 预测结果。

    简化版评测：检查预测是否是有效的 JSON 函数调用，
    以及函数名是否在可用函数列表中。

    Args:
        predictions: 模型预测列表
        samples: BFCL 测试样本

    Returns:
        评测指标字典
    """
    from eval.metrics import parse_function_call

    total = len(predictions)
    parsed = 0
    name_correct = 0

    for pred, sample in zip(predictions, samples):
        call = parse_function_call(pred)
        if call is not None:
            parsed += 1
            # 检查函数名是否在可用函数中
            functions = sample.get("function", sample.get("functions", []))
            valid_names = {f.get("name", "") for f in functions}
            if call.get("name") in valid_names:
                name_correct += 1

    return {
        "total": total,
        "parse_rate": parsed / total if total > 0 else 0,
        "name_accuracy": name_correct / total if total > 0 else 0,
    }


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description="在 BFCL 基准上评测函数调用模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="本地模型路径",
    )
    parser.add_argument(
        "--api-base", type=str, default=None,
        help="vLLM API 地址（如 http://localhost:8000/v1）",
    )
    parser.add_argument(
        "--model-name", type=str, default="Qwen/Qwen3-0.6B",
        help="模型名称",
    )
    parser.add_argument(
        "--output-dir", type=str, default="eval/bfcl_results",
        help="结果输出目录",
    )
    parser.add_argument(
        "--data-dir", type=str, default="eval/bfcl_data",
        help="BFCL 数据缓存目录",
    )

    args = parser.parse_args()

    if args.api_base is None and args.model_path is None:
        print("[错误] 请指定 --model-path 或 --api-base")
        sys.exit(1)

    print("=" * 60)
    print("  BFCL 基准评测")
    print("=" * 60)
    print()

    # Step 1: 下载测试数据
    print("[*] 准备 BFCL 测试数据...")
    all_data = download_bfcl_data(args.data_dir)
    print()

    # Step 2: 逐类别评测
    os.makedirs(args.output_dir, exist_ok=True)
    overall_results = {}

    for category, samples in all_data.items():
        if not samples:
            print(f"[!] 跳过 {category}（无数据）")
            continue

        print(f"[*] 评测类别: {category} ({len(samples)} 条)")

        predictions = run_bfcl_inference(
            samples,
            model_path=args.model_path,
            api_base=args.api_base,
            model_name=args.model_name,
        )

        results = evaluate_bfcl_results(predictions, samples)
        overall_results[category] = results

        # 保存该类别的预测结果
        pred_file = os.path.join(args.output_dir, f"{category}_predictions.jsonl")
        with open(pred_file, "w", encoding="utf-8") as f:
            for i, pred in enumerate(predictions):
                f.write(json.dumps({"index": i, "prediction": pred}, ensure_ascii=False) + "\n")

        print(f"    解析率: {results['parse_rate']:.4f}")
        print(f"    函数名准确率: {results['name_accuracy']:.4f}")
        print()

    # Step 3: 汇总结果
    print("=" * 60)
    print("  BFCL 评测汇总")
    print("=" * 60)
    print()
    print(f"  {'类别':<45} {'解析率':>8} {'名称准确率':>10}")
    print(f"  {'-'*45} {'-'*8} {'-'*10}")
    for cat, res in overall_results.items():
        print(f"  {cat:<45} {res['parse_rate']:>8.4f} {res['name_accuracy']:>10.4f}")
    print()

    # 保存汇总
    summary_file = os.path.join(args.output_dir, "bfcl_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, ensure_ascii=False, indent=2)
    print(f"[✓] 汇总结果保存到: {summary_file}")
    print()


if __name__ == "__main__":
    main()
