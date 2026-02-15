#!/usr/bin/env python3
"""
============================================================
evaluate.py - 离线评测主脚本
============================================================
功能：读取模型的预测结果，与标准答案对比，计算各项评测指标。

背景知识：
-----------
"离线评测"是指在模型训练完成后，用一组测试数据来评估模型的质量。
与"在线评测"（实时测试服务性能）不同，离线评测关注的是：
- 模型的输出格式是否正确？
- 调用的函数是否正确？
- 参数是否填对了？

使用流程：
-----------
1. 先用 run_inference.py 生成预测结果（保存为 JSONL 文件）
2. 运行本脚本计算指标

使用方式：
-----------
    python eval/evaluate.py \\
        --predictions eval/predictions.jsonl \\
        --ground-truth data/processed/val.json \\
        --output eval/results.json

输入文件格式：
-----------
predictions.jsonl（每行一个 JSON 对象）：
    {"prediction": "模型输出的文本", "index": 0}
    {"prediction": "模型输出的文本", "index": 1}

val.json（LLaMA-Factory sharegpt 格式的数组）：
    [{"conversations": [...], "system": "...", "tools": "..."}, ...]
============================================================
"""

import argparse
import json
import sys
from pathlib import Path

# 导入我们的指标计算模块
from eval.metrics import compute_all_metrics, parse_function_call


def load_predictions(filepath: str) -> list[str]:
    """
    从 JSONL 文件加载模型预测结果。

    JSONL = JSON Lines，每行一个 JSON 对象。
    这种格式适合大量数据，因为不需要一次性读入整个数组。

    Args:
        filepath: JSONL 文件路径

    Returns:
        预测文本列表
    """
    predictions = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                predictions.append(obj.get("prediction", ""))
            except json.JSONDecodeError:
                predictions.append("")
    return predictions


def load_ground_truth(filepath: str) -> tuple[list[dict], list[list[dict]]]:
    """
    从验证集文件加载标准答案和工具列表。

    解析 sharegpt 格式的数据，提取：
    1. 标准答案（function_call 角色的内容）
    2. 可用工具列表

    Args:
        filepath: 验证集 JSON 文件路径

    Returns:
        (references, tools_list)
        - references: 标准答案列表，每个元素是 {"name": ..., "arguments": ...}
        - tools_list: 每个样本对应的工具列表
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = []
    tools_list = []

    for sample in data:
        conversations = sample.get("conversations", [])
        tools_str = sample.get("tools", "[]")

        # 解析工具列表
        try:
            tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        except json.JSONDecodeError:
            tools = []
        tools_list.append(tools)

        # 从对话中提取第一个 function_call 作为标准答案
        ref = {"name": "", "arguments": {}}
        for conv in conversations:
            if conv.get("from") == "function_call":
                parsed = parse_function_call(conv.get("value", ""))
                if parsed is not None:
                    ref = parsed
                break  # 只取第一个 function_call

        references.append(ref)

    return references, tools_list


def print_results(results: dict) -> None:
    """
    格式化打印评测结果。

    用表格形式展示各项指标，方便阅读。
    """
    print()
    print("=" * 60)
    print("  离线评测结果")
    print("=" * 60)
    print()
    print(f"  {'指标':<25} {'值':>10}")
    print(f"  {'-'*25} {'-'*10}")

    # 指标名称映射（英文→中文）
    metric_names = {
        "parse_rate": "解析成功率 (Parse Rate)",
        "func_name_accuracy": "函数名准确率 (Name Acc)",
        "param_precision": "参数精确率 (Precision)",
        "param_recall": "参数召回率 (Recall)",
        "param_f1": "参数 F1 值 (F1)",
        "schema_hit_rate": "Schema 命中率",
        "exec_rate": "可执行率 (Exec Rate)",
    }

    for key, name in metric_names.items():
        value = results.get(key)
        if value is None:
            print(f"  {name:<25} {'N/A':>10}")
        else:
            print(f"  {name:<25} {value:>10.4f}")

    print()
    print("=" * 60)

    # 简要解读
    parse_r = results.get("parse_rate", 0)
    if parse_r >= 0.95:
        print("  📊 解析率 ≥ 95%，模型已很好地学会了 JSON 输出格式")
    elif parse_r >= 0.80:
        print("  📊 解析率在 80-95%，格式学习基本到位，仍有改进空间")
    else:
        print("  📊 解析率 < 80%，模型尚未充分学会正确格式，建议增加训练")

    print()


def main():
    """主函数"""

    parser = argparse.ArgumentParser(
        description="评测函数调用模型的离线质量指标",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python eval/evaluate.py --predictions eval/predictions.jsonl
  python eval/evaluate.py --predictions eval/predictions.jsonl --ground-truth data/processed/val.json
        """,
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="模型预测结果文件路径（JSONL 格式）",
    )
    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/processed/val.json",
        help="标准答案文件路径（默认: data/processed/val.json）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/results.json",
        help="评测结果输出路径（默认: eval/results.json）",
    )

    args = parser.parse_args()

    # Step 1: 检查文件是否存在
    if not Path(args.predictions).exists():
        print(f"[错误] 预测文件不存在: {args.predictions}")
        print("请先运行: python eval/run_inference.py")
        sys.exit(1)

    if not Path(args.ground_truth).exists():
        print(f"[错误] 标准答案文件不存在: {args.ground_truth}")
        print("请先运行: python scripts/prepare_data.py")
        sys.exit(1)

    # Step 2: 加载数据
    print("[*] 加载预测结果...")
    predictions = load_predictions(args.predictions)
    print(f"    共 {len(predictions)} 条预测")

    print("[*] 加载标准答案...")
    references, tools_list = load_ground_truth(args.ground_truth)
    print(f"    共 {len(references)} 条标准答案")

    # 数量对齐（取较短的）
    n = min(len(predictions), len(references))
    if len(predictions) != len(references):
        print(f"    [警告] 预测数量({len(predictions)}) ≠ 标准答案数量({len(references)})，取前 {n} 条")
    predictions = predictions[:n]
    references = references[:n]
    tools_list = tools_list[:n]

    # Step 3: 计算指标
    print("[*] 计算评测指标...")
    results = compute_all_metrics(predictions, references, tools_list)

    # Step 4: 打印结果
    print_results(results)

    # Step 5: 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[✓] 评测结果已保存到: {output_path}")
    print()


if __name__ == "__main__":
    main()
