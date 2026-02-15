#!/usr/bin/env python3
"""
============================================================
prepare_data.py - 数据准备脚本
============================================================
功能：下载 xlam-function-calling-60k 数据集，并转换为 LLaMA-Factory 可用的格式。

背景知识：
-----------
1. xlam-60k 数据集是 Salesforce 用 APIGen 方法自动生成的函数调用训练数据。
   每条数据包含：用户问题(query)、可用工具列表(tools)、标准答案(answers)。

2. LLaMA-Factory 支持多种数据格式，我们使用 "sharegpt" 格式，它模拟多轮对话：
   - system: 系统提示词（包含可用工具的 JSON Schema）
   - user: 用户的自然语言问题
   - assistant: 模型应该输出的工具调用（JSON 格式）

3. 为什么要做格式转换？
   不同框架对数据格式有不同要求。xlam-60k 的原始格式和 LLaMA-Factory
   需要的 sharegpt 格式不同，所以需要一个转换步骤。

使用方式：
-----------
    python scripts/prepare_data.py

输出文件：
-----------
    data/processed/train.json  - 训练集（90%）
    data/processed/val.json    - 验证集（10%）

注意事项：
-----------
    - xlam-60k 是 gated 数据集，需要先在 HuggingFace 上同意使用条款
    - 首次运行需要 HuggingFace token：huggingface-cli login
============================================================
"""

import json
import os
import random
from pathlib import Path

# tqdm 提供进度条，让你知道处理进度
from tqdm import tqdm

# ============================================================
# 配置参数
# ============================================================

# 数据集名称（HuggingFace 上的路径）
DATASET_NAME_MAIN = "Salesforce/xlam-function-calling-60k"
DATASET_NAME_BACKUP = "NousResearch/hermes-function-calling-v1"

# 输出目录
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "processed"

# 训练集/验证集划分比例（90% 训练，10% 验证）
# 为什么要划分？验证集用来在训练过程中评估模型效果，
# 防止模型"死记硬背"训练数据（即过拟合）。
TRAIN_RATIO = 0.9

# 随机种子（保证每次划分结果一致，方便复现实验）
RANDOM_SEED = 42

# 系统提示词模板
# 这个模板告诉模型：你是一个函数调用助手，下面是可用的工具列表。
# 模型看到这个提示后，就知道自己需要从工具列表中选择合适的函数并填入参数。
SYSTEM_PROMPT_TEMPLATE = (
    "You are a helpful assistant with access to tools. "
    "When you need to call a function, output a JSON object with 'name' and 'arguments' fields."
)


def load_dataset_from_hf():
    """
    从 HuggingFace 下载 xlam-60k 数据集。

    原理：HuggingFace datasets 库会自动处理数据集的下载、缓存和加载。
    第一次运行会从网上下载（需要网络），之后会使用本地缓存。

    Returns:
        dataset: HuggingFace Dataset 对象
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("[错误] 请先安装 datasets 库: pip install datasets")
        raise

    print(f"[*] 正在从 HuggingFace 加载数据集: {DATASET_NAME_MAIN}")
    print("    （首次下载可能需要几分钟，取决于网速）")
    print("    （如果提示需要登录，请运行: huggingface-cli login）")
    print()

    try:
        dataset = load_dataset(DATASET_NAME_MAIN, split="train")
        print(f"[✓] xLAM 数据集加载完成，共 {len(dataset)} 条样本")
        return dataset, "xlam"
    except Exception as e:
        print(f"[!] xLAM 数据集加载失败: {e}")
        print(f"[*] 尝试加载备用数据集: {DATASET_NAME_BACKUP}")
        dataset = load_dataset(DATASET_NAME_BACKUP, split="train")
        print(f"[✓] Hermes 数据集加载完成，共 {len(dataset)} 条样本")
        return dataset, "hermes"


def convert_hermes_sample(sample: dict) -> dict | None:
    """
    将 Hermes 格式转换为 ShareGPT 格式
    Hermes 格式:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "<tool_code>...</tool_code>"}
        ]
    }
    """
    # Hermes 已经是 ShareGPT 格式，但可能需要清洗或调整
    # 这里直接返回，假设格式兼容
    return sample

def convert_sample(sample: dict, dataset_type: str = "xlam") -> dict | None:
    if dataset_type == "hermes":
        return convert_hermes_sample(sample)
    
    # xlam 逻辑保持不变...
    """
    将 xlam-60k 的一条样本转换为 LLaMA-Factory sharegpt 格式。

    xlam-60k 原始格式（每条样本的字段）：
    {
        "query": "用户的自然语言问题",
        "tools": "[{\"name\": \"func1\", \"description\": \"...\", \"parameters\": {...}}, ...]",
        "answers": "[{\"name\": \"func1\", \"arguments\": {\"arg1\": \"value1\"}}]"
    }

     LLaMA-Factory sharegpt 格式（使用 function_call 角色）：
    {
        "conversations": [
            {"from": "human", "value": "用户问题"},
            {"from": "function_call", "value": "工具调用的 JSON"},
            {"from": "observation", "value": "工具执行结果（可选）"},
            {"from": "gpt", "value": "最终回复"}
        ],
        "system": "系统提示词",
        "tools": "工具列表 JSON 字符串"
    }

    注意：我们同时支持两种格式输出，默认使用 function_call 格式，
    因为它能让模型更好地区分"工具调用"和"普通回复"。

    Args:
        sample: xlam-60k 的一条原始样本

    Returns:
        转换后的 sharegpt 格式样本，如果转换失败则返回 None
    """
    try:
        query = sample["query"]
        tools_str = sample["tools"]
        answers_str = sample["answers"]

        # 解析 tools 和 answers（可能是 JSON 字符串或已解析的对象）
        tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
        answers = json.loads(answers_str) if isinstance(answers_str, str) else answers_str

        if not answers:
            return None

        # 构建对话序列
        # LLaMA-Factory 要求严格交替：human → function_call → observation → gpt
        conversations = [{"from": "human", "value": query}]

        # 为每个工具调用添加 function_call + observation 对
        # 多工具调用时，每次调用都需要单独的 function_call 和 observation
        for i, answer in enumerate(answers):
            function_call_obj = {
                "name": answer["name"],
                "arguments": answer.get("arguments", {}),
            }
            conversations.append({
                "from": "function_call",
                "value": json.dumps(function_call_obj, ensure_ascii=False),
            })
            conversations.append({
                "from": "observation",
                "value": json.dumps({"status": "success"}, ensure_ascii=False),
            })

        # 最后的 gpt 回复
        tool_names = [a["name"] for a in answers]
        if len(answers) == 1:
            gpt_response = f"I've called {tool_names[0]} with the specified parameters."
        else:
            names_str = ", ".join(tool_names)
            gpt_response = f"I've called {len(answers)} functions: {names_str}."

        conversations.append({"from": "gpt", "value": gpt_response})

        # 组装最终格式
        converted = {
            "conversations": conversations,
            "system": SYSTEM_PROMPT_TEMPLATE,
            "tools": json.dumps(tools, ensure_ascii=False),
        }

        return converted

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def main():
    """主函数：加载数据 → 转换格式 → 划分训练/验证集 → 保存"""

    # Step 1: 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] 输出目录: {OUTPUT_DIR}")
    print()

    # Step 2: 加载数据集
    dataset, dataset_type = load_dataset_from_hf()
    print()

    # Step 3: 转换格式
    print("[*] 正在转换数据格式...")
    converted_data = []
    skipped = 0

    for sample in tqdm(dataset, desc="转换进度"):
        result = convert_sample(sample, dataset_type)
        if result is not None:
            converted_data.append(result)
        else:
            skipped += 1

    print(f"[✓] 转换完成: {len(converted_data)} 条成功, {skipped} 条跳过")
    print()

    # Step 4: 打印一条样本示例（让你确认格式是否正确）
    if converted_data:
        print("[*] 样本示例（第一条）:")
        print("-" * 60)
        sample = converted_data[0]
        for conv in sample["conversations"]:
            role = conv["from"]
            value = conv["value"]
            # 截断过长的内容，只显示前 200 个字符
            if len(value) > 200:
                value = value[:200] + "..."
            print(f"  [{role}]: {value}")
        print("-" * 60)
        print()

    # Step 5: 随机划分训练集和验证集
    print("[*] 划分训练集和验证集...")
    random.seed(RANDOM_SEED)
    random.shuffle(converted_data)

    split_idx = int(len(converted_data) * TRAIN_RATIO)
    train_data = converted_data[:split_idx]
    val_data = converted_data[split_idx:]

    print(f"    训练集: {len(train_data)} 条 ({TRAIN_RATIO*100:.0f}%)")
    print(f"    验证集: {len(val_data)} 条 ({(1-TRAIN_RATIO)*100:.0f}%)")
    print()

    # Step 6: 保存为 JSON 文件
    train_path = OUTPUT_DIR / "train.json"
    val_path = OUTPUT_DIR / "val.json"

    print(f"[*] 保存训练集到: {train_path}")
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    print(f"[*] 保存验证集到: {val_path}")
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print()
    print("[✓] 数据准备完成！")
    print()
    print("下一步：")
    print("  1. 检查数据: head -50 data/processed/train.json")
    print("  2. 开始训练: bash scripts/train.sh")


if __name__ == "__main__":
    main()
