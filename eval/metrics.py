#!/usr/bin/env python3
"""
============================================================
metrics.py - 函数调用评测指标计算模块
============================================================
功能：计算模型在函数调用任务上的各项离线评测指标。

背景知识：
-----------
评测指标是衡量模型好坏的"尺子"。对于函数调用任务，我们关心：
1. 模型输出的格式对不对？（能否解析为 JSON）
2. 调用的函数名对不对？
3. 参数填得对不对？
4. 能不能真正执行？

每个指标从不同角度衡量模型能力，综合起来才能全面评估。

包含的指标：
-----------
1. parse_rate     - 解析成功率（JSON 格式是否正确）
2. func_name_accuracy - 函数名准确率
3. schema_hit_rate   - Schema 命中率（字段是否齐全、类型是否正确）
4. param_f1         - 参数级 F1 值（精确率、召回率的调和平均）
5. exec_rate        - 可执行率（stub 函数能否执行）
============================================================
"""

import json
from collections import Counter


def parse_function_call(text: str) -> dict | None:
    """
    尝试将文本解析为函数调用对象。

    原理：
        函数调用的标准格式是 JSON：{"name": "函数名", "arguments": {"参数": "值"}}
        这个函数尝试将模型的文本输出解析为这种格式。
        如果解析失败，说明模型输出的格式有问题。

    为什么需要这个函数？
        模型输出的是纯文本字符串，我们需要把它转成结构化的 Python 字典，
        才能进一步检查函数名、参数等是否正确。

    Args:
        text: 模型输出的文本字符串

    Returns:
        解析成功返回 {"name": str, "arguments": dict}
        解析失败返回 None

    示例:
        >>> parse_function_call('{"name": "get_weather", "arguments": {"city": "北京"}}')
        {'name': 'get_weather', 'arguments': {'city': '北京'}}

        >>> parse_function_call('这不是 JSON')
        None
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # 尝试直接解析整个文本为 JSON
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "name" in parsed:
            # 确保有 arguments 字段，如果没有就补一个空的
            if "arguments" not in parsed:
                parsed["arguments"] = {}
            return parsed
    except json.JSONDecodeError:
        pass

    # 有时候模型会在 JSON 前后加一些文字，尝试提取 JSON 部分
    # 找到第一个 { 和最后一个 }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            parsed = json.loads(text[start : end + 1])
            if isinstance(parsed, dict) and "name" in parsed:
                if "arguments" not in parsed:
                    parsed["arguments"] = {}
                return parsed
        except json.JSONDecodeError:
            pass

    return None


def parse_rate(predictions: list[str]) -> float:
    """
    计算解析成功率（Parse Rate）。

    原理：
        最基本的指标——模型的输出能不能被解析为有效的 JSON 函数调用？
        如果连 JSON 格式都不对，那后续的评测都没有意义。
        这个指标衡量的是"模型是否学会了正确的输出格式"。

    为什么这个指标重要？
        在生产环境中，如果模型输出的 JSON 格式有错（比如缺少引号、括号不匹配），
        后续的函数执行系统就无法解析，导致整个流程失败。
        解析成功率直接决定了系统的可用性。

    Args:
        predictions: 模型输出的文本列表

    Returns:
        解析成功的比例（0.0 到 1.0）

    示例:
        >>> parse_rate(['{"name": "f1", "arguments": {}}', '无效JSON', '{"name": "f2", "arguments": {}}'])
        0.6666666666666666
    """
    if not predictions:
        return 0.0

    success = sum(1 for p in predictions if parse_function_call(p) is not None)
    return success / len(predictions)


def func_name_accuracy(
    predictions: list[dict], references: list[dict]
) -> float:
    """
    计算函数名准确率。

    原理：
        检查模型预测的函数名是否与标准答案完全一致。
        这是最直接的指标——你调用的函数对不对？

    为什么要单独衡量函数名？
        即使参数全错，只要函数名对了，至少说明模型理解了用户想做什么。
        函数名错误意味着完全误解了用户意图。

    Args:
        predictions: 已解析的预测结果列表，每个元素是 {"name": ..., "arguments": ...}
        references:  标准答案列表，格式相同

    Returns:
        函数名完全匹配的比例（0.0 到 1.0）

    示例:
        >>> preds = [{"name": "get_weather", "arguments": {}}]
        >>> refs = [{"name": "get_weather", "arguments": {"city": "北京"}}]
        >>> func_name_accuracy(preds, refs)
        1.0
    """
    if not predictions or not references:
        return 0.0

    # 取两个列表中较短的长度（防止长度不一致）
    n = min(len(predictions), len(references))
    correct = sum(
        1
        for i in range(n)
        if predictions[i].get("name") == references[i].get("name")
    )
    return correct / n


def param_f1(
    predictions: list[dict], references: list[dict]
) -> dict:
    """
    计算参数级 F1 值。

    原理：
        F1 = 2 × Precision × Recall / (Precision + Recall)

        - Precision（精确率）：模型预测的参数中，有多少是正确的？
          高精确率 = 模型不会瞎填参数
        - Recall（召回率）：标准答案的参数中，有多少被模型预测到了？
          高召回率 = 模型不会漏掉参数
        - F1 是两者的调和平均，综合衡量精确率和召回率

    为什么用 F1 而不是准确率？
        准确率（Accuracy）只看"全对还是全错"，太粗糙了。
        F1 可以衡量"对了几分之几"，更适合参数可能有多个的情况。

    计算方式：
        把每个参数的 key=value 对视为一个"项目"，
        计算预测集和标准答案集之间的交集大小。

    Args:
        predictions: 已解析的预测结果列表
        references:  标准答案列表

    Returns:
        {"precision": float, "recall": float, "f1": float}

    示例:
        >>> pred = [{"name": "f", "arguments": {"a": 1, "b": 2, "c": 3}}]
        >>> ref = [{"name": "f", "arguments": {"a": 1, "b": 2, "d": 4}}]
        >>> result = param_f1(pred, ref)
        >>> # precision = 2/3 (a,b正确, c错误), recall = 2/3 (a,b命中, d未命中)
    """
    if not predictions or not references:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    total_precision_num = 0  # 精确率分子（正确预测数）
    total_precision_den = 0  # 精确率分母（总预测数）
    total_recall_num = 0     # 召回率分子（正确命中数）
    total_recall_den = 0     # 召回率分母（标准答案总数）

    n = min(len(predictions), len(references))

    for i in range(n):
        pred_args = predictions[i].get("arguments", {})
        ref_args = references[i].get("arguments", {})

        # 将参数转为 (key, value_str) 的集合，方便比较
        # 用 json.dumps 统一值的格式（处理数字、列表等类型差异）
        pred_pairs = {
            (k, json.dumps(v, sort_keys=True, ensure_ascii=False))
            for k, v in pred_args.items()
        }
        ref_pairs = {
            (k, json.dumps(v, sort_keys=True, ensure_ascii=False))
            for k, v in ref_args.items()
        }

        # 交集 = 预测正确的参数
        correct = pred_pairs & ref_pairs

        total_precision_num += len(correct)
        total_precision_den += len(pred_pairs)
        total_recall_num += len(correct)
        total_recall_den += len(ref_pairs)

    # 计算精确率和召回率
    precision = (
        total_precision_num / total_precision_den if total_precision_den > 0 else 0.0
    )
    recall = (
        total_recall_num / total_recall_den if total_recall_den > 0 else 0.0
    )

    # 计算 F1
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def schema_hit_rate(
    predictions: list[dict], tools_list: list[list[dict]]
) -> float:
    """
    计算 Schema 命中率。

    原理：
        检查模型调用的函数是否在可用工具列表中，
        以及是否填写了所有必填参数（required 字段）。

    为什么这个指标重要？
        模型可能会"幻想"一个不存在的函数名，或者漏掉必填参数。
        Schema 命中率衡量的是"模型是否遵守了工具的使用规范"。

    检查步骤：
        1. 函数名是否在工具列表中？
        2. 所有 required 参数是否都有提供？
        3. 如果都满足，则命中

    Args:
        predictions: 已解析的预测结果列表
        tools_list:  每个样本对应的可用工具列表
                     tools_list[i] 是第 i 个样本的工具定义数组

    Returns:
        Schema 命中的比例（0.0 到 1.0）

    示例:
        >>> tools = [[{"name": "get_weather", "parameters": {"required": ["city"]}}]]
        >>> preds = [{"name": "get_weather", "arguments": {"city": "北京"}}]
        >>> schema_hit_rate(preds, tools)
        1.0
    """
    if not predictions or not tools_list:
        return 0.0

    n = min(len(predictions), len(tools_list))
    hit = 0

    for i in range(n):
        pred = predictions[i]
        tools = tools_list[i]
        pred_name = pred.get("name", "")
        pred_args = pred.get("arguments", {})

        # 在工具列表中查找匹配的函数定义
        matched_tool = None
        for tool in tools:
            if tool.get("name") == pred_name:
                matched_tool = tool
                break

        if matched_tool is None:
            # 函数名不在工具列表中
            continue

        # 检查必填参数是否齐全
        params = matched_tool.get("parameters", {})
        required = params.get("required", [])

        all_required_present = all(r in pred_args for r in required)
        if all_required_present:
            hit += 1

    return hit / n


def exec_rate(
    predictions: list[dict], tools_list: list[list[dict]]
) -> float:
    """
    计算可执行率。

    原理：
        模拟执行函数调用：检查所有必填参数是否存在，
        并且参数类型是否符合 Schema 定义。

    为什么要检查可执行性？
        即使函数名对了、参数名对了，如果参数类型错了（比如应该是数字却传了字符串），
        实际执行时也会报错。可执行率衡量"这个调用真的能跑通吗？"

    类型检查规则：
        - string: 值必须是字符串
        - integer/number: 值必须是数字
        - boolean: 值必须是布尔值
        - array: 值必须是列表
        - object: 值必须是字典
        - 如果 Schema 没有定义类型，跳过类型检查

    Args:
        predictions: 已解析的预测结果列表
        tools_list:  每个样本对应的可用工具列表

    Returns:
        可执行的比例（0.0 到 1.0）
    """
    if not predictions or not tools_list:
        return 0.0

    # JSON Schema 类型到 Python 类型的映射
    TYPE_MAP = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    n = min(len(predictions), len(tools_list))
    executable = 0

    for i in range(n):
        pred = predictions[i]
        tools = tools_list[i]
        pred_name = pred.get("name", "")
        pred_args = pred.get("arguments", {})

        # 找到匹配的工具
        matched_tool = None
        for tool in tools:
            if tool.get("name") == pred_name:
                matched_tool = tool
                break

        if matched_tool is None:
            continue

        params = matched_tool.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        # 检查 1：所有必填参数是否存在
        if not all(r in pred_args for r in required):
            continue

        # 检查 2：参数类型是否正确
        type_ok = True
        for param_name, param_value in pred_args.items():
            if param_name in properties:
                expected_type_str = properties[param_name].get("type", "")
                if expected_type_str in TYPE_MAP:
                    expected_type = TYPE_MAP[expected_type_str]
                    if not isinstance(param_value, expected_type):
                        type_ok = False
                        break

        if type_ok:
            executable += 1

    return executable / n


def compute_all_metrics(
    raw_predictions: list[str],
    references: list[dict],
    tools_list: list[list[dict]] | None = None,
) -> dict:
    """
    一次性计算所有指标。

    这是一个便捷函数，调用上面的各个指标函数，汇总结果。

    Args:
        raw_predictions: 模型的原始文本输出列表
        references:      标准答案列表（已解析的字典）
        tools_list:      每个样本的工具定义列表（可选）

    Returns:
        包含所有指标的字典
    """
    results = {}

    # 1. 解析成功率
    results["parse_rate"] = parse_rate(raw_predictions)

    # 2. 解析所有预测
    parsed_predictions = []
    for p in raw_predictions:
        parsed = parse_function_call(p)
        if parsed is not None:
            parsed_predictions.append(parsed)
        else:
            # 解析失败的用空结果占位
            parsed_predictions.append({"name": "", "arguments": {}})

    # 3. 函数名准确率
    results["func_name_accuracy"] = func_name_accuracy(parsed_predictions, references)

    # 4. 参数 F1
    f1_results = param_f1(parsed_predictions, references)
    results["param_precision"] = f1_results["precision"]
    results["param_recall"] = f1_results["recall"]
    results["param_f1"] = f1_results["f1"]

    # 5. Schema 命中率和可执行率（需要工具列表）
    if tools_list is not None:
        results["schema_hit_rate"] = schema_hit_rate(parsed_predictions, tools_list)
        results["exec_rate"] = exec_rate(parsed_predictions, tools_list)
    else:
        results["schema_hit_rate"] = None
        results["exec_rate"] = None

    return results
