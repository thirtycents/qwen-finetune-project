#!/usr/bin/env python3
"""
grpo_reward.py — GRPO (Group Relative Policy Optimization) 奖励函数
====================================================================

什么是 GRPO？
-----------
GRPO 是 DeepSeek 在 DeepSeek-R1 论文中提出的一种强化学习方法
（论文链接: https://arxiv.org/pdf/2402.03300）。

传统 RLHF (人类反馈强化学习) 的流程：
    SFT模型 → 收集人类偏好数据 → 训练奖励模型(RM) → PPO 训练 → 对齐后的模型

GRPO 的简化流程：
    SFT模型 → 定义规则奖励函数 → GRPO 训练 → 对齐后的模型
    
核心区别：
    - PPO 需要一个额外的「奖励模型」来评估输出好坏
    - GRPO 直接用「规则」来打分，不需要训练奖励模型
    - GRPO 通过「组内相对排序」来计算优势值，而不是用绝对分数

GRPO 的核心思想：
    1. 对同一个问题，让模型生成 G 个候选回答（一组）
    2. 用奖励函数对每个回答打分
    3. 在组内做归一化（减均值除标准差），得到相对优势
    4. 优势高的回答 → 增大其生成概率
       优势低的回答 → 减小其生成概率
    
    公式简化版：
        advantage_i = (reward_i - mean(rewards)) / std(rewards)
        loss = -sum(advantage_i * log_prob(response_i))

为什么 GRPO 适合 Function Calling？
    因为 Function Calling 的输出有明确的「对错」标准：
    - JSON 格式是否正确（可以用规则检查）
    - 函数名是否匹配（可以用字符串比较）
    - 参数是否正确（可以用 schema 验证）
    - 执行是否成功（可以用沙箱测试）
    这些都不需要人类判断，可以自动化打分。

本脚本实现的奖励函数：
--------------------
1. json_parse_reward    : JSON 解析成功 → 1.0，否则 → 0.0
2. schema_hit_reward    : 函数名匹配 → 1.0，否则 → 0.0
3. exec_reward          : 参数类型正确 → 1.0，否则 → 0.0（模拟执行）
4. semantic_reward      : 参数值语义匹配 → 0.0~1.0（基于 F1 分数）
5. combined_reward      : 以上四个的加权平均（各 0.25）

使用方法：
---------
这个文件主要作为模块被导入使用（配合 LLaMA-Factory 的 GRPO 训练）。

也可以独立运行来测试奖励函数：
    python scripts/grpo_reward.py --test

与 LLaMA-Factory GRPO 训练的集成：
    在 qwen3_grpo.yaml 中指定 reward_model 为本脚本路径
"""

import json        # JSON 解析和生成
import re          # 正则表达式，用于从文本中提取 JSON
import argparse    # 命令行参数解析
from typing import Any, Optional  # 类型注解，让代码更清晰


# =============================================================================
# 工具函数：从模型输出文本中提取 JSON 对象
# =============================================================================

def extract_json_from_text(text: str) -> Optional[dict]:
    """
    从模型输出的文本中提取第一个 JSON 对象。

    模型的输出可能包含额外的文字说明，我们需要从中找到 JSON 部分。
    例如模型可能输出:
        "I'll help you check the weather. {\"name\": \"get_weather\", \"arguments\": {\"city\": \"Beijing\"}}"
    我们需要提取出:
        {"name": "get_weather", "arguments": {"city": "Beijing"}}

    提取策略（按优先级）：
    1. 先尝试直接解析整个文本（万一它就是纯 JSON）
    2. 用正则表达式找 {...} 结构（支持嵌套）
    3. 找 ```json ... ``` 代码块中的内容

    参数：
        text: 模型输出的原始文本
    
    返回：
        解析成功返回 dict，失败返回 None
    """
    if not text or not text.strip():
        return None

    text = text.strip()

    # ---- 策略 1：直接解析 ----
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass  # 不是纯 JSON，继续尝试其他策略

    # ---- 策略 2：正则提取 {...} ----
    # 这个正则表达式匹配最外层的花括号及其内容
    # {  → 匹配开头的 {
    # [^{}]* → 匹配任意非花括号字符
    # (?:{[^{}]*})* → 匹配嵌套的 {...} 块
    # }  → 匹配结尾的 }
    # 注意：这个简化版本只支持一层嵌套，对大多数 Function Calling 场景够用
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    matches = re.findall(json_pattern, text)

    for match in matches:
        try:
            result = json.loads(match)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    # ---- 策略 3：代码块提取 ----
    # 有些模型会把 JSON 放在 markdown 代码块中
    code_block_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
    code_matches = re.findall(code_block_pattern, text, re.DOTALL)

    for match in code_matches:
        try:
            result = json.loads(match.strip())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None  # 所有策略都失败了


# =============================================================================
# 奖励函数 1：JSON 解析奖励
# =============================================================================

def json_parse_reward(model_output: str, **kwargs: Any) -> float:
    """
    检查模型输出是否能被解析为有效的 JSON 函数调用。

    评分规则：
        - 能解析为 JSON，且包含 "name" 字段 → 1.0
        - 能解析为 JSON，但缺少 "name" 字段 → 0.5（格式对了，但不完整）
        - 无法解析为 JSON → 0.0

    为什么这个奖励很重要？
        Function Calling 的输出必须是结构化的 JSON，
        如果模型输出的不是合法 JSON，后续的所有处理都无法进行。
        这是最基础的质量门槛。

    参数：
        model_output: 模型生成的文本
        **kwargs: 额外参数（保留接口兼容性）

    返回：
        float: 奖励值 0.0 ~ 1.0
    """
    parsed = extract_json_from_text(model_output)

    if parsed is None:
        return 0.0  # 无法解析

    if "name" in parsed:
        return 1.0  # 完整的函数调用格式

    return 0.5  # JSON 格式正确，但缺少函数名


# =============================================================================
# 奖励函数 2：Schema 匹配奖励（函数名是否正确）
# =============================================================================

def schema_hit_reward(
    model_output: str,
    ground_truth: Optional[str] = None,
    available_functions: Optional[list] = None,
    **kwargs: Any,
) -> float:
    """
    检查模型调用的函数名是否正确。

    评分规则：
        - 函数名与标准答案完全匹配 → 1.0
        - 函数名在可用函数列表中（但不是标准答案）→ 0.3
        - 函数名不在可用列表中 → 0.0
        - 无法解析输出 → 0.0

    什么是 Schema？
        在 Function Calling 中，Schema 描述了一个函数的「签名」：
        - 函数名（name）
        - 参数列表（parameters）
        - 每个参数的类型（type）和描述（description）
        
        Schema Hit 就是检查模型是否调用了正确的函数。
        即使参数错了，只要函数名对了，就说明模型理解了用户意图。

    参数：
        model_output: 模型生成的文本
        ground_truth: 标准答案的 JSON 字符串（包含 name 字段）
        available_functions: 可用函数名列表，例如 ["get_weather", "search_web"]
        **kwargs: 额外参数

    返回：
        float: 奖励值 0.0 ~ 1.0
    """
    parsed = extract_json_from_text(model_output)
    if parsed is None or "name" not in parsed:
        return 0.0

    predicted_name = parsed["name"]

    # 如果提供了标准答案，检查是否完全匹配
    if ground_truth is not None:
        gt_parsed = extract_json_from_text(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if gt_parsed and "name" in gt_parsed:
            if predicted_name == gt_parsed["name"]:
                return 1.0
            # 名字不匹配，但在可用函数列表中
            if available_functions and predicted_name in available_functions:
                return 0.3
            return 0.0

    # 没有标准答案，只检查是否在可用函数列表中
    if available_functions:
        return 1.0 if predicted_name in available_functions else 0.0

    # 没有任何参考信息，只要有函数名就给分
    return 0.5


# =============================================================================
# 奖励函数 3：执行奖励（参数类型是否正确）
# =============================================================================

def exec_reward(
    model_output: str,
    ground_truth: Optional[str] = None,
    function_schema: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """
    模拟执行检查：验证参数类型是否符合函数 Schema。

    评分规则：
        - 所有参数类型都正确 → 1.0
        - 部分参数类型正确 → 按正确比例给分
        - 没有参数或全部错误 → 0.0
        - 无法解析 → 0.0

    为什么叫「模拟」执行？
        真正执行函数需要后端环境（数据库、API 等），这在训练时不现实。
        所以我们只检查参数的类型是否正确，作为「能否执行」的近似判断。
        
        例如函数签名要求 city: str, unit: str
        模型输出 {"city": "Beijing", "unit": "celsius"} → 类型都对 → 1.0
        模型输出 {"city": 123, "unit": "celsius"} → city 类型错 → 0.5

    参数：
        model_output: 模型生成的文本
        ground_truth: 标准答案 JSON 字符串
        function_schema: 函数的参数 Schema（可选，格式见下方示例）
        **kwargs: 额外参数

    function_schema 示例：
        {
            "name": "get_weather",
            "parameters": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        }

    返回：
        float: 奖励值 0.0 ~ 1.0
    """
    parsed = extract_json_from_text(model_output)
    if parsed is None or "name" not in parsed:
        return 0.0

    predicted_args = parsed.get("arguments", parsed.get("parameters", {}))
    if not isinstance(predicted_args, dict):
        return 0.0

    # ---- 如果有标准答案，做参数级别的对比 ----
    if ground_truth is not None:
        gt_parsed = extract_json_from_text(ground_truth) if isinstance(ground_truth, str) else ground_truth
        if gt_parsed:
            gt_args = gt_parsed.get("arguments", gt_parsed.get("parameters", {}))
            if isinstance(gt_args, dict) and gt_args:
                # 检查每个标准答案中的参数，预测值的类型是否匹配
                correct = 0
                total = len(gt_args)
                for key, gt_val in gt_args.items():
                    if key in predicted_args:
                        pred_val = predicted_args[key]
                        # 类型匹配检查
                        if type(pred_val) == type(gt_val):
                            correct += 1
                        # 特殊情况：int 和 float 互相兼容
                        elif isinstance(pred_val, (int, float)) and isinstance(gt_val, (int, float)):
                            correct += 1
                return correct / total if total > 0 else 0.0

    # ---- 如果有 Schema，检查参数类型 ----
    if function_schema and "parameters" in function_schema:
        schema_params = function_schema["parameters"]
        if not schema_params:
            return 1.0  # 函数不需要参数

        # Python 类型到 JSON Schema 类型的映射
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }

        correct = 0
        total = len(schema_params)

        for param_name, param_spec in schema_params.items():
            if param_name in predicted_args:
                expected_type = param_spec.get("type", "string")
                python_type = type_map.get(expected_type, str)
                if isinstance(predicted_args[param_name], python_type):
                    correct += 1

        return correct / total if total > 0 else 0.0

    # 没有参考信息，只要有参数就给基础分
    return 0.5 if predicted_args else 0.0


# =============================================================================
# 奖励函数 4：语义匹配奖励（参数值是否正确）
# =============================================================================

def semantic_reward(
    model_output: str,
    ground_truth: Optional[str] = None,
    **kwargs: Any,
) -> float:
    """
    检查模型输出的参数值是否与标准答案语义匹配。

    使用参数级别的 F1 分数来衡量：
        - Precision（精确率）：预测的参数中有多少是正确的
        - Recall（召回率）：标准答案中的参数有多少被预测到了
        - F1 = 2 × Precision × Recall / (Precision + Recall)

    什么是 F1 分数？
        假设标准答案有参数 {A, B, C}，模型预测了 {A, B, D}
        - Precision = 2/3（预测了3个，其中2个正确）
        - Recall = 2/3（标准答案3个，被预测到2个）
        - F1 = 2 × (2/3) × (2/3) / (2/3 + 2/3) = 2/3 ≈ 0.67

    参数值匹配规则：
        - 字符串：忽略大小写和前后空格进行比较
        - 数字：值相等即可（int/float 互相兼容）
        - 布尔值、列表、字典：直接比较
        - None/null：视为匹配

    参数：
        model_output: 模型生成的文本
        ground_truth: 标准答案 JSON 字符串
        **kwargs: 额外参数

    返回：
        float: F1 奖励值 0.0 ~ 1.0
    """
    if ground_truth is None:
        return 0.0  # 没有标准答案，无法评估语义

    parsed = extract_json_from_text(model_output)
    if parsed is None:
        return 0.0

    gt_parsed = extract_json_from_text(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if gt_parsed is None:
        return 0.0

    # 提取参数
    pred_args = parsed.get("arguments", parsed.get("parameters", {}))
    gt_args = gt_parsed.get("arguments", gt_parsed.get("parameters", {}))

    if not isinstance(pred_args, dict) or not isinstance(gt_args, dict):
        return 0.0

    if not gt_args:
        # 标准答案没有参数，预测也没有 → 完美匹配
        return 1.0 if not pred_args else 0.0

    # ---- 计算参数匹配 ----
    def values_match(pred_val: Any, gt_val: Any) -> bool:
        """比较两个参数值是否语义等价"""
        # 都是 None
        if pred_val is None and gt_val is None:
            return True
        if pred_val is None or gt_val is None:
            return False

        # 字符串比较：忽略大小写和空格
        if isinstance(pred_val, str) and isinstance(gt_val, str):
            return pred_val.strip().lower() == gt_val.strip().lower()

        # 数字比较：int 和 float 互相兼容
        if isinstance(pred_val, (int, float)) and isinstance(gt_val, (int, float)):
            return abs(float(pred_val) - float(gt_val)) < 1e-6

        # 其他类型：直接比较
        return pred_val == gt_val

    # 计算 Precision 和 Recall
    # TP (True Positive): 预测正确的参数数量
    tp = sum(
        1 for key in pred_args
        if key in gt_args and values_match(pred_args[key], gt_args[key])
    )

    precision = tp / len(pred_args) if pred_args else 0.0
    recall = tp / len(gt_args) if gt_args else 0.0

    # F1 分数
    if precision + recall == 0:
        return 0.0

    f1 = 2 * precision * recall / (precision + recall)
    return f1


# =============================================================================
# 综合奖励函数
# =============================================================================

def combined_reward(
    model_output: str,
    ground_truth: Optional[str] = None,
    available_functions: Optional[list] = None,
    function_schema: Optional[dict] = None,
    weights: Optional[dict] = None,
    **kwargs: Any,
) -> float:
    """
    综合所有奖励函数，计算加权平均分。

    默认权重（各 0.25，设计文档要求）：
        - json_parse: 0.25  （JSON 格式正确性）
        - schema_hit: 0.25  （函数名匹配）
        - exec: 0.25        （参数类型正确）
        - semantic: 0.25    （参数值匹配）

    总分 = 0.25 × parse + 0.25 × schema + 0.25 × exec + 0.25 × semantic

    为什么平均分配权重？
        四个维度从不同角度评估 Function Calling 的质量：
        1. 格式（parse）：输出是否是合法 JSON → 最基础的要求
        2. 意图（schema）：是否调用了正确的函数 → 理解用户需求
        3. 类型（exec）：参数类型是否正确 → 基本的编程正确性
        4. 语义（semantic）：参数值是否正确 → 深层语义理解
        
        每个维度都很重要，平均分配可以避免模型只优化某一方面。
        实际使用中可以通过 weights 参数调整权重。

    参数：
        model_output: 模型生成的文本
        ground_truth: 标准答案 JSON 字符串
        available_functions: 可用函数名列表
        function_schema: 函数的参数 Schema
        weights: 自定义权重字典，格式 {"json_parse": 0.3, "schema_hit": 0.3, ...}
        **kwargs: 额外参数

    返回：
        float: 综合奖励值 0.0 ~ 1.0
    """
    # 默认权重
    default_weights = {
        "json_parse": 0.25,
        "schema_hit": 0.25,
        "exec": 0.25,
        "semantic": 0.25,
    }

    # 如果提供了自定义权重，合并（自定义 > 默认）
    w = {**default_weights, **(weights or {})}

    # 归一化权重（确保总和为 1）
    total_weight = sum(w.values())
    if total_weight > 0:
        w = {k: v / total_weight for k, v in w.items()}

    # 计算各项奖励
    r_parse = json_parse_reward(model_output)
    r_schema = schema_hit_reward(
        model_output,
        ground_truth=ground_truth,
        available_functions=available_functions,
    )
    r_exec = exec_reward(
        model_output,
        ground_truth=ground_truth,
        function_schema=function_schema,
    )
    r_semantic = semantic_reward(
        model_output,
        ground_truth=ground_truth,
    )

    # 加权平均
    total = (
        w["json_parse"] * r_parse
        + w["schema_hit"] * r_schema
        + w["exec"] * r_exec
        + w["semantic"] * r_semantic
    )

    return total


# =============================================================================
# 批量奖励计算（用于 GRPO 训练循环）
# =============================================================================

def compute_rewards_batch(
    model_outputs: list[str],
    ground_truths: list[str],
    available_functions_list: Optional[list[list]] = None,
    function_schemas: Optional[list[dict]] = None,
    weights: Optional[dict] = None,
) -> list[dict]:
    """
    批量计算一组模型输出的奖励分数。

    在 GRPO 训练中，模型会对同一个问题生成多个候选回答，
    然后用这个函数批量计算每个回答的奖励。

    参数：
        model_outputs: 模型输出列表
        ground_truths: 对应的标准答案列表
        available_functions_list: 每个样本的可用函数列表
        function_schemas: 每个样本的函数 Schema
        weights: 奖励权重

    返回：
        list[dict]: 每个样本的详细奖励分数
        [
            {
                "combined": 0.85,
                "json_parse": 1.0,
                "schema_hit": 1.0,
                "exec": 0.7,
                "semantic": 0.7
            },
            ...
        ]
    """
    results = []

    for i, (output, gt) in enumerate(zip(model_outputs, ground_truths)):
        avail_funcs = (
            available_functions_list[i]
            if available_functions_list and i < len(available_functions_list)
            else None
        )
        schema = (
            function_schemas[i]
            if function_schemas and i < len(function_schemas)
            else None
        )

        result = {
            "combined": combined_reward(
                output,
                ground_truth=gt,
                available_functions=avail_funcs,
                function_schema=schema,
                weights=weights,
            ),
            "json_parse": json_parse_reward(output),
            "schema_hit": schema_hit_reward(
                output, ground_truth=gt, available_functions=avail_funcs
            ),
            "exec": exec_reward(output, ground_truth=gt, function_schema=schema),
            "semantic": semantic_reward(output, ground_truth=gt),
        }
        results.append(result)

    return results


# =============================================================================
# 测试代码：验证奖励函数是否正常工作
# =============================================================================

def run_tests():
    """
    运行奖励函数的自测试。

    包含多种测试场景：
    1. 完美匹配（所有维度都正确）
    2. 格式正确但函数名错误
    3. 参数部分正确
    4. 完全无法解析的输出
    5. 空输出
    """
    print("=" * 60)
    print("  GRPO 奖励函数测试")
    print("=" * 60)

    # 标准答案
    ground_truth = json.dumps({
        "name": "get_weather",
        "arguments": {"city": "Beijing", "unit": "celsius"},
    })

    available_functions = ["get_weather", "search_web", "get_time"]

    function_schema = {
        "name": "get_weather",
        "parameters": {
            "city": {"type": "string"},
            "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
        },
    }

    # 测试用例
    test_cases = [
        {
            "name": "完美匹配",
            "output": '{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}',
            "expected_parse": 1.0,
            "expected_schema": 1.0,
        },
        {
            "name": "带额外文字的正确输出",
            "output": 'I will check the weather for you. {"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}',
            "expected_parse": 1.0,
            "expected_schema": 1.0,
        },
        {
            "name": "函数名错误",
            "output": '{"name": "search_web", "arguments": {"query": "Beijing weather"}}',
            "expected_parse": 1.0,
            "expected_schema": 0.3,  # 在可用列表中但不是正确答案
        },
        {
            "name": "未知函数名",
            "output": '{"name": "unknown_func", "arguments": {"x": 1}}',
            "expected_parse": 1.0,
            "expected_schema": 0.0,
        },
        {
            "name": "参数部分正确",
            "output": '{"name": "get_weather", "arguments": {"city": "Shanghai", "unit": "celsius"}}',
            "expected_parse": 1.0,
            "expected_schema": 1.0,
        },
        {
            "name": "无法解析的输出",
            "output": "I don't know how to help with that.",
            "expected_parse": 0.0,
            "expected_schema": 0.0,
        },
        {
            "name": "空输出",
            "output": "",
            "expected_parse": 0.0,
            "expected_schema": 0.0,
        },
        {
            "name": "JSON 格式但无函数名",
            "output": '{"message": "hello"}',
            "expected_parse": 0.5,
            "expected_schema": 0.0,
        },
    ]

    all_passed = True

    for i, tc in enumerate(test_cases):
        print(f"\n  测试 {i + 1}: {tc['name']}")
        print(f"  输入: {tc['output'][:80]}{'...' if len(tc['output']) > 80 else ''}")

        # 计算各项奖励
        r_parse = json_parse_reward(tc["output"])
        r_schema = schema_hit_reward(
            tc["output"],
            ground_truth=ground_truth,
            available_functions=available_functions,
        )
        r_exec = exec_reward(
            tc["output"],
            ground_truth=ground_truth,
            function_schema=function_schema,
        )
        r_semantic = semantic_reward(tc["output"], ground_truth=ground_truth)
        r_combined = combined_reward(
            tc["output"],
            ground_truth=ground_truth,
            available_functions=available_functions,
            function_schema=function_schema,
        )

        print(f"    json_parse : {r_parse:.2f} (期望: {tc['expected_parse']:.2f}) {'✅' if abs(r_parse - tc['expected_parse']) < 0.01 else '❌'}")
        print(f"    schema_hit : {r_schema:.2f} (期望: {tc['expected_schema']:.2f}) {'✅' if abs(r_schema - tc['expected_schema']) < 0.01 else '❌'}")
        print(f"    exec       : {r_exec:.2f}")
        print(f"    semantic   : {r_semantic:.2f}")
        print(f"    combined   : {r_combined:.2f}")

        if abs(r_parse - tc["expected_parse"]) > 0.01 or abs(r_schema - tc["expected_schema"]) > 0.01:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("  ✅ 所有测试通过！")
    else:
        print("  ❌ 部分测试失败，请检查奖励函数逻辑。")
    print("=" * 60)


# =============================================================================
# 主入口
# =============================================================================

def main():
    """主函数：提供命令行测试入口"""
    parser = argparse.ArgumentParser(description="GRPO 奖励函数")
    parser.add_argument(
        "--test",
        action="store_true",
        help="运行奖励函数自测试",
    )
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        print("使用 --test 参数运行自测试")
        print("或者在其他脚本中导入本模块使用奖励函数：")
        print("  from scripts.grpo_reward import combined_reward")


if __name__ == "__main__":
    main()
