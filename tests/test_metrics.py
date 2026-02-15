#!/usr/bin/env python3
"""
================================================================================
test_metrics.py - eval/metrics.py 和 scripts/grpo_reward.py 的完整测试套件
================================================================================

目的：
    对 eval/metrics.py 和 scripts/grpo_reward.py 中的所有函数进行全面测试。
    
测试范围（25-30 个测试函数）：
    1. eval/metrics.py:
       - parse_function_call: 5 个测试用例
       - parse_rate: 4 个测试用例
       - func_name_accuracy: 4 个测试用例
       - param_f1: 4 个测试用例
       - schema_hit_rate: 4 个测试用例
       - exec_rate: 3 个测试用例
       - compute_all_metrics: 2 个测试用例
    
    2. scripts/grpo_reward.py:
       - extract_json_from_text: 4 个测试用例
       - json_parse_reward: 3 个测试用例
       - schema_hit_reward: 4 个测试用例
       - semantic_reward: 4 个测试用例
       - combined_reward: 3 个测试用例

设计原则：
    - 每个测试函数都有清晰的中文文档说明测试目的
    - 使用 pytest 框架（不使用 unittest）
    - 测试覆盖边界情况和正常情况
    - 测试数据尽可能真实和多样
"""

import sys
import os
import json

# 将项目根目录加入 Python 路径，确保导入正确
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from eval.metrics import (
    parse_function_call,
    parse_rate,
    func_name_accuracy,
    param_f1,
    schema_hit_rate,
    exec_rate,
    compute_all_metrics,
)
from scripts.grpo_reward import (
    extract_json_from_text,
    json_parse_reward,
    schema_hit_reward,
    semantic_reward,
    combined_reward,
)


# =============================================================================
# eval/metrics.py 的测试
# =============================================================================


class TestParseFunction:
    """parse_function_call 函数的测试类"""

    def test_parse_valid_json(self):
        """测试：解析标准的 JSON 格式函数调用"""
        text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        result = parse_function_call(text)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"]["city"] == "Beijing"

    def test_parse_invalid_text(self):
        """测试：纯文本输入无法解析，应返回 None"""
        text = "这不是 JSON 格式，只是普通文本"
        result = parse_function_call(text)
        assert result is None

    def test_parse_json_embedded_in_text(self):
        """测试：JSON 嵌入在文本中，应成功提取 JSON 部分"""
        text = '我会帮你查询天气。{"name": "get_weather", "arguments": {"city": "Beijing"}}好的'
        result = parse_function_call(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_parse_empty_or_none_input(self):
        """测试：空字符串或 None 输入应返回 None"""
        assert parse_function_call("") is None
        assert parse_function_call(None) is None
        assert parse_function_call("   ") is None

    def test_parse_json_without_name_field(self):
        """测试：JSON 解析成功但缺少 'name' 字段，应返回 None"""
        text = '{"arguments": {"city": "Beijing"}}'
        result = parse_function_call(text)
        assert result is None

    def test_parse_json_missing_arguments_field(self):
        """测试：JSON 有 'name' 但没有 'arguments'，应补充空 arguments"""
        text = '{"name": "get_weather"}'
        result = parse_function_call(text)
        assert result is not None
        assert result["name"] == "get_weather"
        assert result["arguments"] == {}


class TestParseRate:
    """parse_rate 函数的测试类"""

    def test_all_valid_predictions(self):
        """测试：所有预测都是有效 JSON，返回 1.0"""
        predictions = [
            '{"name": "f1", "arguments": {}}',
            '{"name": "f2", "arguments": {"a": 1}}',
            '{"name": "f3", "arguments": {}}',
        ]
        result = parse_rate(predictions)
        assert result == 1.0

    def test_all_invalid_predictions(self):
        """测试：所有预测都是无效文本，返回 0.0"""
        predictions = [
            "无效文本",
            "不是 JSON",
            "我无法解析这个",
        ]
        result = parse_rate(predictions)
        assert result == 0.0

    def test_mixed_valid_invalid_predictions(self):
        """测试：预测混合有效和无效，返回成功率"""
        predictions = [
            '{"name": "f1", "arguments": {}}',
            "无效文本",
            '{"name": "f2", "arguments": {}}',
        ]
        result = parse_rate(predictions)
        assert result == pytest.approx(2.0 / 3.0)

    def test_empty_predictions_list(self):
        """测试：空预测列表应返回 0.0"""
        result = parse_rate([])
        assert result == 0.0


class TestFuncNameAccuracy:
    """func_name_accuracy 函数的测试类"""

    def test_exact_match(self):
        """测试：所有函数名都准确匹配，返回 1.0"""
        predictions = [
            {"name": "get_weather", "arguments": {}},
            {"name": "search_web", "arguments": {}},
        ]
        references = [
            {"name": "get_weather", "arguments": {"city": "Beijing"}},
            {"name": "search_web", "arguments": {"query": "AI"}},
        ]
        result = func_name_accuracy(predictions, references)
        assert result == 1.0

    def test_no_match(self):
        """测试：所有函数名都不匹配，返回 0.0"""
        predictions = [
            {"name": "wrong_func1", "arguments": {}},
            {"name": "wrong_func2", "arguments": {}},
        ]
        references = [
            {"name": "get_weather", "arguments": {}},
            {"name": "search_web", "arguments": {}},
        ]
        result = func_name_accuracy(predictions, references)
        assert result == 0.0

    def test_partial_match(self):
        """测试：部分函数名匹配，返回匹配比例"""
        predictions = [
            {"name": "get_weather", "arguments": {}},
            {"name": "wrong_func", "arguments": {}},
        ]
        references = [
            {"name": "get_weather", "arguments": {}},
            {"name": "search_web", "arguments": {}},
        ]
        result = func_name_accuracy(predictions, references)
        assert result == 0.5

    def test_empty_inputs(self):
        """测试：空输入列表应返回 0.0"""
        assert func_name_accuracy([], []) == 0.0
        assert func_name_accuracy([], [{"name": "f"}]) == 0.0


class TestParamF1:
    """param_f1 函数的测试类"""

    def test_perfect_parameter_match(self):
        """测试：参数完全匹配，所有指标都是 1.0"""
        predictions = [{"name": "f", "arguments": {"a": 1, "b": "test"}}]
        references = [{"name": "f", "arguments": {"a": 1, "b": "test"}}]
        result = param_f1(predictions, references)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_partial_parameter_overlap(self):
        """测试：参数部分重叠，计算精确率和召回率"""
        predictions = [{"name": "f", "arguments": {"a": 1, "b": 2, "c": 3}}]
        references = [{"name": "f", "arguments": {"a": 1, "b": 2, "d": 4}}]
        result = param_f1(predictions, references)
        # precision = 2/3 (正确的参数 a, b，错误的参数 c)
        # recall = 2/3 (参考中的 a, b 被找到，d 未找到)
        assert result["precision"] == pytest.approx(2.0 / 3.0)
        assert result["recall"] == pytest.approx(2.0 / 3.0)
        assert result["f1"] == pytest.approx(2.0 / 3.0)

    def test_no_parameter_overlap(self):
        """测试：参数完全不重叠，返回 0 分"""
        predictions = [{"name": "f", "arguments": {"x": 1}}]
        references = [{"name": "f", "arguments": {"y": 2}}]
        result = param_f1(predictions, references)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["f1"] == 0.0

    def test_empty_parameters(self):
        """测试：空参数列表应返回 0 分"""
        result = param_f1([], [])
        assert result == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


class TestSchemaHitRate:
    """schema_hit_rate 函数的测试类"""

    def test_valid_schema_hit(self):
        """测试：函数名和所有必填参数都正确，返回 1.0"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {"required": ["city"]},
                }
            ]
        ]
        predictions = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
        result = schema_hit_rate(predictions, tools_list)
        assert result == 1.0

    def test_missing_required_parameter(self):
        """测试：缺少必填参数，未命中 schema"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {"required": ["city", "unit"]},
                }
            ]
        ]
        predictions = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
        result = schema_hit_rate(predictions, tools_list)
        assert result == 0.0

    def test_unknown_function_name(self):
        """测试：函数名不在工具列表中，返回 0.0"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {"required": ["city"]},
                }
            ]
        ]
        predictions = [{"name": "unknown_func", "arguments": {"city": "Beijing"}}]
        result = schema_hit_rate(predictions, tools_list)
        assert result == 0.0

    def test_multiple_predictions_partial_hit(self):
        """测试：多个预测，部分命中"""
        tools_list = [
            [{"name": "f1", "parameters": {"required": ["a"]}}],
            [{"name": "f2", "parameters": {"required": ["b"]}}],
        ]
        predictions = [
            {"name": "f1", "arguments": {"a": 1}},
            {"name": "f2", "arguments": {"c": 2}},  # 缺少必填参数 b
        ]
        result = schema_hit_rate(predictions, tools_list)
        assert result == 0.5


class TestExecRate:
    """exec_rate 函数的测试类"""

    def test_correct_parameter_types(self):
        """测试：参数类型都正确，返回 1.0"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        ]
        predictions = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
        result = exec_rate(predictions, tools_list)
        assert result == 1.0

    def test_wrong_parameter_types(self):
        """测试：参数类型错误，返回 0.0"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        ]
        predictions = [{"name": "get_weather", "arguments": {"city": 123}}]  # 应该是字符串
        result = exec_rate(predictions, tools_list)
        assert result == 0.0

    def test_missing_required_param_for_exec(self):
        """测试：缺少必填参数，无法执行，返回 0.0"""
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        ]
        predictions = [{"name": "get_weather", "arguments": {}}]
        result = exec_rate(predictions, tools_list)
        assert result == 0.0


class TestComputeAllMetrics:
    """compute_all_metrics 函数的测试类"""

    def test_complete_evaluation_with_tools(self):
        """测试：完整评测，包含所有指标（有工具列表）"""
        raw_predictions = ['{"name": "get_weather", "arguments": {"city": "Beijing"}}']
        references = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
        tools_list = [
            [
                {
                    "name": "get_weather",
                    "parameters": {
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        ]
        
        result = compute_all_metrics(raw_predictions, references, tools_list)
        
        # 验证结果字典包含所有必需的键
        assert "parse_rate" in result
        assert "func_name_accuracy" in result
        assert "param_precision" in result
        assert "param_recall" in result
        assert "param_f1" in result
        assert "schema_hit_rate" in result
        assert "exec_rate" in result
        
        # 验证得分
        assert result["parse_rate"] == 1.0
        assert result["func_name_accuracy"] == 1.0
        assert result["param_f1"] == 1.0
        assert result["schema_hit_rate"] == 1.0
        assert result["exec_rate"] == 1.0

    def test_complete_evaluation_without_tools(self):
        """测试：完整评测，不包含工具列表的相关指标"""
        raw_predictions = ['{"name": "get_weather", "arguments": {"city": "Beijing"}}']
        references = [{"name": "get_weather", "arguments": {"city": "Beijing"}}]
        
        result = compute_all_metrics(raw_predictions, references, tools_list=None)
        
        # 验证 schema 相关指标为 None
        assert result["schema_hit_rate"] is None
        assert result["exec_rate"] is None


# =============================================================================
# scripts/grpo_reward.py 的测试
# =============================================================================


class TestExtractJsonFromText:
    """extract_json_from_text 函数的测试类"""

    def test_pure_json_extraction(self):
        """测试：纯 JSON 输入，直接解析"""
        text = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_embedded_json_extraction(self):
        """测试：JSON 嵌入在文本中，提取 JSON 部分"""
        text = '我会帮你查询天气。{"name": "get_weather", "arguments": {"city": "Beijing"}}谢谢'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_code_block_json_extraction(self):
        """测试：JSON 在 markdown 代码块中，提取 JSON 部分"""
        text = '```json\n{"name": "get_weather", "arguments": {"city": "Beijing"}}\n```'
        result = extract_json_from_text(text)
        assert result is not None
        assert result["name"] == "get_weather"

    def test_invalid_text_extraction(self):
        """测试：无效文本无法提取 JSON，返回 None"""
        text = "这不是 JSON，我无法帮助你"
        result = extract_json_from_text(text)
        assert result is None


class TestJsonParseReward:
    """json_parse_reward 函数的测试类"""

    def test_valid_function_call_output(self):
        """测试：有效的函数调用输出，返回 1.0"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        reward = json_parse_reward(output)
        assert reward == 1.0

    def test_valid_json_without_name(self):
        """测试：有效 JSON 但缺少 'name' 字段，返回 0.5"""
        output = '{"arguments": {"city": "Beijing"}}'
        reward = json_parse_reward(output)
        assert reward == 0.5

    def test_invalid_json_output(self):
        """测试：无效 JSON 输出，返回 0.0"""
        output = "我无法解析这个，不是 JSON"
        reward = json_parse_reward(output)
        assert reward == 0.0


class TestSchemaHitReward:
    """schema_hit_reward 函数的测试类"""

    def test_matching_function_name(self):
        """测试：函数名与标准答案匹配，返回 1.0"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        reward = schema_hit_reward(output, ground_truth=ground_truth)
        assert reward == 1.0

    def test_wrong_name_in_available_list(self):
        """测试：函数名错误但在可用列表中，返回 0.3"""
        output = '{"name": "search_web", "arguments": {"query": "Beijing"}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        available_functions = ["get_weather", "search_web", "get_time"]
        reward = schema_hit_reward(
            output,
            ground_truth=ground_truth,
            available_functions=available_functions,
        )
        assert reward == 0.3

    def test_unknown_function_name_reward(self):
        """测试：函数名不在可用列表中，返回 0.0"""
        output = '{"name": "unknown_func", "arguments": {"x": 1}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        available_functions = ["get_weather", "search_web"]
        reward = schema_hit_reward(
            output,
            ground_truth=ground_truth,
            available_functions=available_functions,
        )
        assert reward == 0.0

    def test_no_ground_truth_with_available_list(self):
        """测试：无标准答案，只检查可用列表，返回 1.0"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        available_functions = ["get_weather", "search_web"]
        reward = schema_hit_reward(
            output,
            ground_truth=None,
            available_functions=available_functions,
        )
        assert reward == 1.0


class TestSemanticReward:
    """semantic_reward 函数的测试类"""

    def test_exact_parameter_match(self):
        """测试：参数值完全匹配，返回 1.0"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing", "unit": "celsius"}}'
        reward = semantic_reward(output, ground_truth=ground_truth)
        assert reward == 1.0

    def test_partial_parameter_match(self):
        """测试：参数部分匹配，计算 F1 分数"""
        output = '{"name": "f", "arguments": {"a": 1, "b": 2, "c": 3}}'
        ground_truth = '{"name": "f", "arguments": {"a": 1, "b": 2, "d": 4}}'
        reward = semantic_reward(output, ground_truth=ground_truth)
        # 2个正确参数，3个预测，2个参考 → F1 = 2/3
        assert reward == pytest.approx(2.0 / 3.0)

    def test_no_parameter_match_semantic(self):
        """测试：参数完全不匹配，返回 0.0"""
        output = '{"name": "f", "arguments": {"x": 1}}'
        ground_truth = '{"name": "f", "arguments": {"y": 2}}'
        reward = semantic_reward(output, ground_truth=ground_truth)
        assert reward == 0.0

    def test_no_ground_truth_semantic(self):
        """测试：无标准答案，无法评估语义，返回 0.0"""
        output = '{"name": "f", "arguments": {"a": 1}}'
        reward = semantic_reward(output, ground_truth=None)
        assert reward == 0.0


class TestCombinedReward:
    """combined_reward 函数的测试类"""

    def test_perfect_score(self):
        """测试：完美的函数调用，应得到高分"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        available_functions = ["get_weather", "search_web"]
        function_schema = {
            "name": "get_weather",
            "parameters": {"city": {"type": "string"}},
        }
        
        reward = combined_reward(
            output,
            ground_truth=ground_truth,
            available_functions=available_functions,
            function_schema=function_schema,
        )
        
        # 所有维度都完美，综合奖励应该是 1.0
        assert reward == pytest.approx(1.0)

    def test_zero_score(self):
        """测试：完全错误的输出，应得到 0 分"""
        output = "无法解析的文本"
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        
        reward = combined_reward(output, ground_truth=ground_truth)
        
        # 无法解析 JSON，所有维度都失败，综合奖励应该接近 0.0
        assert reward == pytest.approx(0.0)

    def test_custom_weights(self):
        """测试：自定义权重，验证加权平均计算正确"""
        output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        ground_truth = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        
        # 使用自定义权重：json_parse 权重最高
        custom_weights = {
            "json_parse": 0.7,
            "schema_hit": 0.1,
            "exec": 0.1,
            "semantic": 0.1,
        }
        
        reward = combined_reward(
            output,
            ground_truth=ground_truth,
            weights=custom_weights,
        )
        
        # 由于 json_parse 权重最高且输出有效，得分应较高
        assert reward > 0.5


# =============================================================================
# 集成测试
# =============================================================================


class TestIntegration:
    """集成测试：验证多个函数共同工作"""

    def test_metrics_and_reward_consistency(self):
        """测试：eval/metrics.py 和 grpo_reward.py 的一致性"""
        # 准备测试数据
        raw_output = '{"name": "get_weather", "arguments": {"city": "Beijing"}}'
        parsed = parse_function_call(raw_output)
        
        # metrics.py 应该解析成功
        assert parsed is not None
        assert parsed["name"] == "get_weather"
        
        # grpo_reward.py 也应该提取成功
        extracted = extract_json_from_text(raw_output)
        assert extracted is not None
        assert extracted["name"] == "get_weather"
        
        # 两者应该提取相同的 name
        assert parsed["name"] == extracted["name"]

    def test_end_to_end_evaluation(self):
        """测试：端到端的评估流程"""
        # 模型输出
        model_outputs = [
            '{"name": "get_weather", "arguments": {"city": "Beijing"}}',
            'I will search for this: {"name": "search_web", "arguments": {"query": "AI"}}',
            "这是无效输出",
        ]
        
        # 标准答案
        references = [
            {"name": "get_weather", "arguments": {"city": "Beijing"}},
            {"name": "search_web", "arguments": {"query": "AI"}},
            {"name": "unknown_func", "arguments": {}},
        ]
        
        # 工具列表
        tools_list = [
            [{"name": "get_weather", "parameters": {"required": ["city"]}}],
            [{"name": "search_web", "parameters": {"required": ["query"]}}],
            [{"name": "unknown_func", "parameters": {"required": ["x"]}}],
        ]
        
        # 计算所有指标
        metrics = compute_all_metrics(model_outputs, references, tools_list)
        
        # 验证解析成功率（前两个成功，第三个失败）
        assert metrics["parse_rate"] == pytest.approx(2.0 / 3.0)
        
        # 验证函数名准确率（前两个正确，第三个错误）
        assert metrics["func_name_accuracy"] == pytest.approx(2.0 / 3.0)
        
        # 验证 schema 命中率
        assert 0 < metrics["schema_hit_rate"] < 1.0


# =============================================================================
# 边界和异常情况测试
# =============================================================================


class TestEdgeCases:
    """边界和异常情况的测试"""

    def test_malformed_json_parsing(self):
        """测试：畸形 JSON 处理"""
        malformed = '{"name": "f", "arguments": {broken}}'
        result = parse_function_call(malformed)
        assert result is None

    def test_nested_json_parsing(self):
        """测试：嵌套 JSON 解析"""
        nested = '{"name": "f", "arguments": {"data": {"inner": "value"}}}'
        result = parse_function_call(nested)
        assert result is not None
        assert result["arguments"]["data"]["inner"] == "value"

    def test_unicode_parameter_values(self):
        """测试：Unicode 字符在参数中的处理"""
        output = '{"name": "f", "arguments": {"city": "北京"}}'
        ground_truth = '{"name": "f", "arguments": {"city": "北京"}}'
        
        reward = semantic_reward(output, ground_truth=ground_truth)
        assert reward == 1.0

    def test_numeric_parameter_types(self):
        """测试：数字参数的类型兼容性（int/float）"""
        tools_list = [
            [
                {
                    "name": "calculate",
                    "parameters": {
                        "properties": {"value": {"type": "number"}},
                        "required": ["value"],
                    },
                }
            ]
        ]
        
        # 使用 float 满足 number 类型
        predictions = [{"name": "calculate", "arguments": {"value": 3.14}}]
        result = exec_rate(predictions, tools_list)
        assert result == 1.0
        
        # 使用 int 也满足 number 类型
        predictions = [{"name": "calculate", "arguments": {"value": 42}}]
        result = exec_rate(predictions, tools_list)
        assert result == 1.0


if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
