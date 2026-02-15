# AGENTS.md — Qwen3-0.6B Function Calling 项目

## 项目概述

本项目将 Qwen3-0.6B 基座模型微调为 Function Calling 助手，包含完整的训练、评估、部署和监控流程。

## 代码规范

- **语言**：Python 3.10+，Shell (Bash)
- **注释**：所有代码注释使用中文，面向零基础读者
- **风格**：PEP 8，行长度不做严格限制（ruff --ignore E501）
- **类型**：无强制类型标注，但关键函数有 docstring 说明参数类型
- **测试**：pytest 框架，51 个测试用例覆盖核心逻辑

## 模块结构

### scripts/ — 核心脚本

| 文件 | 功能 | 入口 |
|------|------|------|
| setup_env.sh | 环境初始化（conda + pip + GPU 检测） | `bash scripts/setup_env.sh` |
| prepare_data.py | xlam-60k 数据下载 + 格式转换 | `python scripts/prepare_data.py` |
| train.sh | LLaMA-Factory LoRA SFT 训练启动器 | `bash scripts/train.sh` |
| merge_lora.py | LoRA 适配器合并到基座模型 | `python scripts/merge_lora.py` |
| export_model.py | 模型验证/信息报告/模型卡生成 | `python scripts/export_model.py {verify,info,card}` |
| grpo_reward.py | GRPO 5 种奖励函数 | 被训练框架调用 |
| serve.sh | 本地 vLLM 推理服务启动 | `bash scripts/serve.sh` |
| deploy.sh | 多模式部署（docker/k8s/helm） | `bash scripts/deploy.sh {docker,k8s,helm}` |
| benchmark.sh | 性能压测 Shell 包装器 | `bash scripts/benchmark.sh {--quick,--full}` |
| convert_gguf.sh | HF → GGUF 量化转换 | `bash scripts/convert_gguf.sh` |

### eval/ — 评估模块

| 文件 | 功能 |
|------|------|
| metrics.py | 6 个评测指标：parse_rate, func_name_accuracy, param_f1, schema_hit_rate, exec_rate, compute_all_metrics |
| evaluate.py | 命令行评估入口（支持本地模型和 API） |
| run_inference.py | 推理运行器（本地 transformers + OpenAI API） |
| run_bfcl.py | BFCL 基准集成 |
| benchmark.py | 异步负载测试（aiohttp，SSE 流式，全矩阵压测） |

### configs/ — 配置文件

| 文件 | 用途 |
|------|------|
| dataset_info.json | LLaMA-Factory 数据集注册表（sharegpt 格式） |
| qwen3_lora_sft.yaml | LoRA SFT 训练配置 |
| qwen3_grpo.yaml | GRPO 强化学习配置 |

### deploy/ — 部署配置

- `docker/` — Dockerfile + docker-compose（vLLM + Prometheus + Grafana）
- `helm/` — Kubernetes Helm Chart
- `k8s/` — 原生 Kubernetes manifests
- `monitoring/` — Prometheus 抓取配置 + Grafana 仪表盘 JSON

### tests/ — 测试

- `test_metrics.py` — 51 个 pytest 用例，覆盖 eval/metrics.py 和 scripts/grpo_reward.py

## 关键路径约定

| 用途 | 路径 |
|------|------|
| 训练输出 | `outputs/qwen3-0.6b-fc-lora` |
| 合并模型 | `outputs/qwen3-0.6b-fc-merged` |
| GRPO 输出 | `outputs/qwen3-0.6b-fc-grpo` |
| GGUF 输出 | `outputs/gguf` |
| 训练数据 | `data/processed/{train,val}.json` |
| 评测结果 | `eval/results.json`, `eval/predictions.jsonl` |
| 压测结果 | `eval/benchmark_results.json`, `eval/benchmark_charts.png` |

## 依赖关系

```
prepare_data.py → data/processed/ → train.sh (LLaMA-Factory)
                                         ↓
                               outputs/.../lora adapter
                                         ↓
                               merge_lora.py → merged model
                                         ↓
                                GRPO 训练 (llamafactory-cli)
                                         ↓
                     ┌───────────────────┼───────────────────┐
                     ↓                   ↓                   ↓
               serve.sh/deploy.sh   evaluate.py        convert_gguf.sh
                     ↓                   ↓                   ↓
               benchmark.py         results.json        outputs/gguf/
```

## 系统提示词 (System Prompt)

全项目统一使用以下系统提示词：

```
You are a helpful assistant with access to tools. When you need to call a function, output a JSON object with 'name' and 'arguments' fields.
```

## Conda 环境

- 环境名：`qwen-fc`
- Python：3.10
- GPU：NVIDIA GPU（12GB 显存，如 RTX 4070）
