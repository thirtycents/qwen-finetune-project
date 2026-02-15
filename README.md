# 🛠️ Qwen3-0.6B Function Calling 微调项目

> **一句话概括**：将 Qwen3-0.6B 基座模型通过 SFT + GRPO 强化学习微调为高质量 AI Agent 工具调用（Function Calling）助手，并部署到 vLLM 生产环境。
>
> **One-liner**: Fine-tune Qwen3-0.6B into a production-grade AI Agent Function Calling assistant via SFT + GRPO reinforcement learning, with full vLLM deployment pipeline.

---

## 📖 目录

- [项目简介](#项目简介)
- [三条命令复现](#三条命令复现)
- [环境准备](#环境准备)
- [数据准备](#数据准备)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [部署服务](#部署服务)
- [GRPO 强化学习](#grpo-强化学习必须)
- [面试准备文档](#面试准备文档)

---

## 项目简介

### 什么是 Function Calling？

Function Calling（函数调用）是让大语言模型（LLM）学会"调用工具"的能力。当用户提问时，模型不是直接回答，而是输出一个结构化的 JSON。

**示例**：
```
用户：北京今天天气怎么样？
模型输出：{"name": "get_weather", "arguments": {"city": "北京", "unit": "celsius"}}
```

### 为什么选择 Qwen3-0.6B？

- 小巧高效：0.6B 参数量，单卡即可训练
- 格式学习快：小模型+严格格式=快速收敛
- 端侧可用：量化后可在手机/笔记本离线运行

### 技术栈

```
训练框架：LLaMA-Factory
训练数据：Salesforce/xlam-function-calling-60k（60K 样本）
推理引擎：vLLM（PagedAttention）
部署平台：Docker + Kubernetes + Helm
硬件要求：NVIDIA GPU（≥12GB 显存）
```

---

## 🚀 三条命令复现

```bash
# 1️⃣ 环境准备（首次运行）
bash scripts/setup_env.sh

# 2️⃣ 训练（SFT → LoRA合并 → GRPO 全流程）
bash scripts/train.sh

# 3️⃣ 启动服务
bash scripts/serve.sh --model outputs/qwen3-0.6b-fc-merged
```

---

## 环境准备

### 硬件要求

| 组件 | 最低要求 |
|------|---------|
| GPU | ≥12GB 显存（RTX 4070/3060/4090）|
| 内存 | 16 GB |
| 硬盘 | 20 GB |

### 一键环境搭建

```bash
bash scripts/setup_env.sh
```

这个脚本会自动：
1. 创建 conda 环境 `qwen-fc`
2. 安装 PyTorch（CUDA）
3. 安装 LLaMA-Factory
4. 安装 vLLM
5. 检测 GPU

### 手动安装

```bash
conda create -n qwen-fc python=3.10 -y
conda activate qwen-fc
pip install -r requirements.txt
pip install llamafactory[torch,metrics]
pip install vllm

# 验证 GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 数据准备

### 数据集

使用 [Salesforce/xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)（需登录 HuggingFace 接受条款）

### 运行数据准备

```bash
conda activate qwen-fc
python scripts/prepare_data.py
```

**输出**：
```
data/processed/
├── train.json（约 54,000 条）
└── val.json（约 6,000 条）
```

---

## 模型训练

### 开始训练

```bash
conda activate qwen-fc
bash scripts/train.sh
```

这个脚本会自动执行：
1. **SFT**：LoRA 微调（~4-6 小时）
2. **合并**：LoRA → 基座模型
3. **GRPO**：强化学习对齐（~2-4 小时）

### 训练配置

| 参数 | 值 |
|------|-----|
| 模型 | Qwen/Qwen3-0.6B |
| LoRA rank | 32 |
| batch_size | 2 |
| learning_rate | 2e-4 |
| epochs | 3 |

### 训练输出

```
outputs/qwen3-0.6b-fc-lora/      # LoRA 适配器
outputs/qwen3-0.6b-fc-merged/    # 合并后的模型
outputs/qwen3-0.6b-fc-grpo/      # GRPO 后的模型
```

---

## 模型评估

```bash
# 评估合并后的模型
python eval/evaluate.py \
    --model_path outputs/qwen3-0.6b-fc-merged \
    --test_data data/processed/val.json \
    --output_dir eval/
```

### 评估指标

| 指标 | 含义 | 目标 |
|------|------|------|
| Parse Rate | JSON 解析成功率 | 95%+ |
| Schema Hit | 满足 schema 约束 | 88%+ |
| Func Accuracy | 函数名准确率 | 90%+ |
| Exec Rate | 可执行比例 | 80%+ |

---

## 部署服务

### 本地服务

```bash
# 启动 vLLM
bash scripts/serve.sh --model outputs/qwen3-0.6b-fc-merged

# 测试
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-0.6b-fc-merged",
    "messages": [{"role": "user", "content": "北京天气如何？"}]
  }'
```

### Docker 部署

```bash
bash scripts/deploy.sh docker
```

服务地址：
- vLLM: http://localhost:8000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (admin/admin)

---

## GRPO 强化学习（必须）

GRPO 已集成到 `train.sh` 中，自动执行。

### 奖励函数

| 奖励 | 权重 |
|------|------|
| JSON 解析 | 0.25 |
| Schema 命中 | 0.25 |
| 模拟执行 | 0.25 |
| 语义匹配 | 0.25 |

---

## 面试准备文档

| 文档 | 用途 |
|------|------|
| [知识点大纲](docs/知识点大纲.md) | 系统化知识地图 |
| [面试速查卡](docs/面试速查卡.md) | 5 张速查卡 |
| [深度追问](docs/深度追问.md) | 深度追问及答案 |
| [代码走读指南](docs/代码走读指南.md) | 代码走读要点 |

---

## 致谢

- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
- [vLLM](https://github.com/vllm-project/vllm)
- [xlam-function-calling-60k](https://huggingface.co/datasets/Salesforce/xlam-function-calling-60k)
- [Qwen3](https://github.com/QwenLM/Qwen3)

---

<p align="center"><i>Built with ❤️</i></p>
