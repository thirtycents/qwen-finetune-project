#!/usr/bin/env python3
"""
benchmark.py — vLLM 推理服务性能压测工具
==========================================

什么是性能压测？
--------------
压测（Load Testing）就是模拟大量用户同时访问服务，
测量服务在不同负载下的性能表现。

我们关注的核心指标：
1. TTFT (Time To First Token)  : 从发送请求到收到第一个 token 的时间
   - 这直接影响用户体验，用户等越久越不耐烦
   - 目标：P95 < 500ms

2. 吞吐量 (Throughput)         : 每秒生成多少个 token
   - 越高越好，意味着服务能处理更多请求
   - 目标：> 30 tokens/s/request

3. 端到端延迟 (E2E Latency)    : 从发送请求到收到完整回复的总时间
   - P50 = 中位数（50% 的请求在这个时间内完成）
   - P95 = 95 百分位（95% 的请求在这个时间内完成）

4. 成功率 (Success Rate)       : 请求成功的比例
   - 高并发时如果显存不足，请求可能失败
   - 目标：> 99%

压测矩阵（来自项目设计文档）：
    并发数 × 上下文长度
    并发数: {1, 2, 4, 8, 16, 32}
    上下文长度: {256, 1024, 2048, 4096}

使用方法：
---------
# 快速测试（单个并发）
python eval/benchmark.py --base-url http://localhost:8000

# 完整压测矩阵
python eval/benchmark.py --base-url http://localhost:8000 --full-matrix

# 自定义并发数
python eval/benchmark.py --base-url http://localhost:8000 --concurrency 1 2 4 8

# 保存结果
python eval/benchmark.py --base-url http://localhost:8000 --output eval/benchmark_results.json
"""

import argparse
import asyncio       # Python 异步编程库，用于并发发送请求
import json
import os
import statistics    # Python 标准库，计算中位数、百分位等统计量
import sys
import time
from dataclasses import dataclass, field, asdict  # 数据类，简化数据结构定义
from typing import Optional

# aiohttp 是异步 HTTP 客户端库
# 与同步的 requests 库不同，aiohttp 可以同时发送多个请求而不阻塞
# 这正是压测需要的：模拟多个并发用户
try:
    import aiohttp
except ImportError:
    print("[错误] aiohttp 未安装。请运行: pip install aiohttp")
    sys.exit(1)


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class RequestResult:
    """
    单个请求的结果。

    @dataclass 是 Python 3.7+ 的语法糖，自动生成 __init__、__repr__ 等方法。
    相当于定义了一个带有类型注解的简单类。
    """
    success: bool               # 请求是否成功
    ttft: float                 # Time To First Token（秒）
    total_latency: float        # 总延迟（秒）
    output_tokens: int          # 生成的 token 数量
    tokens_per_second: float    # 生成速度（tokens/s）
    error: Optional[str] = None # 错误信息（如果失败）


@dataclass
class BenchmarkResult:
    """
    一组压测的汇总结果。
    """
    concurrency: int            # 并发数
    context_length: int         # 输入上下文长度
    num_requests: int           # 总请求数
    successful: int             # 成功请求数
    failed: int                 # 失败请求数
    success_rate: float         # 成功率 (0~1)

    # 延迟统计
    ttft_p50: float             # TTFT 中位数
    ttft_p95: float             # TTFT 95 百分位
    ttft_avg: float             # TTFT 平均值

    latency_p50: float          # 总延迟中位数
    latency_p95: float          # 总延迟 95 百分位
    latency_avg: float          # 总延迟平均值

    throughput_avg: float       # 平均吞吐量 (tokens/s)
    total_throughput: float     # 总吞吐量 (所有请求的 tokens/s 之和)

    duration: float             # 总测试时间（秒）
    errors: list = field(default_factory=list)  # 错误列表


# =============================================================================
# 生成测试 Prompt
# =============================================================================

def generate_test_prompt(context_length: int) -> list[dict]:
    """
    生成指定长度的测试 prompt。

    context_length 不是精确的 token 数，而是近似的字符数。
    一般英文中 1 token ≈ 4 个字符，所以 context_length=256 大约是 64 tokens。

    我们使用 Function Calling 场景的 prompt，与项目的实际用途一致。
    """
    # 基础系统消息
    system_message = (
        "You are a helpful assistant with access to tools. "
        "When you need to call a function, output a JSON object with 'name' and 'arguments' fields."
    )

    # 可用工具的描述（增加上下文长度）
    tools_description = """
Available functions:
- get_weather(city: str, unit: str = "celsius") -> str: Get the current weather for a city
- search_web(query: str, num_results: int = 5) -> list: Search the web for information
- get_stock_price(symbol: str) -> dict: Get the current stock price
- send_email(to: str, subject: str, body: str) -> bool: Send an email
- calculate(expression: str) -> float: Evaluate a mathematical expression
- get_time(timezone: str = "UTC") -> str: Get the current time in a timezone
- translate(text: str, source_lang: str, target_lang: str) -> str: Translate text
- get_news(topic: str, count: int = 5) -> list: Get latest news articles
"""

    # 根据目标长度填充上下文
    user_message = f"Please help me with the following task.\n\n{tools_description}\n"

    # 如果需要更长的上下文，添加额外的对话历史
    if context_length > 512:
        extra_turns = (context_length - 512) // 200
        for i in range(extra_turns):
            user_message += f"\nPrevious interaction {i+1}: The user asked about various topics and the assistant provided helpful responses using the available tools.\n"

    user_message += "\nNow, what's the weather in Tokyo and the current time in PST?"

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message[:context_length]},
    ]

    return messages


# =============================================================================
# 异步请求发送
# =============================================================================

async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    messages: list[dict],
    max_tokens: int = 128,
) -> RequestResult:
    """
    发送单个推理请求并测量性能。

    使用 vLLM 的 OpenAI 兼容 API：
    POST /v1/chat/completions

    为了测量 TTFT，使用 stream=True（流式输出）：
    - 服务器一生成一个 token 就立即返回（不等全部生成完）
    - 第一个 chunk 到达的时间就是 TTFT

    参数：
        session: aiohttp 会话（复用 TCP 连接）
        base_url: vLLM 服务地址
        messages: 对话消息列表
        max_tokens: 最大生成 token 数
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": "qwen3-fc",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.1,  # 低温度，输出更确定
        "stream": True,      # 流式输出，用于测量 TTFT
    }

    start_time = time.monotonic()  # monotonic 不受系统时间调整影响
    first_token_time = None
    output_tokens = 0

    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                error_text = await response.text()
                return RequestResult(
                    success=False,
                    ttft=0.0,
                    total_latency=time.monotonic() - start_time,
                    output_tokens=0,
                    tokens_per_second=0.0,
                    error=f"HTTP {response.status}: {error_text[:200]}",
                )

            # 流式读取响应
            # SSE (Server-Sent Events) 格式：每个 chunk 以 "data: " 开头
            async for line in response.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data: "):
                    continue
                if decoded == "data: [DONE]":
                    break

                try:
                    data = json.loads(decoded[6:])  # 去掉 "data: " 前缀
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            if first_token_time is None:
                                first_token_time = time.monotonic()
                            # 简单估算 token 数（实际应该用 tokenizer 计算）
                            output_tokens += 1
                except json.JSONDecodeError:
                    continue

        end_time = time.monotonic()
        total_latency = end_time - start_time

        ttft = (first_token_time - start_time) if first_token_time else total_latency

        # 计算生成速度
        generation_time = (end_time - first_token_time) if first_token_time else total_latency
        tokens_per_second = output_tokens / generation_time if generation_time > 0 else 0.0

        return RequestResult(
            success=True,
            ttft=ttft,
            total_latency=total_latency,
            output_tokens=output_tokens,
            tokens_per_second=tokens_per_second,
        )

    except asyncio.TimeoutError:
        return RequestResult(
            success=False,
            ttft=0.0,
            total_latency=time.monotonic() - start_time,
            output_tokens=0,
            tokens_per_second=0.0,
            error="Request timeout",
        )
    except Exception as e:
        return RequestResult(
            success=False,
            ttft=0.0,
            total_latency=time.monotonic() - start_time,
            output_tokens=0,
            tokens_per_second=0.0,
            error=str(e),
        )


# =============================================================================
# 压测核心函数
# =============================================================================

async def run_benchmark(
    base_url: str,
    concurrency: int,
    context_length: int,
    num_requests: int = 20,
    max_tokens: int = 128,
    timeout: int = 120,
) -> BenchmarkResult:
    """
    运行一组压测。

    工作流程：
    1. 创建 num_requests 个测试 prompt
    2. 使用 asyncio.Semaphore 控制并发数
    3. 并发发送所有请求
    4. 收集结果并计算统计数据

    asyncio.Semaphore 是什么？
        信号量（Semaphore）限制同时运行的协程数量。
        例如 Semaphore(4) 表示最多 4 个请求同时发送。
        第 5 个请求会等待前面某个请求完成后才发送。

    参数：
        base_url: vLLM 服务地址
        concurrency: 并发数
        context_length: 输入上下文长度
        num_requests: 总请求数
        max_tokens: 每个请求最大生成 token 数
        timeout: 请求超时时间（秒）
    """
    print(f"\n  并发={concurrency}, 上下文={context_length}, 请求数={num_requests}")

    # 创建信号量控制并发
    semaphore = asyncio.Semaphore(concurrency)

    # 生成测试 prompt
    messages = generate_test_prompt(context_length)

    async def bounded_request(session: aiohttp.ClientSession) -> RequestResult:
        """带并发限制的请求"""
        async with semaphore:
            return await send_request(session, base_url, messages, max_tokens)

    # 创建 aiohttp 会话（复用 TCP 连接）
    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    client_timeout = aiohttp.ClientTimeout(total=timeout)

    start_time = time.monotonic()

    async with aiohttp.ClientSession(connector=connector, timeout=client_timeout) as session:
        # 并发发送所有请求
        tasks = [bounded_request(session) for _ in range(num_requests)]
        results = await asyncio.gather(*tasks)

    duration = time.monotonic() - start_time

    # ---- 统计分析 ----
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]

    if not successful:
        return BenchmarkResult(
            concurrency=concurrency,
            context_length=context_length,
            num_requests=num_requests,
            successful=0,
            failed=len(failed),
            success_rate=0.0,
            ttft_p50=0.0,
            ttft_p95=0.0,
            ttft_avg=0.0,
            latency_p50=0.0,
            latency_p95=0.0,
            latency_avg=0.0,
            throughput_avg=0.0,
            total_throughput=0.0,
            duration=duration,
            errors=[r.error or "Unknown" for r in failed],
        )

    # 提取各指标的值
    ttfts = sorted([r.ttft for r in successful])
    latencies = sorted([r.total_latency for r in successful])
    throughputs = [r.tokens_per_second for r in successful]

    # 计算百分位数
    # percentile 的原理：将数据排序后，取第 N% 位置的值
    def percentile(data: list[float], p: float) -> float:
        """计算第 p 百分位数"""
        if not data:
            return 0.0
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        c = f + 1 if f + 1 < len(data) else f
        d = k - f
        return data[f] + d * (data[c] - data[f])

    return BenchmarkResult(
        concurrency=concurrency,
        context_length=context_length,
        num_requests=num_requests,
        successful=len(successful),
        failed=len(failed),
        success_rate=len(successful) / num_requests,
        ttft_p50=percentile(ttfts, 50),
        ttft_p95=percentile(ttfts, 95),
        ttft_avg=statistics.mean(ttfts),
        latency_p50=percentile(latencies, 50),
        latency_p95=percentile(latencies, 95),
        latency_avg=statistics.mean(latencies),
        throughput_avg=statistics.mean(throughputs) if throughputs else 0.0,
        total_throughput=sum(throughputs),
        duration=duration,
        errors=[r.error or "Unknown" for r in failed],
    )


# =============================================================================
# 结果展示
# =============================================================================

def print_results(results: list[BenchmarkResult]):
    """以表格形式打印压测结果"""
    print("\n" + "=" * 100)
    print("  压测结果汇总")
    print("=" * 100)

    # 表头
    header = (
        f"{'并发':>4s}  {'上下文':>6s}  {'成功率':>6s}  "
        f"{'TTFT P50':>9s}  {'TTFT P95':>9s}  "
        f"{'延迟 P50':>9s}  {'延迟 P95':>9s}  "
        f"{'速度 avg':>9s}  {'总耗时':>7s}"
    )
    print(header)
    print("-" * 100)

    for r in results:
        row = (
            f"{r.concurrency:>4d}  {r.context_length:>6d}  {r.success_rate:>5.1%}  "
            f"{r.ttft_p50:>8.3f}s  {r.ttft_p95:>8.3f}s  "
            f"{r.latency_p50:>8.3f}s  {r.latency_p95:>8.3f}s  "
            f"{r.throughput_avg:>7.1f}t/s  {r.duration:>6.1f}s"
        )
        print(row)

    print("=" * 100)

    # 打印失败的错误信息
    all_errors = []
    for r in results:
        all_errors.extend(r.errors)
    if all_errors:
        print(f"\n  ⚠️ 共有 {len(all_errors)} 个失败的请求")
        # 去重显示错误类型
        unique_errors = set(all_errors)
        for err in unique_errors:
            count = all_errors.count(err)
            print(f"    [{count}x] {err[:100]}")


def save_results(results: list[BenchmarkResult], output_path: str):
    """保存结果到 JSON 文件"""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [asdict(r) for r in results],
    }
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n  结果已保存到: {output_path}")


# =============================================================================
# 图表生成（可选）
# =============================================================================

def generate_charts(results: list[BenchmarkResult], output_dir: str = "eval"):
    """
    生成性能图表（需要安装 matplotlib）。

    图表包括：
    1. TTFT vs 并发数
    2. 吞吐量 vs 并发数
    3. 成功率 vs 并发数
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  [提示] matplotlib 未安装，跳过图表生成。")
        print("  安装方法: pip install matplotlib")
        return

    # 按上下文长度分组
    context_lengths = sorted(set(r.context_length for r in results))
    concurrencies = sorted(set(r.concurrency for r in results))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ctx_len in context_lengths:
        ctx_results = [r for r in results if r.context_length == ctx_len]
        ctx_results.sort(key=lambda r: r.concurrency)

        conc = [r.concurrency for r in ctx_results]

        # 图 1: TTFT
        axes[0].plot(conc, [r.ttft_p50 for r in ctx_results], marker="o", label=f"P50 ctx={ctx_len}")
        axes[0].plot(conc, [r.ttft_p95 for r in ctx_results], marker="s", linestyle="--", label=f"P95 ctx={ctx_len}")

        # 图 2: 吞吐量
        axes[1].plot(conc, [r.throughput_avg for r in ctx_results], marker="o", label=f"ctx={ctx_len}")

        # 图 3: 成功率
        axes[2].plot(conc, [r.success_rate * 100 for r in ctx_results], marker="o", label=f"ctx={ctx_len}")

    axes[0].set_xlabel("并发数")
    axes[0].set_ylabel("TTFT (秒)")
    axes[0].set_title("Time To First Token")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("并发数")
    axes[1].set_ylabel("Tokens/s")
    axes[1].set_title("平均吞吐量")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("并发数")
    axes[2].set_ylabel("成功率 (%)")
    axes[2].set_title("请求成功率")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0, 105)

    plt.tight_layout()

    chart_path = os.path.join(output_dir, "benchmark_charts.png")
    plt.savefig(chart_path, dpi=150)
    print(f"\n  图表已保存到: {chart_path}")
    plt.close()


# =============================================================================
# 主函数
# =============================================================================

async def main_async(args):
    """异步主函数"""
    print("=" * 60)
    print("  Qwen3-FC 推理服务性能压测")
    print("=" * 60)
    print(f"  服务地址: {args.base_url}")

    # 确定测试矩阵
    if args.full_matrix:
        concurrencies = [1, 2, 4, 8, 16, 32]
        context_lengths = [256, 1024, 2048, 4096]
    else:
        concurrencies = args.concurrency
        context_lengths = args.context_length

    total_tests = len(concurrencies) * len(context_lengths)
    print(f"  并发数: {concurrencies}")
    print(f"  上下文长度: {context_lengths}")
    print(f"  每组请求数: {args.num_requests}")
    print(f"  总测试组数: {total_tests}")
    print("=" * 60)

    # 先检查服务是否可用
    print("\n检查服务状态...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{args.base_url}/health") as resp:
                if resp.status == 200:
                    print("  ✅ 服务正常")
                else:
                    print(f"  ❌ 服务异常 (HTTP {resp.status})")
                    return
    except Exception as e:
        print(f"  ❌ 无法连接到服务: {e}")
        print(f"  请确保服务已启动: bash scripts/serve.sh")
        return

    # 运行压测
    all_results = []
    test_num = 0

    for ctx_len in context_lengths:
        for conc in concurrencies:
            test_num += 1
            print(f"\n[{test_num}/{total_tests}] 测试中...")

            result = await run_benchmark(
                base_url=args.base_url,
                concurrency=conc,
                context_length=ctx_len,
                num_requests=args.num_requests,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
            )
            all_results.append(result)

            # 打印单组结果
            print(f"    成功: {result.successful}/{result.num_requests}, "
                  f"TTFT P50: {result.ttft_p50:.3f}s, "
                  f"速度: {result.throughput_avg:.1f} t/s")

    # 打印汇总结果
    print_results(all_results)

    # 保存结果
    if args.output:
        save_results(all_results, args.output)

    # 生成图表
    if args.chart:
        generate_charts(all_results)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="vLLM 推理服务性能压测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000",
        help="vLLM 服务地址 (默认: http://localhost:8000)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        nargs="+",  # 接受多个值
        default=[1, 4, 8],
        help="并发数列表 (默认: 1 4 8)",
    )
    parser.add_argument(
        "--context-length",
        type=int,
        nargs="+",
        default=[256, 1024],
        help="上下文长度列表 (默认: 256 1024)",
    )
    parser.add_argument(
        "--full-matrix",
        action="store_true",
        help="运行完整压测矩阵 (并发 1~32 × 上下文 256~4096)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=20,
        help="每组测试的请求数 (默认: 20)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=128,
        help="每个请求最大生成 token 数 (默认: 128)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="请求超时时间（秒）(默认: 120)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval/benchmark_results.json",
        help="结果保存路径 (默认: eval/benchmark_results.json)",
    )
    parser.add_argument(
        "--chart",
        action="store_true",
        help="生成性能图表（需要 matplotlib）",
    )

    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
