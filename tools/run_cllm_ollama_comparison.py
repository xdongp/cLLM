#!/usr/bin/env python3
"""
cLLM vs Ollama 性能对比测试脚本
测试方案: 160请求，5并发，50 tokens
适用场景: 大规模性能对比测试
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

import requests

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('cllm-ollama-comparison')

def check_server(url: str, name: str) -> bool:
    """检查服务器是否运行"""
    try:
        if name == "Ollama":
            response = requests.get(f"{url}/api/tags", timeout=5)
        else:
            response = requests.get(f"{url}/health", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def run_cllm_test(
    server_url: str,
    requests_count: int,
    concurrency: int,
    max_tokens: int,
    output_file: str = None
) -> Dict[str, Any]:
    """运行cLLM测试"""
    logger.info("=" * 60)
    logger.info("运行cLLM性能测试")
    logger.info("=" * 60)
    logger.info(f"测试参数: {requests_count}请求, {concurrency}并发, {max_tokens} tokens")
    logger.info("")

    # 导入测试模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cllm_optimized_benchmark import run_benchmark

    # 运行测试
    results = run_benchmark(
        server_url=server_url,
        test_type="all",
        requests=requests_count,
        concurrency=concurrency,
        max_tokens=max_tokens,
        output_file=output_file
    )

    return results

def run_ollama_test(
    server_url: str,
    model: str,
    requests_count: int,
    concurrency: int,
    max_tokens: int,
    output_file: str = None
) -> Dict[str, Any]:
    """运行Ollama测试"""
    logger.info("=" * 60)
    logger.info("运行Ollama性能测试")
    logger.info("=" * 60)
    logger.info(f"测试参数: {requests_count}请求, {concurrency}并发, {max_tokens} tokens")
    logger.info(f"模型: {model}")
    logger.info("")

    # 导入测试模块
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ollama_benchmark import run_benchmark as run_ollama_benchmark

    # 运行测试
    results = run_ollama_benchmark(
        server_url=server_url,
        model=model,
        test_type="all",
        requests=requests_count,
        concurrency=concurrency,
        max_tokens=max_tokens,
        output_file=output_file
    )

    return results

def generate_report(
    cllm_results: Dict[str, Any],
    ollama_results: Dict[str, Any],
    requests_count: int,
    concurrency: int,
    max_tokens: int,
    model: str
) -> str:
    """生成测试报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    report_file = f"docs/testing/cllm_vs_ollama_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

    report = f"""# cLLM vs Ollama 性能对比测试报告

**测试时间**: {timestamp}
**测试环境**: {os.uname().sysname} {os.uname().machine}
**测试方案**: {requests_count}请求, {concurrency}并发, {max_tokens} tokens
**模型**: {model}

## 测试结果摘要

### cLLM测试结果

| 指标 | 顺序测试 | 并发测试 |
|------|---------|---------|
| 总请求数 | {cllm_results.get('sequential', {}).get('total_requests', 0)} | {cllm_results.get('concurrent', {}).get('total_requests', 0)} |
| 成功请求数 | {cllm_results.get('sequential', {}).get('successful_requests', 0)} | {cllm_results.get('concurrent', {}).get('successful_requests', 0)} |
| 失败请求数 | {cllm_results.get('sequential', {}).get('failed_requests', 0)} | {cllm_results.get('concurrent', {}).get('failed_requests', 0)} |
| 成功率 | {cllm_results.get('sequential', {}).get('success_rate', 'N/A')}% | {cllm_results.get('concurrent', {}).get('success_rate', 'N/A')}% |
| 平均响应时间 | {cllm_results.get('sequential', {}).get('avg_response_time', 'N/A')}s | {cllm_results.get('concurrent', {}).get('avg_response_time', 'N/A')}s |
| 最小响应时间 | {cllm_results.get('sequential', {}).get('min_response_time', 'N/A')}s | {cllm_results.get('concurrent', {}).get('min_response_time', 'N/A')}s |
| 最大响应时间 | {cllm_results.get('sequential', {}).get('max_response_time', 'N/A')}s | {cllm_results.get('concurrent', {}).get('max_response_time', 'N/A')}s |
| 总测试时间 | {cllm_results.get('sequential', {}).get('total_time', 'N/A')}s | {cllm_results.get('concurrent', {}).get('total_time', 'N/A')}s |
| 平均吞吐量 | {cllm_results.get('sequential', {}).get('avg_throughput', 'N/A')} t/s | {cllm_results.get('concurrent', {}).get('avg_throughput', 'N/A')} t/s |
| 平均tokens/sec | {cllm_results.get('sequential', {}).get('avg_tokens_per_second', 'N/A')} t/s | {cllm_results.get('concurrent', {}).get('avg_tokens_per_second', 'N/A')} t/s |
| 总处理token数 | {cllm_results.get('sequential', {}).get('total_tokens', 0)} | {cllm_results.get('concurrent', {}).get('total_tokens', 0)} |
| 平均生成token数 | {cllm_results.get('sequential', {}).get('avg_generated_tokens', 'N/A')} | {cllm_results.get('concurrent', {}).get('avg_generated_tokens', 'N/A')} |

### Ollama测试结果

| 指标 | 顺序测试 | 并发测试 |
|------|---------|---------|
| 总请求数 | {ollama_results.get('sequential', {}).get('total_requests', 0)} | {ollama_results.get('concurrent', {}).get('total_requests', 0)} |
| 成功请求数 | {ollama_results.get('sequential', {}).get('successful_requests', 0)} | {ollama_results.get('concurrent', {}).get('successful_requests', 0)} |
| 失败请求数 | {ollama_results.get('sequential', {}).get('failed_requests', 0)} | {ollama_results.get('concurrent', {}).get('failed_requests', 0)} |
| 成功率 | {ollama_results.get('sequential', {}).get('success_rate', 'N/A')}% | {ollama_results.get('concurrent', {}).get('success_rate', 'N/A')}% |
| 平均响应时间 | {ollama_results.get('sequential', {}).get('avg_response_time', 'N/A')}s | {ollama_results.get('concurrent', {}).get('avg_response_time', 'N/A')}s |
| 最小响应时间 | {ollama_results.get('sequential', {}).get('min_response_time', 'N/A')}s | {ollama_results.get('concurrent', {}).get('min_response_time', 'N/A')}s |
| 最大响应时间 | {ollama_results.get('sequential', {}).get('max_response_time', 'N/A')}s | {ollama_results.get('concurrent', {}).get('max_response_time', 'N/A')}s |
| 总测试时间 | {ollama_results.get('sequential', {}).get('total_time', 'N/A')}s | {ollama_results.get('concurrent', {}).get('total_time', 'N/A')}s |
| 平均吞吐量 | {ollama_results.get('sequential', {}).get('avg_throughput', 'N/A')} t/s | {ollama_results.get('concurrent', {}).get('avg_throughput', 'N/A')} t/s |
| 平均tokens/sec | {ollama_results.get('sequential', {}).get('avg_tokens_per_second', 'N/A')} t/s | {ollama_results.get('concurrent', {}).get('avg_tokens_per_second', 'N/A')} t/s |
| 总处理token数 | {ollama_results.get('sequential', {}).get('total_tokens', 0)} | {ollama_results.get('concurrent', {}).get('total_tokens', 0)} |
| 平均生成token数 | {ollama_results.get('sequential', {}).get('avg_generated_tokens', 'N/A')} | {ollama_results.get('concurrent', {}).get('avg_generated_tokens', 'N/A')} |

## 性能对比分析

### 顺序测试对比

| 指标 | cLLM | Ollama | 优势方 |
|------|-------|---------|--------|
| 平均响应时间 | {cllm_results.get('sequential', {}).get('avg_response_time', 'N/A')}s | {ollama_results.get('sequential', {}).get('avg_response_time', 'N/A')}s | {'cLLM' if float(cllm_results.get('sequential', {}).get('avg_response_time', 999)) < float(ollama_results.get('sequential', {}).get('avg_response_time', 999)) else 'Ollama'} |
| 吞吐量 | {cllm_results.get('sequential', {}).get('avg_throughput', 'N/A')} t/s | {ollama_results.get('sequential', {}).get('avg_throughput', 'N/A')} t/s | {'cLLM' if float(cllm_results.get('sequential', {}).get('avg_throughput', 0)) > float(ollama_results.get('sequential', {}).get('avg_throughput', 0)) else 'Ollama'} |
| 成功率 | {cllm_results.get('sequential', {}).get('success_rate', 'N/A')}% | {ollama_results.get('sequential', {}).get('success_rate', 'N/A')}% | {'cLLM' if float(cllm_results.get('sequential', {}).get('success_rate', 0)) > float(ollama_results.get('sequential', {}).get('success_rate', 0)) else 'Ollama'} |

### 并发测试对比

| 指标 | cLLM | Ollama | 优势方 |
|------|-------|---------|--------|
| 平均响应时间 | {cllm_results.get('concurrent', {}).get('avg_response_time', 'N/A')}s | {ollama_results.get('concurrent', {}).get('avg_response_time', 'N/A')}s | {'cLLM' if float(cllm_results.get('concurrent', {}).get('avg_response_time', 999)) < float(ollama_results.get('concurrent', {}).get('avg_response_time', 999)) else 'Ollama'} |
| 吞吐量 | {cllm_results.get('concurrent', {}).get('avg_throughput', 'N/A')} t/s | {ollama_results.get('concurrent', {}).get('avg_throughput', 'N/A')} t/s | {'cLLM' if float(cllm_results.get('concurrent', {}).get('avg_throughput', 0)) > float(ollama_results.get('concurrent', {}).get('avg_throughput', 0)) else 'Ollama'} |
| 成功率 | {cllm_results.get('concurrent', {}).get('success_rate', 'N/A')}% | {ollama_results.get('concurrent', {}).get('success_rate', 'N/A')}% | {'cLLM' if float(cllm_results.get('concurrent', {}).get('success_rate', 0)) > float(ollama_results.get('concurrent', {}).get('success_rate', 0)) else 'Ollama'} |

## 相关文件

- cLLM测试脚本: [tools/cllm_optimized_benchmark.py](tools/cllm_optimized_benchmark.py)
- Ollama测试脚本: [tools/ollama_benchmark.py](tools/ollama_benchmark.py)
- 自动化测试脚本: [tools/run_cllm_ollama_comparison.py](tools/run_cllm_ollama_comparison.py)
- 自动化测试脚本(Bash): [tools/run_cllm_ollama_comparison.sh](tools/run_cllm_ollama_comparison.sh)

---
报告生成时间: {timestamp}"""

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    return report_file

def main():
    parser = argparse.ArgumentParser(
        description='cLLM vs Ollama 性能对比测试脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 默认测试: 160请求，5并发，50 tokens
  python3 tools/run_cllm_ollama_comparison.py

  # 自定义测试参数
  python3 tools/run_cllm_ollama_comparison.py --requests 100 --concurrency 10 --max-tokens 100

  # 测试特定模型
  python3 tools/run_cllm_ollama_comparison.py --model qwen3:0.6b

  # 保存JSON结果
  python3 tools/run_cllm_ollama_comparison.py --save-results
        """
    )

    parser.add_argument(
        '--cllm-url',
        default='http://localhost:18085',
        help='cLLM服务器URL (默认: http://localhost:18085)'
    )

    parser.add_argument(
        '--ollama-url',
        default='http://localhost:11434',
        help='Ollama服务器URL (默认: http://localhost:11434)'
    )

    parser.add_argument(
        '--model',
        default='qwen3:0.6b',
        help='Ollama模型名称 (默认: qwen3:0.6b)'
    )

    parser.add_argument(
        '--requests',
        type=int,
        default=160,
        help='测试请求数 (默认: 160)'
    )

    parser.add_argument(
        '--concurrency',
        type=int,
        default=5,
        help='并发数 (默认: 5)'
    )

    parser.add_argument(
        '--max-tokens',
        type=int,
        default=50,
        help='最大生成token数 (默认: 50)'
    )

    parser.add_argument(
        '--save-results',
        action='store_true',
        help='保存测试结果为JSON文件'
    )

    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='跳过服务器状态检查'
    )

    args = parser.parse_args()

    # 打印欢迎信息
    logger.info("=" * 60)
    logger.info("cLLM vs Ollama 性能对比测试")
    logger.info("=" * 60)
    logger.info("")

    # 检查服务器状态
    if not args.skip_check:
        logger.info("[1/4] 检查cLLM服务器状态...")
        if check_server(args.cllm_url, "cLLM"):
            logger.info(f"✓ cLLM服务器运行正常 ({args.cllm_url})")
        else:
            logger.error(f"✗ cLLM服务器未运行")
            logger.error(f"  启动命令: ./build/bin/cllm_server --config config/config_gpu.yaml")
            sys.exit(1)
        logger.info("")

        logger.info("[2/4] 检查Ollama服务器状态...")
        if check_server(args.ollama_url, "Ollama"):
            logger.info(f"✓ Ollama服务器运行正常 ({args.ollama_url})")
        else:
            logger.error(f"✗ Ollama服务器未运行")
            logger.error(f"  启动命令: ollama serve")
            sys.exit(1)
        logger.info("")

    # 打印测试参数
    logger.info("测试参数:")
    logger.info(f"  请求数: {args.requests}")
    logger.info(f"  并发数: {args.concurrency}")
    logger.info(f"  最大token数: {args.max_tokens}")
    logger.info(f"  Ollama模型: {args.model}")
    logger.info("")

    # 创建results目录
    os.makedirs("results", exist_ok=True)

    # 运行cLLM测试
    logger.info("[3/4] 运行cLLM性能测试...")
    cllm_output_file = f"results/cllm_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" if args.save_results else None
    cllm_results = run_cllm_test(
        server_url=args.cllm_url,
        requests_count=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        output_file=cllm_output_file
    )
    logger.info("✓ cLLM测试完成")
    logger.info("")

    # 运行Ollama测试
    logger.info("[4/4] 运行Ollama性能测试...")
    ollama_output_file = f"results/ollama_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json" if args.save_results else None
    ollama_results = run_ollama_test(
        server_url=args.ollama_url,
        model=args.model,
        requests_count=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        output_file=ollama_output_file
    )
    logger.info("✓ Ollama测试完成")
    logger.info("")

    # 生成报告
    logger.info("[5/4] 生成测试报告...")
    report_file = generate_report(
        cllm_results=cllm_results,
        ollama_results=ollama_results,
        requests_count=args.requests,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        model=args.model
    )
    logger.info(f"✓ 测试报告已生成: {report_file}")
    logger.info("")

    # 总结
    logger.info("=" * 60)
    logger.info("测试完成！")
    logger.info("=" * 60)
    logger.info("")
    logger.info("测试结果:")
    logger.info(f"  - cLLM: 已完成 {args.requests} 个请求的顺序和并发测试")
    logger.info(f"  - Ollama: 已完成 {args.requests} 个请求的顺序和并发测试")
    logger.info("")
    logger.info("查看详细结果:")
    logger.info(f"  - 控制台输出（已显示）")
    if args.save_results:
        logger.info(f"  - JSON结果文件: results/")
    logger.info(f"  - 测试报告: {report_file}")
    logger.info("")
    logger.info("下次测试直接运行:")
    logger.info(f"  python3 tools/run_cllm_ollama_comparison.py")
    logger.info("")

if __name__ == "__main__":
    main()
