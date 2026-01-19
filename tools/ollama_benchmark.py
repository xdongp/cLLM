#!/usr/bin/env python3
"""
Ollama Benchmark Tool
用于测试Ollama API的性能
"""

import argparse
import json
import time
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import requests

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger('ollama-benchmark')

class OllamaBenchmark:
    def __init__(self, server_url: str, model: str = "qwen3:0.6b"):
        self.server_url = server_url.rstrip('/')
        self.model = model
        self.api_url = f"{self.server_url}/api/generate"

    def test_api_sequential(self, num_requests: int, max_tokens: int) -> Dict[str, Any]:
        """顺序测试API性能"""
        logger.info(f"Running Ollama sequential test: {num_requests} requests, {max_tokens} tokens each...")

        response_times = []
        total_tokens = []
        tokens_per_second_list = []
        prompt_tokens = []
        generated_tokens = []

        for i in range(num_requests):
            start_time = time.time()

            try:
                payload = {
                    "model": self.model,
                    "prompt": "Hello, how are you?",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    response_time = time.time() - start_time
                    
                    # 提取token信息 - 使用context字段获取真实token数
                    if 'context' in data:
                        # context包含所有tokens（prompt + generated）
                        # 假设prompt大约10个tokens，剩余是生成的
                        # 更准确的方法是计算生成的tokens数
                        generated_token_count = len(data['context']) - 10
                        if generated_token_count < 0:
                            generated_token_count = len(data['context'])
                    elif 'response' in data:
                        # 备用方法：按字符数估算（中文每个字符约1个token）
                        generated_token_count = len(data['response'])
                    else:
                        generated_token_count = max_tokens

                    response_times.append(response_time)
                    total_tokens.append(generated_token_count)
                    tokens_per_second_list.append(generated_token_count / response_time)
                    generated_tokens.append(generated_token_count)
                    
                    logger.info(f"  Request {i+1}/{num_requests}: ✓ {response_time:.2f}s - Generated: {generated_token_count} tokens")
                else:
                    logger.error(f"  Request {i+1}/{num_requests}: ✗ HTTP {response.status_code}")

            except Exception as e:
                logger.error(f"  Request {i+1}/{num_requests}: ✗ Error: {str(e)}")

        total_time = sum(response_times)
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        avg_throughput = sum(total_tokens) / total_time if total_time > 0 else 0
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
        total_tokens_processed = sum(total_tokens)
        avg_generated_tokens = sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0

        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("")
        logger.info("Ollama API Sequential Test Statistics:")
        logger.info("--------------------------------------------------")
        logger.info(f"  Total requests: {num_requests}")
        logger.info(f"  Successful requests: {len(response_times)}")
        logger.info(f"  Failed requests: {num_requests - len(response_times)}")
        logger.info(f"  Avg response time: {avg_response_time:.2f}s")
        logger.info(f"  Min response time: {min_response_time:.2f}s")
        logger.info(f"  Max response time: {max_response_time:.2f}s")
        logger.info(f"  Avg throughput: {avg_throughput:.2f} tokens/sec")
        logger.info(f"  Avg tokens per second: {avg_tokens_per_second:.2f} tokens/sec")
        logger.info(f"  Total tokens processed: {total_tokens_processed}")
        logger.info(f"  Avg generated tokens: {avg_generated_tokens:.2f}")

        return {
            "total_requests": num_requests,
            "successful_requests": len(response_times),
            "failed_requests": num_requests - len(response_times),
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "avg_throughput": avg_throughput,
            "avg_tokens_per_second": avg_tokens_per_second,
            "total_tokens_processed": total_tokens_processed,
            "avg_generated_tokens": avg_generated_tokens,
            "total_time": total_time
        }

    def test_api_concurrent(self, num_requests: int, concurrency: int, max_tokens: int) -> Dict[str, Any]:
        """并发测试API性能"""
        logger.info(f"Running Ollama concurrent test: {num_requests} requests, {concurrency} concurrency, {max_tokens} tokens each...")

        response_times = []
        total_tokens = []
        tokens_per_second_list = []
        prompt_tokens = []
        generated_tokens = []

        def send_request(request_id: int):
            start_time = time.time()

            try:
                payload = {
                    "model": self.model,
                    "prompt": "Hello, how are you?",
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens
                    }
                }

                response = requests.post(
                    self.api_url,
                    json=payload,
                    timeout=60
                )

                if response.status_code == 200:
                    data = response.json()
                    response_time = time.time() - start_time
                    
                    # 提取token信息 - 使用context字段获取真实token数
                    if 'context' in data:
                        # context包含所有tokens（prompt + generated）
                        # 假设prompt大约10个tokens，剩余是生成的
                        generated_token_count = len(data['context']) - 10
                        if generated_token_count < 0:
                            generated_token_count = len(data['context'])
                    elif 'response' in data:
                        # 备用方法：按字符数估算（中文每个字符约1个token）
                        generated_token_count = len(data['response'])
                    else:
                        generated_token_count = max_tokens

                    return {
                        "request_id": request_id,
                        "success": True,
                        "response_time": response_time,
                        "generated_tokens": generated_token_count,
                        "tokens_per_second": generated_token_count / response_time
                    }
                else:
                    logger.error(f"  Request {request_id+1}/{num_requests}: ✗ HTTP {response.status_code}")
                    return {
                        "request_id": request_id,
                        "success": False,
                        "response_time": 0,
                        "generated_tokens": 0,
                        "tokens_per_second": 0
                    }

            except Exception as e:
                logger.error(f"  Request {request_id+1}/{num_requests}: ✗ Error: {str(e)}")
                return {
                    "request_id": request_id,
                    "success": False,
                    "response_time": 0,
                    "generated_tokens": 0,
                    "tokens_per_second": 0
                }

        start_time = time.time()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(send_request, i): i for i in range(num_requests)}

            for future in as_completed(futures):
                result = future.result()
                if result['success']:
                    response_times.append(result['response_time'])
                    total_tokens.append(result['generated_tokens'])
                    tokens_per_second_list.append(result['tokens_per_second'])
                    generated_tokens.append(result['generated_tokens'])
                    
                    logger.info(f"  Request {result['request_id']+1}/{num_requests}: ✓ {result['response_time']:.2f}s - Generated: {result['generated_tokens']} tokens")

        total_time = time.time() - start_time
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        avg_throughput = sum(total_tokens) / total_time if total_time > 0 else 0
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
        total_tokens_processed = sum(total_tokens)
        avg_generated_tokens = sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0

        logger.info(f"Total time: {total_time:.2f}s")
        logger.info("")
        logger.info("Ollama API Concurrent Test Statistics:")
        logger.info("--------------------------------------------------")
        logger.info(f"  Total requests: {num_requests}")
        logger.info(f"  Successful requests: {len(response_times)}")
        logger.info(f"  Failed requests: {num_requests - len(response_times)}")
        logger.info(f"  Avg response time: {avg_response_time:.2f}s")
        logger.info(f"  Min response time: {min_response_time:.2f}s")
        logger.info(f"  Max response time: {max_response_time:.2f}s")
        logger.info(f"  Avg throughput: {avg_throughput:.2f} tokens/sec")
        logger.info(f"  Avg tokens per second: {avg_tokens_per_second:.2f} tokens/sec")
        logger.info(f"  Total tokens processed: {total_tokens_processed}")
        logger.info(f"  Avg generated tokens: {avg_generated_tokens:.2f}")

        return {
            "total_requests": num_requests,
            "successful_requests": len(response_times),
            "failed_requests": num_requests - len(response_times),
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "avg_throughput": avg_throughput,
            "avg_tokens_per_second": avg_tokens_per_second,
            "total_tokens_processed": total_tokens_processed,
            "avg_generated_tokens": avg_generated_tokens,
            "total_time": total_time
        }

def load_prompts_from_file(file_path: str) -> List[str]:
    """从文件加载prompts，每行一个"""
    if not os.path.exists(file_path):
        logger.warning(f"Prompt file not found: {file_path}")
        return []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            prompts = [line.strip() for line in f if line.strip()]
        logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
        return prompts
    except Exception as e:
        logger.error(f"Failed to load prompts from {file_path}: {e}")
        return []


def get_prompts(prompts_file: str, num_requests: int) -> List[str]:
    """获取prompts列表，支持从文件读取或使用默认prompts"""
    # 尝试从文件加载
    prompts = load_prompts_from_file(prompts_file) if prompts_file else []
    
    # 如果文件加载失败或没有指定文件，使用默认prompts
    if not prompts:
        logger.info("Using default prompts")
        prompts = [
            "什么是人工智能？",
            "机器学习的应用领域有哪些？",
            "深度学习和神经网络的关系是什么？",
            "自然语言处理的主要挑战是什么？",
            "计算机视觉在自动驾驶中的应用有哪些？"
        ]
    
    # 如果prompts数量不足，循环使用
    expanded_prompts = []
    for i in range(num_requests):
        expanded_prompts.append(prompts[i % len(prompts)])
    
    return expanded_prompts


def main():
    parser = argparse.ArgumentParser(description="Ollama Benchmark Tool")
    parser.add_argument("--server-url", type=str, default="http://localhost:11434", help="Ollama server URL")
    parser.add_argument("--model", type=str, default="qwen3:0.6b", help="Model name")
    parser.add_argument("--test-type", choices=["api-sequential", "api-concurrent", "all"], 
                        default="all", help="Test type")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency level")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output-file", type=str, default="", help="Output JSON file for test results")
    parser.add_argument("--prompts-file", type=str, default="data/test_prompts_500.txt", help="File containing prompts (one per line)")

    args = parser.parse_args()

    logger.info("="*50)
    logger.info("Ollama Benchmark Tool")
    logger.info("="*50)
    logger.info(f"Server URL: {args.server_url}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Number of requests: {args.requests}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info(f"Prompts file: {args.prompts_file}")
    logger.info("="*50)

    benchmark = OllamaBenchmark(args.server_url, args.model)

    # 获取prompts列表
    prompts = get_prompts(args.prompts_file, args.requests)

    seq_stats = None
    conc_stats = None

    if args.test_type in ["api-sequential", "all"]:
        seq_stats = benchmark.test_api_sequential(args.requests, args.max_tokens)
        logger.info("")

    if args.test_type in ["api-concurrent", "all"]:
        conc_stats = benchmark.test_api_concurrent(args.requests, args.concurrency, args.max_tokens)

    if args.output_file:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        output_data = {
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "server_url": args.server_url,
            "model": args.model,
            "requests": args.requests,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "sequential": seq_stats,
            "concurrent": conc_stats
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to: {args.output_file}")

if __name__ == "__main__":
    main()
