#!/usr/bin/env python3
"""
统一基准测试脚本 - 用于测试xLLM在不同场景下的性能表现
统一了API服务器测试和直接模型测试的策略
"""

import torch
import requests
import json
import time
import argparse
import logging
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xLLM-unified-benchmark")

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.model_executor import ModelExecutor
from python.model_executor_optimized import OptimizedModelExecutor
from python.sampler import Sampler

class UnifiedBenchmarkTester:
    """统一基准测试器"""
    
    def __init__(self, model_path: str = None, quantization: str = None, server_url: str = None):
        self.model_path = model_path
        self.quantization = quantization
        self.server_url = server_url
        
        # 初始化模型执行器（如果提供了模型路径）
        if model_path:
            logger.info(f"Initializing model executor with path: {model_path}")
            self.model_executor = ModelExecutor(
                model_path=model_path,
                quantization=quantization,
                use_c_sampler=True,
                enable_compile=True
            )
            
            self.optimized_executor = OptimizedModelExecutor(
                model_path=model_path,
                quantization=quantization,
                use_c_sampler=True,
                enable_compile=True
            )
        else:
            self.model_executor = None
            self.optimized_executor = None
        
        # 初始化API测试器（如果提供了服务器URL）
        if server_url:
            self.api_base_url = server_url
            self.generate_url = f"{server_url}/generate"
            self.health_url = f"{server_url}/health"
        else:
            self.api_base_url = None
            self.generate_url = None
            self.health_url = None
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        if not self.api_base_url:
            return False
        
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def send_api_request(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> Dict[str, Any]:
        """发送API请求"""
        if not self.api_base_url:
            return {
                "success": False,
                "error": "API server URL not provided"
            }
        
        payload = {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        start_time = time.time()
        try:
            response = requests.post(
                self.generate_url,
                headers={"Content-Type": "application/json"},
                data=json.dumps(payload),
                timeout=300  # 增加超时时间到300秒，支持生成更多token
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("text", "")
                
                # 改进token数估算：中文按字符数，英文按单词数
                if any(ord(c) > 127 for c in generated_text):
                    # 包含非ASCII字符（如中文），按字符数估算
                    estimated_tokens = len(generated_text)
                else:
                    # 纯英文，按单词数估算
                    estimated_tokens = len(generated_text.split())
                
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": estimated_tokens,
                    "total_tokens": len(prompt) + estimated_tokens,
                    "start_time": start_time,
                    "end_time": end_time,
                    "finish_reason": "length"  # 假设是因为达到max_tokens结束
                }
            else:
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": f"HTTP {response.status_code}"
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    def direct_model_request(self, input_ids: List[int], max_new_tokens: int, 
                           temperature: float = 0.7, executor_type: str = "standard") -> Dict[str, Any]:
        """直接模型请求"""
        if not self.model_executor:
            return {
                "success": False,
                "error": "Model executor not initialized"
            }
        
        executor = self.model_executor if executor_type == "standard" else self.optimized_executor
        input_tensor = torch.tensor([input_ids])
        
        start_time = time.time()
        try:
            # 检查执行器是否有generate_batch方法，如果没有则尝试generate方法
            if hasattr(executor, 'generate_batch'):
                # 使用generate_batch方法
                batch_inputs = [{
                    'input_ids': input_tensor,
                    'max_new_tokens': max_new_tokens,
                    'temperature': temperature,
                    'do_sample': True
                }]
                outputs = executor.generate_batch(batch_inputs)
            elif hasattr(executor, 'generate'):
                # 使用generate方法（如果存在）
                outputs = executor.generate(
                    input_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )
            else:
                # 如果都没有，返回错误
                return {
                    "success": False,
                    "response_time": 0,
                    "error": f"Executor does not have generate or generate_batch method"
                }
            
            end_time = time.time()
            
            if isinstance(outputs, torch.Tensor):
                generated_tokens = len(outputs[0]) - len(input_ids)  # 计算新生成的token数
            elif isinstance(outputs, list) and len(outputs) > 0:
                # 如果输出是列表，计算第一个输出的token数
                if isinstance(outputs[0], torch.Tensor):
                    generated_tokens = len(outputs[0][0]) - len(input_ids) if len(outputs[0]) > 0 else max_new_tokens
                else:
                    generated_tokens = max_new_tokens  # 假设成功生成了所有token
            else:
                generated_tokens = max_new_tokens  # 假设成功生成了所有token
            
            return {
                "success": True,
                "response_time": end_time - start_time,
                "prompt_tokens": len(input_ids),
                "generated_tokens": generated_tokens,
                "total_tokens": len(input_ids) + generated_tokens,
                "start_time": start_time,
                "end_time": end_time
            }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e)
            }
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计数据"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r["success"]]
        failed_requests = len(results) - len(successful_results)
        
        if not successful_results:
            return {"failed_requests": failed_requests}
        
        response_times = [r["response_time"] for r in successful_results]
        total_tokens = [r["total_tokens"] for r in successful_results]
        generated_tokens = [r["generated_tokens"] for r in successful_results]
        
        # 计算实际吞吐量：总生成token数 / 总测试时间（从第一个请求开始到最后一个请求结束）
        if len(successful_results) > 0:
            first_request_start = min(r["start_time"] for r in successful_results)
            last_request_end = max(r["end_time"] for r in successful_results)
            total_test_time = last_request_end - first_request_start
            actual_throughput = sum(generated_tokens) / total_test_time if total_test_time > 0 else 0
        else:
            actual_throughput = 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_throughput": actual_throughput,
            "total_tokens_processed": sum(total_tokens),
            "avg_generated_tokens": sum(generated_tokens) / len(generated_tokens)
        }
    
    def run_api_sequential_test(self, num_requests: int, max_tokens: int, prompts: List[str]) -> List[Dict[str, Any]]:
        """运行API顺序性能测试"""
        logger.info(f"Running API sequential test: {num_requests} requests, {max_tokens} tokens each...")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            result = self.send_api_request(prompt, max_tokens)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            logger.info(f"  Request {i+1}/{num_requests}: {status} {result['response_time']:.2f}s")
        
        total_time = time.time() - start_time
        
        return results
    
    def run_api_concurrent_test(self, num_requests: int, max_tokens: int, concurrency: int, 
                              prompts: List[str]) -> List[Dict[str, Any]]:
        """运行API并发性能测试"""
        logger.info(f"Running API concurrent test: {num_requests} requests, {concurrency} concurrency, {max_tokens} tokens each...")
        
        results = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_index = {
                executor.submit(self.send_api_request, prompts[i % len(prompts)], max_tokens): i 
                for i in range(num_requests)
            }
            
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)
                index = future_to_index[future]
                status = "✓" if result["success"] else "✗"
                logger.info(f"  Request {index+1}/{num_requests}: {status} {result['response_time']:.2f}s")
        
        total_time = time.time() - start_time
        
        return results
    
    def run_direct_model_test(self, num_requests: int, input_lengths: List[int], max_new_tokens: int, 
                            executor_type: str = "standard") -> List[Dict[str, Any]]:
        """运行直接模型测试"""
        logger.info(f"Running direct model test ({executor_type}): {num_requests} requests, max_new_tokens={max_new_tokens}...")
        
        results = []
        start_time = time.time()
        
        for i in range(num_requests):
            input_len = input_lengths[i % len(input_lengths)]
            input_ids = [100 + (i * 10 + j) % 1000 for j in range(input_len)]  # 生成不同输入
            
            result = self.direct_model_request(input_ids, max_new_tokens, executor_type=executor_type)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            logger.info(f"  Request {i+1}/{num_requests}: input_len={input_len}, {status} {result['response_time']:.2f}s")
        
        total_time = time.time() - start_time
        
        return results
    
    def run_token_length_test(self, token_lengths: List[int], max_new_tokens: int = 10, 
                            executor_type: str = "standard", iterations: int = 3) -> Dict[int, Any]:
        """运行不同token长度的测试"""
        logger.info(f"Running token length test ({executor_type}): {token_lengths} input lengths, {max_new_tokens} new tokens...")
        
        results = {}
        
        for length in token_lengths:
            logger.info(f"\nTesting input length: {length}")
            
            # 生成测试输入
            test_input = [100 + (i % 1000) for i in range(length)]
            
            # 预热
            logger.info("Warming up...")
            for _ in range(2):
                self.direct_model_request(test_input[:5], 3, executor_type=executor_type)
            
            # 测试推理
            times = []
            tokens_per_iteration = []
            
            for i in range(iterations):
                logger.info(f"Iteration {i+1}/{iterations}")
                start_time = time.time()
                
                result = self.direct_model_request(test_input, max_new_tokens, executor_type=executor_type)
                
                end_time = time.time()
                if result["success"]:
                    duration = end_time - start_time
                    times.append(duration)
                    tokens_per_iteration.append(result["generated_tokens"])
                    
                    logger.info(f"Generated {result['generated_tokens']} tokens in {duration:.4f} seconds")
                else:
                    logger.error(f"Failed: {result['error']}")
            
            # 计算统计数据
            if times:
                avg_time = sum(times) / len(times)
                avg_tokens = sum(tokens_per_iteration) / len(tokens_per_iteration)
                tokens_per_second = sum(tokens_per_iteration) / sum(times) if sum(times) > 0 else 0
            else:
                avg_time = 0
                avg_tokens = 0
                tokens_per_second = 0
            
            results[length] = {
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "tokens_per_second": tokens_per_second,
                "individual_times": times
            }
            
            logger.info(f"\nInput length {length} - Results:")
            logger.info(f"  Average time per iteration: {avg_time:.4f} seconds")
            logger.info(f"  Average tokens generated: {avg_tokens:.1f}")
            logger.info(f"  Tokens per second: {tokens_per_second:.2f}")
        
        return results
    
    def print_statistics(self, stats: Dict[str, Any], test_name: str):
        """打印统计结果"""
        logger.info(f"\n{test_name} Statistics:")
        logger.info("-" * 50)
        
        if not stats:
            logger.info("  No results")
            return
        
        if stats.get("failed_requests", 0) == stats.get("total_requests", 0):
            logger.info(f"  All requests failed: {stats['failed_requests']} requests")
            return
        
        logger.info(f"  Total requests: {stats.get('total_requests', 0)}")
        logger.info(f"  Successful requests: {stats.get('successful_requests', 0)}")
        logger.info(f"  Failed requests: {stats.get('failed_requests', 0)}")
        logger.info(f"  Avg response time: {stats.get('avg_response_time', 0):.2f}s")
        logger.info(f"  Min response time: {stats.get('min_response_time', 0):.2f}s")
        logger.info(f"  Max response time: {stats.get('max_response_time', 0):.2f}s")
        logger.info(f"  Avg throughput: {stats.get('avg_throughput', 0):.2f} tokens/sec")
        logger.info(f"  Total tokens processed: {stats.get('total_tokens_processed', 0)}")
        logger.info(f"  Avg generated tokens: {stats.get('avg_generated_tokens', 0):.2f}")
    
    def print_token_length_summary(self, results: Dict[int, Any], test_name: str):
        """打印token长度测试摘要"""
        logger.info(f"\n{test_name} Summary:")
        logger.info("="*80)
        logger.info(f"{'Input Length':<15} {'Avg Time (s)':<15} {'Tokens/s':<15} {'Avg Gen Tokens':<15}")
        logger.info("-"*80)
        
        for length, data in results.items():
            logger.info(f"{length:<15} {data['avg_time']:<15.4f} {data['tokens_per_second']:<15.2f} {data['avg_tokens']:<15.1f}")
        
        logger.info("="*80)


def main():
    parser = argparse.ArgumentParser(description="xLLM Unified Benchmark Tool")
    parser.add_argument("--model-path", type=str, help="Path to the model for direct testing")
    parser.add_argument("--quantization", type=str, help="Quantization type (fp16, int8, int4)")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="xLLM server URL for API testing")
    parser.add_argument("--test-type", choices=["api-sequential", "api-concurrent", "direct-model", "token-length", "all"], 
                       default="all", help="Test type")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens for API tests")
    parser.add_argument("--max-new-tokens", type=int, default=10, help="Max new tokens for direct model tests")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations for token length tests")
    
    args = parser.parse_args()
    
    # 检查是否提供了必要的参数
    if not args.model_path and not args.server_url:
        logger.error("Either --model-path or --server-url must be provided")
        return
    
    # 创建测试器
    tester = UnifiedBenchmarkTester(
        model_path=args.model_path,
        quantization=args.quantization,
        server_url=args.server_url
    )
    
    # 测试提示词
    prompts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
        "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
        "自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
    ]
    
    # Print test configuration
    logger.info("=" * 50)
    logger.info("xLLM Unified Benchmark Tool")
    logger.info("=" * 50)
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Server URL: {args.server_url}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Quantization: {args.quantization}")
    logger.info("=" * 50)
    
    try:
        # API测试
        if args.test_type in ["api-sequential", "api-concurrent", "all"]:
            if args.server_url:
                # 检查服务器健康状态
                if not tester.check_server_health():
                    logger.error("Error: Cannot connect to xLLM server, please ensure the server is running")
                    return
                
                if args.test_type in ["api-sequential", "all"]:
                    # 顺序测试
                    seq_results = tester.run_api_sequential_test(args.requests, args.max_tokens, prompts)
                    seq_stats = tester.calculate_statistics(seq_results)
                    tester.print_statistics(seq_stats, "API Sequential Test")
                
                if args.test_type in ["api-concurrent", "all"]:
                    # 并发测试
                    conc_results = tester.run_api_concurrent_test(args.requests, args.max_tokens, args.concurrency, prompts)
                    conc_stats = tester.calculate_statistics(conc_results)
                    tester.print_statistics(conc_stats, "API Concurrent Test")
            else:
                logger.warning("Server URL not provided, skipping API tests")
        
        # 直接模型测试
        if args.test_type in ["direct-model", "token-length", "all"]:
            if args.model_path:
                if args.test_type in ["direct-model", "all"]:
                    # 直接模型测试 - 标准执行器
                    input_lengths = [50, 100, 200]  # 不同输入长度
                    direct_results_standard = tester.run_direct_model_test(
                        args.requests, input_lengths, args.max_new_tokens, "standard"
                    )
                    direct_stats_standard = tester.calculate_statistics(direct_results_standard)
                    tester.print_statistics(direct_stats_standard, "Direct Model Test (Standard)")
                    
                    # 直接模型测试 - 优化执行器
                    direct_results_optimized = tester.run_direct_model_test(
                        args.requests, input_lengths, args.max_new_tokens, "optimized"
                    )
                    direct_stats_optimized = tester.calculate_statistics(direct_results_optimized)
                    tester.print_statistics(direct_stats_optimized, "Direct Model Test (Optimized)")
                
                if args.test_type in ["token-length", "all"]:
                    # Token长度测试
                    token_lengths = [50, 100, 200, 300, 400, 500]
                    
                    # 标准执行器测试
                    token_results_standard = tester.run_token_length_test(
                        token_lengths, args.max_new_tokens, "standard", args.iterations
                    )
                    tester.print_token_length_summary(token_results_standard, "Token Length Test (Standard)")
                    
                    # 优化执行器测试
                    token_results_optimized = tester.run_token_length_test(
                        token_lengths, args.max_new_tokens, "optimized", args.iterations
                    )
                    tester.print_token_length_summary(token_results_optimized, "Token Length Test (Optimized)")
            else:
                logger.warning("Model path not provided, skipping direct model tests")
    
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        logger.error(e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()