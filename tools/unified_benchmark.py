#!/usr/bin/env python3
"""
统一基准测试脚本 - 支持 cLLM 和 Ollama
确保测试方法和计算方式完全一致
"""

import requests
import json
import time
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("unified-benchmark")


class UnifiedBenchmarkTester:
    """统一基准测试器 - 支持 cLLM 和 Ollama"""
    
    def __init__(self, server_type: str, server_url: str, model: Optional[str] = None):
        """
        初始化测试器
        server_type: "cllm" 或 "ollama"
        server_url: 服务器URL
        model: 模型名称（仅Ollama需要）
        """
        self.server_type = server_type
        self.server_url = server_url.rstrip('/')
        self.model = model
        
        if server_type == "cllm":
            self.generate_url = f"{self.server_url}/generate"
            self.health_url = f"{self.server_url}/health"
        elif server_type == "ollama":
            self.generate_url = f"{self.server_url}/api/generate"
            self.health_url = f"{self.server_url}/api/tags"  # Ollama没有health端点，使用tags
        else:
            raise ValueError(f"Unsupported server type: {server_type}")
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            if self.server_type == "cllm":
                response = requests.get(self.health_url, timeout=5)
                return response.status_code == 200
            else:  # ollama
                response = requests.get(self.health_url, timeout=5)
                return response.status_code == 200
        except Exception:
            return False
    
    def send_api_request(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> Dict[str, Any]:
        """发送API请求 - 统一接口"""
        start_time = time.time()
        
        try:
            if self.server_type == "cllm":
                payload = {
                    "prompt": prompt,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": False
                }
                response = requests.post(
                    self.generate_url,
                    headers={"Content-Type": "application/json"},
                    data=json.dumps(payload),
                    timeout=600
                )
            else:  # ollama
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature
                    }
                }
                response = requests.post(
                    self.generate_url,
                    json=payload,
                    timeout=600
                )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            if response.status_code == 200:
                result = response.json()
                
                # 统一提取token信息
                if self.server_type == "cllm":
                    data = result.get("data", {})
                    generated_text = data.get("text", "")
                    tokens_per_second = data.get("tokens_per_second", 0)
                    
                    # 估算生成的tokens
                    if tokens_per_second > 0:
                        generated_tokens = int(tokens_per_second * response_time)
                    elif any(ord(c) > 127 for c in generated_text):
                        generated_tokens = len(generated_text)
                    else:
                        generated_tokens = len(generated_text.split())
                    
                    prompt_tokens = len(prompt)
                else:  # ollama
                    generated_text = result.get("response", "")
                    # 从context字段提取真实token数
                    if 'context' in result:
                        # context包含所有tokens（prompt + generated）
                        # 假设prompt大约10个tokens，剩余是生成的
                        generated_tokens = len(result['context']) - 10
                        if generated_tokens < 0:
                            generated_tokens = len(result['context'])
                    elif generated_text:
                        # 备用方法：按字符数估算（中文每个字符约1个token）
                        generated_tokens = len(generated_text)
                    else:
                        generated_tokens = max_tokens
                    
                    prompt_tokens = 10  # Ollama的prompt tokens估算
                    tokens_per_second = generated_tokens / response_time if response_time > 0 else 0
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "total_tokens": prompt_tokens + generated_tokens,
                    "start_time": start_time,
                    "end_time": end_time,
                    "tokens_per_second": tokens_per_second
                }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "error": f"HTTP {response.status_code}",
                    "start_time": start_time,
                    "end_time": end_time
                }
        except Exception as e:
            end_time = time.time()
            return {
                "success": False,
                "response_time": end_time - start_time,
                "error": str(e),
                "start_time": start_time,
                "end_time": end_time
            }
    
    def calculate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计数据 - 统一计算方法"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get("success", False)]
        failed_requests = len(results) - len(successful_results)
        
        if not successful_results:
            return {"failed_requests": failed_requests}
        
        response_times = [r["response_time"] for r in successful_results]
        total_tokens = [r["total_tokens"] for r in successful_results]
        generated_tokens = [r["generated_tokens"] for r in successful_results]
        tokens_per_second_list = [r.get("tokens_per_second", 0) for r in successful_results]
        
        # 统一时间计算：使用第一个请求开始到最后一个请求结束的时间
        if len(successful_results) > 0:
            first_request_start = min(r["start_time"] for r in successful_results)
            last_request_end = max(r["end_time"] for r in successful_results)
            total_test_time = last_request_end - first_request_start
            # 总吞吐量 = 总生成tokens / 总测试时间
            actual_throughput = sum(generated_tokens) / total_test_time if total_test_time > 0 else 0
        else:
            total_test_time = 0
            actual_throughput = 0
        
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "avg_throughput": actual_throughput,  # 总吞吐量
            "avg_tokens_per_second": avg_tokens_per_second,  # 单请求平均速度
            "total_tokens_processed": sum(total_tokens),
            "total_generated_tokens": sum(generated_tokens),
            "avg_generated_tokens": sum(generated_tokens) / len(generated_tokens) if generated_tokens else 0,
            "total_test_time": total_test_time
        }
    
    def run_api_sequential_test(self, num_requests: int, max_tokens: int, prompts: List[str]) -> List[Dict[str, Any]]:
        """运行API顺序性能测试"""
        logger.info(f"Running API sequential test: {num_requests} requests, {max_tokens} tokens each...")
        
        results = []
        test_start_time = time.time()
        
        for i in range(num_requests):
            prompt = prompts[i % len(prompts)]
            result = self.send_api_request(prompt, max_tokens)
            results.append(result)
            status = "✓" if result["success"] else "✗"
            logger.info(f"  Request {i+1}/{num_requests}: {status} {result['response_time']:.2f}s - Generated: {result.get('generated_tokens', 0)} tokens")
        
        test_total_time = time.time() - test_start_time
        logger.info(f"Total time: {test_total_time:.2f}s")
        
        return results
    
    def run_api_concurrent_test(self, num_requests: int, max_tokens: int, concurrency: int, 
                              prompts: List[str]) -> List[Dict[str, Any]]:
        """运行API并发性能测试"""
        logger.info(f"Running API concurrent test: {num_requests} requests, {concurrency} concurrency, {max_tokens} tokens each...")
        
        results = []
        test_start_time = time.time()
        
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
                logger.info(f"  Request {index+1}/{num_requests}: {status} {result['response_time']:.2f}s - Generated: {result.get('generated_tokens', 0)} tokens")
        
        test_total_time = time.time() - test_start_time
        logger.info(f"Total time: {test_total_time:.2f}s")
        
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
        logger.info(f"  Avg tokens per second: {stats.get('avg_tokens_per_second', 0):.2f} tokens/sec")
        logger.info(f"  Total tokens processed: {stats.get('total_tokens_processed', 0)}")
        logger.info(f"  Total generated tokens: {stats.get('total_generated_tokens', 0)}")
        logger.info(f"  Avg generated tokens: {stats.get('avg_generated_tokens', 0):.2f}")
        logger.info(f"  Total test time: {stats.get('total_test_time', 0):.2f}s")


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
            "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
            "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
            "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
            "自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。",
            "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
        ]
    
    # 如果prompts数量不足，循环使用
    expanded_prompts = []
    for i in range(num_requests):
        expanded_prompts.append(prompts[i % len(prompts)])
    
    return expanded_prompts


def main():
    parser = argparse.ArgumentParser(description="Unified Benchmark Tool for cLLM and Ollama")
    parser.add_argument("--server-type", choices=["cllm", "ollama"], required=True, 
                       help="Server type: cllm or ollama")
    parser.add_argument("--server-url", type=str, 
                       default="http://localhost:8080", 
                       help="Server URL (default: http://localhost:8080 for cLLM, http://localhost:11434 for Ollama)")
    parser.add_argument("--model", type=str, default="qwen3:0.6b", 
                       help="Model name (required for Ollama)")
    parser.add_argument(
        "--test-type",
        choices=["api-sequential", "api-concurrent", "all"],
        default="all",
        help="Test type",
    )
    parser.add_argument("--requests", type=int, default=72, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrency level")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--output-file", type=str, default="", help="Output JSON file for test results")
    parser.add_argument("--prompts-file", type=str, default="", help="File containing prompts (one per line)")
    
    args = parser.parse_args()
    
    # 设置默认URL
    if args.server_type == "ollama" and args.server_url == "http://localhost:8080":
        args.server_url = "http://localhost:11434"
    
    tester = UnifiedBenchmarkTester(
        server_type=args.server_type,
        server_url=args.server_url,
        model=args.model if args.server_type == "ollama" else None
    )
    
    # 获取prompts列表
    prompts = get_prompts(args.prompts_file, args.requests)
    
    logger.info("=" * 50)
    logger.info(f"{args.server_type.upper()} Unified Benchmark Tool")
    logger.info("=" * 50)
    logger.info(f"Server type: {args.server_type}")
    logger.info(f"Server URL: {args.server_url}")
    if args.server_type == "ollama":
        logger.info(f"Model: {args.model}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Number of requests: {args.requests}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info("=" * 50)
    
    try:
        seq_stats = None
        conc_stats = None
        if args.test_type in ["api-sequential", "api-concurrent", "all"]:
            if not tester.check_server_health():
                logger.error(f"Error: Cannot connect to {args.server_type} server, please ensure that the server is running")
                return
            
            if args.test_type in ["api-sequential", "all"]:
                seq_results = tester.run_api_sequential_test(args.requests, args.max_tokens, prompts)
                seq_stats = tester.calculate_statistics(seq_results)
                tester.print_statistics(seq_stats, f"{args.server_type.upper()} API Sequential Test")
            
            if args.test_type in ["api-concurrent", "all"]:
                conc_results = tester.run_api_concurrent_test(args.requests, args.max_tokens, args.concurrency, prompts)
                conc_stats = tester.calculate_statistics(conc_results)
                tester.print_statistics(conc_stats, f"{args.server_type.upper()} API Concurrent Test")

        if args.output_file:
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_data = {
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "server_type": args.server_type,
                "server_url": args.server_url,
                "model": args.model if args.server_type == "ollama" else None,
                "requests": args.requests,
                "concurrency": args.concurrency,
                "max_tokens": args.max_tokens,
                "sequential": seq_stats,
                "concurrent": conc_stats
            }
            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to: {args.output_file}")
    
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
