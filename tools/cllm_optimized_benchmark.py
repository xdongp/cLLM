#!/usr/bin/env python3
"""
cLLM优化基准测试脚本 - 对标Ollama性能
使用更大的max_tokens参数以提升吞吐量
"""

import requests
import json
import time
import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("cLLM-optimized-benchmark")


class CLMLOptimizedBenchmarkTester:
    """cLLM优化基准测试器"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.generate_url = f"{server_url}/generate"
        self.health_url = f"{server_url}/health"
    
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def send_api_request(self, prompt: str, max_tokens: int, temperature: float = 0.7) -> Dict[str, Any]:
        """发送API请求"""
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
                timeout=600
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                data = result.get("data", {})
                generated_text = data.get("text", "")
                tokens_per_second = data.get("tokens_per_second", 0)
                response_time = end_time - start_time
                
                if tokens_per_second > 0:
                    estimated_tokens = int(tokens_per_second * response_time)
                elif any(ord(c) > 127 for c in generated_text):
                    estimated_tokens = len(generated_text)
                else:
                    estimated_tokens = len(generated_text.split())
                
                return {
                    "success": True,
                    "response_time": response_time,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": estimated_tokens,
                    "total_tokens": len(prompt) + estimated_tokens,
                    "start_time": start_time,
                    "end_time": end_time,
                    "finish_reason": "length",
                    "tokens_per_second": tokens_per_second
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
        tokens_per_second_list = [r.get("tokens_per_second", 0) for r in successful_results]
        
        if len(successful_results) > 0:
            first_request_start = min(r["start_time"] for r in successful_results)
            last_request_end = max(r["end_time"] for r in successful_results)
            total_test_time = last_request_end - first_request_start
            actual_throughput = sum(generated_tokens) / total_test_time if total_test_time > 0 else 0
        else:
            actual_throughput = 0
        
        avg_tokens_per_second = sum(tokens_per_second_list) / len(tokens_per_second_list) if tokens_per_second_list else 0
        
        return {
            "total_requests": len(results),
            "successful_requests": len(successful_results),
            "failed_requests": failed_requests,
            "avg_response_time": sum(response_times) / len(response_times),
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "avg_throughput": actual_throughput,
            "avg_tokens_per_second": avg_tokens_per_second,
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
            logger.info(f"  Request {i+1}/{num_requests}: {status} {result['response_time']:.2f}s - Generated: {result.get('generated_tokens', 0)} tokens")
        
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time:.2f}s")
        
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
                logger.info(f"  Request {index+1}/{num_requests}: {status} {result['response_time']:.2f}s - Generated: {result.get('generated_tokens', 0)} tokens")
        
        total_time = time.time() - start_time
        logger.info(f"Total time: {total_time:.2f}s")
        
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
        logger.info(f"  Avg generated tokens: {stats.get('avg_generated_tokens', 0):.2f}")


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
    parser = argparse.ArgumentParser(description="cLLM Optimized Benchmark Tool")
    parser.add_argument("--server-url", type=str, default="http://localhost:8080", help="cLLM server URL")
    parser.add_argument("--test-type", choices=["api-sequential", "api-concurrent", "all"], 
                       default="all", help="Test type")
    parser.add_argument("--requests", type=int, default=10, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=5, help="Concurrency level")
    parser.add_argument("--max-tokens", type=int, default=600, help="Max tokens to generate (optimized for Ollama comparison)")
    parser.add_argument("--output-file", type=str, default="", help="Output JSON file for test results")
    parser.add_argument("--prompts-file", type=str, default="data/test_prompts_500.txt", help="File containing prompts (one per line)")
    
    args = parser.parse_args()
    
    tester = CLMLOptimizedBenchmarkTester(server_url=args.server_url)
    
    # 获取prompts列表
    prompts = get_prompts(args.prompts_file, args.requests)
    
    logger.info("=" * 50)
    logger.info("cLLM Optimized Benchmark Tool")
    logger.info("=" * 50)
    logger.info(f"Server URL: {args.server_url}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Number of requests: {args.requests}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Max tokens: {args.max_tokens} (optimized for Ollama comparison)")
    logger.info("=" * 50)
    
    try:
        seq_stats = None
        conc_stats = None
        if args.test_type in ["api-sequential", "api-concurrent", "all"]:
            if not tester.check_server_health():
                logger.error("Error: Cannot connect to cLLM server, please ensure that cLLM is running")
                return
            
            if args.test_type in ["api-sequential", "all"]:
                seq_results = tester.run_api_sequential_test(args.requests, args.max_tokens, prompts)
                seq_stats = tester.calculate_statistics(seq_results)
                tester.print_statistics(seq_stats, "cLLM API Sequential Test (Optimized)")
            
            if args.test_type in ["api-concurrent", "all"]:
                conc_results = tester.run_api_concurrent_test(args.requests, args.max_tokens, args.concurrency, prompts)
                conc_stats = tester.calculate_statistics(conc_results)
                tester.print_statistics(conc_stats, "cLLM API Concurrent Test (Optimized)")

        if args.output_file:
            output_dir = os.path.dirname(args.output_file)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            output_data = {
                "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "server_url": args.server_url,
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