#!/usr/bin/env python3
"""
API基准测试脚本 - 用于测试xLLM API服务器的性能表现
"""

import requests
import json
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("xLLM-api-benchmark")


class APIBenchmarkTester:
    """API基准测试器"""
    
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
                timeout=300
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("text", "")
                
                if any(ord(c) > 127 for c in generated_text):
                    estimated_tokens = len(generated_text)
                else:
                    estimated_tokens = len(generated_text.split())
                
                return {
                    "success": True,
                    "response_time": end_time - start_time,
                    "prompt_tokens": len(prompt),
                    "generated_tokens": estimated_tokens,
                    "total_tokens": len(prompt) + estimated_tokens,
                    "start_time": start_time,
                    "end_time": end_time,
                    "finish_reason": "length"
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
                logger.info(f"  Request {index+1}/{num_requests}: {status} {result['response_time']:.2f}s")
        
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
        logger.info(f"  Total tokens processed: {stats.get('total_tokens_processed', 0)}")
        logger.info(f"  Avg generated tokens: {stats.get('avg_generated_tokens', 0):.2f}")


def main():
    parser = argparse.ArgumentParser(description="xLLM API Benchmark Tool")
    parser.add_argument("--server-url", type=str, default="http://localhost:8000", help="xLLM server URL")
    parser.add_argument("--test-type", choices=["api-sequential", "api-concurrent", "all"], 
                       default="all", help="Test type")
    parser.add_argument("--requests", type=int, default=20, help="Number of requests")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency level")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    
    args = parser.parse_args()
    
    tester = APIBenchmarkTester(server_url=args.server_url)
    
    prompts = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
        "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
        "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
        "自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。",
        "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
    ]
    
    logger.info("=" * 50)
    logger.info("xLLM API Benchmark Tool")
    logger.info("=" * 50)
    logger.info(f"Server URL: {args.server_url}")
    logger.info(f"Test type: {args.test_type}")
    logger.info(f"Number of requests: {args.requests}")
    logger.info(f"Concurrency: {args.concurrency}")
    logger.info(f"Max tokens: {args.max_tokens}")
    logger.info("=" * 50)
    
    try:
        if args.test_type in ["api-sequential", "api-concurrent", "all"]:
            if not tester.check_server_health():
                logger.error("Error: Cannot connect to xLLM server, please ensure the server is running")
                return
            
            if args.test_type in ["api-sequential", "all"]:
                seq_results = tester.run_api_sequential_test(args.requests, args.max_tokens, prompts)
                seq_stats = tester.calculate_statistics(seq_results)
                tester.print_statistics(seq_stats, "API Sequential Test")
            
            if args.test_type in ["api-concurrent", "all"]:
                conc_results = tester.run_api_concurrent_test(args.requests, args.max_tokens, args.concurrency, prompts)
                conc_stats = tester.calculate_statistics(conc_results)
                tester.print_statistics(conc_stats, "API Concurrent Test")
    
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
