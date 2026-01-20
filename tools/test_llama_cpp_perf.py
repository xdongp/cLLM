#!/usr/bin/env python3
"""
直接测试 llama.cpp 的性能上限
使用 llama.cpp 的 Python 绑定（如果可用）或通过 subprocess 调用
"""

import subprocess
import time
import json
import sys
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import statistics

def test_llama_cpp_direct(model_path: str, prompt: str, max_tokens: int = 50) -> Dict:
    """使用 llama.cpp 直接测试单个请求的性能"""
    # 这里我们需要一个简单的测试程序
    # 暂时返回模拟结果
    start_time = time.time()
    
    # 模拟推理时间（实际应该调用 llama.cpp）
    time.sleep(0.1)  # 模拟推理
    
    end_time = time.time()
    response_time = end_time - start_time
    
    return {
        "success": True,
        "tokens_generated": max_tokens,
        "response_time": response_time
    }

def run_concurrent_test(model_path: str, num_requests: int, concurrency: int, max_tokens: int):
    """运行并发测试"""
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a short story about a robot.",
        "What are the benefits of exercise?",
    ]
    
    results = []
    start_time = time.time()
    
    def process_request(request_id: int):
        prompt = prompts[request_id % len(prompts)]
        return test_llama_cpp_direct(model_path, prompt, max_tokens)
    
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(process_request, i) for i in range(num_requests)]
        
        for future in as_completed(futures):
            results.append(future.result())
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # 计算统计信息
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    
    if successful:
        response_times = [r["response_time"] for r in successful]
        total_tokens = sum(r["tokens_generated"] for r in successful)
        
        print("\n" + "="*50)
        print("llama.cpp Direct Performance Test Results")
        print("="*50)
        print(f"Total requests: {num_requests}")
        print(f"Successful requests: {len(successful)} ({100.0*len(successful)/num_requests:.1f}%)")
        print(f"Failed requests: {len(failed)}")
        print(f"Avg response time: {statistics.mean(response_times):.2f}s")
        print(f"Min response time: {min(response_times):.2f}s")
        print(f"Max response time: {max(response_times):.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print(f"Avg throughput: {total_tokens/total_time:.2f} tokens/sec")
        print(f"Avg tokens per second: {total_tokens/sum(response_times):.2f} tokens/sec")
        print(f"Total tokens processed: {total_tokens}")
        print(f"Avg generated tokens: {total_tokens/len(successful):.2f}")
        print("="*50)
    else:
        print("All requests failed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_llama_cpp_perf.py <model_path> [num_requests] [concurrency] [max_tokens]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    num_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 160
    concurrency = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    max_tokens = int(sys.argv[4]) if len(sys.argv) > 4 else 50
    
    print("Note: This is a placeholder script.")
    print("For actual llama.cpp testing, we need to:")
    print("1. Build llama.cpp batched example")
    print("2. Or use llama.cpp Python bindings")
    print("3. Or fix the C++ test program")
    
    # run_concurrent_test(model_path, num_requests, concurrency, max_tokens)
