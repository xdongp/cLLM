#!/usr/bin/env python3
"""
Concurrent Benchmark Script for cLLM Server
Tests with 2 and 4 concurrent threads
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8080"
PROMPTS = [
    ("Hello Prompt", "Hello, I am a large language model."),
    ("Math Prompt", "What is 123 + 456? Please reason step by step."),
    ("Chinese Prompt", "请用中文介绍一下你自己。"),
]

MAX_TOKENS = 50
TEMPERATURE = 0.7
ROUNDS = 8

def test_single_request(prompt_name, prompt):
    """Test a single request"""
    start_time = time.time()
    try:
        response = requests.post(
            f"{SERVER_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE
            },
            timeout=30
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return {
                    "success": True,
                    "prompt_name": prompt_name,
                    "elapsed": elapsed,
                    "tokens_per_sec": data.get("data", {}).get("tokens_per_second", 0)
                }
        
        return {
            "success": False,
            "prompt_name": prompt_name,
            "elapsed": elapsed,
            "error": f"Status: {response.status_code}, Response: {response.text[:100]}"
        }
    except Exception as e:
        return {
            "success": False,
            "prompt_name": prompt_name,
            "elapsed": time.time() - start_time,
            "error": str(e)
        }

def run_concurrent_test(num_threads, total_requests):
    """Run concurrent benchmark test"""
    print(f"\n{'='*60}")
    print(f"Testing with {num_threads} concurrent threads, {total_requests} total requests")
    print(f"{'='*60}")
    
    all_tasks = []
    for round_num in range(ROUNDS):
        for prompt_name, prompt in PROMPTS:
            all_tasks.append((prompt_name, prompt))
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(test_single_request, p_name, p) for p_name, p in all_tasks]
        
        completed = 0
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            completed += 1
            
            if result["success"]:
                print(f"[OK] {result['prompt_name']}: {result['elapsed']:.2f}s, {result['tokens_per_sec']:.1f} tok/s")
            else:
                print(f"[FAIL] {result['prompt_name']}: {result['error']}")
    
    elapsed_total = time.time() - start_time
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"\n--- Results Summary ---")
    print(f"Success: {success_count}/{len(results)}")
    print(f"Failed: {fail_count}/{len(results)}")
    print(f"Total time: {elapsed_total:.2f}s")
    print(f"Requests/sec: {len(results)/elapsed_total:.2f}")
    
    return {
        "num_threads": num_threads,
        "total_requests": len(results),
        "success": success_count,
        "failed": fail_count,
        "duration": elapsed_total,
        "req_per_sec": len(results) / elapsed_total
    }

def main():
    print("="*60)
    print("cLLM Concurrent Benchmark")
    print("="*60)
    
    # Health check first
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        print(f"Server health check: {resp.status_code}")
        model_info = requests.get(f"{SERVER_URL}/model/info", timeout=5)
        print(f"Model: {model_info.json().get('data', {}).get('model_name', 'Unknown')}")
    except Exception as e:
        print(f"Server not available: {e}")
        return
    
    # Test 2 concurrent threads
    test1 = run_concurrent_test(num_threads=2, total_requests=24)
    
    # Test 4 concurrent threads
    test2 = run_concurrent_test(num_threads=4, total_requests=24)
    
    # Summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Threads':<10} {'Requests':<10} {'Success':<10} {'Duration':<10} {'Req/sec':<10}")
    print("-"*50)
    print(f"{test1['num_threads']:<10} {test1['total_requests']:<10} {test1['success']:<10} {test1['duration']:.2f}s     {test1['req_per_sec']:.2f}")
    print(f"{test2['num_threads']:<10} {test2['total_requests']:<10} {test2['success']:<10} {test2['duration']:.2f}s     {test2['req_per_sec']:.2f}")

if __name__ == "__main__":
    main()
