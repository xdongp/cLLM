#!/usr/bin/env python3
"""
Sequential Concurrent Benchmark for cLLM Server
Tests with 2 and 4 concurrent threads separately
"""

import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

SERVER_URL = "http://localhost:8080"
PROMPTS = [
    ("Hello", "Hello, I am a large language model."),
    ("Math", "What is 123 + 456? Please reason step by step."),
    ("Chinese", "请用中文介绍一下你自己。"),
]

MAX_TOKENS = 50
TEMPERATURE = 0.7
ROUNDS = 8

def test_single_request(prompt_name, prompt, round_num):
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
            timeout=60
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return {
                    "success": True,
                    "prompt_name": prompt_name,
                    "round": round_num,
                    "elapsed": elapsed,
                    "tokens_per_sec": data.get("data", {}).get("tokens_per_second", 0),
                    "text_preview": data.get("data", {}).get("text", "")[:50]
                }
        
        return {
            "success": False,
            "prompt_name": prompt_name,
            "round": round_num,
            "elapsed": elapsed,
            "error": f"Status: {response.status_code}"
        }
    except Exception as e:
        return {
            "success": False,
            "prompt_name": prompt_name,
            "round": round_num,
            "elapsed": time.time() - start_time,
            "error": str(e)[:100]
        }

def run_test(num_threads, test_name):
    """Run a single concurrent test"""
    print(f"\n{'='*60}")
    print(f"Test: {test_name} - {num_threads} threads, {ROUNDS * 3} requests")
    print(f"{'='*60}")
    
    all_tasks = []
    for round_num in range(ROUNDS):
        for prompt_name, prompt in PROMPTS:
            all_tasks.append((prompt_name, prompt, round_num))
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for p_name, prompt, r_num in all_tasks:
            futures.append(executor.submit(test_single_request, p_name, prompt, r_num))
        
        for future in futures:
            result = future.result()
            results.append(result)
    
    elapsed_total = time.time() - start_time
    
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count
    
    print(f"\n--- {test_name} Results ---")
    print(f"Success: {success_count}/{len(results)}")
    print(f"Failed: {fail_count}/{len(results)}")
    print(f"Duration: {elapsed_total:.2f}s")
    print(f"Avg req time: {elapsed_total/len(results):.2f}s")
    print(f"Throughput: {len(results)/elapsed_total:.2f} req/s")
    
    # Show success rate per prompt
    for p_name, _ in PROMPTS:
        p_results = [r for r in results if r["prompt_name"] == p_name]
        p_success = sum(1 for r in p_results if r["success"])
        print(f"  {p_name}: {p_success}/{len(p_results)}")
    
    return {
        "test_name": test_name,
        "num_threads": num_threads,
        "total": len(results),
        "success": success_count,
        "failed": fail_count,
        "duration": elapsed_total,
        "throughput": len(results) / elapsed_total
    }

def main():
    print("="*60)
    print("cLLM Sequential Concurrent Benchmark")
    print("="*60)
    
    # Health check
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("Server not available")
            return
        model_info = requests.get(f"{SERVER_URL}/model/info", timeout=5)
        model = model_info.json().get('data', {}).get('model_name', 'Unknown')
        print(f"Server ready - Model: {model}")
    except Exception as e:
        print(f"Server error: {e}")
        return
    
    results = []
    
    # Test 1: 2 threads
    results.append(run_test(2, "2-Threads"))
    time.sleep(3)  # Cool down
    
    # Check server
    try:
        resp = requests.get(f"{SERVER_URL}/health", timeout=5)
        if resp.status_code != 200:
            print("Server crashed after 2-threads test!")
            return
    except:
        print("Server crashed after 2-threads test!")
        return
    
    # Test 2: 4 threads
    results.append(run_test(4, "4-Threads"))
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"{'Test':<15} {'Threads':<8} {'Requests':<10} {'Success':<10} {'Duration':<10} {'Req/s':<10}")
    print("-"*65)
    for r in results:
        print(f"{r['test_name']:<15} {r['num_threads']:<8} {r['total']:<10} {r['success']:<10} {r['duration']:.2f}s     {r['throughput']:.2f}")

if __name__ == "__main__":
    main()
