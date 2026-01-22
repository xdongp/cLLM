#!/usr/bin/env python3
"""
API并发测试脚本
"""

import subprocess
import json
import time
import sys

API_URL = "http://0.0.0.0:8080"
NUM_REQUESTS = 72
CONCURRENCY = 24
MAX_TOKENS = 50

PROMPTS = [
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。",
    "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。",
    "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。",
    "自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。",
    "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
]

print("=" * 50)
print("xLLM API Concurrent Test")
print("=" * 50)
print(f"API URL: {API_URL}")
print(f"Number of requests: {NUM_REQUESTS}")
print(f"Concurrency: {CONCURRENCY}")
print(f"Max tokens: {MAX_TOKENS}")
print("=" * 50)
print()

def send_request(prompt):
    """发送单个API请求"""
    payload = {
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": MAX_TOKENS,
        "stream": False
    }
    
    start_time = time.time()
    try:
        result = subprocess.run(
            ["curl", "-s", "-X", "POST", f"{API_URL}/generate",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=300
        )
        end_time = time.time()
        duration = end_time - start_time
        
        try:
            response = json.loads(result.stdout)
            if response.get("success"):
                return True, duration, response.get("data", {}).get("tokens_per_second", 0)
            else:
                return False, duration, response.get("error", "Unknown error")
        except json.JSONDecodeError:
            return False, duration, f"Invalid JSON: {result.stdout}"
    except subprocess.TimeoutExpired:
        end_time = time.time()
        return False, end_time - start_time, "Timeout"
    except Exception as e:
        end_time = time.time()
        return False, end_time - start_time, str(e)

def main():
    """主函数"""
    import random
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    print("Running concurrent test...")
    print()
    
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
        futures = []
        for i in range(NUM_REQUESTS):
            prompt = PROMPTS[i % len(PROMPTS)]
            future = executor.submit(send_request, prompt)
            futures.append((i + 1, future))
        
        for request_id, future in futures:
            success, duration, info = future.result()
            results.append((success, duration, info))
            status = "✓" if success else "✗"
            if success:
                print(f"Request {request_id}/{NUM_REQUESTS}: {status} {duration:.2f}s - {info:.1f} tokens/s")
            else:
                print(f"Request {request_id}/{NUM_REQUESTS}: {status} {duration:.2f}s - {info}")
    
    total_time = time.time() - start_time
    
    print()
    print("=" * 50)
    print("Test Statistics")
    print("=" * 50)
    
    successful = [r for r in results if r[0]]
    failed = [r for r in results if not r[0]]
    
    print(f"Total requests: {NUM_REQUESTS}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_time:.2f}s")
    
    if successful:
        durations = [r[1] for r in successful]
        tps_values = [r[2] for r in successful]
        print(f"Avg response time: {sum(durations) / len(durations):.2f}s")
        print(f"Min response time: {min(durations):.2f}s")
        print(f"Max response time: {max(durations):.2f}s")
        print(f"Avg tokens/s: {sum(tps_values) / len(tps_values):.1f}")
        print(f"Total tokens/s: {sum(tps_values) / total_time:.1f}")
    
    print("=" * 50)

if __name__ == "__main__":
    main()
