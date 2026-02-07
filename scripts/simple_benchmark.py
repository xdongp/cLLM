#!/usr/bin/env python3
"""
简单的性能测试脚本
"""
import requests
import time
import json
import statistics

BASE_URL = "http://localhost:18080"

def test_health():
    """测试健康检查端点"""
    response = requests.get(f"{BASE_URL}/health")
    return response.json()

def test_generate(prompt, max_tokens=50):
    """测试文本生成"""
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens}
    )
    elapsed = time.time() - start_time
    return response.json(), elapsed

def test_encode(text):
    """测试编码端点"""
    start_time = time.time()
    response = requests.post(
        f"{BASE_URL}/encode",
        json={"text": text}
    )
    elapsed = time.time() - start_time
    return response.json(), elapsed

def benchmark_generate(prompt, max_tokens, iterations=5):
    """基准测试文本生成"""
    times = []
    tokens_per_second = []
    
    print(f"\n=== Benchmark: generate {max_tokens} tokens ===")
    print(f"Prompt: '{prompt}'")
    print(f"Iterations: {iterations}")
    
    # Warmup
    print("Warming up...")
    test_generate(prompt, max_tokens=10)
    
    # Benchmark
    for i in range(iterations):
        result, elapsed = test_generate(prompt, max_tokens)
        times.append(elapsed)
        
        # Calculate tokens per second
        if result.get('success') and 'data' in result:
            generated_text = result['data'].get('generated_text', '')
            # Rough estimate: 1 token ≈ 0.75 Chinese characters or 4 English chars
            estimated_tokens = len(generated_text) // 3  # Conservative estimate for Chinese
            tps = estimated_tokens / elapsed if elapsed > 0 else 0
            tokens_per_second.append(tps)
            print(f"  Run {i+1}: {elapsed:.3f}s, ~{tps:.1f} tokens/s")
        else:
            print(f"  Run {i+1}: {elapsed:.3f}s (error: {result.get('error', 'unknown')})")
    
    if times:
        print(f"\nResults:")
        print(f"  Average time: {statistics.mean(times):.3f}s")
        print(f"  Min time: {min(times):.3f}s")
        print(f"  Max time: {max(times):.3f}s")
        if tokens_per_second:
            print(f"  Average throughput: {statistics.mean(tokens_per_second):.1f} tokens/s")
    
    return times, tokens_per_second

def main():
    print("=" * 60)
    print("cLLM Simple Performance Test")
    print("=" * 60)
    
    # Test health
    print("\n1. Testing health endpoint...")
    try:
        health = test_health()
        print(f"   Status: {health}")
    except Exception as e:
        print(f"   Error: {e}")
        return
    
    # Test encode
    print("\n2. Testing encode endpoint...")
    try:
        result, elapsed = test_encode("你好，世界！")
        if result.get('success'):
            tokens = result['data'].get('tokens', [])
            print(f"   Encoded {len(tokens)} tokens in {elapsed:.3f}s")
        else:
            print(f"   Error: {result.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Test generate - short
    print("\n3. Testing generate (short)...")
    try:
        result, elapsed = test_generate("你好", max_tokens=20)
        if result.get('success'):
            text = result['data'].get('generated_text', '')
            print(f"   Generated in {elapsed:.3f}s:")
            print(f"   '{text[:100]}...' " if len(text) > 100 else f"   '{text}'")
        else:
            print(f"   Error: {result.get('error')}")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Benchmark
    print("\n4. Running benchmark...")
    try:
        times, tps = benchmark_generate(
            prompt="请介绍一下人工智能的发展历程",
            max_tokens=100,
            iterations=3
        )
    except Exception as e:
        print(f"   Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
