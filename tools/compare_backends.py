#!/usr/bin/env python3
"""
对比 Kylin 和 llama_cpp 后端的输出
用于调试和定位问题
"""

import requests
import json
import sys
import time

BASE_URL = "http://localhost:8080"

def test_backend(backend_type, prompt, max_tokens=5, temperature=0.0):
    """测试指定后端"""
    # 修改配置
    config_path = "config/config.yaml"
    with open(config_path, 'r') as f:
        config = f.read()
    
    # 替换后端类型
    import re
    config = re.sub(
        r'type:\s*["\']?(\w+)["\']?',
        f'type: "{backend_type}"',
        config,
        count=1
    )
    
    with open(config_path, 'w') as f:
        f.write(config)
    
    print(f"\n{'='*60}")
    print(f"Testing {backend_type} backend")
    print(f"{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"Max tokens: {max_tokens}, Temperature: {temperature}")
    
    # 等待服务器重启（需要手动重启）
    print("\n⚠️  请手动重启服务器以应用配置更改")
    input("按 Enter 继续...")
    
    # 发送请求
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                result = data["data"]
                print(f"\n✅ Success:")
                print(f"   Text: {result['text']}")
                print(f"   Tokens: {result['generated_tokens']}")
                print(f"   Time: {result['response_time']:.3f}s")
                return result
            else:
                print(f"\n❌ Error: {data.get('error', 'Unknown error')}")
                return None
        else:
            print(f"\n❌ HTTP {response.status_code}: {response.text}")
            return None
    except Exception as e:
        print(f"\n❌ Exception: {e}")
        return None

def compare_embeddings():
    """对比 embedding 输出（需要从日志中提取）"""
    print("\n" + "="*60)
    print("Embedding 对比")
    print("="*60)
    print("请查看服务器日志中的 [Kylin Debug] Embedding stats")
    print("对比 llama_cpp 和 kylin 的 embedding 统计信息")

def compare_layer0():
    """对比第一层输出（需要从日志中提取）"""
    print("\n" + "="*60)
    print("Layer 0 输出对比")
    print("="*60)
    print("请查看服务器日志中的 [Kylin Debug] Layer 0 output stats")
    print("对比 llama_cpp 和 kylin 的第一层输出统计信息")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 compare_backends.py <prompt>")
        print("Example: python3 compare_backends.py 'Hello'")
        sys.exit(1)
    
    prompt = sys.argv[1]
    
    print("="*60)
    print("Backend Comparison Tool")
    print("="*60)
    print("\n此工具将帮助对比 Kylin 和 llama_cpp 后端的输出")
    print("请确保服务器正在运行")
    
    # 测试 Kylin
    kylin_result = test_backend("kylin", prompt, max_tokens=3, temperature=0.0)
    
    # 测试 llama_cpp
    llama_result = test_backend("llama_cpp", prompt, max_tokens=3, temperature=0.0)
    
    # 对比结果
    print("\n" + "="*60)
    print("结果对比")
    print("="*60)
    if kylin_result and llama_result:
        print(f"\nKylin:     {kylin_result['text']}")
        print(f"llama_cpp: {llama_result['text']}")
        print(f"\n差异: {'相同' if kylin_result['text'] == llama_result['text'] else '不同'}")
    
    # 提示查看日志
    compare_embeddings()
    compare_layer0()
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)

if __name__ == "__main__":
    main()
