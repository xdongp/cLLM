#!/usr/bin/env python3
"""
测试 Kylin Backend 使用 HuggingFace 模型

用法:
    python tools/test_kylin_hf.py [--server URL] [--prompt PROMPT]
"""

import argparse
import requests
import json
import time
import sys


def test_health(base_url: str) -> bool:
    """检查服务健康状态"""
    try:
        resp = requests.get(f"{base_url}/health", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            print(f"✅ 服务健康: {data.get('status', 'unknown')}")
            print(f"   Backend: {data.get('backend', 'unknown')}")
            return True
        else:
            print(f"❌ 服务不健康: HTTP {resp.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到服务: {base_url}")
        return False
    except Exception as e:
        print(f"❌ 健康检查失败: {e}")
        return False


def test_generate(base_url: str, prompt: str, max_tokens: int = 20) -> dict:
    """测试生成接口"""
    print(f"\n📝 测试生成 (max_tokens={max_tokens}):")
    print(f"   Prompt: \"{prompt}\"")
    
    try:
        start_time = time.time()
        resp = requests.post(
            f"{base_url}/generate",
            json={
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.9
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")
            tokens_generated = data.get("tokens_generated", 0)
            print(f"✅ 生成成功 ({elapsed:.2f}s)")
            print(f"   生成文本: \"{text[:200]}{'...' if len(text) > 200 else ''}\"")
            print(f"   Token 数: {tokens_generated}")
            if tokens_generated > 0:
                print(f"   速度: {tokens_generated / elapsed:.1f} tokens/s")
            return {"success": True, "data": data, "elapsed": elapsed}
        else:
            print(f"❌ 生成失败: HTTP {resp.status_code}")
            print(f"   响应: {resp.text[:500]}")
            return {"success": False, "error": resp.text}
            
    except requests.exceptions.Timeout:
        print(f"❌ 请求超时")
        return {"success": False, "error": "timeout"}
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return {"success": False, "error": str(e)}


def test_encode(base_url: str, text: str) -> dict:
    """测试编码接口"""
    print(f"\n🔤 测试编码:")
    print(f"   文本: \"{text}\"")
    
    try:
        resp = requests.post(
            f"{base_url}/encode",
            json={"text": text},
            timeout=10
        )
        
        if resp.status_code == 200:
            data = resp.json()
            tokens = data.get("tokens", [])
            print(f"✅ 编码成功")
            print(f"   Token IDs: {tokens[:10]}{'...' if len(tokens) > 10 else ''}")
            print(f"   Token 数量: {len(tokens)}")
            return {"success": True, "tokens": tokens}
        else:
            print(f"❌ 编码失败: HTTP {resp.status_code}")
            return {"success": False, "error": resp.text}
            
    except Exception as e:
        print(f"❌ 编码失败: {e}")
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="测试 Kylin Backend (HuggingFace 模型)")
    parser.add_argument("--server", default="http://localhost:8080", help="服务器地址")
    parser.add_argument("--prompt", default="Hello, how are you?", help="测试 prompt")
    parser.add_argument("--max-tokens", type=int, default=20, help="最大生成 token 数")
    args = parser.parse_args()
    
    base_url = args.server.rstrip("/")
    
    print("=" * 60)
    print("🚀 Kylin Backend (HuggingFace) 测试")
    print("=" * 60)
    print(f"服务地址: {base_url}")
    
    # 1. 健康检查
    print("\n" + "-" * 40)
    if not test_health(base_url):
        print("\n⚠️  服务未启动，请先运行:")
        print("   cd build && ./bin/cllm_server")
        sys.exit(1)
    
    # 2. 编码测试
    print("\n" + "-" * 40)
    test_encode(base_url, args.prompt)
    
    # 3. 生成测试
    print("\n" + "-" * 40)
    result = test_generate(base_url, args.prompt, args.max_tokens)
    
    # 4. 多轮测试
    if result.get("success"):
        print("\n" + "-" * 40)
        print("🔄 多轮生成测试:")
        
        prompts = [
            "What is the capital of France?",
            "Explain quantum computing in simple terms.",
            "Write a haiku about programming."
        ]
        
        for i, p in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}]")
            test_generate(base_url, p, 30)
    
    print("\n" + "=" * 60)
    print("✨ 测试完成")
    print("=" * 60)


if __name__ == "__main__":
    main()
