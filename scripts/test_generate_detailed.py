#!/usr/bin/env python3
"""
详细的生成测试，检查返回的所有字段
"""
import requests
import json

BASE_URL = "http://localhost:18080"

def test_generate_detailed():
    """测试生成并打印详细信息"""
    prompt = "hello"
    max_tokens = 10
    
    print(f"\n{'='*60}")
    print(f"测试生成: prompt='{prompt}', max_tokens={max_tokens}")
    print(f"{'='*60}")
    
    # 先编码 prompt
    encode_resp = requests.post(
        f"{BASE_URL}/encode",
        json={"text": prompt}
    )
    encode_data = encode_resp.json()
    print(f"\n1. Encode 结果:")
    print(f"   Prompt tokens: {encode_data['data']['tokens']}")
    
    # 生成
    print(f"\n2. 生成请求...")
    gen_resp = requests.post(
        f"{BASE_URL}/generate",
        json={"prompt": prompt, "max_tokens": max_tokens}
    )
    gen_data = gen_resp.json()
    
    print(f"\n3. 生成结果:")
    print(f"   success: {gen_data.get('success')}")
    print(f"   generated_tokens: {gen_data['data'].get('generated_tokens')}")
    print(f"   text: '{gen_data['data'].get('text')}'")
    print(f"   response_time: {gen_data['data'].get('response_time'):.3f}s")
    print(f"   tokens_per_second: {gen_data['data'].get('tokens_per_second'):.2f}")
    print(f"   id: {gen_data['data'].get('id')}")
    
    # 检查是否有错误
    if not gen_data.get('success'):
        print(f"\n   Error: {gen_data.get('error')}")
    
    return gen_data

def test_decode():
    """测试解码端点（如果存在）"""
    # 尝试解码一些已知的 tokens
    test_tokens = [
        [14990],  # "hello"
        [1879],   # " world"
        [11],     # ","
        [0],      # "!" or special
        [14990, 1879],  # "hello world"
    ]
    
    print(f"\n{'='*60}")
    print("测试解码 (通过 encode/decode 推断)")
    print(f"{'='*60}")
    
    for tokens in test_tokens:
        # 由于服务器没有 decode 端点，我们通过其他方式测试
        print(f"   Tokens {tokens}: 需要手动验证解码结果")

def main():
    print("cLLM 生成测试 - 详细版")
    
    try:
        # 健康检查
        health = requests.get(f"{BASE_URL}/health").json()
        print(f"\n健康检查: {health}")
        
        if not health.get('success') or not health['data'].get('model_loaded'):
            print("模型未加载，无法测试")
            return
        
        # 运行测试
        test_generate_detailed()
        test_decode()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
