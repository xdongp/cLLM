#!/usr/bin/env python3
"""
调试 /generate 接口 - 查看生成的 token ID
"""

import requests
import json

def test_generate_with_debug():
    """测试生成接口并查看详细信息"""
    base_url = "http://localhost:8080"
    generate_url = f"{base_url}/generate"
    
    payload = {
        "prompt": "Hello",
        "max_tokens": 5,
        "temperature": 0.0,
        "stream": False
    }
    
    print(f"发送请求: {json.dumps(payload, indent=2)}")
    
    response = requests.post(generate_url, json=payload, timeout=10)
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
        
        data = result.get("data", {})
        generated_text = data.get("text", "")
        
        print(f"\n生成的文本: '{generated_text}'")
        print(f"文本长度: {len(generated_text)}")
        print(f"文本字符: {list(generated_text)}")
        
        # 检查是否有乱码
        try:
            generated_text.encode('utf-8').decode('utf-8')
            print("✅ 文本是有效的 UTF-8")
        except:
            print("❌ 文本不是有效的 UTF-8")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    test_generate_with_debug()
