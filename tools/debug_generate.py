#!/usr/bin/env python3
"""
调试 /generate 接口生成的 token IDs
"""

import requests
import json

def test_with_debug():
    url = "http://localhost:8080/generate"
    
    # 简单测试
    payload = {
        "prompt": "Hello",
        "max_tokens": 3,
        "temperature": 0.0,
        "stream": False
    }
    
    print("发送请求:", json.dumps(payload, indent=2))
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        print(f"状态码: {response.status_code}")
        print(f"响应: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            text = result.get("data", {}).get("text", "")
            print(f"\n生成的文本: '{text}'")
            print(f"文本长度: {len(text)}")
            print(f"文本字节: {text.encode('utf-8')}")
            
            # 尝试分析文本
            for i, char in enumerate(text):
                print(f"  字符 {i}: '{char}' (ord={ord(char)}, hex=0x{ord(char):04x})")
                
    except Exception as e:
        print(f"错误: {e}")

if __name__ == "__main__":
    test_with_debug()
