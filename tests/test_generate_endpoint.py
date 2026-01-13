#!/usr/bin/env python3
"""
测试 /generate 接口的脚本
"""
import requests
import json
import sys

def test_generate_endpoint():
    """测试 /generate 接口"""
    print("="*60)
    print("测试 cLLM 服务器 /generate 接口")
    print("="*60)
    
    # 测试配置
    base_url = "http://localhost:8080"
    generate_url = f"{base_url}/generate"
    
    # 测试数据
    test_data = {
        "prompt": "hello",
        "max_tokens": 3,
        "temperature": 0.7
    }
    
    print(f"\n测试请求:")
    print(f"URL: {generate_url}")
    print(f"请求数据: {json.dumps(test_data, indent=2)}")
    
    try:
        # 发送请求
        response = requests.post(generate_url, json=test_data, timeout=10)
        
        print(f"\n响应结果:")
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            # 解析JSON响应
            try:
                response_data = response.json()
                print(f"响应体 (JSON): {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                
                # 验证响应结构
                if "text" in response_data:
                    print(f"\n✅ 成功: 接口返回了生成的文本")
                    print(f"生成文本: {response_data['text']}")
                    return True
                else:
                    print(f"\n❌ 失败: 响应中缺少 'text' 字段")
                    return False
            except json.JSONDecodeError:
                print(f"\n响应体 (原始): {response.text}")
                print(f"❌ 失败: 响应不是有效的JSON格式")
                return False
        else:
            print(f"\n响应体: {response.text}")
            print(f"❌ 失败: 请求失败，状态码 {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print(f"\n❌ 失败: 无法连接到服务器 {base_url}")
        print("请确保服务器已启动并在端口 8080 上运行")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ 失败: 请求超时")
        return False
    except Exception as e:
        print(f"\n❌ 失败: 发生未知错误: {e}")
        return False

def main():
    """主函数"""
    success = test_generate_endpoint()
    print("\n" + "="*60)
    if success:
        print("✅ 测试通过!")
        return 0
    else:
        print("❌ 测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())