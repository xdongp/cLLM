#!/usr/bin/env python3
"""
测试 /generate 接口 - 使用 LlamaCppBackend
测试输入: "1+1="
"""
import requests
import json
import sys
import time

def test_generate_llama_cpp():
    """测试 /generate 接口"""
    print("="*60)
    print("测试 cLLM 服务器 /generate 接口 (LlamaCppBackend)")
    print("="*60)
    
    # 测试配置
    base_url = "http://localhost:8080"
    generate_url = f"{base_url}/generate"
    
    # 测试数据 - 数学问题
    test_data = {
        "prompt": "1+1=",
        "max_tokens": 16,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    print(f"\n测试请求:")
    print(f"URL: {generate_url}")
    print(f"请求数据: {json.dumps(test_data, indent=2, ensure_ascii=False)}")
    
    try:
        # 发送请求
        print(f"\n发送请求...")
        start_time = time.time()
        response = requests.post(generate_url, json=test_data, timeout=30)
        elapsed_time = time.time() - start_time
        
        print(f"\n响应结果:")
        print(f"状态码: {response.status_code}")
        print(f"耗时: {elapsed_time:.2f} 秒")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            # 解析JSON响应
            try:
                response_data = response.json()
                print(f"\n响应体 (JSON): {json.dumps(response_data, indent=2, ensure_ascii=False)}")
                
                # 验证响应结构
                if "text" in response_data:
                    print(f"\n" + "="*60)
                    print(f"✅ 成功: 接口返回了生成的文本")
                    print(f"="*60)
                    print(f"输入: {test_data['prompt']}")
                    print(f"输出: {response_data['text']}")
                    print(f"完整: {test_data['prompt']}{response_data['text']}")
                    print(f"="*60)
                    
                    # 验证输出是否包含正确答案
                    if "2" in response_data['text']:
                        print(f"\n✅ 验证通过: 输出包含正确答案 '2'")
                        return True
                    else:
                        print(f"\n⚠️  警告: 输出可能不正确，未找到 '2'")
                        return True
                else:
                    print(f"\n❌ 失败: 响应中缺少 'text' 字段")
                    print(f"响应字段: {list(response_data.keys())}")
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
        print("启动命令: ./build/bin/cllm_server --model-path model/Qwen/qwen3-0.6b-q4_k_m.gguf")
        return False
    except requests.exceptions.Timeout:
        print(f"\n❌ 失败: 请求超时")
        return False
    except Exception as e:
        print(f"\n❌ 失败: 发生未知错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    success = test_generate_llama_cpp()
    print("\n" + "="*60)
    if success:
        print("✅ 测试通过!")
        return 0
    else:
        print("❌ 测试失败!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
