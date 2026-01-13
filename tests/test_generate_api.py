#!/usr/bin/env python3
import requests
import json

# 测试/generate接口
def test_generate_api():
    print("Testing /generate API with input 'hello'...")
    
    url = "http://localhost:8080/generate"
    headers = {
        "Content-Type": "application/json"
    }
    data = {
        "prompt": "hello",
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print(f"Response Body: {response.text}")
        
        # 解析JSON响应
        if response.status_code == 200:
            try:
                json_response = response.json()
                print("\nJSON Response:")
                print(json.dumps(json_response, indent=2))
                
                # 检查响应结构
                if "choices" in json_response and len(json_response["choices"]) > 0:
                    text = json_response["choices"][0]["text"]
                    print(f"\nGenerated Text: {text}")
                    return True
            except json.JSONDecodeError:
                print("\nError: Response is not valid JSON")
        else:
            print(f"\nError: Request failed with status code {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is the server running?")
    except Exception as e:
        print(f"Error: {e}")
    
    return False

if __name__ == "__main__":
    test_generate_api()
