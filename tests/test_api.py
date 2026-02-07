"""
xLLM API 完整测试套件
测试所有 API 端点：健康检查、文本生成（流式/非流式）、文本编码
"""
import unittest
import requests
import json
import time
import sys
import os
from typing import List, Dict, Any

# API 配置
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 300


class TestAPIHealth(unittest.TestCase):
    """健康检查端点测试"""
    
    def test_health_check(self):
        """测试健康检查端点"""
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
        self.assertIn("model_loaded", data)
        self.assertIsInstance(data["model_loaded"], bool)
        print(f"✅ 健康检查通过: {data}")


class TestAPIEncode(unittest.TestCase):
    """文本编码端点测试"""
    
    def test_encode_simple_text(self):
        """测试简单文本编码"""
        text = "Hello, world!"
        response = requests.post(
            f"{API_BASE_URL}/encode",
            json={"text": text},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertIn("length", data)
        self.assertIsInstance(data["tokens"], list)
        self.assertIsInstance(data["length"], int)
        self.assertGreater(data["length"], 0)
        print(f"✅ 文本编码成功: '{text}' -> {data['tokens']} (长度: {data['length']})")
    
    def test_encode_chinese_text(self):
        """测试中文文本编码"""
        text = "你好，世界！"
        response = requests.post(
            f"{API_BASE_URL}/encode",
            json={"text": text},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("tokens", data)
        self.assertGreater(data["length"], 0)
        print(f"✅ 中文文本编码成功: '{text}' -> {data['tokens']} (长度: {data['length']})")
    
    def test_encode_empty_text(self):
        """测试空文本编码"""
        text = ""
        response = requests.post(
            f"{API_BASE_URL}/encode",
            json={"text": text},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["length"], 0)
        print(f"✅ 空文本编码成功: 长度 {data['length']}")
    
    def test_encode_long_text(self):
        """测试长文本编码"""
        text = "This is a long text. " * 50
        response = requests.post(
            f"{API_BASE_URL}/encode",
            json={"text": text},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["length"], 100)
        print(f"✅ 长文本编码成功: 长度 {data['length']}")
    
    def test_encode_invalid_text_type(self):
        """测试无效的文本类型"""
        response = requests.post(
            f"{API_BASE_URL}/encode",
            json={"text": 123},
            timeout=30
        )
        
        self.assertEqual(response.status_code, 400)
        print(f"✅ 无效文本类型测试通过: 返回 400 错误")


class TestAPIGenerate(unittest.TestCase):
    """文本生成端点测试（非流式）"""
    
    def test_generate_simple_prompt(self):
        """测试简单提示生成"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "The capital of France is",
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("id", data)
        self.assertIn("text", data)
        self.assertIn("tokens", data)
        self.assertIn("response_time", data)
        self.assertIn("tokens_per_second", data)
        self.assertIsInstance(data["text"], str)
        self.assertGreater(len(data["text"]), 0)
        print(f"✅ 简单提示生成成功: {data['text'][:50]}...")
        print(f"   响应时间: {data['response_time']}s, 速度: {data['tokens_per_second']} tokens/s")
    
    def test_generate_with_messages(self):
        """测试使用 messages 参数生成"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "messages": [
                    {"role": "user", "content": "What is 2+2?"}
                ],
                "max_tokens": 20,
                "temperature": 0.7
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)
        print(f"✅ Messages 生成成功: {data['text'][:50]}...")
    
    def test_generate_chinese_prompt(self):
        """测试中文提示生成"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "中国的首都是",
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 0)
        print(f"✅ 中文提示生成成功: {data['text'][:50]}...")
    
    def test_generate_with_low_temperature(self):
        """测试低温度生成（更确定性）"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "The sky is",
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        print(f"✅ 低温度生成成功: {data['text'][:50]}...")
    
    def test_generate_with_high_temperature(self):
        """测试高温度生成（更随机）"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "Once upon a time",
                "max_tokens": 15,
                "temperature": 1.0
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        print(f"✅ 高温度生成成功: {data['text'][:50]}...")
    
    def test_generate_long_output(self):
        """测试生成长文本"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "Write a short story about a robot:",
                "max_tokens": 100,
                "temperature": 0.8
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("text", data)
        self.assertGreater(len(data["text"]), 50)
        print(f"✅ 长文本生成成功: {len(data['text'])} 字符")
    
    def test_generate_missing_prompt_and_messages(self):
        """测试缺少 prompt 和 messages 参数"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "max_tokens": 10,
                "temperature": 0.7
            },
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 400)
        print(f"✅ 缺少参数测试通过: 返回 400 错误")


class TestAPIGenerateStream(unittest.TestCase):
    """流式生成端点测试"""
    
    def test_stream_simple_prompt(self):
        """测试简单提示流式生成"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "The quick brown fox",
                "max_tokens": 20,
                "temperature": 0.7,
                "stream": True
            },
            stream=True,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        tokens_received = []
        done_received = False
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        done_received = True
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'token' in data:
                            tokens_received.append(data['token'])
                        if data.get('done', False):
                            done_received = True
                    except json.JSONDecodeError:
                        pass
        
        self.assertGreater(len(tokens_received), 0)
        self.assertTrue(done_received)
        full_text = ''.join(tokens_received)
        print(f"✅ 流式生成成功: {full_text[:50]}... ({len(tokens_received)} tokens)")
    
    def test_stream_chinese_prompt(self):
        """测试中文提示流式生成"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "人工智能是",
                "max_tokens": 15,
                "temperature": 0.7,
                "stream": True
            },
            stream=True,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        tokens_received = []
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'token' in data:
                            tokens_received.append(data['token'])
                    except json.JSONDecodeError:
                        pass
        
        self.assertGreater(len(tokens_received), 0)
        full_text = ''.join(tokens_received)
        print(f"✅ 中文流式生成成功: {full_text[:50]}... ({len(tokens_received)} tokens)")
    
    def test_stream_long_output(self):
        """测试流式生成长文本"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            json={
                "prompt": "Explain quantum computing:",
                "max_tokens": 50,
                "temperature": 0.7,
                "stream": True
            },
            stream=True,
            timeout=TIMEOUT
        )
        
        self.assertEqual(response.status_code, 200)
        
        tokens_received = []
        
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line.startswith('data: '):
                    data_str = line[6:]
                    if data_str == '[DONE]':
                        break
                    
                    try:
                        data = json.loads(data_str)
                        if 'token' in data:
                            tokens_received.append(data['token'])
                    except json.JSONDecodeError:
                        pass
        
        self.assertGreater(len(tokens_received), 10)
        full_text = ''.join(tokens_received)
        print(f"✅ 长文本流式生成成功: {len(full_text)} 字符 ({len(tokens_received)} tokens)")


class TestAPIPerformance(unittest.TestCase):
    """API 性能测试"""
    
    def test_concurrent_requests(self):
        """测试并发请求"""
        import concurrent.futures
        
        def make_request(prompt):
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.7
                },
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            return response.status_code == 200, elapsed
        
        prompts = [
            "What is AI?",
            "Hello world",
            "Python is",
            "The future is",
            "Technology"
        ]
        
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, prompt) for prompt in prompts]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        
        success_count = sum(1 for success, _ in results if success)
        avg_time = sum(elapsed for _, elapsed in results) / len(results)
        
        self.assertEqual(success_count, 5)
        print(f"✅ 并发请求测试通过: 5/5 成功")
        print(f"   总时间: {total_time:.2f}s, 平均响应时间: {avg_time:.2f}s")
    
    def test_response_time(self):
        """测试响应时间"""
        prompts = [
            "Hello",
            "How are you?",
            "What is this?"
        ]
        
        response_times = []
        for prompt in prompts:
            start = time.time()
            response = requests.post(
                f"{API_BASE_URL}/generate",
                json={
                    "prompt": prompt,
                    "max_tokens": 10,
                    "temperature": 0.7
                },
                timeout=TIMEOUT
            )
            elapsed = time.time() - start
            response_times.append(elapsed)
            
            self.assertEqual(response.status_code, 200)
        
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        
        print(f"✅ 响应时间测试通过")
        print(f"   平均响应时间: {avg_response_time:.2f}s")
        print(f"   最大响应时间: {max_response_time:.2f}s")


class TestAPIErrorHandling(unittest.TestCase):
    """API 错误处理测试"""
    
    def test_invalid_json(self):
        """测试无效 JSON"""
        response = requests.post(
            f"{API_BASE_URL}/generate",
            data="invalid json",
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        self.assertIn(response.status_code, [400, 422])
        print(f"✅ 无效 JSON 测试通过: 返回 {response.status_code}")
    
    def test_missing_endpoint(self):
        """测试不存在的端点"""
        response = requests.get(
            f"{API_BASE_URL}/nonexistent",
            timeout=10
        )
        
        self.assertEqual(response.status_code, 404)
        print(f"✅ 不存在端点测试通过: 返回 404")
    
    def test_invalid_method(self):
        """测试无效的 HTTP 方法"""
        response = requests.get(
            f"{API_BASE_URL}/generate",
            timeout=10
        )
        
        self.assertIn(response.status_code, [405, 404])
        print(f"✅ 无效方法测试通过: 返回 {response.status_code}")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 70)
    print("🧪 xLLM API 测试套件")
    print("=" * 70)
    print(f"📍 API 地址: {API_BASE_URL}")
    print("=" * 70 + "\n")
    
    # 检查服务器是否运行
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code != 200:
            print(f"❌ 服务器未正常运行: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ 无法连接到服务器: {e}")
        print(f"   请确保服务器正在运行: {API_BASE_URL}")
        return
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestAPIHealth))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIEncode))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIGenerate))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIGenerateStream))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIPerformance))
    suite.addTests(loader.loadTestsFromTestCase(TestAPIErrorHandling))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print("\n" + "=" * 70)
    print("📊 测试总结")
    print("=" * 70)
    print(f"总测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
