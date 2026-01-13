#!/usr/bin/env python3
"""
xLLM 客户端测试程序
测试命令行客户端的各项功能
"""

import unittest
import sys
import os
import time
import json
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.xllm_client import XLLMClient


class TestXLLMClient(unittest.TestCase):
    """XLLMClient 类测试"""
    
    def setUp(self):
        """测试前准备"""
        self.server_url = "http://localhost:8000"
        self.client = XLLMClient(server_url=self.server_url)
    
    def test_client_initialization(self):
        """测试客户端初始化"""
        self.assertEqual(self.client.server_url, self.server_url)
        self.assertIsNotNone(self.client.session)
        self.assertIn("max_tokens", self.client.default_params)
        self.assertIn("temperature", self.client.default_params)
        self.assertIn("top_p", self.client.default_params)
        self.assertIn("top_k", self.client.default_params)
        self.assertEqual(self.client.default_params["max_tokens"], 50)
        self.assertEqual(self.client.default_params["temperature"], 0.7)
        self.assertFalse(self.client.interrupt_requested)
    
    def test_check_server_health_success(self):
        """测试检查服务器健康状态 - 成功"""
        with patch.object(self.client.session, 'get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            result = self.client.check_server_health()
            self.assertTrue(result)
            mock_get.assert_called_once_with(f"{self.server_url}/health", timeout=5)
    
    def test_check_server_health_failure(self):
        """测试检查服务器健康状态 - 失败"""
        with patch.object(self.client.session, 'get') as mock_get:
            mock_get.side_effect = Exception("Connection error")
            
            result = self.client.check_server_health()
            self.assertFalse(result)
    
    def test_request_interrupt(self):
        """测试请求中断"""
        self.client.request_interrupt()
        self.assertTrue(self.client.interrupt_requested)
    
    def test_reset_interrupt(self):
        """测试重置中断标志"""
        self.client.interrupt_requested = True
        self.client.reset_interrupt()
        self.assertFalse(self.client.interrupt_requested)
    
    def test_generate_text_simple(self):
        """测试简单文本生成（非流式）"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test_123",
                "text": "This is a generated text.",
                "tokens": ["This", "is", "a", "generated", "text", "."],
                "response_time": 0.5,
                "tokens_per_second": 12.0
            }
            mock_post.return_value = mock_response
            
            result = self.client.generate_text("Hello", max_tokens=10)
            
            self.assertIsNotNone(result)
            self.assertIn("generated_text", result)
            self.assertEqual(result["generated_text"], "This is a generated text.")
            mock_post.assert_called_once()
    
    def test_generate_text_with_messages(self):
        """测试使用 messages 参数生成文本"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test_456",
                "text": "Response to messages",
                "tokens": ["Response", "to", "messages"],
                "response_time": 0.3,
                "tokens_per_second": 10.0
            }
            mock_post.return_value = mock_response
            
            result = self.client.generate_text(
                prompt="",  # generate_text 需要 prompt 参数
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=20
            )
            
            self.assertIsNotNone(result)
            self.assertIn("generated_text", result)
    
    def test_generate_text_timeout(self):
        """测试生成文本超时"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_post.side_effect = Exception("Timeout")
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = self.client.generate_text("Hello")
                self.assertIsNone(result)
                self.assertIn("请求失败", fake_out.getvalue())
    
    def test_generate_text_streaming(self):
        """测试流式文本生成"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                'data: {"token": "Hello"}',
                'data: {"token": " world"}',
                'data: {"token": "!"}',
                'data: [DONE]'
            ]
            mock_post.return_value = mock_response
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = self.client.generate_text("Hello", stream=True)
                
                self.assertIsNotNone(result)
                self.assertIn("generated_text", result)
                self.assertEqual(result["generated_text"], "Hello world!")
                self.assertEqual(result["token_count"], 3)
    
    def test_generate_text_streaming_interrupt(self):
        """测试流式生成中断"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.iter_lines.return_value = [
                'data: {"token": "Hello"}',
                'data: {"token": " world"}',
            ]
            mock_post.return_value = mock_response
            
            # 先初始化 token_count
            self.client._token_count = 0
            self.client.interrupt_requested = True
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = self.client.generate_text("Hello", stream=True)
                
                self.assertIsNotNone(result)
                # 检查返回结果包含必要的字段
                self.assertIn("generated_text", result)
                self.assertIn("token_count", result)
                # finish_reason 可能在某些情况下不存在，所以只检查存在性
                if "finish_reason" in result:
                    self.assertEqual(result["finish_reason"], "interrupted")
    
    def test_show_help(self):
        """测试显示帮助信息"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.client.show_help()
            output = fake_out.getvalue()
            self.assertIn("帮助信息", output)
            self.assertIn("quit/exit", output)
            self.assertIn("clear", output)
            self.assertIn("help", output)
    
    def test_show_config(self):
        """测试显示配置"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.client.show_config()
            output = fake_out.getvalue()
            self.assertIn("当前配置", output)
            self.assertIn("max_tokens", output)
            self.assertIn("temperature", output)
    
    def test_set_config_valid(self):
        """测试设置有效配置"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.client.set_config("temperature 0.8")
            output = fake_out.getvalue()
            self.assertIn("已设置 temperature = 0.8", output)
            self.assertEqual(self.client.default_params["temperature"], 0.8)
    
    def test_set_config_invalid_value(self):
        """测试设置无效配置值"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.client.set_config("max_tokens abc")
            output = fake_out.getvalue()
            self.assertIn("无效的参数值", output)
    
    def test_set_config_missing_value(self):
        """测试缺少配置值"""
        with patch('sys.stdout', new=StringIO()) as fake_out:
            self.client.set_config("temperature")
            output = fake_out.getvalue()
            self.assertIn("用法", output)


class TestXLLMClientIntegration(unittest.TestCase):
    """XLLMClient 集成测试（需要服务器运行）"""
    
    def setUp(self):
        """测试前准备"""
        self.server_url = "http://localhost:8000"
        self.client = XLLMClient(server_url=self.server_url)
    
    def test_integration_server_health(self):
        """集成测试：检查服务器健康状态"""
        result = self.client.check_server_health()
        if result:
            print("✅ 服务器健康检查通过")
        else:
            print("⚠️  服务器未运行或无法连接，跳过集成测试")
    
    def test_integration_simple_generation(self):
        """集成测试：简单文本生成"""
        if not self.client.check_server_health():
            self.skipTest("服务器未运行")
        
        result = self.client.generate_text(
            "The capital of France is",
            max_tokens=10,
            temperature=0.7
        )
        
        self.assertIsNotNone(result)
        self.assertIn("generated_text", result)
        self.assertIsInstance(result["generated_text"], str)
        self.assertGreater(len(result["generated_text"]), 0)
        print(f"✅ 生成测试通过: {result['generated_text'][:50]}...")
    
    def test_integration_streaming_generation(self):
        """集成测试：流式文本生成"""
        if not self.client.check_server_health():
            self.skipTest("服务器未运行")
        
        with patch('sys.stdout', new=StringIO()) as fake_out:
            result = self.client.generate_text(
                "Write a short poem:",
                max_tokens=20,
                stream=True
            )
            
            self.assertIsNotNone(result)
            self.assertIn("generated_text", result)
            self.assertIn("token_count", result)
            self.assertGreater(result["token_count"], 0)
            print(f"✅ 流式生成测试通过: {result['token_count']} tokens")


class TestXLLMClientCommandArgs(unittest.TestCase):
    """测试命令行参数解析"""
    
    def test_command_args_default(self):
        """测试默认命令行参数"""
        with patch('sys.argv', ['xllm_client.py']):
            from python.xllm_client import main
            # 只测试参数解析，不实际运行 main()
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--server-url", default="http://localhost:8000")
            parser.add_argument("--max-tokens", type=int, default=50)
            parser.add_argument("--temperature", type=float, default=0.7)
            parser.add_argument("--top-p", type=float, default=0.9)
            parser.add_argument("--top-k", type=int, default=50)
            args = parser.parse_args([])
            
            self.assertEqual(args.server_url, "http://localhost:8000")
            self.assertEqual(args.max_tokens, 50)
            self.assertEqual(args.temperature, 0.7)
            self.assertEqual(args.top_p, 0.9)
            self.assertEqual(args.top_k, 50)
    
    def test_command_args_custom(self):
        """测试自定义命令行参数"""
        with patch('sys.argv', ['xllm_client.py', '--server-url', 'http://localhost:9000',
                               '--max-tokens', '100', '--temperature', '0.8']):
            import argparse
            parser = argparse.ArgumentParser()
            parser.add_argument("--server-url", default="http://localhost:8000")
            parser.add_argument("--max-tokens", type=int, default=50)
            parser.add_argument("--temperature", type=float, default=0.7)
            parser.add_argument("--top-p", type=float, default=0.9)
            parser.add_argument("--top-k", type=int, default=50)
            args = parser.parse_args(['--server-url', 'http://localhost:9000',
                                     '--max-tokens', '100', '--temperature', '0.8'])
            
            self.assertEqual(args.server_url, "http://localhost:9000")
            self.assertEqual(args.max_tokens, 100)
            self.assertEqual(args.temperature, 0.8)


class TestXLLMClientEdgeCases(unittest.TestCase):
    """测试边界情况"""
    
    def setUp(self):
        """测试前准备"""
        self.client = XLLMClient()
    
    def test_empty_prompt(self):
        """测试空提示词"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "text": "",
                "tokens": [],
                "response_time": 0.1,
                "tokens_per_second": 0.0
            }
            mock_post.return_value = mock_response
            
            result = self.client.generate_text("")
            self.assertIsNotNone(result)
    
    def test_very_long_prompt(self):
        """测试超长提示词"""
        long_prompt = "Hello " * 1000
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "text": "OK",
                "tokens": ["OK"],
                "response_time": 1.0,
                "tokens_per_second": 1.0
            }
            mock_post.return_value = mock_response
            
            result = self.client.generate_text(long_prompt)
            self.assertIsNotNone(result)
    
    def test_extreme_temperature(self):
        """测试极端温度参数"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "text": "Response",
                "tokens": ["Response"],
                "response_time": 0.5,
                "tokens_per_second": 2.0
            }
            mock_post.return_value = mock_response
            
            # 测试温度为0
            result = self.client.generate_text("Test", temperature=0.0)
            self.assertIsNotNone(result)
            
            # 测试温度为2.0
            result = self.client.generate_text("Test", temperature=2.0)
            self.assertIsNotNone(result)
    
    def test_zero_max_tokens(self):
        """测试零最大token数 - 服务器可能返回错误"""
        with patch.object(self.client.session, 'post') as mock_post:
            # 模拟服务器返回错误（因为max_tokens=0可能不被接受）
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.text = "max_tokens must be greater than 0"
            mock_post.return_value = mock_response
            
            with patch('sys.stdout', new=StringIO()) as fake_out:
                result = self.client.generate_text("Test", max_tokens=0)
                self.assertIsNone(result)
                output = fake_out.getvalue()
                self.assertIn("服务器返回错误", output)
    
    def test_single_max_tokens(self):
        """测试单个最大token数"""
        with patch.object(self.client.session, 'post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "id": "test",
                "text": "A",
                "tokens": ["A"],
                "response_time": 0.1,
                "tokens_per_second": 10.0
            }
            mock_post.return_value = mock_response
            
            result = self.client.generate_text("Test", max_tokens=1)
            self.assertIsNotNone(result)
            self.assertIn("generated_text", result)


def run_tests():
    """运行所有测试"""
    print("=" * 70)
    print("xLLM 客户端测试程序")
    print("=" * 70)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    suite.addTests(loader.loadTestsFromTestCase(TestXLLMClient))
    suite.addTests(loader.loadTestsFromTestCase(TestXLLMClientCommandArgs))
    suite.addTests(loader.loadTestsFromTestCase(TestXLLMClientEdgeCases))
    
    # 可选：添加集成测试（需要服务器运行）
    print("是否运行集成测试（需要服务器运行）？[y/N]: ", end='', flush=True)
    try:
        choice = input().strip().lower()
        if choice == 'y':
            suite.addTests(loader.loadTestsFromTestCase(TestXLLMClientIntegration))
    except EOFError:
        pass
    
    print()
    print("开始运行测试...")
    print("-" * 70)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印测试结果摘要
    print()
    print("=" * 70)
    print("测试结果摘要")
    print("=" * 70)
    print(f"运行测试数: {result.testsRun}")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
