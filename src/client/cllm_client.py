#!/usr/bin/env python3
"""
cLLM 交互式客户端
支持流式和非流式文本生成
"""

import os
import requests
import json
import time
import argparse
from typing import Dict, Any, Optional


# 默认服务器地址
DEFAULT_SERVER_URL = "http://localhost:8080"


class CLLMClient:
    """cLLM 客户端类"""
    
    def __init__(self, server_url: str = None):
        if server_url is None:
            server_url = os.environ.get("CLLM_SERVER_URL", DEFAULT_SERVER_URL)
        self.server_url = server_url
        self.session = requests.Session()
        self.default_params = {
            "max_tokens": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50
        }
        self.interrupt_requested = False
        self._token_count = 0
        
    def check_server_health(self) -> bool:
        """检查服务器健康状态"""
        try:
            timeout = float(os.environ.get("CLLM_CLIENT_HEALTH_TIMEOUT", "5"))
            response = self.session.get(f"{self.server_url}/health", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False
    
    def request_interrupt(self):
        """请求中断当前生成"""
        self.interrupt_requested = True
    
    def reset_interrupt(self):
        """重置中断标志"""
        self.interrupt_requested = False
    
    def generate_text(self, prompt: str, stream: bool = False, **kwargs) -> Optional[Dict[str, Any]]:
        """
        生成文本
        
        Args:
            prompt: 输入提示词
            stream: 是否使用流式输出
            **kwargs: 其他参数 (max_tokens, temperature, top_p, top_k)
        
        Returns:
            包含生成结果的字典，失败返回 None
        """
        self.reset_interrupt()
        
        # 合并参数
        params = self.default_params.copy()
        params.update(kwargs)
        params["prompt"] = prompt
        
        if stream:
            return self._generate_streaming(params)
        else:
            return self._generate_non_streaming(params)
    
    def _generate_non_streaming(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """非流式生成"""
        try:
            timeout = float(os.environ.get("CLLM_CLIENT_GENERATE_TIMEOUT", "120"))
            response = self.session.post(
                f"{self.server_url}/generate",
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=json.dumps(params, ensure_ascii=False).encode('utf-8'),
                timeout=timeout
            )
            
            if response.status_code == 200:
                response.encoding = 'utf-8'
                result = response.json()
                # 统一提取生成的文本
                if 'data' in result and isinstance(result['data'], dict):
                    data = result['data']
                    if 'text' in data:
                        result['generated_text'] = data['text']
                    if 'generated_tokens' in data:
                        result['token_count'] = data['generated_tokens']
                elif 'text' in result:
                    result['generated_text'] = result['text']
                return result
            else:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(f"   错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器")
            return None
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None

    def _generate_streaming(self, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """流式生成"""
        params["stream"] = True
        self._token_count = 0
        
        try:
            response = self.session.post(
                f"{self.server_url}/generate_stream",
                headers={"Content-Type": "application/json; charset=utf-8"},
                data=json.dumps(params, ensure_ascii=False).encode('utf-8'),
                stream=True,
                timeout=300
            )
            
            if response.status_code == 200:
                print("🤖 服务器: ", end='', flush=True)
                
                full_text = ""
                buffer = ""
                
                for chunk in response.iter_content(chunk_size=None, decode_unicode=False):
                    if self.interrupt_requested:
                        print("\n✅ 生成已中断")
                        return {"generated_text": full_text, "finish_reason": "interrupted", "token_count": self._token_count}
                    
                    if not chunk:
                        continue
                    
                    try:
                        buffer += chunk.decode('utf-8')
                    except UnicodeDecodeError:
                        continue
                    
                    # 处理缓冲区中的完整行
                    while '\n' in buffer:
                        line, buffer = buffer.split('\n', 1)
                        line = line.strip()
                        
                        if not line:
                            continue
                        
                        if line.startswith('data: '):
                            data_str = line[6:]
                            if data_str.strip() == '[DONE]':
                                break
                            try:
                                data = json.loads(data_str)
                                if "token" in data:
                                    token_text = data["token"]
                                    full_text += token_text
                                    self._token_count += 1
                                    print(token_text, end='', flush=True)
                                if data.get("done", False):
                                    print()
                                    return {"generated_text": full_text, "token_count": self._token_count}
                            except json.JSONDecodeError:
                                continue
                
                print()
                return {"generated_text": full_text, "token_count": self._token_count}
            else:
                print(f"❌ 服务器返回错误: {response.status_code}")
                print(f"   错误信息: {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print("❌ 请求超时")
            return None
        except requests.exceptions.ConnectionError:
            print("❌ 无法连接到服务器")
            return None
        except Exception as e:
            print(f"❌ 流式请求失败: {e}")
            return None
    
    def show_help(self):
        """显示帮助信息"""
        print()
        print("=" * 50)
        print("📖 cLLM 客户端帮助")
        print("=" * 50)
        print()
        print("基本用法:")
        print("  直接输入问题，按回车发送")
        print()
        print("命令列表:")
        print("  help          - 显示此帮助信息")
        print("  config        - 查看当前配置")
        print("  set <k> <v>   - 设置参数 (如: set max_tokens 200)")
        print("  stream on     - 开启流式输出 (逐字显示)")
        print("  stream off    - 关闭流式输出 (一次显示)")
        print("  clear         - 清屏")
        print("  quit/exit/q   - 退出程序")
        print()
        print("可设置的参数:")
        print("  max_tokens    - 最大生成token数 (整数)")
        print("  temperature   - 温度，控制随机性 (0.0-2.0)")
        print("  top_p         - Top-p 采样参数 (0.0-1.0)")
        print("  top_k         - Top-k 采样参数 (整数)")
        print()
        print("快捷键:")
        print("  Ctrl+C        - 中断当前生成")
        print("=" * 50)
        print()
    
    def show_config(self):
        """显示当前配置"""
        print()
        print("⚙️  当前配置:")
        print(f"  服务器地址: {self.server_url}")
        for key, value in self.default_params.items():
            print(f"  {key}: {value}")
        print()
    
    def set_config(self, config_str: str) -> bool:
        """设置配置参数"""
        parts = config_str.strip().split()
        if len(parts) < 2:
            print("❌ 用法: set <参数名> <参数值>")
            print("   例如: set max_tokens 200")
            return False
        
        param_name = parts[0]
        param_value_str = parts[1]
        
        # 类型转换
        try:
            if param_name in ['max_tokens', 'top_k']:
                param_value = int(param_value_str)
            elif param_name in ['temperature', 'top_p']:
                param_value = float(param_value_str)
            else:
                param_value = param_value_str
        except ValueError:
            print(f"❌ 无效的参数值: {param_value_str}")
            return False
        
        self.default_params[param_name] = param_value
        print(f"✅ 已设置 {param_name} = {param_value}")
        return True

    def chat_loop(self):
        """交互式聊天循环"""
        print()
        print("🚀 cLLM 交互式客户端")
        print("=" * 50)
        print(f"服务器: {self.server_url}")
        print("输入 'help' 查看帮助，'quit' 退出")
        print("=" * 50)
        
        # 检查服务器状态
        if not self.check_server_health():
            print()
            print("❌ 无法连接到服务器，请确保 cLLM 服务器正在运行")
            print(f"   服务器地址: {self.server_url}")
            return
        
        print("✅ 已连接到服务器")
        print()
        
        # 默认启用流式输出
        streaming_enabled = True
        
        while True:
            try:
                user_input = input("💬 您: ").strip()
                
                if not user_input:
                    continue
                
                # 处理命令
                cmd = user_input.lower()
                
                if cmd in ['quit', 'exit', 'q']:
                    self.request_interrupt()
                    print("👋 再见！")
                    break
                    
                elif cmd == 'clear':
                    os.system('clear' if os.name != 'nt' else 'cls')
                    continue
                    
                elif cmd == 'help':
                    self.show_help()
                    continue
                    
                elif cmd == 'config':
                    self.show_config()
                    continue
                    
                elif cmd.startswith('set '):
                    self.set_config(user_input[4:])
                    continue
                    
                elif cmd == 'stream on':
                    streaming_enabled = True
                    print("✅ 已启用流式输出")
                    continue
                    
                elif cmd == 'stream off':
                    streaming_enabled = False
                    print("✅ 已禁用流式输出")
                    continue
                
                # 发送生成请求
                if not streaming_enabled:
                    print("⏳ 正在生成...", end='', flush=True)
                
                start_time = time.time()
                result = self.generate_text(user_input, stream=streaming_enabled)
                end_time = time.time()
                
                if result:
                    # 提取生成的文本
                    generated_text = result.get("generated_text", "")
                    
                    # 非流式模式显示结果
                    if not streaming_enabled:
                        print()
                        print(f"🤖 服务器: {generated_text}")
                    
                    # 统计信息
                    response_time = end_time - start_time
                    token_count = result.get("token_count", self._token_count)
                    if token_count == 0 and generated_text:
                        token_count = len(generated_text) // 2 + 1  # 估算
                    
                    speed = token_count / response_time if response_time > 0 else 0
                    print(f"📈 统计: {token_count} tokens, {response_time:.2f}s, {speed:.1f} t/s")
                else:
                    if not streaming_enabled:
                        print()
                    print("❌ 生成失败，请重试")
                
                print()
                
            except KeyboardInterrupt:
                print("\n✅ 已中断")
                self.reset_interrupt()
                continue
            except EOFError:
                print("\n👋 再见！")
                break


def main():
    parser = argparse.ArgumentParser(
        description="cLLM 交互式客户端",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python cllm_client.py --server-url http://localhost:8085
  python cllm_client.py --max-tokens 200 --temperature 0.8
        """
    )
    parser.add_argument(
        "--server-url", 
        default=DEFAULT_SERVER_URL,
        help=f"cLLM 服务器地址 (默认: {DEFAULT_SERVER_URL})"
    )
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=100,
        help="最大生成token数 (默认: 100)"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7,
        help="温度参数 (默认: 0.7)"
    )
    parser.add_argument(
        "--top-p", 
        type=float, 
        default=0.9,
        help="Top-p 采样参数 (默认: 0.9)"
    )
    parser.add_argument(
        "--top-k", 
        type=int, 
        default=50,
        help="Top-k 采样参数 (默认: 50)"
    )
    
    args = parser.parse_args()
    
    # 创建客户端
    client = CLLMClient(args.server_url)
    
    # 设置参数
    client.default_params.update({
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k
    })
    
    # 开始聊天
    client.chat_loop()


if __name__ == "__main__":
    main()
