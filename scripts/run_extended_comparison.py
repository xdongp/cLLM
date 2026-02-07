#!/usr/bin/env python3
"""
扩展的 CPU vs GPU 对比测试脚本

使用更长的输入和更多的token生成来观察更明显的输出模式
"""

import subprocess
import json
import time
import os
import sys

class ExtendedComparisonTester:
    def __init__(self, model_path=None, timeout=120):
        self.model_path = model_path or "model/Qwen/Qwen3-0.6B"
        self.timeout = timeout
        self.cpu_port = 18080
        self.gpu_port = 8080
    
    def start_server(self, config_file, port):
        """启动服务器"""
        cmd = [
            "./build/bin/cllm_server",
            "--model-path", self.model_path,
            "--config", config_file,
            "--log-level", "error"
        ]
        
        print(f"启动服务器 (端口{port})...")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/Users/dannypan/PycharmProjects/cLLM"
        )
        
        # 等待服务器启动
        start_time = time.time()
        while time.time() - start_time < self.timeout:
            try:
                import urllib.request
                req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        print(f"服务器 (端口{port}) 已就绪!")
                        return process
            except:
                time.sleep(1)
        
        raise TimeoutError(f"服务器 (端口{port}) 启动超时")
    
    def stop_server(self, process):
        """停止服务器"""
        print("停止服务器...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    
    def generate_text(self, prompt, port, max_tokens=20, temperature=0.0):
        """调用生成API"""
        import urllib.request
        
        data = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_output(self, text):
        """分析输出质量"""
        import re
        
        if not text:
            return {
                "has_gibberish": True,
                "non_ascii_ratio": 0.0,
                "special_char_ratio": 0.0,
                "word_count": 0,
                "length": 0
            }
        
        length = len(text)
        words = text.split()
        word_count = len(words)
        
        # 非ASCII字符比例
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / length if length > 0 else 0.0
        
        # 特殊字符比例
        special_chars = re.findall(r'[^\w\s]', text)
        special_ratio = len(special_chars) / length if length > 0 else 0.0
        
        # 判断是否有乱码
        has_gibberish = non_ascii_ratio > 0.7 or special_ratio > 0.5
        
        return {
            "has_gibberish": has_gibberish,
            "non_ascii_ratio": non_ascii_ratio,
            "special_char_ratio": special_ratio,
            "word_count": word_count,
            "length": length
        }
    
    def run_test(self, prompt, max_tokens=20):
        """运行单个测试"""
        print(f"\n" + "-"*80)
        print(f"测试输入: '{prompt}'")
        print(f"生成token数: {max_tokens}")
        print("-"*80)
        
        # 测试CPU
        print("\n=== CPU测试 ===")
        cpu_response = self.generate_text(prompt, self.cpu_port, max_tokens)
        
        if cpu_response.get("success"):
            cpu_text = cpu_response.get("data", {}).get("text", "")
            cpu_tokens = cpu_response.get("data", {}).get("generated_tokens", 0)
            cpu_time = cpu_response.get("data", {}).get("response_time", 0)
            cpu_analysis = self.analyze_output(cpu_text)
            
            print(f"CPU输出: {repr(cpu_text[:200])}...")
            print(f"CPU生成token数: {cpu_tokens}")
            print(f"CPU响应时间: {cpu_time:.2f}ms")
            print(f"CPU乱码检测: {'✗' if cpu_analysis['has_gibberish'] else '✓'}")
            print(f"CPU非ASCII比例: {cpu_analysis['non_ascii_ratio']:.4f}")
            print(f"CPU特殊字符比例: {cpu_analysis['special_char_ratio']:.4f}")
        else:
            cpu_text = ""
            cpu_tokens = 0
            cpu_time = 0
            print(f"CPU错误: {cpu_response.get('error', 'Unknown error')}")
        
        # 测试GPU
        print("\n=== GPU测试 ===")
        gpu_response = self.generate_text(prompt, self.gpu_port, max_tokens)
        
        if gpu_response.get("success"):
            gpu_text = gpu_response.get("data", {}).get("text", "")
            gpu_tokens = gpu_response.get("data", {}).get("generated_tokens", 0)
            gpu_time = gpu_response.get("data", {}).get("response_time", 0)
            gpu_analysis = self.analyze_output(gpu_text)
            
            print(f"GPU输出: {repr(gpu_text[:200])}...")
            print(f"GPU生成token数: {gpu_tokens}")
            print(f"GPU响应时间: {gpu_time:.2f}ms")
            print(f"GPU乱码检测: {'✗' if gpu_analysis['has_gibberish'] else '✓'}")
            print(f"GPU非ASCII比例: {gpu_analysis['non_ascii_ratio']:.4f}")
            print(f"GPU特殊字符比例: {gpu_analysis['special_char_ratio']:.4f}")
        else:
            gpu_text = ""
            gpu_tokens = 0
            gpu_time = 0
            print(f"GPU错误: {gpu_response.get('error', 'Unknown error')}")
        
        # 对比
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            print(f"\n=== 性能对比 ===")
            print(f"GPU比CPU快 {speedup:.2f}x")
    
    def run_all_tests(self):
        """运行所有测试"""
        # 测试用例
        test_cases = [
            ("Hello, how are you?", 20),
            ("What is the capital of France?", 20),
            ("Python is a programming language", 20),
            ("1+1=2, 2+2=", 20),
            ("你好，今天天气怎么样？", 20)
        ]
        
        # 启动服务器
        cpu_process = None
        gpu_process = None
        
        try:
            # 启动CPU服务器
            cpu_process = self.start_server("config/config_kylin_cpu.yaml", self.cpu_port)
            time.sleep(5)
            
            # 启动GPU服务器
            gpu_process = self.start_server("config/config_kylin_gpu.yaml", self.gpu_port)
            time.sleep(5)
            
            # 运行测试
            for prompt, max_tokens in test_cases:
                self.run_test(prompt, max_tokens)
                
        finally:
            if gpu_process:
                self.stop_server(gpu_process)
            if cpu_process:
                self.stop_server(cpu_process)


def main():
    """主函数"""
    if not os.path.exists("./build/bin/cllm_server"):
        print("错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    tester = ExtendedComparisonTester()
    tester.run_all_tests()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
