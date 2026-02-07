#!/usr/bin/env python3
"""
中文输入调试脚本

专门测试CPU和GPU在处理中文输入时的表现
"""

import subprocess
import json
import time
import os
import sys

class ChineseInputDebugger:
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
    
    def generate_text(self, prompt, port, max_tokens=10, temperature=0.0):
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
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def analyze_output(self, text):
        """分析输出质量"""
        import re
        
        if not text:
            return {
                "has_gibberish": True,
                "has_repeats": True,
                "chinese_chars": 0,
                "total_chars": 0,
                "special_chars": 0
            }
        
        total_chars = len(text)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        special_chars = sum(1 for c in text if not c.isalnum() and not c.isspace())
        
        # 检查是否有重复模式
        words = text.split()
        has_repeats = len(set(words)) < len(words) / 2 if words else False
        
        # 检查是否有大量重复的 copyright
        copyright_count = text.count('copyright')
        has_copyright_spam = copyright_count > 3
        
        # 检查是否有乱码
        has_gibberish = has_copyright_spam or (chinese_chars == 0 and total_chars > 0)
        
        return {
            "has_gibberish": has_gibberish,
            "has_repeats": has_repeats,
            "has_copyright_spam": has_copyright_spam,
            "chinese_chars": chinese_chars,
            "total_chars": total_chars,
            "special_chars": special_chars,
            "copyright_count": copyright_count
        }
    
    def run_test(self, prompt, max_tokens=10):
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
            print(f"CPU中文字符数: {cpu_analysis['chinese_chars']}/{cpu_analysis['total_chars']}")
            print(f"CPU特殊字符数: {cpu_analysis['special_chars']}")
            print(f"CPU重复检测: {'✗' if cpu_analysis['has_repeats'] else '✓'}")
            print(f"CPU copyright spam: {'✗' if cpu_analysis['has_copyright_spam'] else '✓'}")
        else:
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
            print(f"GPU中文字符数: {gpu_analysis['chinese_chars']}/{gpu_analysis['total_chars']}")
            print(f"GPU特殊字符数: {gpu_analysis['special_chars']}")
            print(f"GPU重复检测: {'✗' if gpu_analysis['has_repeats'] else '✓'}")
            print(f"GPU copyright spam: {'✗' if gpu_analysis['has_copyright_spam'] else '✓'}")
            print(f"GPU copyright count: {gpu_analysis['copyright_count']}")
        else:
            print(f"GPU错误: {gpu_response.get('error', 'Unknown error')}")
    
    def run_all_tests(self):
        """运行所有测试"""
        # 中文测试用例
        chinese_tests = [
            "你好",
            "今天天气",
            "我爱中国",
            "北京",
            "上海"
        ]
        
        # 英文对比测试
        english_tests = [
            "hello",
            "today",
            "I love",
            "Beijing",
            "Shanghai"
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
            
            # 测试中文输入
            print("\n" + "="*80)
            print("测试中文输入")
            print("="*80)
            for prompt in chinese_tests:
                self.run_test(prompt, 10)
            
            # 测试英文输入
            print("\n" + "="*80)
            print("测试英文输入")
            print("="*80)
            for prompt in english_tests:
                self.run_test(prompt, 10)
                
        finally:
            if gpu_process:
                self.stop_server(gpu_process)
            if cpu_process:
                self.stop_server(cpu_process)
    
    def analyze_patterns(self):
        """分析模式"""
        print("\n" + "="*80)
        print("分析模式")
        print("="*80)
        
        print("观察到的问题模式:")
        print("1. GPU在处理中文输入时输出大量重复的 'copyright' 字符串")
        print("2. CPU在处理中文输入时能够输出中文字符，但质量不稳定")
        print("3. GPU在处理英文输入时表现相对正常")
        print("\n可能的根因:")
        print("1. 权重处理问题: 中文token的权重在GPU上处理错误")
        print("2. 词汇表问题: GPU可能使用了错误的词汇表或token映射")
        print("3. Attention计算问题: 中文token的注意力计算异常")
        print("4. 内存管理问题: GPU内存中的某些区域被意外重复访问")
        print("\n建议的调试步骤:")
        print("1. 检查中文token的Embedding输出")
        print("2. 验证Attention层对中文token的处理")
        print("3. 检查GPU内存中的权重值")
        print("4. 对比CPU和GPU的词汇表处理")


def main():
    """主函数"""
    if not os.path.exists("./build/bin/cllm_server"):
        print("错误: 请在项目根目录运行此脚本")
        sys.exit(1)
    
    debugger = ChineseInputDebugger()
    debugger.run_all_tests()
    debugger.analyze_patterns()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
