#!/usr/bin/env python3
"""
CPU vs GPU 实际推理对比测试脚本

此脚本用于在真实服务器上对比CPU和GPU的推理输出，收集详细的对比数据。

使用方法:
    python3 cpu_gpu_comparison_test.py

输出:
    - 对比报告 (markdown格式)
    - 详细的输出差异分析
"""

import subprocess
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class CPUGPUComparisonTest:
    def __init__(self):
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "cpu_results": {},
            "gpu_results": {},
            "comparisons": []
        }
        self.base_url = "http://127.0.0.1:8080"
        
    def start_server(self, config_path: str, timeout: int = 120) -> subprocess.Popen:
        """启动服务器"""
        cmd = [
            "./build/bin/cllm_server",
            "--model-path", "model/Qwen/Qwen3-0.6B",
            "--config", config_path,
            "--log-level", "error"
        ]
        
        print(f"启动服务器: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd="/Users/dannypan/PycharmProjects/cLLM"
        )
        
        # 等待服务器启动
        print(f"等待服务器启动 (最多{timeout}秒)...")
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                import urllib.request
                req = urllib.request.Request(f"{self.base_url}/health")
                with urllib.request.urlopen(req, timeout=2) as response:
                    if response.status == 200:
                        print("服务器已就绪!")
                        return process
            except:
                time.sleep(1)
        
        raise TimeoutError("服务器启动超时")
    
    def stop_server(self, process: subprocess.Popen):
        """停止服务器"""
        print("停止服务器...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("服务器已停止")
    
    def generate_text(self, prompt: str, max_tokens: int = 30, temperature: float = 0.0) -> Dict:
        """调用生成API"""
        import urllib.request
        import urllib.error
        
        data = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }).encode('utf-8')
        
        req = urllib.request.Request(
            f"{self.base_url}/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method='POST'
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read().decode('utf-8'))
        except urllib.error.HTTPError as e:
            return {"success": False, "error": f"HTTP {e.code}: {e.reason}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_cpu_inference(self, prompts: List[str]) -> Dict:
        """测试CPU推理"""
        print("\n" + "="*60)
        print("测试 CPU 推理")
        print("="*60)
        
        process = None
        try:
            process = self.start_server("config/config_kylin_cpu.yaml")
            
            results = {}
            for prompt in prompts:
                print(f"\n测试输入: '{prompt}'")
                response = self.generate_text(prompt)
                
                if response.get("success"):
                    data = response.get("data", {})
                    results[prompt] = {
                        "text": data.get("text", ""),
                        "generated_tokens": data.get("generated_tokens", 0),
                        "response_time": data.get("response_time", 0),
                        "tokens_per_second": data.get("tokens_per_second", 0)
                    }
                    print(f"输出: {data.get('text', '')[:100]}...")
                    print(f"Token数: {data.get('generated_tokens', 0)}")
                    print(f"响应时间: {data.get('response_time', 0):.3f}s")
                else:
                    results[prompt] = {"error": response.get("error", "Unknown error")}
                    print(f"错误: {response.get('error', 'Unknown error')}")
            
            return results
            
        finally:
            if process:
                self.stop_server(process)
    
    def test_gpu_inference(self, prompts: List[str]) -> Dict:
        """测试GPU推理"""
        print("\n" + "="*60)
        print("测试 GPU 推理")
        print("="*60)
        
        process = None
        try:
            process = self.start_server("config/config_kylin_gpu.yaml")
            
            results = {}
            for prompt in prompts:
                print(f"\n测试输入: '{prompt}'")
                response = self.generate_text(prompt)
                
                if response.get("success"):
                    data = response.get("data", {})
                    results[prompt] = {
                        "text": data.get("text", ""),
                        "generated_tokens": data.get("generated_tokens", 0),
                        "response_time": data.get("response_time", 0),
                        "tokens_per_second": data.get("tokens_per_second", 0)
                    }
                    print(f"输出: {data.get('text', '')[:100]}...")
                    print(f"Token数: {data.get('generated_tokens', 0)}")
                    print(f"响应时间: {data.get('response_time', 0):.3f}s")
                else:
                    results[prompt] = {"error": response.get("error", "Unknown error")}
                    print(f"错误: {response.get('error', 'Unknown error')}")
            
            return results
            
        finally:
            if process:
                self.stop_server(process)
    
    def analyze_text_quality(self, text: str) -> Dict:
        """分析文本质量"""
        import re
        
        # 检查是否有乱码特征
        # 1. 检查是否包含大量非ASCII字符
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = non_ascii_count / len(text) if text else 0
        
        # 2. 检查是否有连续的特殊字符
        special_chars = re.findall(r'[^\w\s]', text)
        special_ratio = len(special_chars) / len(text) if text else 0
        
        # 3. 检查是否有重复的字符模式
        words = text.split()
        unique_words = set(words)
        repetition_ratio = 1 - len(unique_words) / len(words) if words else 0
        
        # 4. 检查是否有意义不明的混合语言
        has_gibberish = non_ascii_ratio > 0.5 or special_ratio > 0.3 or repetition_ratio > 0.5
        
        return {
            "length": len(text),
            "non_ascii_ratio": non_ascii_ratio,
            "special_char_ratio": special_ratio,
            "repetition_ratio": repetition_ratio,
            "word_count": len(words),
            "unique_words": len(unique_words),
            "has_gibberish": has_gibberish,
            "quality_score": max(0, 1 - (non_ascii_ratio + special_ratio + repetition_ratio) / 3)
        }
    
    def compare_results(self, cpu_results: Dict, gpu_results: Dict, prompts: List[str]) -> List[Dict]:
        """对比CPU和GPU结果"""
        comparisons = []
        
        for prompt in prompts:
            cpu_result = cpu_results.get(prompt, {})
            gpu_result = gpu_results.get(prompt, {})
            
            if "error" in cpu_result or "error" in gpu_result:
                comparison = {
                    "prompt": prompt,
                    "status": "ERROR",
                    "cpu_error": cpu_result.get("error"),
                    "gpu_error": gpu_result.get("error")
                }
            else:
                cpu_text = cpu_result.get("text", "")
                gpu_text = gpu_result.get("text", "")
                
                cpu_quality = self.analyze_text_quality(cpu_text)
                gpu_quality = self.analyze_text_quality(gpu_text)
                
                comparison = {
                    "prompt": prompt,
                    "status": "OK",
                    "cpu_output": cpu_text[:200],  # 截断显示
                    "gpu_output": gpu_text[:200],
                    "cpu_quality": cpu_quality,
                    "gpu_quality": gpu_quality,
                    "cpu_tokens": cpu_result.get("generated_tokens", 0),
                    "gpu_tokens": gpu_result.get("generated_tokens", 0),
                    "cpu_time": cpu_result.get("response_time", 0),
                    "gpu_time": gpu_result.get("response_time", 0),
                    "cpu_tps": cpu_result.get("tokens_per_second", 0),
                    "gpu_tps": gpu_result.get("tokens_per_second", 0),
                    "quality_match": abs(cpu_quality["quality_score"] - gpu_quality["quality_score"]) < 0.3,
                    "cpu_has_gibberish": cpu_quality["has_gibberish"],
                    "gpu_has_gibberish": gpu_quality["has_gibberish"]
                }
            
            comparisons.append(comparison)
        
        return comparisons
    
    def generate_report(self, comparisons: List[Dict], output_file: str = "cpu_gpu_comparison_report.md"):
        """生成对比报告"""
        report = []
        report.append("# CPU vs GPU 推理对比测试报告")
        report.append(f"\n生成时间: {self.results['timestamp']}")
        report.append(f"\n模型: Qwen/Qwen3-0.6B")
        report.append("\n" + "="*60 + "\n")
        
        # 汇总统计
        total_tests = len(comparisons)
        quality_mismatch = sum(1 for c in comparisons if not c.get("quality_match", True))
        cpu_gibberish = sum(1 for c in comparisons if c.get("cpu_has_gibberish", False))
        gpu_gibberish = sum(1 for c in comparisons if c.get("gpu_has_gibberish", False))
        
        report.append("## 汇总统计\n")
        report.append(f"- 总测试数: {total_tests}")
        report.append(f"- 质量不匹配: {quality_mismatch}")
        report.append(f"- CPU乱码: {cpu_gibberish}")
        report.append(f"- GPU乱码: {gpu_gibberish}")
        report.append("")
        
        # 详细对比
        report.append("## 详细对比\n")
        
        for i, comp in enumerate(comparisons, 1):
            report.append(f"### 测试 {i}: '{comp['prompt']}'\n")
            
            if comp.get("status") == "ERROR":
                report.append(f"**状态**: ERROR")
                if comp.get("cpu_error"):
                    report.append(f"- CPU错误: {comp['cpu_error']}")
                if comp.get("gpu_error"):
                    report.append(f"- GPU错误: {comp['gpu_error']}")
            else:
                report.append(f"**状态**: {comp['status']}")
                report.append(f"**质量匹配**: {'✓' if comp['quality_match'] else '✗'}")
                report.append("")
                
                report.append("**CPU输出**:")
                report.append(f"```\n{comp['cpu_output']}\n```")
                report.append(f"- Token数: {comp['cpu_tokens']}")
                report.append(f"- 响应时间: {comp['cpu_time']:.3f}s")
                report.append(f"- 速度: {comp['cpu_tps']:.1f} tokens/s")
                report.append(f"- 质量评分: {comp['cpu_quality']['quality_score']:.2f}")
                report.append(f"- 乱码: {'✗' if comp['cpu_has_gibberish'] else '✓'}")
                report.append("")
                
                report.append("**GPU输出**:")
                report.append(f"```\n{comp['gpu_output']}\n```")
                report.append(f"- Token数: {comp['gpu_tokens']}")
                report.append(f"- 响应时间: {comp['gpu_time']:.3f}s")
                report.append(f"- 速度: {comp['gpu_tps']:.1f} tokens/s")
                report.append(f"- 质量评分: {comp['gpu_quality']['quality_score']:.2f}")
                report.append(f"- 乱码: {'✗' if comp['gpu_has_gibberish'] else '✓'}")
            
            report.append("")
        
        # 保存报告
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"\n报告已保存到: {output_file}")
        return '\n'.join(report)
    
    def run(self):
        """运行完整对比测试"""
        print("="*60)
        print("CPU vs GPU 实际推理对比测试")
        print("="*60)
        
        # 测试用例
        test_prompts = [
            "hello",
            "hi",
            "你好",
            "what is AI",
            "什么是机器学习",
            "1+1=",
            "The capital of France is",
            "人工智能的未来发展"
        ]
        
        # 测试CPU推理
        cpu_results = self.test_cpu_inference(test_prompts)
        self.results["cpu_results"] = cpu_results
        
        # 等待端口释放
        print("\n等待端口释放...")
        time.sleep(3)
        
        # 测试GPU推理
        gpu_results = self.test_gpu_inference(test_prompts)
        self.results["gpu_results"] = gpu_results
        
        # 对比结果
        print("\n" + "="*60)
        print("对比分析")
        print("="*60)
        comparisons = self.compare_results(cpu_results, gpu_results, test_prompts)
        self.results["comparisons"] = comparisons
        
        # 生成报告
        report = self.generate_report(comparisons)
        
        # 打印摘要
        print("\n" + "="*60)
        print("测试摘要")
        print("="*60)
        
        for comp in comparisons:
            if comp.get("status") == "OK":
                match = "✓" if comp["quality_match"] else "✗"
                cpu_ok = "✓" if not comp["cpu_has_gibberish"] else "✗"
                gpu_ok = "✓" if not comp["gpu_has_gibberish"] else "✗"
                print(f"'{comp['prompt'][:20]}...': 匹配{match} | CPU{cpu_ok} | GPU{gpu_ok}")
        
        return self.results


def main():
    """主函数"""
    # 检查是否在正确的目录
    if not os.path.exists("./build/bin/cllm_server"):
        print("错误: 请在项目根目录运行此脚本")
        print("当前目录:", os.getcwd())
        sys.exit(1)
    
    # 运行测试
    tester = CPUGPUComparisonTest()
    results = tester.run()
    
    print("\n测试完成!")


if __name__ == "__main__":
    main()
