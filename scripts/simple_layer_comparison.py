#!/usr/bin/env python3
"""
简单的逐层输出对比测试脚本
"""

import subprocess
import sys
import os

def run_cpu_test():
    """运行 CPU 测试"""
    print("\n" + "="*60)
    print("运行 CPU 测试")
    print("="*60)
    
    cmd = [
        "./bin/kylin_test_suite",
        "--stage=19"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/Users/dannypan/PycharmProjects/cLLM/build",
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.stdout

def run_gpu_test():
    """运行 GPU 测试"""
    print("\n" + "="*60)
    print("运行 GPU 测试")
    print("="*60)
    
    # GPU 测试需要设置环境变量
    env = os.environ.copy()
    env['CLLM_DEVICE'] = 'gpu'
    
    cmd = [
        "./bin/kylin_test_suite",
        "--stage=19"
    ]
    
    result = subprocess.run(
        cmd,
        cwd="/Users/dannypan/PycharmProjects/cLLM/build",
        capture_output=True,
        text=True,
        env=env
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.stdout

def main():
    """主函数"""
    print("逐层输出对比测试")
    
    # 运行 CPU 测试
    cpu_output = run_cpu_test()
    
    # 运行 GPU 测试
    gpu_output = run_gpu_test()
    
    # 分析输出
    print("\n" + "="*60)
    print("输出分析")
    print("="*60)
    
    # 查找 LAYER_DEBUG 输出
    cpu_debug_lines = [line for line in cpu_output.split('\n') if 'LAYER_DEBUG' in line]
    gpu_debug_lines = [line for line in gpu_output.split('\n') if 'LAYER_DEBUG' in line]
    
    print(f"\nCPU 调试输出行数: {len(cpu_debug_lines)}")
    print(f"GPU 调试输出行数: {len(gpu_debug_lines)}")
    
    if cpu_debug_lines:
        print("\nCPU 调试输出:")
        for line in cpu_debug_lines[:10]:  # 只显示前 10 行
            print(f"  {line}")
    
    if gpu_debug_lines:
        print("\nGPU 调试输出:")
        for line in gpu_debug_lines[:10]:  # 只显示前 10 行
            print(f"  {line}")
    
    if not cpu_debug_lines and not gpu_debug_lines:
        print("\n⚠️  未找到 LAYER_DEBUG 输出")
        print("可能的原因:")
        print("1. 调试代码未被触发")
        print("2. 日志级别设置不正确")
        print("3. 测试程序未调用 forward 函数")

if __name__ == "__main__":
    main()
