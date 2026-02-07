#!/usr/bin/env python3
"""
CPU vs GPU 逐层输出对比测试
"""

import subprocess
import sys
import os
import re

def run_command(cmd, env=None):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, env=env)
    return result.stdout + result.stderr

def extract_generation_steps(output):
    """提取生成的 token 步骤"""
    steps = []
    # 匹配 "Step  N: Token  XXXX ("")" 格式
    pattern = r'Step\s+(\d+):\s+Token\s+(\d+)\s*\("([^"]*)"\)'
    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            step_num = int(match.group(1))
            token_id = int(match.group(2))
            token_text = match.group(3)
            steps.append((step_num, token_id, token_text))
    return steps

def main():
    model_path = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B"
    test_input = "1+1="
    build_dir = "/Users/dannypan/PycharmProjects/cLLM/build"

    print("=" * 70)
    print("CPU vs GPU 逐层输出对比测试")
    print("=" * 70)
    print(f"输入: '{test_input}'")
    print(f"预期 CPU 输出: 2,1")  # 1+1=2
    print()

    env = os.environ.copy()
    env["CLLM_MODEL_PATH"] = model_path

    # CPU 推理
    print(">>> 运行 CPU 推理...")
    cpu_output = run_command(
        f"{build_dir}/bin/show_model_output --model {model_path} --input '{test_input}' --device cpu --max_tokens 5",
        env=env
    )

    # GPU 推理
    print(">>> 运行 GPU 推理...")
    gpu_output = run_command(
        f"{build_dir}/bin/show_model_output --model {model_path} --input '{test_input}' --device gpu --max_tokens 5",
        env=env
    )

    # 提取生成步骤
    cpu_steps = extract_generation_steps(cpu_output)
    gpu_steps = extract_generation_steps(gpu_output)

    print()
    print("=" * 70)
    print("对比结果")
    print("=" * 70)

    print(f"\nCPU 生成步骤 ({len(cpu_steps)} tokens):")
    for step, token_id, text in cpu_steps:
        print(f"  Step {step}: Token {token_id:6d} -> '{text}'")

    print(f"\nGPU 生成步骤 ({len(gpu_steps)} tokens):")
    for step, token_id, text in gpu_steps:
        print(f"  Step {step}: Token {token_id:6d} -> '{text}'")

    # 逐 token 对比
    print()
    print("=" * 70)
    print("逐 token 差异分析")
    print("=" * 70)

    max_steps = max(len(cpu_steps), len(gpu_steps))
    all_match = True

    for i in range(max_steps):
        cpu_token = cpu_steps[i] if i < len(cpu_steps) else None
        gpu_token = gpu_steps[i] if i < len(gpu_steps) else None

        if cpu_token and gpu_token:
            step_num, cpu_id, cpu_text = cpu_token
            _, gpu_id, gpu_text = gpu_token

            if cpu_id == gpu_id:
                print(f"  Step {step_num}: ✅ CPU='{cpu_text}' == GPU='{gpu_text}'")
            else:
                print(f"  Step {step_num}: ❌ CPU='{cpu_text}' (id={cpu_id}) vs GPU='{gpu_text}' (id={gpu_id})")
                all_match = False
        elif cpu_token:
            print(f"  Step {i+1}: ❌ CPU 有 token '{cpu_token[2]}'，GPU 没有")
            all_match = False
        else:
            print(f"  Step {i+1}: ❌ GPU 有 token '{gpu_token[2]}'，CPU 没有")
            all_match = False

    print()
    if all_match and len(cpu_steps) == len(gpu_steps):
        print("=" * 70)
        print("✅ 所有生成的 token 完全匹配!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("❌ 发现差异! GPU 输出与 CPU 不一致")
        print("=" * 70)
        print()
        print("问题定位建议:")
        print("  1. 检查 Embedding 层输出 (第一层)")
        print("  2. 检查 QKV 投影计算")
        print("  3. 检查 Attention 计算")
        print("  4. 检查 FFN 输出")
        print("  5. 检查 Final Norm 和 LM Head")
        print()
        print("建议: 添加层对比测试，比较每一层的 CPU/GPU 输出")

if __name__ == "__main__":
    main()
