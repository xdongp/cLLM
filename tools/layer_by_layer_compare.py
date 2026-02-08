#!/usr/bin/env python3
"""
逐层对比 CPU 和 GPU 的输出
"""

import subprocess
import re
import sys

def run_test(device, input_text="hello"):
    """运行测试并解析输出"""
    cmd = [
        "/Users/dannypan/PycharmProjects/cLLM/build/bin/show_model_output",
        "--input", input_text,
        "--device", device,
        "--max_tokens", "1"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stderr  # 日志输出到 stderr

    # 解析调试输出
    data = {}

    # FFN Input
    match = re.search(rf'{device.upper()} DEBUG\] FFN Input - first 5=\[([^\]]+)\]', output)
    if match:
        data['ffn_input'] = [float(x) for x in match.group(1).split(', ')]

    # Post Attention RMS Norm (Raw) - GPU only
    match = re.search(r'GPU DEBUG\] Post Attention RMS Norm \(Raw\) - first 5=\[([^\]]+)\]', output)
    if match:
        data['post_norm_raw'] = [float(x) for x in match.group(1).split(', ')]

    # Post Attention RMS Norm (Weighted)
    match = re.search(rf'{device.upper()} DEBUG\] Post Attention RMS Norm(?: \(Weighted\))? - first 5=\[([^\]]+)\]', output)
    if match:
        data['post_norm_weighted'] = [float(x) for x in match.group(1).split(', ')]

    # FFN Gate
    match = re.search(rf'{device.upper()} DEBUG\] FFN Gate - first 5=\[([^\]]+)\]', output)
    if match:
        data['ffn_gate'] = [float(x) for x in match.group(1).split(', ')]

    # FFN Up
    match = re.search(rf'{device.upper()} DEBUG\] FFN Up - first 5=\[([^\]]+)\]', output)
    if match:
        data['ffn_up'] = [float(x) for x in match.group(1).split(', ')]

    # FFN Down
    match = re.search(rf'{device.upper()} DEBUG\] FFN Down - first 5=\[([^\]]+)\]', output)
    if match:
        data['ffn_down'] = [float(x) for x in match.group(1).split(', ')]

    return data

def compare_values(cpu_vals, gpu_vals, name):
    """对比两个值列表"""
    if not cpu_vals or not gpu_vals:
        return

    print(f"\n{name}:")
    print(f"  CPU: {cpu_vals}")
    print(f"  GPU: {gpu_vals}")

    if len(cpu_vals) != len(gpu_vals):
        print(f"  ⚠️  长度不一致: CPU={len(cpu_vals)}, GPU={len(gpu_vals)}")
        return

    diffs = [abs(c - g) for c, g in zip(cpu_vals, gpu_vals)]
    max_diff = max(diffs)
    mean_diff = sum(diffs) / len(diffs)

    print(f"  差异: max={max_diff:.6f}, mean={mean_diff:.6f}")

    if max_diff < 0.01:
        print(f"  ✅ 基本一致")
    elif max_diff < 0.1:
        print(f"  ⚠️  有差异")
    else:
        print(f"  ❌ 差异很大")

def main():
    print("=" * 60)
    print("CPU vs GPU 逐层对比测试")
    print("=" * 60)

    print("\n【CPU 测试】")
    cpu_data = run_test("cpu")

    print("\n【GPU 测试】")
    gpu_data = run_test("gpu")

    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    # 对比各层输出
    if 'ffn_input' in cpu_data and 'ffn_input' in gpu_data:
        compare_values(cpu_data['ffn_input'], gpu_data['ffn_input'], "FFN Input")

    if 'post_norm_weighted' in cpu_data and 'post_norm_weighted' in gpu_data:
        compare_values(cpu_data['post_norm_weighted'], gpu_data['post_norm_weighted'], "Post Attention RMS Norm (Weighted)")

    if 'ffn_gate' in cpu_data and 'ffn_gate' in gpu_data:
        compare_values(cpu_data['ffn_gate'], gpu_data['ffn_gate'], "FFN Gate")

    if 'ffn_up' in cpu_data and 'ffn_up' in gpu_data:
        compare_values(cpu_data['ffn_up'], gpu_data['ffn_up'], "FFN Up")

    if 'ffn_down' in cpu_data and 'ffn_down' in gpu_data:
        compare_values(cpu_data['ffn_down'], gpu_data['ffn_down'], "FFN Down")

    print("\n" + "=" * 60)
    print("分析:")
    print("=" * 60)

    # 检查 GPU 的 RMS Norm Raw 是否为零
    if 'post_norm_raw' in gpu_data:
        if all(v == 0.0 for v in gpu_data['post_norm_raw']):
            print("❌ GPU Post Attention RMS Norm (Raw) 全是零！")
            print("   这说明 ggml_rms_norm 的输出没有被正确计算")
        else:
            print("✅ GPU Post Attention RMS Norm (Raw) 正常")

if __name__ == "__main__":
    main()
