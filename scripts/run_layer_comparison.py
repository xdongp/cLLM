#!/usr/bin/env python3
"""
逐层输出对比测试脚本
对比 CPU 和 GPU 在 Embedding、Attention 和 FFN 层的输出
"""

import subprocess
import re
import sys
from pathlib import Path

def run_test(device_type, test_input="你好"):
    """
    运行测试并捕获输出
    
    Args:
        device_type: "CPU" 或 "GPU"
        test_input: 测试输入文本
    
    Returns:
        包含所有调试输出的列表
    """
    print(f"\n{'='*60}")
    print(f"运行 {device_type} 测试，输入: '{test_input}'")
    print(f"{'='*60}")
    
    # 构建测试命令
    cmd = [
        "./bin/kylin_test_suite",
        "--stage", "13",
        "--device", device_type.lower(),
        "--input", test_input
    ]
    
    # 运行命令
    result = subprocess.run(
        cmd,
        cwd="/Users/dannypan/PycharmProjects/cLLM/build",
        capture_output=True,
        text=True
    )
    
    # 提取调试输出
    debug_lines = []
    for line in result.stdout.split('\n'):
        if '[LAYER_DEBUG]' in line:
            debug_lines.append(line)
    
    print(f"\n找到 {len(debug_lines)} 条调试输出")
    
    return debug_lines

def parse_debug_output(lines):
    """
    解析调试输出
    
    Args:
        lines: 调试输出行列表
    
    Returns:
        包含解析结果的字典
    """
    result = {
        'embedding': {},
        'attention': {},
        'ffn': {}
    }
    
    for line in lines:
        # 解析 Embedding 输出
        if 'Embedding:' in line:
            match = re.search(r'token_id=(\d+), min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)', line)
            if match:
                result['embedding'] = {
                    'token_id': int(match.group(1)),
                    'min': float(match.group(2)),
                    'max': float(match.group(3)),
                    'mean': float(match.group(4))
                }
        
        # 解析 Embedding 前 10 个值
        if 'Embedding first 10 values:' in line:
            match = re.search(r'\[([-\d., ]+)\]', line)
            if match:
                values = [float(x.strip()) for x in match.group(1).split(',')]
                result['embedding']['first_10'] = values
        
        # 解析 Attention 输出
        if 'QKV Projection:' in line:
            match = re.search(r'Layer (\d+) QKV Projection: Q\[min=([-\d.]+),max=([-\d.]+),mean=([-\d.]+)\], K\[min=([-\d.]+),max=([-\d.]+),mean=([-\d.]+)\], V\[min=([-\d.]+),max=([-\d.]+),mean=([-\d.]+)\]', line)
            if match:
                layer_idx = int(match.group(1))
                result['attention'][layer_idx] = {
                    'q': {'min': float(match.group(2)), 'max': float(match.group(3)), 'mean': float(match.group(4))},
                    'k': {'min': float(match.group(5)), 'max': float(match.group(6)), 'mean': float(match.group(7))},
                    'v': {'min': float(match.group(8)), 'max': float(match.group(9)), 'mean': float(match.group(10))}
                }
        
        # 解析 Attention 输出
        if 'Attention Output:' in line:
            match = re.search(r'Layer (\d+) Attention Output: min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)', line)
            if match:
                layer_idx = int(match.group(1))
                if layer_idx not in result['attention']:
                    result['attention'][layer_idx] = {}
                result['attention'][layer_idx]['output'] = {
                    'min': float(match.group(2)),
                    'max': float(match.group(3)),
                    'mean': float(match.group(4))
                }
        
        # 解析 FFN 输出
        if 'FFN Output:' in line:
            match = re.search(r'Layer (\d+) FFN Output: min=([-\d.]+), max=([-\d.]+), mean=([-\d.]+)', line)
            if match:
                layer_idx = int(match.group(1))
                result['ffn'][layer_idx] = {
                    'min': float(match.group(2)),
                    'max': float(match.group(3)),
                    'mean': float(match.group(4))
                }
    
    return result

def compare_outputs(cpu_data, gpu_data):
    """
    对比 CPU 和 GPU 的输出
    
    Args:
        cpu_data: CPU 输出数据
        gpu_data: GPU 输出数据
    
    Returns:
        对比结果
    """
    print("\n" + "="*60)
    print("输出对比分析")
    print("="*60)
    
    # 对比 Embedding 层
    print("\n📊 Embedding 层对比:")
    if cpu_data['embedding'] and gpu_data['embedding']:
        cpu_emb = cpu_data['embedding']
        gpu_emb = gpu_data['embedding']
        
        print(f"  CPU:  min={cpu_emb.get('min', 'N/A'):.6f}, max={cpu_emb.get('max', 'N/A'):.6f}, mean={cpu_emb.get('mean', 'N/A'):.6f}")
        print(f"  GPU:  min={gpu_emb.get('min', 'N/A'):.6f}, max={gpu_emb.get('max', 'N/A'):.6f}, mean={gpu_emb.get('mean', 'N/A'):.6f}")
        
        # 计算差异
        if 'min' in cpu_emb and 'min' in gpu_emb:
            min_diff = abs(cpu_emb['min'] - gpu_emb['min'])
            max_diff = abs(cpu_emb['max'] - gpu_emb['max'])
            mean_diff = abs(cpu_emb['mean'] - gpu_emb['mean'])
            
            print(f"  差异: min={min_diff:.6f}, max={max_diff:.6f}, mean={mean_diff:.6f}")
            
            # 检查是否匹配
            if min_diff < 0.001 and max_diff < 0.001 and mean_diff < 0.001:
                print("  ✅ Embedding 层输出匹配")
            else:
                print("  ⚠️  Embedding 层输出存在差异")
        
        # 对比前 10 个值
        if 'first_10' in cpu_emb and 'first_10' in gpu_emb:
            print(f"\n  前 10 个值对比:")
            cpu_vals = cpu_emb['first_10']
            gpu_vals = gpu_emb['first_10']
            max_val_diff = 0
            for i, (c, g) in enumerate(zip(cpu_vals, gpu_vals)):
                diff = abs(c - g)
                max_val_diff = max(max_val_diff, diff)
                status = "✅" if diff < 0.001 else "⚠️"
                print(f"    [{i}] CPU={c:.6f}, GPU={g:.6f}, diff={diff:.6f} {status}")
            print(f"  最大值差异: {max_val_diff:.6f}")
    
    # 对比 Attention 层
    print("\n📊 Attention 层对比:")
    for layer_idx in sorted(set(list(cpu_data['attention'].keys()) + list(gpu_data['attention'].keys()))):
        print(f"\n  Layer {layer_idx}:")
        if layer_idx in cpu_data['attention']:
            cpu_attn = cpu_data['attention'][layer_idx]
            if 'q' in cpu_attn:
                print(f"    CPU Q:  min={cpu_attn['q']['min']:.6f}, max={cpu_attn['q']['max']:.6f}, mean={cpu_attn['q']['mean']:.6f}")
                print(f"    CPU K:  min={cpu_attn['k']['min']:.6f}, max={cpu_attn['k']['max']:.6f}, mean={cpu_attn['k']['mean']:.6f}")
                print(f"    CPU V:  min={cpu_attn['v']['min']:.6f}, max={cpu_attn['v']['max']:.6f}, mean={cpu_attn['v']['mean']:.6f}")
            if 'output' in cpu_attn:
                print(f"    CPU Out: min={cpu_attn['output']['min']:.6f}, max={cpu_attn['output']['max']:.6f}, mean={cpu_attn['output']['mean']:.6f}")
        
        if layer_idx in gpu_data['attention']:
            gpu_attn = gpu_data['attention'][layer_idx]
            if 'q' in gpu_attn:
                print(f"    GPU Q:  min={gpu_attn['q']['min']:.6f}, max={gpu_attn['q']['max']:.6f}, mean={gpu_attn['q']['mean']:.6f}")
                print(f"    GPU K:  min={gpu_attn['k']['min']:.6f}, max={gpu_attn['k']['max']:.6f}, mean={gpu_attn['k']['mean']:.6f}")
                print(f"    GPU V:  min={gpu_attn['v']['min']:.6f}, max={gpu_attn['v']['max']:.6f}, mean={gpu_attn['v']['mean']:.6f}")
            if 'output' in gpu_attn:
                print(f"    GPU Out: min={gpu_attn['output']['min']:.6f}, max={gpu_attn['output']['max']:.6f}, mean={gpu_attn['output']['mean']:.6f}")
        
        # 计算差异
        if layer_idx in cpu_data['attention'] and layer_idx in gpu_data['attention']:
            cpu_attn = cpu_data['attention'][layer_idx]
            gpu_attn = gpu_data['attention'][layer_idx]
            
            if 'output' in cpu_attn and 'output' in gpu_attn:
                min_diff = abs(cpu_attn['output']['min'] - gpu_attn['output']['min'])
                max_diff = abs(cpu_attn['output']['max'] - gpu_attn['output']['max'])
                mean_diff = abs(cpu_attn['output']['mean'] - gpu_attn['output']['mean'])
                
                print(f"    差异: min={min_diff:.6f}, max={max_diff:.6f}, mean={mean_diff:.6f}")
                
                if min_diff < 0.001 and max_diff < 0.001 and mean_diff < 0.001:
                    print(f"    ✅ Layer {layer_idx} Attention 输出匹配")
                else:
                    print(f"    ⚠️  Layer {layer_idx} Attention 输出存在差异")
    
    # 对比 FFN 层
    print("\n📊 FFN 层对比:")
    for layer_idx in sorted(set(list(cpu_data['ffn'].keys()) + list(gpu_data['ffn'].keys()))):
        print(f"\n  Layer {layer_idx}:")
        if layer_idx in cpu_data['ffn']:
            cpu_ffn = cpu_data['ffn'][layer_idx]
            print(f"    CPU: min={cpu_ffn['min']:.6f}, max={cpu_ffn['max']:.6f}, mean={cpu_ffn['mean']:.6f}")
        
        if layer_idx in gpu_data['ffn']:
            gpu_ffn = gpu_data['ffn'][layer_idx]
            print(f"    GPU: min={gpu_ffn['min']:.6f}, max={gpu_ffn['max']:.6f}, mean={gpu_ffn['mean']:.6f}")
        
        # 计算差异
        if layer_idx in cpu_data['ffn'] and layer_idx in gpu_data['ffn']:
            cpu_ffn = cpu_data['ffn'][layer_idx]
            gpu_ffn = gpu_data['ffn'][layer_idx]
            
            min_diff = abs(cpu_ffn['min'] - gpu_ffn['min'])
            max_diff = abs(cpu_ffn['max'] - gpu_ffn['max'])
            mean_diff = abs(cpu_ffn['mean'] - gpu_ffn['mean'])
            
            print(f"    差异: min={min_diff:.6f}, max={max_diff:.6f}, mean={mean_diff:.6f}")
            
            if min_diff < 0.001 and max_diff < 0.001 and mean_diff < 0.001:
                print(f"    ✅ Layer {layer_idx} FFN 输出匹配")
            else:
                print(f"    ⚠️  Layer {layer_idx} FFN 输出存在差异")

def main():
    """主函数"""
    test_input = "你好"
    
    # 运行 CPU 测试
    cpu_lines = run_test("CPU", test_input)
    cpu_data = parse_debug_output(cpu_lines)
    
    # 运行 GPU 测试
    gpu_lines = run_test("GPU", test_input)
    gpu_data = parse_debug_output(gpu_lines)
    
    # 对比输出
    compare_outputs(cpu_data, gpu_data)
    
    print("\n" + "="*60)
    print("测试完成")
    print("="*60)

if __name__ == "__main__":
    main()
