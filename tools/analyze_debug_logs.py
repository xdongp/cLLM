#!/usr/bin/env python3
"""
分析并对比 Kylin 和 llama_cpp 的调试日志
提取 embedding 和 layer 0 输出的统计信息
"""

import re
import sys
from pathlib import Path

def parse_kylin_debug_log(log_file):
    """解析 Kylin 调试日志"""
    results = {
        'embedding': {},
        'layer0': {},
        'output': None
    }
    
    try:
        with open(log_file, 'r') as f:
            content = f.read()
            
        # 提取 embedding 统计信息
        emb_stats_match = re.search(
            r'\[Kylin Debug\] Embedding stats: min=([\d.-]+), max=([\d.-]+), mean=([\d.-]+), nan=(\d+), inf=(\d+), shape=\[(\d+),(\d+)\]',
            content
        )
        if emb_stats_match:
            results['embedding'] = {
                'min': float(emb_stats_match.group(1)),
                'max': float(emb_stats_match.group(2)),
                'mean': float(emb_stats_match.group(3)),
                'nan': int(emb_stats_match.group(4)),
                'inf': int(emb_stats_match.group(5)),
                'shape': (int(emb_stats_match.group(6)), int(emb_stats_match.group(7)))
            }
        
        # 提取 embedding 前10个值
        emb_values_match = re.search(
            r'\[Kylin Debug\] Embedding first \d+ values: (.+)',
            content
        )
        if emb_values_match:
            values_str = emb_values_match.group(1)
            results['embedding']['first_values'] = [float(x) for x in values_str.split()]
        
        # 提取 layer 0 统计信息
        layer0_stats_match = re.search(
            r'\[Kylin Debug\] Layer 0 output stats: min=([\d.-]+), max=([\d.-]+), mean=([\d.-]+), nan=(\d+), inf=(\d+), shape=\[(\d+),(\d+)\]',
            content
        )
        if layer0_stats_match:
            results['layer0'] = {
                'min': float(layer0_stats_match.group(1)),
                'max': float(layer0_stats_match.group(2)),
                'mean': float(layer0_stats_match.group(3)),
                'nan': int(layer0_stats_match.group(4)),
                'inf': int(layer0_stats_match.group(5)),
                'shape': (int(layer0_stats_match.group(6)), int(layer0_stats_match.group(7)))
            }
        
        # 提取 layer 0 前10个值
        layer0_values_match = re.search(
            r'\[Kylin Debug\] Layer 0 output first \d+ values: (.+)',
            content
        )
        if layer0_values_match:
            values_str = layer0_values_match.group(1)
            results['layer0']['first_values'] = [float(x) for x in values_str.split()]
        
    except Exception as e:
        print(f"Error parsing Kylin log: {e}", file=sys.stderr)
    
    return results

def print_comparison(kylin_results, llama_results=None):
    """打印对比结果"""
    print("=" * 70)
    print("Debug Output Comparison")
    print("=" * 70)
    print()
    
    # Embedding 对比
    print("📊 Embedding Statistics:")
    print("-" * 70)
    if kylin_results.get('embedding'):
        emb = kylin_results['embedding']
        print(f"Kylin:")
        print(f"  Shape:      {emb['shape']}")
        print(f"  Min:        {emb['min']:.6f}")
        print(f"  Max:        {emb['max']:.6f}")
        print(f"  Mean:       {emb['mean']:.6f}")
        print(f"  NaN count:  {emb['nan']}")
        print(f"  Inf count:  {emb['inf']}")
        if 'first_values' in emb:
            print(f"  First 10:   {', '.join(f'{v:.6f}' for v in emb['first_values'][:10])}")
    else:
        print("Kylin: No data found")
    
    print()
    print("llama_cpp: (Not available - using internal llama.cpp API)")
    print()
    
    # Layer 0 对比
    print("📊 Layer 0 Output Statistics:")
    print("-" * 70)
    if kylin_results.get('layer0'):
        layer0 = kylin_results['layer0']
        print(f"Kylin:")
        print(f"  Shape:      {layer0['shape']}")
        print(f"  Min:        {layer0['min']:.6f}")
        print(f"  Max:        {layer0['max']:.6f}")
        print(f"  Mean:       {layer0['mean']:.6f}")
        print(f"  NaN count:  {layer0['nan']}")
        print(f"  Inf count:  {layer0['inf']}")
        if 'first_values' in layer0:
            print(f"  First 10:   {', '.join(f'{v:.6f}' for v in layer0['first_values'][:10])}")
    else:
        print("Kylin: No data found")
    
    print()
    print("llama_cpp: (Not available - using internal llama.cpp API)")
    print()
    
    # 分析
    print("🔍 Analysis:")
    print("-" * 70)
    if kylin_results.get('embedding') and kylin_results.get('layer0'):
        emb = kylin_results['embedding']
        layer0 = kylin_results['layer0']
        
        # 检查是否有异常值
        issues = []
        if emb['nan'] > 0 or emb['inf'] > 0:
            issues.append(f"Embedding has {emb['nan']} NaN and {emb['inf']} Inf values")
        if layer0['nan'] > 0 or layer0['inf'] > 0:
            issues.append(f"Layer 0 has {layer0['nan']} NaN and {layer0['inf']} Inf values")
        
        # 检查数值范围
        if abs(emb['max']) > 10 or abs(emb['min']) > 10:
            issues.append(f"Embedding values out of normal range: [{emb['min']:.3f}, {emb['max']:.3f}]")
        if abs(layer0['max']) > 10 or abs(layer0['min']) > 10:
            issues.append(f"Layer 0 values out of normal range: [{layer0['min']:.3f}, {layer0['max']:.3f}]")
        
        if issues:
            print("⚠️  Potential issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✅ No obvious issues detected in statistics")
        
        # 数值分布分析
        print()
        print("📈 Value Distribution:")
        emb_range = emb['max'] - emb['min']
        layer0_range = layer0['max'] - layer0['min']
        print(f"  Embedding range: {emb_range:.6f} (mean={emb['mean']:.6f})")
        print(f"  Layer 0 range:   {layer0_range:.6f} (mean={layer0['mean']:.6f})")
        
        if abs(emb['mean']) > 0.1:
            print(f"  ⚠️  Embedding mean is not close to zero: {emb['mean']:.6f}")
        if abs(layer0['mean']) > 0.1:
            print(f"  ⚠️  Layer 0 mean is not close to zero: {layer0['mean']:.6f}")
    
    print()
    print("=" * 70)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_debug_logs.py <kylin_log_file> [llama_log_file]")
        print("Example: python3 analyze_debug_logs.py /tmp/kylin_debug.log")
        sys.exit(1)
    
    kylin_log = sys.argv[1]
    llama_log = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not Path(kylin_log).exists():
        print(f"Error: Kylin log file not found: {kylin_log}", file=sys.stderr)
        sys.exit(1)
    
    kylin_results = parse_kylin_debug_log(kylin_log)
    llama_results = None
    
    if llama_log and Path(llama_log).exists():
        # 未来可以解析 llama_cpp 日志（如果有调试信息）
        pass
    
    print_comparison(kylin_results, llama_results)

if __name__ == "__main__":
    main()
