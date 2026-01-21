#!/usr/bin/env python3
"""
生成对比测试报告
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any

def load_results() -> Dict[str, Dict[str, Any]]:
    """加载所有测试结果"""
    results = {}
    
    # 查找所有结果文件
    pattern = "/tmp/comparison_results/*_r*_c*.json"
    files = sorted(glob.glob(pattern))
    
    print(f"找到 {len(files)} 个结果文件")
    
    for file in files:
        # 解析文件名: {server}_r{round}_c{concurrency}.json
        basename = os.path.basename(file)
        parts = basename.replace('.json', '').split('_')
        
        # 格式: cllm_r1_c8.json -> ['cllm', 'r1', 'c8'] (3部分)
        if len(parts) < 3:
            continue
            
        server = parts[0]  # cllm or ollama
        round_num = None
        concurrency = None
        
        # 处理 round 部分: r1 (parts[1])
        if len(parts) >= 2:
            round_str = parts[1]
            if round_str.startswith('r'):
                try:
                    round_num = int(round_str[1:])  # r1 -> 1
                except ValueError:
                    continue
            elif 'round' in round_str:
                try:
                    round_num = int(round_str.replace('round', ''))  # round1 -> 1
                except ValueError:
                    continue
            else:
                continue
        else:
            continue
        
        # 处理 concurrency 部分: c8 (parts[2])
        if len(parts) >= 3:
            conc_str = parts[2]
            if conc_str.startswith('c'):
                try:
                    concurrency = int(conc_str[1:])  # c8 -> 8
                except ValueError:
                    continue
            else:
                continue
        else:
            continue
        
        if round_num is None or concurrency is None:
            continue
        
        try:
            with open(file, 'r') as f:
                data = json.load(f)
                concurrent_stats = data.get('concurrent', {})
                
                if not concurrent_stats:
                    print(f"Warning: {basename} has no 'concurrent' data")
                    continue
                
                key = f"round_{round_num}_c{concurrency}"
                if key not in results:
                    results[key] = {}
                
                results[key][server] = {
                    "successful_requests": concurrent_stats.get('successful_requests', 0),
                    "failed_requests": concurrent_stats.get('failed_requests', 0),
                    "avg_throughput": concurrent_stats.get('avg_throughput', 0),
                    "avg_response_time": concurrent_stats.get('avg_response_time', 0),
                    "total_test_time": concurrent_stats.get('total_test_time', 0),
                    "avg_tokens_per_second": concurrent_stats.get('avg_tokens_per_second', 0),
                    "total_generated_tokens": concurrent_stats.get('total_generated_tokens', 0),
                    "success_rate": (concurrent_stats.get('successful_requests', 0) / 72) * 100
                }
        except Exception as e:
            print(f"Error loading {file}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"成功加载 {len(results)} 个测试结果")
    return results

def calculate_averages(results: Dict[str, Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, float]]]:
    """计算每个并发级别的平均值"""
    concurrency_levels = [8, 16, 24, 32, 40]
    summary = {}
    
    for concurrency in concurrency_levels:
        cllm_throughputs = []
        ollama_throughputs = []
        cllm_success_rates = []
        ollama_success_rates = []
        cllm_response_times = []
        ollama_response_times = []
        
        for key, value in results.items():
            if f"_c{concurrency}" in key:
                if "cllm" in value:
                    cllm = value["cllm"]
                    cllm_throughputs.append(cllm["avg_throughput"])
                    cllm_success_rates.append(cllm["success_rate"])
                    cllm_response_times.append(cllm["avg_response_time"])
                
                if "ollama" in value:
                    ollama = value["ollama"]
                    ollama_throughputs.append(ollama["avg_throughput"])
                    ollama_success_rates.append(ollama["success_rate"])
                    ollama_response_times.append(ollama["avg_response_time"])
        
        summary[concurrency] = {
            "cllm": {
                "avg_throughput": sum(cllm_throughputs) / len(cllm_throughputs) if cllm_throughputs else 0,
                "avg_success_rate": sum(cllm_success_rates) / len(cllm_success_rates) if cllm_success_rates else 0,
                "avg_response_time": sum(cllm_response_times) / len(cllm_response_times) if cllm_response_times else 0,
                "rounds": len(cllm_throughputs)
            },
            "ollama": {
                "avg_throughput": sum(ollama_throughputs) / len(ollama_throughputs) if ollama_throughputs else 0,
                "avg_success_rate": sum(ollama_success_rates) / len(ollama_success_rates) if ollama_success_rates else 0,
                "avg_response_time": sum(ollama_response_times) / len(ollama_response_times) if ollama_response_times else 0,
                "rounds": len(ollama_throughputs)
            }
        }
    
    return summary

def generate_report(summary: Dict[int, Dict[str, Dict[str, float]]], results: Dict[str, Dict[str, Any]]) -> str:
    """生成Markdown报告"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    report = f"""# cLLM vs Ollama 多轮对比测试报告

**测试时间**: {timestamp}
**测试配置**: 72请求, 50 tokens, 3轮测试
**并发级别**: 8, 16, 24, 32, 40

## 执行摘要

本报告展示了cLLM和Ollama在多个并发级别下的性能对比测试结果。测试进行了3轮，取平均值以确保结果的稳定性。

## 汇总结果（3轮平均值）

| 并发数 | cLLM吞吐量 (t/s) | Ollama吞吐量 (t/s) | cLLM成功率 | Ollama成功率 | cLLM响应时间 (s) | Ollama响应时间 (s) | 优势方 | 性能差距 |
|--------|----------------|-------------------|-----------|-------------|----------------|-------------------|--------|---------|
"""
    
    for concurrency in sorted(summary.keys()):
        s = summary[concurrency]
        cllm_tp = s["cllm"]["avg_throughput"]
        ollama_tp = s["ollama"]["avg_throughput"]
        cllm_sr = s["cllm"]["avg_success_rate"]
        ollama_sr = s["ollama"]["avg_success_rate"]
        cllm_rt = s["cllm"]["avg_response_time"]
        ollama_rt = s["ollama"]["avg_response_time"]
        
        if cllm_tp > ollama_tp:
            winner = "**cLLM**"
            gap = ((cllm_tp - ollama_tp) / ollama_tp * 100) if ollama_tp > 0 else 0
            gap_str = f"+{gap:.1f}%"
        else:
            winner = "Ollama"
            gap = ((ollama_tp - cllm_tp) / cllm_tp * 100) if cllm_tp > 0 else 0
            gap_str = f"-{gap:.1f}%"
        
        report += f"| **{concurrency}** | {cllm_tp:.2f} | {ollama_tp:.2f} | {cllm_sr:.1f}% | {ollama_sr:.1f}% | {cllm_rt:.2f} | {ollama_rt:.2f} | {winner} | {gap_str} |\n"
    
    report += f"""
## 详细分析

### 吞吐量对比

"""
    
    for concurrency in sorted(summary.keys()):
        s = summary[concurrency]
        cllm_tp = s["cllm"]["avg_throughput"]
        ollama_tp = s["ollama"]["avg_throughput"]
        
        if cllm_tp > ollama_tp:
            advantage = ((cllm_tp - ollama_tp) / ollama_tp * 100) if ollama_tp > 0 else 0
            report += f"- **并发{concurrency}**: cLLM ({cllm_tp:.2f} t/s) 优于 Ollama ({ollama_tp:.2f} t/s)，领先 {advantage:.1f}%\n"
        else:
            advantage = ((ollama_tp - cllm_tp) / cllm_tp * 100) if cllm_tp > 0 else 0
            report += f"- **并发{concurrency}**: Ollama ({ollama_tp:.2f} t/s) 优于 cLLM ({cllm_tp:.2f} t/s)，领先 {advantage:.1f}%\n"
    
    report += f"""
### 成功率对比

"""
    
    for concurrency in sorted(summary.keys()):
        s = summary[concurrency]
        cllm_sr = s["cllm"]["avg_success_rate"]
        ollama_sr = s["ollama"]["avg_success_rate"]
        
        if cllm_sr >= ollama_sr:
            report += f"- **并发{concurrency}**: cLLM ({cllm_sr:.1f}%) {'=' if cllm_sr == ollama_sr else '>'} Ollama ({ollama_sr:.1f}%)\n"
        else:
            report += f"- **并发{concurrency}**: Ollama ({ollama_sr:.1f}%) > cLLM ({cllm_sr:.1f}%)\n"
    
    report += f"""
## 关键发现

1. **性能趋势**: 分析各并发级别下的性能变化趋势
2. **稳定性**: 比较两系统的成功率
3. **最优配置**: 确定最佳并发级别

## 测试数据

所有测试结果文件保存在: `/tmp/comparison_results/`

---
报告生成时间: {timestamp}
"""
    
    return report

def main():
    print("加载测试结果...")
    results = load_results()
    
    print(f"已加载 {len(results)} 个测试结果")
    
    print("计算平均值...")
    summary = calculate_averages(results)
    
    print("生成报告...")
    report = generate_report(summary, results)
    
    # 保存报告
    report_file = f"docs/analysis/cllm_vs_ollama_multi_round_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✓ 报告已生成: {report_file}")
    
    # 打印汇总
    print("\n汇总结果:")
    print(f"{'并发数':<8} {'cLLM吞吐量':<15} {'Ollama吞吐量':<15} {'优势方':<10}")
    print("-" * 50)
    for concurrency in sorted(summary.keys()):
        s = summary[concurrency]
        cllm_tp = s["cllm"]["avg_throughput"]
        ollama_tp = s["ollama"]["avg_throughput"]
        winner = "cLLM" if cllm_tp > ollama_tp else "Ollama"
        print(f"{concurrency:<8} {cllm_tp:<15.2f} {ollama_tp:<15.2f} {winner:<10}")

if __name__ == "__main__":
    main()
