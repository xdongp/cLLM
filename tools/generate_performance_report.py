#!/usr/bin/env python3
"""
生成 cLLM vs Ollama 性能对比报告
"""

import json
from typing import Dict, Any, List

def load_corrected_data(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """加载修正后的数据"""
    data = {}
    for file in files:
        with open(file, 'r', encoding='utf-8') as f:
            result = json.load(f)
            server_type = result.get('server_type')
            concurrency = result.get('concurrency')
            key = f"{server_type}_{concurrency}"
            data[key] = result
    return data

def generate_report(data: Dict[str, Dict[str, Any]]) -> str:
    """生成性能对比报告"""
    report = []
    report.append("=" * 80)
    report.append("cLLM vs Ollama 性能对比报告")
    report.append("=" * 80)
    report.append("")
    report.append("测试环境:")
    report.append("  - 模型: Qwen3-0.6B (q4_k_m)")
    report.append("  - 请求类型: 并发测试")
    report.append("  - 每次请求数: 72")
    report.append("  - 最大生成tokens: 50")
    report.append("  - 数据修正: 排除超过50 tokens的请求")
    report.append("")
    report.append("=" * 80)
    report.append("")
    
    # 按并发级别分组
    concurrencies = [8, 16, 24, 32]
    
    for concurrency in concurrencies:
        report.append(f"并发级别: {concurrency}")
        report.append("-" * 80)
        
        cllm_key = f"cllm_{concurrency}"
        ollama_key = f"ollama_{concurrency}"
        
        cllm_data = data.get(cllm_key, {}).get('concurrent', {})
        ollama_data = data.get(ollama_key, {}).get('concurrent', {})
        
        # 检查数据是否有效
        if cllm_data.get('successful_requests', 0) <= 0:
            report.append(f"  cLLM: 数据无效（有效请求数 <= 0）")
            report.append(f"    说明: 在{concurrency}并发下，几乎所有请求都超过了50 tokens限制")
            report.append("")
            continue
        
        report.append(f"  {'指标':<30} {'cLLM':>15} {'Ollama':>15} {'差距':>15}")
        report.append(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
        
        # 总请求数
        report.append(f"  {'总请求数':<30} {cllm_data.get('total_requests', 0):>15} {ollama_data.get('total_requests', 0):>15} {'':>15}")
        
        # 有效请求数
        cllm_valid = cllm_data.get('successful_requests', 0)
        ollama_valid = ollama_data.get('successful_requests', 0)
        valid_diff = cllm_valid - ollama_valid
        valid_diff_str = f"{valid_diff:+d}" if valid_diff != 0 else "-"
        report.append(f"  {'有效请求数':<30} {cllm_valid:>15} {ollama_valid:>15} {valid_diff_str:>15}")
        
        # 超限请求数
        cllm_failed = cllm_data.get('failed_requests', 0)
        ollama_failed = ollama_data.get('failed_requests', 0)
        failed_diff = cllm_failed - ollama_failed
        failed_diff_str = f"{failed_diff:+d}" if failed_diff != 0 else "-"
        report.append(f"  {'超限请求数':<30} {cllm_failed:>15} {ollama_failed:>15} {failed_diff_str:>15}")
        
        # 平均响应时间
        cllm_avg_time = cllm_data.get('avg_response_time', 0)
        ollama_avg_time = ollama_data.get('avg_response_time', 0)
        time_diff = cllm_avg_time - ollama_avg_time
        time_diff_pct = (time_diff / ollama_avg_time * 100) if ollama_avg_time > 0 else 0
        time_diff_str = f"{time_diff:+.2f}s ({time_diff_pct:+.1f}%)" if time_diff != 0 else "-"
        report.append(f"  {'平均响应时间':<30} {cllm_avg_time:>14.2f}s {ollama_avg_time:>14.2f}s {time_diff_str:>15}")
        
        # 最小响应时间
        cllm_min_time = cllm_data.get('min_response_time', 0)
        ollama_min_time = ollama_data.get('min_response_time', 0)
        min_diff = cllm_min_time - ollama_min_time
        min_diff_str = f"{min_diff:+.2f}s" if min_diff != 0 else "-"
        report.append(f"  {'最小响应时间':<30} {cllm_min_time:>14.2f}s {ollama_min_time:>14.2f}s {min_diff_str:>15}")
        
        # 最大响应时间
        cllm_max_time = cllm_data.get('max_response_time', 0)
        ollama_max_time = ollama_data.get('max_response_time', 0)
        max_diff = cllm_max_time - ollama_max_time
        max_diff_str = f"{max_diff:+.2f}s" if max_diff != 0 else "-"
        report.append(f"  {'最大响应时间':<30} {cllm_max_time:>14.2f}s {ollama_max_time:>14.2f}s {max_diff_str:>15}")
        
        # 平均吞吐量
        cllm_throughput = cllm_data.get('avg_throughput', 0)
        ollama_throughput = ollama_data.get('avg_throughput', 0)
        throughput_diff = cllm_throughput - ollama_throughput
        throughput_diff_pct = (throughput_diff / ollama_throughput * 100) if ollama_throughput > 0 else 0
        throughput_diff_str = f"{throughput_diff:+.2f} ({throughput_diff_pct:+.1f}%)" if throughput_diff != 0 else "-"
        report.append(f"  {'平均吞吐量':<30} {cllm_throughput:>14.2f}t/s {ollama_throughput:>14.2f}t/s {throughput_diff_str:>15}")
        
        # 平均 tokens per second
        cllm_tps = cllm_data.get('avg_tokens_per_second', 0)
        ollama_tps = ollama_data.get('avg_tokens_per_second', 0)
        tps_diff = cllm_tps - ollama_tps
        tps_diff_pct = (tps_diff / ollama_tps * 100) if ollama_tps > 0 else 0
        tps_diff_str = f"{tps_diff:+.2f} ({tps_diff_pct:+.1f}%)" if tps_diff != 0 else "-"
        report.append(f"  {'平均tokens/sec':<30} {cllm_tps:>14.2f}t/s {ollama_tps:>14.2f}t/s {tps_diff_str:>15}")
        
        # 总生成tokens
        cllm_total_gen = cllm_data.get('total_generated_tokens', 0)
        ollama_total_gen = ollama_data.get('total_generated_tokens', 0)
        total_gen_diff = cllm_total_gen - ollama_total_gen
        total_gen_diff_pct = (total_gen_diff / ollama_total_gen * 100) if ollama_total_gen > 0 else 0
        total_gen_diff_str = f"{total_gen_diff:+d} ({total_gen_diff_pct:+.1f}%)" if total_gen_diff != 0 else "-"
        report.append(f"  {'总生成tokens':<30} {cllm_total_gen:>15} {ollama_total_gen:>15} {total_gen_diff_str:>15}")
        
        # 总测试时间
        cllm_total_time = cllm_data.get('total_test_time', 0)
        ollama_total_time = ollama_data.get('total_test_time', 0)
        total_time_diff = cllm_total_time - ollama_total_time
        total_time_diff_pct = (total_time_diff / ollama_total_time * 100) if ollama_total_time > 0 else 0
        total_time_diff_str = f"{total_time_diff:+.2f}s ({total_time_diff_pct:+.1f}%)" if total_time_diff != 0 else "-"
        report.append(f"  {'总测试时间':<30} {cllm_total_time:>14.2f}s {ollama_total_time:>14.2f}s {total_time_diff_str:>15}")
        
        report.append("")
    
    # 总结
    report.append("=" * 80)
    report.append("总结")
    report.append("=" * 80)
    report.append("")
    
    # 8并发总结
    cllm_8 = data.get('cllm_8', {}).get('concurrent', {})
    ollama_8 = data.get('ollama_8', {}).get('concurrent', {})
    
    if cllm_8.get('successful_requests', 0) > 0:
        report.append("8并发:")
        cllm_throughput_8 = cllm_8.get('avg_throughput', 0)
        ollama_throughput_8 = ollama_8.get('avg_throughput', 0)
        if cllm_throughput_8 > ollama_throughput_8:
            report.append(f"  - cLLM 吞吐量比 Ollama 低 {((ollama_throughput_8 - cllm_throughput_8) / ollama_throughput_8 * 100):.1f}%")
        else:
            report.append(f"  - cLLM 吞吐量比 Ollama 高 {((cllm_throughput_8 - ollama_throughput_8) / ollama_throughput_8 * 100):.1f}%")
        
        cllm_time_8 = cllm_8.get('avg_response_time', 0)
        ollama_time_8 = ollama_8.get('avg_response_time', 0)
        if cllm_time_8 > ollama_time_8:
            report.append(f"  - cLLM 平均响应时间比 Ollama 慢 {((cllm_time_8 - ollama_time_8) / ollama_time_8 * 100):.1f}%")
        else:
            report.append(f"  - cLLM 平均响应时间比 Ollama 快 {((ollama_time_8 - cllm_time_8) / ollama_time_8 * 100):.1f}%")
        report.append("")
    
    # 高并发问题分析
    report.append("高并发问题分析:")
    report.append("  - cLLM 在16/24/32并发下，几乎所有请求都超过了50 tokens限制")
    report.append("  - 这表明 cLLM 在高并发下对 max_tokens 参数的控制存在问题")
    report.append("  - 可能原因：")
    report.append("    1. 批处理调度器在高并发下没有正确应用 max_tokens 限制")
    report.append("    2. KV cache 管理在高并发下可能存在问题")
    report.append("    3. 请求状态管理在高并发下可能存在竞争条件")
    report.append("  - Ollama 在所有并发级别下都有约30个请求超过了50 tokens限制")
    report.append("  - 这表明 Ollama 的 max_tokens 控制也不完全准确")
    report.append("")
    
    # 建议
    report.append("建议:")
    report.append("  1. 修复 cLLM 的 max_tokens 限制问题")
    report.append("     - 在批处理调度器中正确应用 max_tokens 限制")
    report.append("     - 确保在每次生成迭代后检查是否达到 max_tokens")
    report.append("     - 在达到 max_tokens 时立即停止生成")
    report.append("  2. 优化高并发性能")
    report.append("     - 改进批处理调度算法")
    report.append("     - 优化 KV cache 管理")
    report.append("     - 减少锁竞争")
    report.append("  3. 添加更详细的日志")
    report.append("     - 记录每个请求的 max_tokens 设置")
    report.append("     - 记录实际生成的 tokens 数")
    report.append("     - 记录停止生成的原因")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)

def main():
    files = [
        '/tmp/cllm_benchmark_8_corrected.json',
        '/tmp/cllm_benchmark_16_corrected.json',
        '/tmp/cllm_benchmark_24_corrected.json',
        '/tmp/cllm_benchmark_32_corrected.json',
        '/tmp/ollama_benchmark_8_corrected.json',
        '/tmp/ollama_benchmark_16_corrected.json',
        '/tmp/ollama_benchmark_24_corrected.json',
        '/tmp/ollama_benchmark_32_corrected.json'
    ]
    
    data = load_corrected_data(files)
    report = generate_report(data)
    
    # 保存报告
    report_file = '/tmp/cllm_vs_ollama_performance_report.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n报告已保存到: {report_file}")

if __name__ == '__main__':
    main()
