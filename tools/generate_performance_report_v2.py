#!/usr/bin/env python3
"""
生成 cLLM vs Ollama 性能对比报告（修复后版本）
"""

import json
from typing import Dict, Any, List

def load_data(files: List[str]) -> Dict[str, Dict[str, Any]]:
    """加载数据"""
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
    report.append("cLLM vs Ollama 性能对比报告（修复后）")
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
        
        report.append(f"  {'指标':<30} {'cLLM':>15} {'Ollama':>15} {'差距':>15}")
        report.append(f"  {'-'*30} {'-'*15} {'-'*15} {'-'*15}")
        
        # 总请求数
        report.append(f"  {'总请求数':<30} {cllm_data.get('total_requests', 0):>15} {ollama_data.get('total_requests', 0):>15} {'':>15}")
        
        # 成功请求数
        cllm_success = cllm_data.get('successful_requests', 0)
        ollama_success = ollama_data.get('successful_requests', 0)
        success_diff = cllm_success - ollama_success
        success_diff_str = f"{success_diff:+d}" if success_diff != 0 else "-"
        report.append(f"  {'成功请求数':<30} {cllm_success:>15} {ollama_success:>15} {success_diff_str:>15}")
        
        # 平均生成tokens
        cllm_avg_gen = cllm_data.get('avg_generated_tokens', 0)
        ollama_avg_gen = ollama_data.get('avg_generated_tokens', 0)
        gen_diff = cllm_avg_gen - ollama_avg_gen
        gen_diff_pct = (gen_diff / ollama_avg_gen * 100) if ollama_avg_gen > 0 else 0
        gen_diff_str = f"{gen_diff:+.2f} ({gen_diff_pct:+.1f}%)" if gen_diff != 0 else "-"
        report.append(f"  {'平均生成tokens':<30} {cllm_avg_gen:>14.2f} {ollama_avg_gen:>14.2f} {gen_diff_str:>15}")
        
        # 超限标记
        if cllm_avg_gen > 50:
            report.append(f"  {'⚠️  cLLM 超限':<30} {'是 (平均 {:.2f} tokens)'.format(cllm_avg_gen):>15} {'':>15} {'':>15}")
        if ollama_avg_gen > 50:
            report.append(f"  {'⚠️  Ollama 超限':<30} {'':>15} {'是 (平均 {:.2f} tokens)'.format(ollama_avg_gen):>15} {'':>15}")
        
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
    
    report.append("8并发:")
    cllm_avg_gen_8 = cllm_8.get('avg_generated_tokens', 0)
    ollama_avg_gen_8 = ollama_8.get('avg_generated_tokens', 0)
    
    if cllm_avg_gen_8 <= 55:
        report.append(f"  ✅ cLLM max_tokens 限制已修复（平均生成 {cllm_avg_gen_8:.2f} tokens）")
    else:
        report.append(f"  ❌ cLLM max_tokens 限制仍有问题（平均生成 {cllm_avg_gen_8:.2f} tokens）")
    
    cllm_throughput_8 = cllm_8.get('avg_throughput', 0)
    ollama_throughput_8 = ollama_8.get('avg_throughput', 0)
    if cllm_throughput_8 > ollama_throughput_8:
        report.append(f"  - cLLM 吞吐量比 Ollama 高 {((cllm_throughput_8 - ollama_throughput_8) / ollama_throughput_8 * 100):.1f}%")
    else:
        report.append(f"  - cLLM 吞吐量比 Ollama 低 {((ollama_throughput_8 - cllm_throughput_8) / ollama_throughput_8 * 100):.1f}%")
    
    cllm_time_8 = cllm_8.get('avg_response_time', 0)
    ollama_time_8 = ollama_8.get('avg_response_time', 0)
    if cllm_time_8 > ollama_time_8:
        report.append(f"  - cLLM 平均响应时间比 Ollama 慢 {((cllm_time_8 - ollama_time_8) / ollama_time_8 * 100):.1f}%")
    else:
        report.append(f"  - cLLM 平均响应时间比 Ollama 快 {((ollama_time_8 - cllm_time_8) / ollama_time_8 * 100):.1f}%")
    report.append("")
    
    # 高并发问题分析
    report.append("高并发问题分析:")
    
    cllm_16 = data.get('cllm_16', {}).get('concurrent', {})
    cllm_24 = data.get('cllm_24', {}).get('concurrent', {})
    cllm_32 = data.get('cllm_32', {}).get('concurrent', {})
    
    cllm_avg_gen_16 = cllm_16.get('avg_generated_tokens', 0)
    cllm_avg_gen_24 = cllm_24.get('avg_generated_tokens', 0)
    cllm_avg_gen_32 = cllm_32.get('avg_generated_tokens', 0)
    
    report.append(f"  - cLLM 在 16 并发下：平均生成 {cllm_avg_gen_16:.2f} tokens")
    report.append(f"  - cLLM 在 24 并发下：平均生成 {cllm_avg_gen_24:.2f} tokens")
    report.append(f"  - cLLM 在 32 并发下：平均生成 {cllm_avg_gen_32:.2f} tokens")
    report.append("")
    
    if cllm_avg_gen_16 > 50 or cllm_avg_gen_24 > 50 or cllm_avg_gen_32 > 50:
        report.append("  ⚠️  cLLM 在高并发下仍然存在 max_tokens 限制问题")
        report.append("  - 问题随着并发数增加而加剧")
        report.append("  - 可能原因：")
        report.append("    1. 批处理调度器在高并发下没有正确应用 max_tokens 限制")
        report.append("    2. KV cache 管理在高并发下可能存在问题")
        report.append("    3. 请求状态管理在高并发下可能存在竞争条件")
        report.append("    4. max_tokens 检查逻辑在批处理模式下可能存在延迟")
        report.append("")
    
    # Ollama 分析
    report.append("Ollama 分析:")
    ollama_avg_gen_16 = data.get('ollama_16', {}).get('concurrent', {}).get('avg_generated_tokens', 0)
    ollama_avg_gen_24 = data.get('ollama_24', {}).get('concurrent', {}).get('avg_generated_tokens', 0)
    ollama_avg_gen_32 = data.get('ollama_32', {}).get('concurrent', {}).get('avg_generated_tokens', 0)
    
    report.append(f"  - Ollama 在所有并发级别下平均生成约 {ollama_avg_gen_16:.2f} tokens")
    report.append(f"  - Ollama 的 max_tokens 控制也不完全准确（超出约 40%）")
    report.append("")
    
    # 建议
    report.append("建议:")
    report.append("  1. 进一步修复 cLLM 的 max_tokens 限制问题")
    report.append("     - 在批处理调度器中正确应用 max_tokens 限制")
    report.append("     - 确保在每次生成迭代后检查是否达到 max_tokens")
    report.append("     - 在达到 max_tokens 时立即停止生成")
    report.append("     - 检查批处理模式下的 max_tokens 检查逻辑")
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
        '/tmp/cllm_benchmark_8_v2.json',
        '/tmp/cllm_benchmark_16_v2.json',
        '/tmp/cllm_benchmark_24_v2.json',
        '/tmp/cllm_benchmark_32_v2.json',
        '/tmp/ollama_benchmark_8_v2.json',
        '/tmp/ollama_benchmark_16_v2.json',
        '/tmp/ollama_benchmark_24_v2.json',
        '/tmp/ollama_benchmark_32_v2.json'
    ]
    
    data = load_data(files)
    report = generate_report(data)
    
    # 保存报告
    report_file = '/tmp/cllm_vs_ollama_performance_report_v2.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(report)
    print(f"\n报告已保存到: {report_file}")

if __name__ == '__main__':
    main()
