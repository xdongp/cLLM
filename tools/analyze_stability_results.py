#!/usr/bin/env python3
"""
分析稳定性测试结果，识别响应时间方差和最大响应时间的根本原因
"""

import json
import statistics
from datetime import datetime
from typing import List, Dict


def load_results(file_path: str) -> Dict:
    """加载测试结果"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def analyze_response_time_distribution(response_times: List[float]) -> Dict:
    """分析响应时间分布"""
    if not response_times:
        return {}
    
    sorted_times = sorted(response_times)
    mean = statistics.mean(response_times)
    std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
    
    # 识别异常值（超过3个标准差）
    outliers = [rt for rt in response_times if rt > mean + 3 * std_dev]
    
    # 分析时间间隔模式
    intervals = []
    for i in range(1, len(sorted_times)):
        intervals.append(sorted_times[i] - sorted_times[i-1])
    
    # 识别批次模式（响应时间集中的区间）
    batch_patterns = {}
    for rt in response_times:
        key = round(rt, 1)  # 四舍五入到小数点后1位
        batch_patterns[key] = batch_patterns.get(key, 0) + 1
    
    return {
        'mean': mean,
        'std_dev': std_dev,
        'variance': std_dev ** 2,
        'cv': std_dev / mean if mean > 0 else 0,
        'outlier_count': len(outliers),
        'outlier_percentage': (len(outliers) / len(response_times)) * 100,
        'outliers': outliers[:10],  # 只保留前10个
        'interval_analysis': {
            'min_interval': min(intervals) if intervals else 0,
            'max_interval': max(intervals) if intervals else 0,
            'mean_interval': statistics.mean(intervals) if intervals else 0,
            'std_interval': statistics.stdev(intervals) if len(intervals) > 1 else 0
        },
        'batch_patterns': dict(sorted(batch_patterns.items(), key=lambda x: x[1], reverse=True)[:10])
    }


def analyze_max_response_time_causes(test_results: Dict) -> Dict:
    """分析最大响应时间的根本原因"""
    raw_data = test_results.get('raw_data', {})
    response_times = raw_data.get('response_times', [])
    timestamps = raw_data.get('timestamps', [])
    
    if not response_times:
        return {}
    
    max_rt = max(response_times)
    max_indices = [i for i, rt in enumerate(response_times) if rt == max_rt]
    
    # 分析最大响应时间发生的时间点
    max_timestamps = [timestamps[i] for i in max_indices if i < len(timestamps)]
    
    # 分析前后的响应时间
    context_analysis = []
    for idx in max_indices:
        before = response_times[max(0, idx-5):idx]
        after = response_times[idx+1:min(len(response_times), idx+6)]
        
        context_analysis.append({
            'position': idx,
            'max_rt': response_times[idx],
            'before_mean': statistics.mean(before) if before else 0,
            'after_mean': statistics.mean(after) if after else 0,
            'before_count': len(before),
            'after_count': len(after)
        })
    
    # 识别可能的原因
    possible_causes = []
    
    # 原因1: 系统初始化/预热
    if max_indices and max_indices[0] < 10:
        possible_causes.append({
            'cause': '系统初始化/预热',
            'evidence': f"最大响应时间发生在第{max_indices[0]+1}个请求（前10个）",
            'confidence': '高'
        })
    
    # 原因2: 批处理重组
    if len(max_indices) > 1:
        interval_between = max_indices[1] - max_indices[0]
        if interval_between < 20:
            possible_causes.append({
                'cause': '批处理重组开销',
                'evidence': f"多个最大响应时间集中出现，间隔{interval_between}个请求",
                'confidence': '高'
            })
    
    # 原因3: 资源竞争
    if max_rt > statistics.mean(response_times) * 2:
        possible_causes.append({
            'cause': '资源竞争/锁竞争',
            'evidence': f"最大响应时间({max_rt:.2f}s)是平均值({statistics.mean(response_times):.2f}s)的{max_rt/statistics.mean(response_times):.1f}倍",
            'confidence': '中'
        })
    
    # 原因4: 序列ID池耗尽
    if test_results.get('test_name', '').startswith('高并发'):
        possible_causes.append({
            'cause': '序列ID池可能耗尽',
            'evidence': f"在高并发场景下({test_results.get('total_requests', 0)}个请求)，序列ID池可能成为瓶颈",
            'confidence': '中'
        })
    
    # 原因5: GPU/CPU资源限制
    if max_rt > 10:
        possible_causes.append({
            'cause': 'GPU/CPU资源限制',
            'evidence': f"响应时间超过10秒，可能是硬件资源瓶颈",
            'confidence': '中'
        })
    
    return {
        'max_response_time': max_rt,
        'occurrences': len(max_indices),
        'timestamps': max_timestamps,
        'context_analysis': context_analysis[:5],  # 只保留前5个
        'possible_causes': possible_causes
    }


def generate_analysis_report(results: Dict) -> str:
    """生成分析报告"""
    report = ["="*80]
    report.append("CLLM系统稳定性测试 - 根本原因分析报告")
    report.append("="*80)
    report.append(f"\n测试时间: {results.get('test_time', 'N/A')}")
    report.append(f"测试数量: {results.get('summary', {}).get('total_tests', 0)}")
    report.append(f"平均稳定性分数: {results.get('summary', {}).get('overall_stability_score', 0) * 100:.2f}%")
    report.append(f"平均方差: {results.get('summary', {}).get('overall_variance', 0):.2f}")
    report.append(f"最大响应时间: {results.get('summary', {}).get('overall_max_rt', 0):.2f}秒")
    
    report.append("\n" + "="*80)
    report.append("各测试详细分析")
    report.append("="*80)
    
    for test in results.get('tests', []):
        report.append(f"\n{'─'*80}")
        report.append(f"测试名称: {test.get('test_name', 'N/A')}")
        report.append(f"{'─'*80}")
        
        # 基本统计
        rt_stats = test.get('response_time_stats', {})
        report.append(f"\n📊 响应时间统计:")
        report.append(f"  平均值: {rt_stats.get('mean', 0):.2f}秒")
        report.append(f"  中位数: {rt_stats.get('median', 0):.2f}秒")
        report.append(f"  最大值: {rt_stats.get('max', 0):.2f}秒")
        report.append(f"  标准差: {rt_stats.get('std_dev', 0):.2f}秒")
        report.append(f"  方差: {rt_stats.get('variance', 0):.2f}")
        report.append(f"  变异系数(CV): {rt_stats.get('cv', 0) * 100:.2f}%")
        report.append(f"  稳定性分数: {rt_stats.get('stability_score', 0) * 100:.2f}%")
        
        # 百分位数
        percentiles = test.get('percentiles', {})
        report.append(f"\n📈 百分位数分析:")
        report.append(f"  P50: {percentiles.get('p50', 0):.2f}秒")
        report.append(f"  P90: {percentiles.get('p90', 0):.2f}秒")
        report.append(f"  P95: {percentiles.get('p95', 0):.2f}秒")
        report.append(f"  P99: {percentiles.get('p99', 0):.2f}秒")
        
        # 响应时间分布分析
        raw_data = test.get('raw_data', {})
        response_times = raw_data.get('response_times', [])
        if response_times:
            dist_analysis = analyze_response_time_distribution(response_times)
            report.append(f"\n📊 分布分析:")
            report.append(f"  异常值数量: {dist_analysis.get('outlier_count', 0)}个 ({dist_analysis.get('outlier_percentage', 0):.2f}%)")
            report.append(f"  时间间隔分析:")
            report.append(f"    最小间隔: {dist_analysis.get('interval_analysis', {}).get('min_interval', 0):.3f}秒")
            report.append(f"    最大间隔: {dist_analysis.get('interval_analysis', {}).get('max_interval', 0):.3f}秒")
            report.append(f"    平均间隔: {dist_analysis.get('interval_analysis', {}).get('mean_interval', 0):.3f}秒")
            report.append(f"    间隔标准差: {dist_analysis.get('interval_analysis', {}).get('std_interval', 0):.3f}秒")
            
            # 批次模式
            batch_patterns = dist_analysis.get('batch_patterns', {})
            if batch_patterns:
                report.append(f"\n📦 响应时间集中模式:")
                for rt, count in list(batch_patterns.items())[:5]:
                    report.append(f"    {rt:.1f}秒: {count}次")
        
        # 最大响应时间原因分析
        max_analysis = analyze_max_response_time_causes(test)
        report.append(f"\n🔍 最大响应时间根本原因分析:")
        report.append(f"  最大响应时间: {max_analysis.get('max_response_time', 0):.2f}秒")
        report.append(f"  出现次数: {max_analysis.get('occurrences', 0)}次")
        
        # 上下文分析
        context = max_analysis.get('context_analysis', [])
        if context:
            report.append(f"\n  上下文分析:")
            for ctx in context[:3]:
                report.append(f"    位置#{ctx['position']}: {ctx['max_rt']:.2f}秒")
                report.append(f"      前5个平均: {ctx['before_mean']:.2f}秒")
                report.append(f"      后5个平均: {ctx['after_mean']:.2f}秒")
        
        # 可能的原因
        causes = max_analysis.get('possible_causes', [])
        if causes:
            report.append(f"\n  可能的根本原因:")
            for cause in causes:
                report.append(f"    • {cause.get('cause', 'N/A')}")
                report.append(f"      证据: {cause.get('evidence', 'N/A')}")
                report.append(f"      置信度: {cause.get('confidence', 'N/A')}")
        
        # 错误分析
        error_analysis = test.get('error_analysis', {})
        if error_analysis.get('error_types', {}):
            report.append(f"\n❌ 错误分析:")
            for error_type, count in error_analysis.get('error_types', {}).items():
                report.append(f"  {error_type}: {count}次")
    
    report.append("\n" + "="*80)
    report.append("综合优化建议")
    report.append("="*80)
    
    # 生成优化建议
    recommendations = []
    
    # 根据方差分析建议
    overall_variance = results.get('summary', {}).get('overall_variance', 0)
    if overall_variance > 5:
        recommendations.append({
            'priority': '高',
            'title': '降低批处理重组频率',
            'description': '当前方差较高({:.2f})，表明批处理重组过于频繁。建议调整BATCH_REGROUP_THRESHOLD参数。'.format(overall_variance),
            'action': '减小BATCH_REGROUP_THRESHOLD值，或增加MIN_EFFICIENT_BATCH_SIZE'
        })
    
    # 根据最大响应时间分析建议
    overall_max_rt = results.get('summary', {}).get('overall_max_rt', 0)
    if overall_max_rt > 10:
        recommendations.append({
            'priority': '高',
            'title': '优化资源分配',
            'description': '最大响应时间过长({:.2f}秒)，可能存在资源竞争或序列ID池瓶颈。'.format(overall_max_rt),
            'action': '检查n_seq_max配置，考虑增加序列ID池大小；优化线程池配置'
        })
    
    # 根据稳定性分数分析建议
    overall_stability = results.get('summary', {}).get('overall_stability_score', 0)
    if overall_stability < 0.85:
        recommendations.append({
            'priority': '中',
            'title': '提高系统稳定性',
            'description': '稳定性分数较低({:.2f}%)，需要优化请求调度和批处理策略。'.format(overall_stability * 100),
            'action': '实现自适应批处理大小；优化请求队列管理；添加请求优先级机制'
        })
    
    # 通用建议
    recommendations.append({
        'priority': '中',
        'title': '添加预热机制',
        'description': '前几个请求响应时间较长，建议添加系统预热机制。',
        'action': '在服务器启动时发送几个预热请求；缓存初始计算结果'
    })
    
    recommendations.append({
        'priority': '低',
        'title': '实现请求优先级',
        'description': '避免长请求阻塞短请求，提高整体响应时间一致性。',
        'action': '添加请求优先级队列；实现动态超时调整'
    })
    
    for rec in recommendations:
        report.append(f"\n{'🔴' if rec['priority'] == '高' else '🟡' if rec['priority'] == '中' else '🟢'} {rec['title']} ({rec['priority']}优先级)")
        report.append(f"  描述: {rec['description']}")
        report.append(f"  行动: {rec['action']}")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


def main():
    """主函数"""
    result_file = "/tmp/stability_test_comprehensive_20260121_223609.json"
    
    try:
        results = load_results(result_file)
        
        report = generate_analysis_report(results)
        
        print(report)
        
        # 保存报告
        report_file = f"/tmp/stability_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n\n报告已保存到: {report_file}")
        
    except FileNotFoundError:
        print(f"错误: 未找到文件 {result_file}")
    except Exception as e:
        print(f"错误: {str(e)}")


if __name__ == "__main__":
    main()
