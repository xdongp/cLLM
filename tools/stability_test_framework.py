#!/usr/bin/env python3
"""
CLLM系统稳定性测试框架
用于测量和分析请求响应时间的稳定性和一致性
"""

import time
import json
import statistics
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import threading


class StabilityTestFramework:
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.results = []
        self.lock = threading.Lock()
        
    def send_request(self, prompt: str, max_tokens: int = 50, timeout: int = 300) -> Dict:
        """发送单个请求并记录响应时间"""
        url = f"{self.base_url}/generate"
        headers = {"Content-Type": "application/json"}
        data = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        start_time = time.time()
        try:
            response = requests.post(url, headers=headers, json=data, timeout=timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "tokens_generated": len(result.get("text", "").split()),
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "response_time": response_time,
                    "status_code": response.status_code,
                    "error": f"HTTP {response.status_code}",
                    "timestamp": datetime.now().isoformat()
                }
        except requests.exceptions.RequestException as e:
            response_time = time.time() - start_time
            return {
                "success": False,
                "response_time": response_time,
                "status_code": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_concurrent_test(self, num_requests: int, concurrency: int, 
                           max_tokens: int = 50, prompt: str = "Hello") -> List[Dict]:
        """运行并发测试"""
        print(f"\n{'='*60}")
        print(f"运行并发测试: {num_requests}个请求, 并发度{concurrency}")
        print(f"{'='*60}")
        
        responses = []
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(self.send_request, prompt, max_tokens) 
                      for _ in range(num_requests)]
            
            for i, future in enumerate(as_completed(futures)):
                try:
                    result = future.result()
                    responses.append(result)
                    if (i + 1) % 10 == 0 or (i + 1) == num_requests:
                        print(f"  已完成: {i + 1}/{num_requests} 请求")
                except Exception as e:
                    responses.append({
                        "success": False,
                        "response_time": 0,
                        "status_code": 0,
                        "error": f"Future exception: {str(e)}",
                        "timestamp": datetime.now().isoformat()
                    })
        
        total_time = time.time() - start_time
        print(f"测试完成, 总耗时: {total_time:.2f}秒")
        
        return responses
    
    def analyze_results(self, responses: List[Dict], test_name: str) -> Dict:
        """分析测试结果,计算统计指标"""
        if not responses:
            return {}
        
        success_responses = [r for r in responses if r["success"]]
        failed_responses = [r for r in responses if not r["success"]]
        
        response_times = [r["response_time"] for r in success_responses]
        
        if not response_times:
            return {
                "test_name": test_name,
                "total_requests": len(responses),
                "success_count": 0,
                "failed_count": len(responses),
                "success_rate": 0,
                "error": "所有请求失败",
                "response_time_stats": {
                    "mean": 0,
                    "median": 0,
                    "min": 0,
                    "max": 0,
                    "std_dev": 0,
                    "variance": 0,
                    "cv": 0,
                    "stability_score": 0
                },
                "percentiles": {
                    "p50": 0,
                    "p90": 0,
                    "p95": 0,
                    "p99": 0
                },
                "throughput": {
                    "requests_per_second": 0,
                    "tokens_per_second": 0
                },
                "max_response_time_analysis": {
                    "max_rt": 0,
                    "count": 0,
                    "samples": []
                },
                "error_analysis": {
                    "error_types": {},
                    "failed_responses": failed_responses[:5]
                },
                "raw_data": {
                    "response_times": [],
                    "timestamps": []
                }
            }
        
        # 计算统计指标
        mean_rt = statistics.mean(response_times)
        median_rt = statistics.median(response_times)
        min_rt = min(response_times)
        max_rt = max(response_times)
        std_dev = statistics.stdev(response_times) if len(response_times) > 1 else 0
        variance = statistics.variance(response_times) if len(response_times) > 1 else 0
        
        # 计算百分位数（纯Python实现）
        sorted_times = sorted(response_times)
        n = len(sorted_times)
        
        def percentile(p):
            idx = (p / 100) * (n - 1)
            if idx.is_integer():
                return sorted_times[int(idx)]
            else:
                lower = sorted_times[int(idx)]
                upper = sorted_times[int(idx) + 1]
                return lower + (upper - lower) * (idx - int(idx))
        
        p50 = percentile(50)
        p90 = percentile(90)
        p95 = percentile(95)
        p99 = percentile(99)
        
        # 计算稳定性指标
        cv = std_dev / mean_rt  # 变异系数
        stability_score = 1 / (1 + cv)  # 稳定性分数(0-1)
        
        # 分析最大响应时间的原因
        max_rt_indices = [i for i, rt in enumerate(response_times) if rt == max_rt]
        max_rt_samples = [success_responses[i] for i in max_rt_indices[:3]]
        
        # 分析失败请求
        error_types = {}
        for r in failed_responses:
            error_msg = r.get("error", "Unknown")
            error_types[error_msg] = error_types.get(error_msg, 0) + 1
        
        analysis = {
            "test_name": test_name,
            "total_requests": len(responses),
            "success_count": len(success_responses),
            "failed_count": len(failed_responses),
            "success_rate": len(success_responses) / len(responses),
            
            # 响应时间统计
            "response_time_stats": {
                "mean": mean_rt,
                "median": median_rt,
                "min": min_rt,
                "max": max_rt,
                "std_dev": std_dev,
                "variance": variance,
                "cv": cv,
                "stability_score": stability_score
            },
            
            # 百分位数
            "percentiles": {
                "p50": p50,
                "p90": p90,
                "p95": p95,
                "p99": p99
            },
            
            # 吞吐量
            "throughput": {
                "requests_per_second": len(success_responses) / sum(response_times) if sum(response_times) > 0 else 0,
                "tokens_per_second": sum(r.get("tokens_generated", 0) for r in success_responses) / sum(response_times) if sum(response_times) > 0 else 0
            },
            
            # 最大响应时间分析
            "max_response_time_analysis": {
                "max_rt": max_rt,
                "count": len(max_rt_indices),
                "samples": max_rt_samples
            },
            
            # 错误分析
            "error_analysis": {
                "error_types": error_types,
                "failed_responses": failed_responses[:5]  # 只保留前5个失败示例
            },
            
            "raw_data": {
                "response_times": response_times,
                "timestamps": [r["timestamp"] for r in success_responses]
            }
        }
        
        return analysis
    
    def print_analysis_summary(self, analysis: Dict):
        """打印分析摘要"""
        print(f"\n{'='*60}")
        print(f"测试结果摘要: {analysis['test_name']}")
        print(f"{'='*60}")
        
        print(f"\n📊 基本统计:")
        print(f"  总请求数: {analysis['total_requests']}")
        print(f"  成功数: {analysis['success_count']}")
        print(f"  失败数: {analysis['failed_count']}")
        print(f"  成功率: {analysis['success_rate']*100:.2f}%")
        
        print(f"\n⏱️  响应时间统计 (秒):")
        rt_stats = analysis['response_time_stats']
        print(f"  平均值: {rt_stats['mean']:.2f}")
        print(f"  中位数: {rt_stats['median']:.2f}")
        print(f"  最小值: {rt_stats['min']:.2f}")
        print(f"  最大值: {rt_stats['max']:.2f}")
        print(f"  标准差: {rt_stats['std_dev']:.2f}")
        print(f"  方差: {rt_stats['variance']:.2f}")
        print(f"  变异系数(CV): {rt_stats['cv']*100:.2f}%")
        print(f"  稳定性分数: {rt_stats['stability_score']*100:.2f}%")
        
        print(f"\n📈 百分位数 (秒):")
        percentiles = analysis['percentiles']
        print(f"  P50: {percentiles['p50']:.2f}")
        print(f"  P90: {percentiles['p90']:.2f}")
        print(f"  P95: {percentiles['p95']:.2f}")
        print(f"  P99: {percentiles['p99']:.2f}")
        
        print(f"\n⚡ 吞吐量:")
        throughput = analysis['throughput']
        print(f"  请求/秒: {throughput['requests_per_second']:.2f}")
        print(f"  Token/秒: {throughput['tokens_per_second']:.2f}")
        
        if analysis['failed_count'] > 0:
            print(f"\n❌ 错误分析:")
            for error_type, count in analysis['error_analysis']['error_types'].items():
                print(f"  {error_type}: {count}次")
        
        print(f"\n{'='*60}")
    
    def save_results(self, analysis: Dict, output_file: str):
        """保存分析结果到JSON文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存到: {output_file}")
    
    def compare_benchmarks(self, baseline: Dict, optimized: Dict) -> Dict:
        """比较基准测试和优化后的测试结果"""
        baseline_rt = baseline['response_time_stats']
        optimized_rt = optimized['response_time_stats']
        
        improvements = {
            'variance_improvement': ((baseline_rt['variance'] - optimized_rt['variance']) / baseline_rt['variance']) * 100 if baseline_rt['variance'] > 0 else 0,
            'max_rt_improvement': ((baseline_rt['max'] - optimized_rt['max']) / baseline_rt['max']) * 100 if baseline_rt['max'] > 0 else 0,
            'stability_improvement': ((optimized_rt['stability_score'] - baseline_rt['stability_score']) / baseline_rt['stability_score']) * 100 if baseline_rt['stability_score'] > 0 else 0,
            'cv_improvement': ((baseline_rt['cv'] - optimized_rt['cv']) / baseline_rt['cv']) * 100 if baseline_rt['cv'] > 0 else 0,
        }
        
        return {
            'baseline': baseline_rt,
            'optimized': optimized_rt,
            'improvements': improvements,
            'target_achieved': improvements['stability_improvement'] >= 20
        }


def main():
    """主函数: 运行稳定性测试"""
    framework = StabilityTestFramework()
    
    # 测试配置
    test_configs = [
        {"name": "低并发稳定性测试", "requests": 100, "concurrency": 8, "max_tokens": 50},
        {"name": "中并发稳定性测试", "requests": 150, "concurrency": 16, "max_tokens": 50},
        {"name": "高并发稳定性测试", "requests": 200, "concurrency": 24, "max_tokens": 50},
    ]
    
    prompt = "请介绍一下人工智能的发展历史和未来趋势"
    
    all_analyses = []
    
    for config in test_configs:
        print(f"\n{'#'*60}")
        print(f"开始: {config['name']}")
        print(f"{'#'*60}")
        
        responses = framework.run_concurrent_test(
            num_requests=config['requests'],
            concurrency=config['concurrency'],
            max_tokens=config['max_tokens'],
            prompt=prompt
        )
        
        analysis = framework.analyze_results(responses, config['name'])
        framework.print_analysis_summary(analysis)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"/tmp/stability_test_{config['name']}_{timestamp}.json"
        framework.save_results(analysis, output_file)
        
        all_analyses.append(analysis)
    
    # 生成综合报告
    comprehensive_report = {
        "test_time": datetime.now().isoformat(),
        "tests": all_analyses,
        "summary": {
            "total_tests": len(all_analyses),
            "overall_stability_score": statistics.mean([a['response_time_stats']['stability_score'] for a in all_analyses]),
            "overall_variance": statistics.mean([a['response_time_stats']['variance'] for a in all_analyses]),
            "overall_max_rt": max([a['response_time_stats']['max'] for a in all_analyses])
        }
    }
    
    report_file = f"/tmp/stability_test_comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"综合报告已生成: {report_file}")
    print(f"{'='*60}")
    
    print(f"\n📊 综合统计:")
    print(f"  平均稳定性分数: {comprehensive_report['summary']['overall_stability_score']*100:.2f}%")
    print(f"  平均方差: {comprehensive_report['summary']['overall_variance']:.2f}")
    print(f"  最大响应时间: {comprehensive_report['summary']['overall_max_rt']:.2f}秒")


if __name__ == "__main__":
    main()
