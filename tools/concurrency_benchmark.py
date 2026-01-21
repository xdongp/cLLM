#!/usr/bin/env python3
"""
cLLM 并发性能基准测试工具

用于测试不同并发级别下的系统性能，识别性能拐点和瓶颈
"""

import time
import json
import requests
import threading
import queue
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import statistics
import os
import argparse

@dataclass
class RequestResult:
    """单个请求的结果"""
    request_id: int
    concurrency_level: int
    timestamp: float
    latency_ms: float
    tokens_generated: int
    tokens_per_second: float
    success: bool
    error: Optional[str] = None

@dataclass
class BenchmarkResult:
    """基准测试结果"""
    concurrency_level: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    throughput_tokens_per_second: float
    throughput_requests_per_second: float
    latency_ms: Dict[str, float]  # min, max, avg, p50, p95, p99
    tokens_generated: Dict[str, float]  # min, max, avg
    error_rate: float
    duration_seconds: float

class ConcurrencyBenchmark:
    """并发基准测试器"""
    
    def __init__(
        self,
        server_url: str = "http://localhost:8080",
        output_dir: str = "/tmp/cllm_benchmark",
        max_tokens: int = 50,
        prompt: Optional[str] = None
    ):
        self.server_url = server_url
        self.output_dir = output_dir
        self.max_tokens = max_tokens
        self.prompt = prompt or "Hello!"
        self.results: List[RequestResult] = []
        
        os.makedirs(output_dir, exist_ok=True)
    
    def run_concurrency_test(
        self,
        concurrency: int,
        total_requests: int
    ) -> BenchmarkResult:
        """运行指定并发级别的测试"""
        print(f"\n{'='*60}")
        print(f"运行并发测试: concurrency={concurrency}, requests={total_requests}")
        print(f"{'='*60}")
        
        results = []
        request_queue = queue.Queue()
        
        # 填充请求队列
        for i in range(total_requests):
            request_queue.put(i)
        
        # 创建工作线程
        threads = []
        thread_results = []
        results_lock = threading.Lock()
        
        def worker():
            while not request_queue.empty():
                try:
                    request_id = request_queue.get(timeout=1)
                except queue.Empty:
                    break
                
                result = self._make_request(request_id, concurrency)
                with results_lock:
                    thread_results.append(result)
                request_queue.task_done()
        
        # 启动线程
        start_time = time.time()
        for i in range(concurrency):
            t = threading.Thread(target=worker, daemon=True)
            threads.append(t)
            t.start()
        
        # 等待所有请求完成
        request_queue.join()
        end_time = time.time()
        
        # 收集结果
        results = thread_results
        self.results.extend(results)
        
        # 分析结果
        benchmark_result = self._analyze_results(results, concurrency, end_time - start_time)
        
        print(f"\n并发 {concurrency} 测试完成:")
        print(f"  吞吐量: {benchmark_result.throughput_tokens_per_second:.1f} tokens/s")
        print(f"  延迟 (P95): {benchmark_result.latency_ms['p95']:.1f}ms")
        print(f"  错误率: {benchmark_result.error_rate:.2%}")
        print(f"  耗时: {benchmark_result.duration_seconds:.2f}s")
        
        return benchmark_result
    
    def _make_request(self, request_id: int, concurrency: int) -> RequestResult:
        """发送单个请求"""
        url = f"{self.server_url}/generate"
        
        payload = {
            "prompt": self.prompt,
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        
        start_time = time.time()
        
        try:
            response = requests.post(
                url,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                latency_ms = (time.time() - start_time) * 1000
                
                tokens_generated = data.get("generated_tokens", 0)
                tokens_per_second = data.get("tokens_per_second", 0)
                
                return RequestResult(
                    request_id=request_id,
                    concurrency_level=concurrency,
                    timestamp=start_time,
                    latency_ms=latency_ms,
                    tokens_generated=tokens_generated,
                    tokens_per_second=tokens_per_second,
                    success=True
                )
            else:
                latency_ms = (time.time() - start_time) * 1000
                return RequestResult(
                    request_id=request_id,
                    concurrency_level=concurrency,
                    timestamp=start_time,
                    latency_ms=latency_ms,
                    tokens_generated=0,
                    tokens_per_second=0,
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}"
                )
                
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return RequestResult(
                request_id=request_id,
                concurrency_level=concurrency,
                timestamp=start_time,
                latency_ms=latency_ms,
                tokens_generated=0,
                tokens_per_second=0,
                success=False,
                error=str(e)
            )
    
    def _analyze_results(
        self,
        results: List[RequestResult],
        concurrency: int,
        duration: float
    ) -> BenchmarkResult:
        """分析测试结果"""
        if not results:
            return BenchmarkResult(
                concurrency_level=concurrency,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                throughput_tokens_per_second=0,
                throughput_requests_per_second=0,
                latency_ms={},
                tokens_generated={},
                error_rate=0,
                duration_seconds=duration
            )
        
        successful = [r for r in results if r.success]
        failed = [r for r in results if not r.success]
        
        total_requests = len(results)
        successful_requests = len(successful)
        failed_requests = len(failed)
        error_rate = failed_requests / total_requests if total_requests > 0 else 0
        
        # 吞吐量计算
        total_tokens = sum(r.tokens_generated for r in successful)
        throughput_tokens = total_tokens / duration if duration > 0 else 0
        throughput_requests = successful_requests / duration if duration > 0 else 0
        
        # 延迟统计
        if successful:
            latencies = [r.latency_ms for r in successful]
            latencies_sorted = sorted(latencies)
            
            latency_stats = {
                "min": min(latencies),
                "max": max(latencies),
                "avg": statistics.mean(latencies),
                "p50": statistics.median(latencies),
                "p95": self._percentile(latencies, 95),
                "p99": self._percentile(latencies, 99)
            }
        else:
            latency_stats = {}
        
        # Token生成统计
        if successful:
            tokens = [r.tokens_generated for r in successful]
            tokens_stats = {
                "min": min(tokens),
                "max": max(tokens),
                "avg": statistics.mean(tokens)
            }
        else:
            tokens_stats = {}
        
        return BenchmarkResult(
            concurrency_level=concurrency,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            throughput_tokens_per_second=throughput_tokens,
            throughput_requests_per_second=throughput_requests,
            latency_ms=latency_stats,
            tokens_generated=tokens_stats,
            error_rate=error_rate,
            duration_seconds=duration
        )
    
    def run_benchmark_suite(
        self,
        concurrency_levels: List[int],
        requests_per_level: int = 72
    ) -> List[BenchmarkResult]:
        """运行完整的基准测试套件"""
        print(f"{'='*60}")
        print(f"cLLM 并发基准测试套件")
        print(f"{'='*60}")
        print(f"服务器: {self.server_url}")
        print(f"并发级别: {concurrency_levels}")
        print(f"每级别请求数: {requests_per_level}")
        print(f"Max tokens: {self.max_tokens}")
        print(f"输出目录: {self.output_dir}")
        
        all_results = []
        
        for concurrency in concurrency_levels:
            result = self.run_concurrency_test(concurrency, requests_per_level)
            all_results.append(result)
        
        self.generate_comprehensive_report(all_results)
        
        return all_results
    
    def generate_comprehensive_report(self, results: List[BenchmarkResult]):
        """生成综合报告"""
        report = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "server_url": self.server_url,
                "max_tokens": self.max_tokens,
                "prompt_length": len(self.prompt)
            },
            "benchmark_results": [asdict(r) for r in results],
            "analysis": self._analyze_benchmark_results(results),
            "recommendations": self._generate_recommendations(results)
        }
        
        # 保存JSON报告
        report_path = os.path.join(
            self.output_dir,
            f"concurrency_benchmark_{int(time.time())}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"综合报告已保存: {report_path}")
        print(f"{'='*60}")
        
        # 打印摘要
        self._print_benchmark_summary(results)
    
    def _analyze_benchmark_results(self, results: List[BenchmarkResult]) -> Dict:
        """分析基准测试结果"""
        if not results:
            return {}
        
        # 吞吐量分析
        throughputs = [r.throughput_tokens_per_second for r in results]
        concurrencies = [r.concurrency_level for r in results]
        
        # 找到最大吞吐量和对应的并发级别
        max_throughput = max(throughputs)
        optimal_concurrency = concurrencies[throughputs.index(max_throughput)]
        
        # 计算吞吐量增长趋势
        throughput_growth = []
        for i in range(1, len(throughputs)):
            growth = (throughputs[i] - throughputs[i-1]) / throughputs[i-1] * 100
            throughput_growth.append(growth)
        
        # 延迟分析
        p95_latencies = [r.latency_ms.get('p95', 0) for r in results if r.latency_ms]
        
        # 错误率分析
        error_rates = [r.error_rate for r in results]
        
        return {
            "throughput": {
                "max_throughput": max_throughput,
                "optimal_concurrency": optimal_concurrency,
                "throughput_trend": throughputs,
                "growth_trend": throughput_growth,
                "saturation_point": self._find_saturation_point(results)
            },
            "latency": {
                "p95_trend": p95_latencies,
                "max_p95": max(p95_latencies) if p95_latencies else 0
            },
            "error_rate": {
                "max_error_rate": max(error_rates),
                "error_trend": error_rates
            }
        }
    
    def _find_saturation_point(self, results: List[BenchmarkResult]) -> Optional[int]:
        """找到性能饱和点"""
        if len(results) < 3:
            return None
        
        throughputs = [r.throughput_tokens_per_second for r in results]
        
        # 当吞吐量增长 < 5% 时认为达到饱和
        for i in range(2, len(throughputs)):
            growth = (throughputs[i] - throughputs[i-1]) / throughputs[i-1] * 100
            if growth < 5:
                return results[i].concurrency_level
        
        return None
    
    def _generate_recommendations(self, results: List[BenchmarkResult]) -> List[Dict]:
        """生成优化建议"""
        recommendations = []
        
        if not results:
            return recommendations
        
        analysis = self._analyze_benchmark_results(results)
        
        # 基于吞吐量的建议
        optimal_concurrency = analysis['throughput']['optimal_concurrency']
        recommendations.append({
            "type": "optimal_concurrency",
            "priority": "high",
            "description": f"最佳并发级别为 {optimal_concurrency}",
            "suggestion": f"在生产环境中，将并发请求数设置为 {optimal_concurrency} 以获得最大吞吐量"
        })
        
        # 基于延迟的建议
        max_p95 = analysis['latency']['max_p95']
        if max_p95 > 5000:  # 超过5秒
            recommendations.append({
                "type": "latency_optimization",
                "priority": "medium",
                "description": f"P95 延迟较高 ({max_p95:.0f}ms)",
                "suggestion": "考虑优化推理引擎或增加硬件资源"
            })
        
        # 基于错误率的建议
        max_error = analysis['error_rate']['max_error_rate']
        if max_error > 0.05:  # 超过5%
            recommendations.append({
                "type": "error_rate",
                "priority": "high",
                "description": f"错误率较高 ({max_error:.2%})",
                "suggestion": "检查系统资源限制，考虑增加线程池大小或优化错误处理"
            })
        
        # 饱和点建议
        saturation_point = analysis['throughput']['saturation_point']
        if saturation_point:
            recommendations.append({
                "type": "saturation_point",
                "priority": "medium",
                "description": f"性能饱和点在并发 {saturation_point}",
                "suggestion": f"超过并发 {saturation_point} 后，吞吐量增长有限，建议优化系统架构"
            })
        
        return recommendations
    
    def _print_benchmark_summary(self, results: List[BenchmarkResult]):
        """打印基准测试摘要"""
        print("\n" + "="*60)
        print("cLLM 并发基准测试摘要")
        print("="*60)
        
        # 打印表格
        print(f"\n{'并发级别':<12} {'吞吐量(t/s)':<15} {'P95延迟(ms)':<15} {'错误率':<10} {'耗时(s)':<10}")
        print("-" * 60)
        
        for result in results:
            throughput = f"{result.throughput_tokens_per_second:.1f}"
            p95 = f"{result.latency_ms.get('p95', 0):.0f}"
            error_rate = f"{result.error_rate*100:.1f}%"
            duration = f"{result.duration_seconds:.1f}"
            
            print(f"{result.concurrency_level:<12} {throughput:<15} {p95:<15} {error_rate:<10} {duration:<10}")
        
        print("\n" + "="*60)
        
        # 打印分析结果
        analysis = self._analyze_benchmark_results(results)
        
        print(f"\n关键发现:")
        print(f"  最大吞吐量: {analysis['throughput']['max_throughput']:.1f} t/s")
        print(f"  最佳并发: {analysis['throughput']['optimal_concurrency']}")
        
        saturation = analysis['throughput']['saturation_point']
        if saturation:
            print(f"  性能饱和点: 并发 {saturation}")
        
        print(f"  最大P95延迟: {analysis['latency']['max_p95']:.0f}ms")
        print(f"  最大错误率: {analysis['error_rate']['max_error_rate']*100:.1f}%")
        
        # 打印建议
        print(f"\n优化建议:")
        for i, rec in enumerate(self._generate_recommendations(results), 1):
            print(f"  {i}. [{rec['priority'].upper()}] {rec['description']}")
            print(f"     {rec['suggestion']}")
        
        print("\n" + "="*60)
    
    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        data_sorted = sorted(data)
        index = int(len(data_sorted) * p / 100)
        return data_sorted[min(index, len(data_sorted) - 1)]


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="cLLM 并发基准测试工具")
    
    parser.add_argument(
        "--server-url",
        default="http://localhost:8080",
        help="服务器URL（默认: http://localhost:8080）"
    )
    parser.add_argument(
        "--output-dir",
        default="/tmp/cllm_benchmark",
        help="输出目录（默认: /tmp/cllm_benchmark）"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=50,
        help="最大生成token数（默认: 50）"
    )
    parser.add_argument(
        "--prompt",
        default="Hello!",
        help="测试prompt（默认: Hello!）"
    )
    parser.add_argument(
        "--concurrency-levels",
        nargs='+',
        type=int,
        default=[8, 16, 24, 32],
        help="并发级别列表（默认: 8 16 24 32）"
    )
    parser.add_argument(
        "--requests-per-level",
        type=int,
        default=72,
        help="每个并发级别的请求数（默认: 72）"
    )
    
    args = parser.parse_args()
    
    benchmark = ConcurrencyBenchmark(
        server_url=args.server_url,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
        prompt=args.prompt
    )
    
    benchmark.run_benchmark_suite(
        concurrency_levels=args.concurrency_levels,
        requests_per_level=args.requests_per_level
    )


if __name__ == "__main__":
    main()
