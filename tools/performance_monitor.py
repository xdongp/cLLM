#!/usr/bin/env python3
"""
cLLM 性能监控和分析工具

用于收集和分析 cLLM 系统的性能数据，识别瓶颈和优化机会
"""

import time
import json
import subprocess
import threading
import queue
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
from datetime import datetime
import statistics
import os

@dataclass
class PerformanceMetric:
    """性能指标数据类"""
    timestamp: float
    metric_name: str
    value: float
    unit: str
    context: Optional[Dict] = None

@dataclass
class RequestTiming:
    """请求时间统计"""
    request_id: str
    timestamp: float
    phase: str  # receive, queue, tokenization, inference, response
    duration_ms: float
    details: Optional[Dict] = None

@dataclass
class SystemResource:
    """系统资源使用情况"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_mb: float
    gpu_memory_used_mb: Optional[float] = None
    gpu_memory_total_mb: Optional[float] = None

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, output_dir: str = "/tmp/cllm_perf"):
        self.output_dir = output_dir
        self.metrics_queue: queue.Queue = queue.Queue()
        self.request_timings: List[RequestTiming] = []
        self.system_resources: List[SystemResource] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.running = False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        self.start_time = time.time()
    
    def start(self):
        """启动监控"""
        self.running = True
        self.monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitoring_thread.start()
        print(f"[PERF] Performance monitor started (output: {self.output_dir})")
    
    def stop(self):
        """停止监控并生成报告"""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        print(f"[PERF] Performance monitor stopped")
        self.generate_report()
    
    def _monitor_loop(self):
        """监控主循环"""
        while self.running:
            # 收集系统资源（每1秒）
            self._collect_system_resources()
            
            # 处理队列中的指标
            while not self.metrics_queue.empty():
                try:
                    metric = self.metrics_queue.get(timeout=0.1)
                    self._process_metric(metric)
                except queue.Empty:
                    break
            
            time.sleep(1)
    
    def _collect_system_resources(self):
        """收集系统资源使用情况"""
        try:
            # CPU 和内存使用（macOS）
            vm_stat = subprocess.run(
                ["vm_stat"], capture_output=True, text=True
            ).stdout
            
            # 解析 vm_stat 输出
            pages = {}
            for line in vm_stat.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    pages[key.strip()] = int(value.strip().split('.')[0])
            
            # 计算内存使用（每页4KB）
            page_size = 4096
            free_pages = pages.get('Pages free', 0)
            active_pages = pages.get('Pages active', 0)
            inactive_pages = pages.get('Pages inactive', 0)
            wired_pages = pages.get('Pages wired down', 0)
            
            total_memory = (free_pages + active_pages + inactive_pages + wired_pages) * page_size
            used_memory = (active_pages + inactive_pages + wired_pages) * page_size
            
            memory_percent = (used_memory / total_memory) * 100
            memory_mb = used_memory / (1024 * 1024)
            
            # CPU 使用
            cpu_info = subprocess.run(
                ["top", "-l", "1", "-n", "0"], capture_output=True, text=True
            ).stdout
            cpu_percent = 0.0
            for line in cpu_info.split('\n'):
                if 'CPU usage:' in line:
                    parts = line.split()
                    user = float(parts[2].replace('%', ''))
                    sys = float(parts[4].replace('%', ''))
                    cpu_percent = user + sys
                    break
            
            resource = SystemResource(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_mb=memory_mb
            )
            self.system_resources.append(resource)
            
        except Exception as e:
            print(f"[PERF] Error collecting system resources: {e}")
    
    def _process_metric(self, metric: PerformanceMetric):
        """处理性能指标"""
        # 可以在这里添加自定义的指标处理逻辑
        pass
    
    def add_request_timing(self, request_id: str, phase: str, duration_ms: float, details: Optional[Dict] = None):
        """添加请求时间统计"""
        timing = RequestTiming(
            request_id=request_id,
            timestamp=time.time(),
            phase=phase,
            duration_ms=duration_ms,
            details=details
        )
        self.request_timings.append(timing)
    
    def generate_report(self):
        """生成性能分析报告"""
        report = {
            "metadata": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(time.time()).isoformat(),
                "duration_seconds": time.time() - self.start_time,
                "total_requests": len(self.request_timings),
                "total_metrics": len(self.system_resources)
            },
            "system_resources": self._analyze_system_resources(),
            "request_timings": self._analyze_request_timings(),
            "bottlenecks": self._identify_bottlenecks()
        }
        
        # 保存报告
        report_path = os.path.join(
            self.output_dir,
            f"performance_report_{int(self.start_time)}.json"
        )
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[PERF] Performance report saved to: {report_path}")
        
        # 打印摘要
        self._print_summary(report)
    
    def _analyze_system_resources(self) -> Dict:
        """分析系统资源使用情况"""
        if not self.system_resources:
            return {}
        
        cpu_percents = [r.cpu_percent for r in self.system_resources]
        memory_percents = [r.memory_percent for r in self.system_resources]
        memory_mbs = [r.memory_mb for r in self.system_resources]
        
        return {
            "cpu": {
                "min": min(cpu_percents),
                "max": max(cpu_percents),
                "avg": statistics.mean(cpu_percents),
                "p50": statistics.median(cpu_percents),
                "p95": self._percentile(cpu_percents, 95),
                "p99": self._percentile(cpu_percents, 99)
            },
            "memory": {
                "min_mb": min(memory_mbs),
                "max_mb": max(memory_mbs),
                "avg_mb": statistics.mean(memory_mbs),
                "min_percent": min(memory_percents),
                "max_percent": max(memory_percents),
                "avg_percent": statistics.mean(memory_percents)
            }
        }
    
    def _analyze_request_timings(self) -> Dict:
        """分析请求时间统计"""
        if not self.request_timings:
            return {}
        
        # 按阶段分组
        phases = {}
        for timing in self.request_timings:
            if timing.phase not in phases:
                phases[timing.phase] = []
            phases[timing.phase].append(timing.duration_ms)
        
        result = {}
        for phase, durations in phases.items():
            result[phase] = {
                "count": len(durations),
                "min_ms": min(durations),
                "max_ms": max(durations),
                "avg_ms": statistics.mean(durations),
                "p50_ms": statistics.median(durations),
                "p95_ms": self._percentile(durations, 95),
                "p99_ms": self._percentile(durations, 99)
            }
        
        return result
    
    def _identify_bottlenecks(self) -> List[Dict]:
        """识别性能瓶颈"""
        bottlenecks = []
        
        # 检查 CPU 瓶颈
        if self.system_resources:
            cpu_percents = [r.cpu_percent for r in self.system_resources]
            avg_cpu = statistics.mean(cpu_percents)
            if avg_cpu > 80:
                bottlenecks.append({
                    "type": "cpu",
                    "severity": "high",
                    "description": f"CPU 使用率过高 (avg: {avg_cpu:.1f}%)",
                    "suggestion": "考虑增加线程数或优化算法"
                })
        
        # 检查内存瓶颈
        if self.system_resources:
            memory_percents = [r.memory_percent for r in self.system_resources]
            avg_memory = statistics.mean(memory_percents)
            if avg_memory > 85:
                bottlenecks.append({
                    "type": "memory",
                    "severity": "high",
                    "description": f"内存使用率过高 (avg: {avg_memory:.1f}%)",
                    "suggestion": "考虑优化内存使用或增加内存"
                })
        
        # 检查请求阶段瓶颈
        if self.request_timings:
            phases = {}
            for timing in self.request_timings:
                if timing.phase not in phases:
                    phases[timing.phase] = []
                phases[timing.phase].append(timing.duration_ms)
            
            # 找出耗时最长的阶段
            max_duration = 0
            slowest_phase = None
            for phase, durations in phases.items():
                avg_duration = statistics.mean(durations)
                if avg_duration > max_duration:
                    max_duration = avg_duration
                    slowest_phase = phase
            
            if slowest_phase and max_duration > 100:
                bottlenecks.append({
                    "type": "request_phase",
                    "severity": "medium",
                    "description": f"请求阶段 {slowest_phase} 耗时过长 (avg: {max_duration:.1f}ms)",
                    "suggestion": f"优化 {slowest_phase} 阶段的处理逻辑"
                })
        
        return bottlenecks
    
    def _print_summary(self, report: Dict):
        """打印性能摘要"""
        print("\n" + "="*60)
        print("cLLM 性能分析摘要")
        print("="*60)
        
        meta = report["metadata"]
        print(f"\n测试时间: {meta['duration_seconds']:.1f} 秒")
        print(f"总请求数: {meta['total_requests']}")
        
        if report["system_resources"]:
            cpu = report["system_resources"]["cpu"]
            memory = report["system_resources"]["memory"]
            print(f"\nCPU 使用:")
            print(f"  平均值: {cpu['avg']:.1f}%")
            print(f"  最大值: {cpu['max']:.1f}%")
            print(f"  P95: {cpu['p95']:.1f}%")
            print(f"\n内存使用:")
            print(f"  平均值: {memory['avg_percent']:.1f}% ({memory['avg_mb']:.0f} MB)")
            print(f"  最大值: {memory['max_percent']:.1f}% ({memory['max_mb']:.0f} MB)")
        
        if report["request_timings"]:
            print(f"\n请求阶段耗时统计:")
            for phase, stats in report["request_timings"].items():
                print(f"  {phase}:")
                print(f"    平均: {stats['avg_ms']:.1f}ms")
                print(f"    P95: {stats['p95_ms']:.1f}ms")
                print(f"    数量: {stats['count']}")
        
        if report["bottlenecks"]:
            print(f"\n识别到的瓶颈 ({len(report['bottlenecks'])}):")
            for i, bottleneck in enumerate(report["bottlenecks"], 1):
                print(f"  {i}. [{bottleneck['severity'].upper()}] {bottleneck['description']}")
                print(f"     建议: {bottleneck['suggestion']}")
        
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
    import argparse
    
    parser = argparse.ArgumentParser(description="cLLM 性能监控工具")
    parser.add_argument(
        "--output-dir", 
        default="/tmp/cllm_perf",
        help="输出目录（默认: /tmp/cllm_perf）"
    )
    parser.add_argument(
        "--duration", 
        type=int,
        default=60,
        help="监控持续时间（秒，默认: 60）"
    )
    
    args = parser.parse_args()
    
    monitor = PerformanceMonitor(output_dir=args.output_dir)
    monitor.start()
    
    try:
        print(f"[PERF] Monitoring for {args.duration} seconds...")
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print(f"\n[PERF] Monitoring interrupted by user")
    finally:
        monitor.stop()


if __name__ == "__main__":
    main()
