#!/usr/bin/env python3
"""
CPU+GPU协同使用性能测试脚本
测试不同的n_gpu_layers配置，找出最优的CPU和GPU协同方案
"""

import subprocess
import time
import requests
import json
import sys
import os

# 测试配置
TEST_CONFIGS = [
    {"name": "纯CPU", "config": "config_cpu_only.yaml", "n_gpu_layers": 0},
    {"name": "GPU 10层", "config": "config_gpu_10.yaml", "n_gpu_layers": 10},
    {"name": "GPU 20层", "config": "config_gpu_20.yaml", "n_gpu_layers": 20},
    {"name": "GPU 30层", "config": "config_gpu_30.yaml", "n_gpu_layers": 30},
    {"name": "GPU 99层", "config": "config.yaml", "n_gpu_layers": 99},
]

SERVER_URL = "http://localhost:8080"
SERVER_CMD = "./build/bin/cllm_server --config config/{}"
SERVER_LOG = "/tmp/cllm_server_test_{}.log"

def start_server(config_file):
    """启动cLLM服务器"""
    print(f"\n{'='*60}")
    print(f"启动服务器: {config_file}")
    print(f"{'='*60}")
    
    # 停止现有服务器
    subprocess.run(["pkill", "-f", "cllm_server"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    
    # 启动新服务器
    log_file = SERVER_LOG.format(config_file.replace("/", "_"))
    cmd = SERVER_CMD.format(config_file)
    subprocess.Popen(
        cmd,
        shell=True,
        stdout=open(log_file, "w"),
        stderr=subprocess.STDOUT,
        cwd="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM"
    )
    
    # 等待服务器启动
    print("等待服务器启动...")
    for i in range(30):
        try:
            response = requests.get(f"{SERVER_URL}/health", timeout=2)
            if response.status_code == 200:
                print("✓ 服务器启动成功")
                return True
        except:
            time.sleep(1)
    
    print("✗ 服务器启动失败")
    return False

def stop_server():
    """停止cLLM服务器"""
    subprocess.run(["pkill", "-f", "cllm_server"], stderr=subprocess.DEVNULL)
    time.sleep(2)

def run_benchmark(test_name, requests=10, concurrency=5, max_tokens=50):
    """运行基准测试"""
    print(f"\n运行基准测试: {test_name}")
    print(f"请求数: {requests}, 并发数: {concurrency}, 最大tokens: {max_tokens}")
    
    # 运行测试
    cmd = [
        "python3", "unified_benchmark.py",
        "--server-url", SERVER_URL,
        "--test-type", "all",
        "--requests", str(requests),
        "--concurrency", str(concurrency),
        "--max-tokens", str(max_tokens)
    ]
    
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/tools"
    )
    
    # 解析结果
    output = result.stdout
    lines = output.split('\n')
    
    stats = {
        "name": test_name,
        "sequential": {},
        "concurrent": {}
    }
    
    current_section = None
    for line in lines:
        if "Sequential Test Statistics:" in line:
            current_section = "sequential"
        elif "Concurrent Test Statistics:" in line:
            current_section = "concurrent"
        elif current_section and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            
            # 提取数值
            if "tokens/sec" in value:
                try:
                    num_value = float(value.split()[0])
                    stats[current_section][key] = num_value
                except:
                    pass
            elif "s" in value and "time" in key.lower():
                try:
                    num_value = float(value.split()[0])
                    stats[current_section][key] = num_value
                except:
                    pass
            elif key in ["Total requests", "Successful requests", "Failed requests"]:
                try:
                    num_value = int(value)
                    stats[current_section][key] = num_value
                except:
                    pass
    
    return stats

def main():
    """主函数"""
    print("\n" + "="*60)
    print("CPU+GPU协同使用性能测试")
    print("="*60)
    
    all_results = []
    
    for test_config in TEST_CONFIGS:
        # 启动服务器
        if not start_server(test_config["config"]):
            print(f"跳过测试: {test_config['name']}")
            continue
        
        # 运行基准测试
        stats = run_benchmark(
            test_name=test_config["name"],
            requests=10,
            concurrency=5,
            max_tokens=50
        )
        
        stats["n_gpu_layers"] = test_config["n_gpu_layers"]
        all_results.append(stats)
        
        # 停止服务器
        stop_server()
    
    # 生成报告
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)
    
    print("\n顺序测试结果:")
    print(f"{'配置':<15} {'n_gpu_layers':<15} {'吞吐量(t/s)':<15} {'响应时间(s)':<15}")
    print("-"*60)
    for result in all_results:
        seq = result["sequential"]
        throughput = seq.get("Avg throughput", 0)
        response_time = seq.get("Avg response time", 0)
        print(f"{result['name']:<15} {result['n_gpu_layers']:<15} {throughput:<15.2f} {response_time:<15.2f}")
    
    print("\n并发测试结果:")
    print(f"{'配置':<15} {'n_gpu_layers':<15} {'吞吐量(t/s)':<15} {'响应时间(s)':<15} {'成功率':<15}")
    print("-"*75)
    for result in all_results:
        conc = result["concurrent"]
        throughput = conc.get("Avg throughput", 0)
        response_time = conc.get("Avg response time", 0)
        success_rate = 0
        if conc.get("Total requests", 0) > 0:
            success_rate = (conc.get("Successful requests", 0) / conc.get("Total requests", 1)) * 100
        print(f"{result['name']:<15} {result['n_gpu_layers']:<15} {throughput:<15.2f} {response_time:<15.2f} {success_rate:<15.1f}%")
    
    # 找出最优配置
    print("\n" + "="*60)
    print("最优配置分析")
    print("="*60)
    
    # 顺序测试最优
    best_seq = max(all_results, key=lambda x: x["sequential"].get("Avg throughput", 0))
    print(f"\n顺序测试最优配置: {best_seq['name']} (n_gpu_layers={best_seq['n_gpu_layers']})")
    print(f"  吞吐量: {best_seq['sequential'].get('Avg throughput', 0):.2f} t/s")
    print(f"  响应时间: {best_seq['sequential'].get('Avg response time', 0):.2f} s")
    
    # 并发测试最优
    best_conc = max(all_results, key=lambda x: x["concurrent"].get("Avg throughput", 0))
    print(f"\n并发测试最优配置: {best_conc['name']} (n_gpu_layers={best_conc['n_gpu_layers']})")
    print(f"  吞吐量: {best_conc['concurrent'].get('Avg throughput', 0):.2f} t/s")
    print(f"  响应时间: {best_conc['concurrent'].get('Avg response time', 0):.2f} s")
    success_rate = 0
    if best_conc["concurrent"].get("Total requests", 0) > 0:
        success_rate = (best_conc["concurrent"].get("Successful requests", 0) / best_conc["concurrent"].get("Total requests", 1)) * 100
    print(f"  成功率: {success_rate:.1f}%")
    
    # 综合最优
    print(f"\n综合推荐配置: {best_conc['name']} (n_gpu_layers={best_conc['n_gpu_layers']})")
    print("  原因: 并发场景是实际生产环境的主要使用场景")
    
    # 保存结果到JSON
    with open("/tmp/cpu_gpu_hybrid_test_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n详细结果已保存到: /tmp/cpu_gpu_hybrid_test_results.json")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n测试被中断")
        stop_server()
        sys.exit(1)
    finally:
        stop_server()