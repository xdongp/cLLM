#!/usr/bin/env python3
"""
多轮cLLM vs Ollama对比测试脚本
测试并发数: 8, 16, 24, 32, 40
至少完成3轮测试
"""

import subprocess
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Any

def run_benchmark(server_type: str, concurrency: int, round_num: int) -> Dict[str, Any]:
    """运行单次benchmark测试"""
    print(f"\n{'='*60}")
    print(f"【{server_type.upper()}】并发{concurrency} - 第{round_num}轮")
    print(f"{'='*60}\n")
    
    output_file = f"/tmp/{server_type}_round{round_num}_c{concurrency}.json"
    
    if server_type == "cllm":
        cmd = [
            "python3", "tools/unified_benchmark.py",
            "--server-type", "cllm",
            "--server-url", "http://localhost:8080",
            "--test-type", "api-concurrent",
            "--requests", "72",
            "--concurrency", str(concurrency),
            "--max-tokens", "50",
            "--output-file", output_file
        ]
    else:  # ollama
        cmd = [
            "python3", "tools/unified_benchmark.py",
            "--server-type", "ollama",
            "--server-url", "http://localhost:11434",
            "--model", "qwen3:0.6b",
            "--test-type", "api-concurrent",
            "--requests", "72",
            "--concurrency", str(concurrency),
            "--max-tokens", "50",
            "--output-file", output_file
        ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ 测试失败: {result.stderr}")
            return {"success": False, "error": result.stderr}
        
        # 读取结果
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                data = json.load(f)
                concurrent_stats = data.get('concurrent', {})
                return {
                    "success": True,
                    "successful_requests": concurrent_stats.get('successful_requests', 0),
                    "failed_requests": concurrent_stats.get('failed_requests', 0),
                    "avg_throughput": concurrent_stats.get('avg_throughput', 0),
                    "avg_response_time": concurrent_stats.get('avg_response_time', 0),
                    "total_test_time": concurrent_stats.get('total_test_time', 0)
                }
        else:
            return {"success": False, "error": "Output file not found"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "Test timeout"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_cllm_health() -> bool:
    """检查cLLM服务器健康状态"""
    import requests
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_ollama_health() -> bool:
    """检查Ollama服务器健康状态"""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def main():
    print("="*60)
    print("多轮 cLLM vs Ollama 对比测试")
    print("="*60)
    print(f"\n测试配置:")
    print(f"  - 请求数量: 72")
    print(f"  - 每个请求最大tokens: 50")
    print(f"  - 并发数: 8, 16, 24, 32, 40")
    print(f"  - 测试轮数: 至少3轮")
    print()
    
    # 检查服务器
    print("检查服务器状态...")
    if not check_cllm_health():
        print("❌ cLLM服务器未运行，请先启动服务器")
        return
    print("✓ cLLM服务器运行正常")
    
    if not check_ollama_health():
        print("❌ Ollama服务器未运行，请先启动服务器")
        return
    print("✓ Ollama服务器运行正常")
    print()
    
    concurrency_levels = [8, 16, 24, 32, 40]
    min_rounds = 3
    all_results = {}
    
    for round_num in range(1, min_rounds + 1):
        print(f"\n{'#'*60}")
        print(f"第 {round_num} 轮测试")
        print(f"{'#'*60}\n")
        
        round_results = {}
        
        for concurrency in concurrency_levels:
            print(f"\n--- 并发 {concurrency} ---\n")
            
            # cLLM测试
            cllm_result = run_benchmark("cllm", concurrency, round_num)
            if not cllm_result.get("success"):
                print(f"⚠️ cLLM测试失败: {cllm_result.get('error', 'Unknown error')}")
                # 继续测试，但记录错误
                cllm_result = {"success": False, "failed": True}
            else:
                print(f"✓ cLLM: 成功{cllm_result['successful_requests']}/72, "
                      f"吞吐量={cllm_result['avg_throughput']:.2f} t/s")
            
            time.sleep(2)  # 短暂休息
            
            # Ollama测试
            ollama_result = run_benchmark("ollama", concurrency, round_num)
            if not ollama_result.get("success"):
                print(f"⚠️ Ollama测试失败: {ollama_result.get('error', 'Unknown error')}")
                ollama_result = {"success": False, "failed": True}
            else:
                print(f"✓ Ollama: 成功{ollama_result['successful_requests']}/72, "
                      f"吞吐量={ollama_result['avg_throughput']:.2f} t/s")
            
            round_results[concurrency] = {
                "cllm": cllm_result,
                "ollama": ollama_result
            }
            
            time.sleep(3)  # 轮次间休息
        
        all_results[f"round_{round_num}"] = round_results
        
        # 保存中间结果
        output_file = f"/tmp/multi_round_comparison_round{round_num}.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ 第{round_num}轮结果已保存到: {output_file}")
    
    # 生成最终报告
    print(f"\n{'='*60}")
    print("测试完成！生成汇总报告...")
    print(f"{'='*60}\n")
    
    # 计算平均值
    summary = {}
    for concurrency in concurrency_levels:
        cllm_throughputs = []
        ollama_throughputs = []
        cllm_success_rates = []
        ollama_success_rates = []
        
        for round_num in range(1, min_rounds + 1):
            round_key = f"round_{round_num}"
            if round_key in all_results and concurrency in all_results[round_key]:
                cllm = all_results[round_key][concurrency].get("cllm", {})
                ollama = all_results[round_key][concurrency].get("ollama", {})
                
                if cllm.get("success") and "avg_throughput" in cllm:
                    cllm_throughputs.append(cllm["avg_throughput"])
                    success_rate = (cllm["successful_requests"] / 72) * 100
                    cllm_success_rates.append(success_rate)
                
                if ollama.get("success") and "avg_throughput" in ollama:
                    ollama_throughputs.append(ollama["avg_throughput"])
                    success_rate = (ollama["successful_requests"] / 72) * 100
                    ollama_success_rates.append(success_rate)
        
        summary[concurrency] = {
            "cllm": {
                "avg_throughput": sum(cllm_throughputs) / len(cllm_throughputs) if cllm_throughputs else 0,
                "avg_success_rate": sum(cllm_success_rates) / len(cllm_success_rates) if cllm_success_rates else 0,
                "rounds": len(cllm_throughputs)
            },
            "ollama": {
                "avg_throughput": sum(ollama_throughputs) / len(ollama_throughputs) if ollama_throughputs else 0,
                "avg_success_rate": sum(ollama_success_rates) / len(ollama_success_rates) if ollama_success_rates else 0,
                "rounds": len(ollama_throughputs)
            }
        }
    
    # 打印汇总
    print("汇总结果（3轮平均值）:")
    print(f"{'并发数':<8} {'cLLM吞吐量':<15} {'Ollama吞吐量':<15} {'cLLM成功率':<12} {'Ollama成功率':<12} {'优势方':<10}")
    print("-" * 80)
    for concurrency in concurrency_levels:
        s = summary[concurrency]
        cllm_tp = s["cllm"]["avg_throughput"]
        ollama_tp = s["ollama"]["avg_throughput"]
        cllm_sr = s["cllm"]["avg_success_rate"]
        ollama_sr = s["ollama"]["avg_success_rate"]
        winner = "cLLM" if cllm_tp > ollama_tp else "Ollama"
        
        print(f"{concurrency:<8} {cllm_tp:<15.2f} {ollama_tp:<15.2f} {cllm_sr:<12.1f}% {ollama_sr:<12.1f}% {winner:<10}")
    
    # 保存最终结果
    final_output = f"/tmp/multi_round_comparison_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_output, 'w') as f:
        json.dump({
            "summary": summary,
            "all_results": all_results,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2)
    
    print(f"\n✓ 最终结果已保存到: {final_output}")
    print("\n测试完成！")

if __name__ == "__main__":
    main()
