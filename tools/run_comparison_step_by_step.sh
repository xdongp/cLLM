#!/bin/bash
# 分步骤运行对比测试，避免超时

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "=========================================="
echo "cLLM vs Ollama 多轮对比测试"
echo "=========================================="
echo ""
echo "测试配置:"
echo "  - 请求数量: 72"
echo "  - 每个请求最大tokens: 50"
echo "  - 并发数: 8, 16, 24, 32, 40"
echo "  - 测试轮数: 3轮"
echo ""

# 检查服务器
echo "检查服务器状态..."
if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "❌ cLLM服务器未运行"
    exit 1
fi
echo "✓ cLLM服务器运行正常"

if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "❌ Ollama服务器未运行"
    exit 1
fi
echo "✓ Ollama服务器运行正常"
echo ""

# 并发级别
CONCURRENCIES=(8 16 24 32 40)
ROUNDS=3

# 创建结果目录
mkdir -p /tmp/comparison_results

# 运行测试
for round in $(seq 1 $ROUNDS); do
    echo ""
    echo "########################################"
    echo "第 $round 轮测试"
    echo "########################################"
    echo ""
    
    for conc in "${CONCURRENCIES[@]}"; do
        echo "--- 并发 $conc ---"
        
        # cLLM测试
        echo "  [cLLM] 运行中..."
        cllm_output="/tmp/comparison_results/cllm_r${round}_c${conc}.json"
        python3 tools/unified_benchmark.py \
            --server-type cllm \
            --server-url http://localhost:8080 \
            --test-type api-concurrent \
            --requests 72 \
            --concurrency $conc \
            --max-tokens 50 \
            --output-file "$cllm_output" 2>&1 | tail -5
        
        if [ -f "$cllm_output" ]; then
            cllm_stats=$(python3 -c "
import json, sys
try:
    with open('$cllm_output') as f:
        d = json.load(f)
        c = d.get('concurrent', {})
        print(f\"成功: {c.get('successful_requests', 0)}/72, 吞吐量: {c.get('avg_throughput', 0):.2f} t/s\")
except:
    print('解析失败')
")
            echo "  [cLLM] $cllm_stats"
        else
            echo "  [cLLM] ❌ 测试失败"
        fi
        
        sleep 2
        
        # Ollama测试
        echo "  [Ollama] 运行中..."
        ollama_output="/tmp/comparison_results/ollama_r${round}_c${conc}.json"
        python3 tools/unified_benchmark.py \
            --server-type ollama \
            --server-url http://localhost:11434 \
            --model qwen3:0.6b \
            --test-type api-concurrent \
            --requests 72 \
            --concurrency $conc \
            --max-tokens 50 \
            --output-file "$ollama_output" 2>&1 | tail -5
        
        if [ -f "$ollama_output" ]; then
            ollama_stats=$(python3 -c "
import json, sys
try:
    with open('$ollama_output') as f:
        d = json.load(f)
        c = d.get('concurrent', {})
        print(f\"成功: {c.get('successful_requests', 0)}/72, 吞吐量: {c.get('avg_throughput', 0):.2f} t/s\")
except:
    print('解析失败')
")
            echo "  [Ollama] $ollama_stats"
        else
            echo "  [Ollama] ❌ 测试失败"
        fi
        
        echo ""
        sleep 3
    done
    
    echo "✓ 第${round}轮测试完成"
    echo ""
done

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "结果文件保存在: /tmp/comparison_results/"
echo ""
