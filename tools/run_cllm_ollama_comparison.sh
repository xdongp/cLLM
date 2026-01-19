#!/bin/bash

# cLLM vs Ollama 性能对比测试脚本
# 测试方案: 160请求，5并发，50 tokens
# 适用场景: 大规模性能对比测试

set -e

echo "========================================"
echo "cLLM vs Ollama 性能对比测试"
echo "测试方案: 160请求，5并发，50 tokens"
echo "========================================"
echo ""

mkdir -p results

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到python3，请先安装Python 3"
    exit 1
fi

# 测试参数
REQUESTS=160
CONCURRENCY=5
MAX_TOKENS=50
MODEL="qwen3:0.6b"
CLLM_URL=${CLLM_URL:-http://localhost:8080}
OLLAMA_URL=${OLLAMA_URL:-http://localhost:11434}

# 检查cLLM服务器是否运行
echo "[1/4] 检查cLLM服务器状态..."
if curl -s "$CLLM_URL/health" &> /dev/null; then
    echo "✓ cLLM服务器运行正常 ($CLLM_URL)"
else
    echo "✗ cLLM服务器未运行，请先启动cLLM服务器"
    echo "  启动命令: ./build/bin/cllm_server --config config/config_gpu.yaml"
    exit 1
fi
echo ""

# 检查Ollama服务器是否运行
echo "[2/4] 检查Ollama服务器状态..."
if curl -s "$OLLAMA_URL/api/tags" &> /dev/null; then
    echo "✓ Ollama服务器运行正常 ($OLLAMA_URL)"
else
    echo "✗ Ollama服务器未运行，请先启动Ollama"
    echo "  启动命令: ollama serve"
    exit 1
fi
echo ""

echo "测试参数:"
echo "  请求数: $REQUESTS"
echo "  并发数: $CONCURRENCY"
echo "  最大token数: $MAX_TOKENS"
echo "  Ollama模型: $MODEL"
echo ""

# 运行cLLM测试
echo "[3/4] 运行cLLM性能测试..."
echo "  测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  测试类型: 顺序测试 + 并发测试"
echo "  参数: $REQUESTS请求, $CONCURRENCY并发, $MAX_TOKENS tokens"
echo ""

python3 tools/cllm_optimized_benchmark.py \
    --server-url "$CLLM_URL" \
    --test-type all \
    --requests $REQUESTS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS \
    --output-file "results/cllm_test_results_$(date '+%Y%m%d_%H%M%S').json"

echo ""
echo "✓ cLLM测试完成"
echo ""

# 运行Ollama测试
echo "[4/4] 运行Ollama性能测试..."
echo "  测试时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "  测试类型: 顺序测试 + 并发测试"
echo "  参数: $REQUESTS请求, $CONCURRENCY并发, $MAX_TOKENS tokens"
echo ""

python3 tools/ollama_benchmark.py \
    --server-url "$OLLAMA_URL" \
    --model $MODEL \
    --test-type all \
    --requests $REQUESTS \
    --concurrency $CONCURRENCY \
    --max-tokens $MAX_TOKENS \
    --output-file "results/ollama_test_results_$(date '+%Y%m%d_%H%M%S').json"

echo ""
echo "✓ Ollama测试完成"
echo ""


# 总结
echo "========================================"
echo "测试完成！"
echo "========================================"
echo ""
echo "测试结果:"
echo "  - cLLM: 已完成 $REQUESTS 个请求的顺序和并发测试"
echo "  - Ollama: 已完成 $REQUESTS 个请求的顺序和并发测试"
echo ""
echo "查看详细结果:"
echo "  - 控制台输出（已显示）"
echo "  - JSON结果文件: results/"
echo "  - 测试报告: $REPORT_FILE"
echo ""
echo "下次测试直接运行:"
echo "  bash tools/run_cllm_ollama_comparison.sh"
echo ""
