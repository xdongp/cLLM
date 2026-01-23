#!/bin/bash
# Kylin 分阶段测试脚本

MODEL_PATH="${1:-/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf}"
PROMPT="${2:-Hi}"
MAX_TOKENS="${3:-5}"
TEMP="${4:-0.0}"

echo "=========================================="
echo "Kylin Backend Stage-by-Stage Test"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_TOKENS"
echo "Temperature: $TEMP"
echo ""

# 编译测试程序
echo "--- Compiling test program ---"
cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM
if [ ! -f "build/tools/kylin_stage_test" ]; then
    echo "Building kylin_stage_test..."
    # 需要添加到 CMakeLists.txt
    make -j4 2>&1 | tail -5
fi

# 运行测试
echo ""
echo "--- Running Stage Tests ---"
if [ -f "build/tools/kylin_stage_test" ]; then
    ./build/tools/kylin_stage_test "$MODEL_PATH" "$PROMPT" "$MAX_TOKENS" "$TEMP"
else
    echo "⚠️  Test program not found. Using manual stage testing..."
    
    # 手动运行各阶段测试
    echo ""
    echo "Stage 0-1: Basic Environment & Model Loading"
    echo "--------------------------------------------"
    pkill -f cllm_server 2>/dev/null
    ./build/bin/cllm_server --config config/config.yaml > /tmp/kylin_stage_test.log 2>&1 &
    sleep 12
    
    echo ""
    echo "Stage 2: Token Embedding"
    echo "--------------------------------------------"
    curl -s -X POST http://localhost:8080/generate \
      -H "Content-Type: application/json" \
      -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 1, \"temperature\": $TEMP}" \
      | jq -r '.data.text // "ERROR"'
    
    echo ""
    echo "Extracting debug info..."
    grep "\[Kylin Debug\]" /tmp/kylin_stage_test.log | head -10
    
    pkill -f cllm_server 2>/dev/null
fi

echo ""
echo "=========================================="
echo "Test completed. Check logs for details."
echo "=========================================="
