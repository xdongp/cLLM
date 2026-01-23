#!/bin/bash
# 对比 Kylin 和 llama_cpp 后端的调试输出

PROMPT="${1:-Hi}"
MAX_TOKENS="${2:-1}"
TEMP="${3:-0.0}"

KYLIN_LOG="/tmp/kylin_debug.log"
LLAMA_LOG="/tmp/llama_debug.log"

echo "=========================================="
echo "Debug Output Comparison Tool"
echo "=========================================="
echo "Prompt: '$PROMPT'"
echo "Max tokens: $MAX_TOKENS, Temperature: $TEMP"
echo ""

# 测试 Kylin 后端
echo "--- Testing Kylin Backend ---"
sed -i.bak 's/type: ".*"/type: "kylin"/' config/config.yaml
pkill -f cllm_server 2>/dev/null
sleep 2
./build/bin/cllm_server --config config/config.yaml > "$KYLIN_LOG" 2>&1 &
KYLIN_PID=$!
sleep 12

KYLIN_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  2>/dev/null | jq -r '.data.text // "ERROR"')

pkill -f cllm_server 2>/dev/null
sleep 2

echo "Kylin output: $KYLIN_RESULT"
echo ""

# 提取 Kylin 调试信息
echo "=== Kylin Debug Info ==="
echo "Embedding:"
grep "\[Kylin Debug\] Embedding" "$KYLIN_LOG" | tail -3
echo ""
echo "Layer 0 Output:"
grep "\[Kylin Debug\] Layer 0" "$KYLIN_LOG" | tail -3
echo ""

# 测试 llama_cpp 后端
echo "--- Testing llama_cpp Backend ---"
sed -i.bak 's/type: ".*"/type: "llama_cpp"/' config/config.yaml
./build/bin/cllm_server --config config/config.yaml > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
sleep 12

LLAMA_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  2>/dev/null | jq -r '.data.text // "ERROR"')

pkill -f cllm_server 2>/dev/null
sleep 2

echo "llama_cpp output: $LLAMA_RESULT"
echo ""

# 对比结果
echo "=========================================="
echo "Comparison Summary"
echo "=========================================="
echo "Outputs:"
echo "  Kylin:     $KYLIN_RESULT"
echo "  llama_cpp: $LLAMA_RESULT"
if [ "$KYLIN_RESULT" = "$LLAMA_RESULT" ]; then
    echo "  ✅ Results match!"
else
    echo "  ❌ Results differ"
fi
echo ""

# 提取并对比统计信息
echo "=== Detailed Statistics Comparison ==="
echo ""
echo "Embedding Statistics:"
echo "  Kylin:"
grep "\[Kylin Debug\] Embedding stats" "$KYLIN_LOG" | tail -1 | sed 's/^/    /'
echo "  llama_cpp: (not available - using llama.cpp internal API)"
echo ""

echo "Layer 0 Output Statistics:"
echo "  Kylin:"
grep "\[Kylin Debug\] Layer 0 output stats" "$KYLIN_LOG" | tail -1 | sed 's/^/    /'
echo "  llama_cpp: (not available - using llama.cpp internal API)"
echo ""

echo "First 10 Values:"
echo "  Kylin Embedding:"
grep "\[Kylin Debug\] Embedding first" "$KYLIN_LOG" | tail -1 | sed 's/^/    /'
echo "  Kylin Layer 0:"
grep "\[Kylin Debug\] Layer 0 output first" "$KYLIN_LOG" | tail -1 | sed 's/^/    /'
echo ""

# 恢复配置
mv config/config.yaml.bak config/config.yaml 2>/dev/null || true

echo "=========================================="
echo "Full logs saved to:"
echo "  Kylin:     $KYLIN_LOG"
echo "  llama_cpp: $LLAMA_LOG"
echo "=========================================="
