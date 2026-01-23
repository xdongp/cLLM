#!/bin/bash
# 简单的后端对比测试脚本

PROMPT="${1:-Hi}"
MAX_TOKENS="${2:-3}"
TEMP="${3:-0.0}"

echo "=========================================="
echo "Simple Backend Comparison Test"
echo "=========================================="
echo "Prompt: '$PROMPT'"
echo "Max tokens: $MAX_TOKENS, Temperature: $TEMP"
echo ""

# 测试 Kylin 后端
echo "--- Testing Kylin Backend ---"
sed -i.bak 's/type: ".*"/type: "kylin"/' config/config.yaml
pkill -f cllm_server 2>/dev/null
sleep 2
./build/bin/cllm_server --config config/config.yaml > /tmp/kylin_test.log 2>&1 &
sleep 12

KYLIN_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  | jq -r '.data.text // "ERROR"')

echo "Kylin output: $KYLIN_RESULT"
echo ""

# 提取调试信息
echo "Kylin Debug Info:"
grep "Kylin Debug\|Embedding\|Layer 0" /tmp/kylin_test.log | tail -5
echo ""

# 测试 llama_cpp 后端
echo "--- Testing llama_cpp Backend ---"
sed -i.bak 's/type: ".*"/type: "llama_cpp"/' config/config.yaml
pkill -f cllm_server 2>/dev/null
sleep 2
./build/bin/cllm_server --config config/config.yaml > /tmp/llama_test.log 2>&1 &
sleep 12

LLAMA_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  | jq -r '.data.text // "ERROR"')

echo "llama_cpp output: $LLAMA_RESULT"
echo ""

# 对比结果
echo "=========================================="
echo "Comparison:"
echo "=========================================="
echo "Kylin:     $KYLIN_RESULT"
echo "llama_cpp: $LLAMA_RESULT"
if [ "$KYLIN_RESULT" = "$LLAMA_RESULT" ]; then
    echo "✅ Results match!"
else
    echo "❌ Results differ"
fi
echo ""

# 恢复配置
mv config/config.yaml.bak config/config.yaml 2>/dev/null || true
