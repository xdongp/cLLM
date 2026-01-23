#!/bin/bash
# 完整的后端对比测试脚本
# 自动测试两个后端，提取调试信息，并生成对比报告

PROMPT="${1:-Hi}"
MAX_TOKENS="${2:-1}"
TEMP="${3:-0.0}"

KYLIN_LOG="/tmp/kylin_full_comparison.log"
LLAMA_LOG="/tmp/llama_full_comparison.log"
REPORT_FILE="/tmp/comparison_report.txt"

echo "=========================================="
echo "Full Backend Comparison Test"
echo "=========================================="
echo "Prompt: '$PROMPT'"
echo "Max tokens: $MAX_TOKENS, Temperature: $TEMP"
echo ""

# 清理旧日志
rm -f "$KYLIN_LOG" "$LLAMA_LOG" "$REPORT_FILE"

# ========== 测试 Kylin 后端 ==========
echo "--- Step 1: Testing Kylin Backend ---"
sed -i.bak 's/type: ".*"/type: "kylin"/' config/config.yaml
pkill -f cllm_server 2>/dev/null
sleep 2

./build/bin/cllm_server --config config/config.yaml > "$KYLIN_LOG" 2>&1 &
KYLIN_PID=$!
sleep 12

echo "Sending request to Kylin backend..."
KYLIN_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  2>/dev/null | jq -r '.data.text // "ERROR"')

pkill -f cllm_server 2>/dev/null
sleep 2

echo "Kylin output: $KYLIN_RESULT"
echo ""

# ========== 测试 llama_cpp 后端 ==========
echo "--- Step 2: Testing llama_cpp Backend ---"
sed -i.bak 's/type: ".*"/type: "llama_cpp"/' config/config.yaml
./build/bin/cllm_server --config config/config.yaml > "$LLAMA_LOG" 2>&1 &
LLAMA_PID=$!
sleep 12

echo "Sending request to llama_cpp backend..."
LLAMA_RESULT=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": $MAX_TOKENS, \"temperature\": $TEMP}" \
  2>/dev/null | jq -r '.data.text // "ERROR"')

pkill -f cllm_server 2>/dev/null
sleep 2

echo "llama_cpp output: $LLAMA_RESULT"
echo ""

# ========== 生成对比报告 ==========
echo "--- Step 3: Generating Comparison Report ---"
{
    echo "=========================================="
    echo "Backend Comparison Report"
    echo "=========================================="
    echo "Test Parameters:"
    echo "  Prompt: '$PROMPT'"
    echo "  Max tokens: $MAX_TOKENS"
    echo "  Temperature: $TEMP"
    echo ""
    echo "Output Comparison:"
    echo "  Kylin:     $KYLIN_RESULT"
    echo "  llama_cpp: $LLAMA_RESULT"
    if [ "$KYLIN_RESULT" = "$LLAMA_RESULT" ]; then
        echo "  Status: ✅ Results match"
    else
        echo "  Status: ❌ Results differ"
    fi
    echo ""
    echo "=========================================="
    echo "Detailed Statistics (Kylin)"
    echo "=========================================="
    echo ""
} > "$REPORT_FILE"

# 使用 Python 分析工具生成详细报告
python3 tools/analyze_debug_logs.py "$KYLIN_LOG" >> "$REPORT_FILE" 2>&1

# 添加日志文件位置
{
    echo ""
    echo "=========================================="
    echo "Log Files"
    echo "=========================================="
    echo "Kylin log:     $KYLIN_LOG"
    echo "llama_cpp log: $LLAMA_LOG"
    echo "Report file:   $REPORT_FILE"
    echo "=========================================="
} >> "$REPORT_FILE"

# 显示报告
cat "$REPORT_FILE"

# 恢复配置
mv config/config.yaml.bak config/config.yaml 2>/dev/null || true

echo ""
echo "=========================================="
echo "Report saved to: $REPORT_FILE"
echo "=========================================="
