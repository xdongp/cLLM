#!/bin/bash
# Stage 2-3 详细验证测试脚本

MODEL_PATH="${1:-/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf}"
PROMPT="${2:-Hi}"
LOG_FILE="/tmp/kylin_stage2_3_test.log"

echo "=========================================="
echo "Stage 2-3 详细验证测试"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Prompt: $PROMPT"
echo "Log file: $LOG_FILE"
echo ""

# 确保使用 Kylin 后端和 CPU
echo "--- 配置检查 ---"
grep -A 2 "backend:" config/config.yaml | head -3

# 停止现有服务器
pkill -f cllm_server 2>/dev/null
sleep 2

# 启动服务器
echo ""
echo "--- 启动服务器 ---"
./build/bin/cllm_server --config config/config.yaml > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务器启动
echo "等待服务器启动..."
sleep 12

# 测试请求
echo ""
echo "--- 发送测试请求 ---"
RESPONSE=$(curl -s -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"$PROMPT\", \"max_tokens\": 1, \"temperature\": 0.0}")

echo "Response: $RESPONSE" | jq -r '.data.text // "ERROR"'

# 提取 Stage 2 信息
echo ""
echo "=========================================="
echo "Stage 2: Token Embedding 验证"
echo "=========================================="
grep "Stage 2: Embedding" "$LOG_FILE" -A 15 | head -20

# 提取 Stage 3 信息
echo ""
echo "=========================================="
echo "Stage 3: Layer 0 详细验证"
echo "=========================================="
grep "Stage 3" "$LOG_FILE" -A 15 | head -80

# 检查是否有错误
echo ""
echo "=========================================="
echo "错误检查"
echo "=========================================="
ERROR_COUNT=$(grep -i "error\|exception\|failed" "$LOG_FILE" | wc -l)
if [ "$ERROR_COUNT" -gt 0 ]; then
    echo "⚠️  发现 $ERROR_COUNT 个错误/异常："
    grep -i "error\|exception\|failed" "$LOG_FILE" | head -10
else
    echo "✅ 未发现错误"
fi

# 停止服务器
pkill -f cllm_server 2>/dev/null

echo ""
echo "=========================================="
echo "测试完成"
echo "=========================================="
echo "详细日志: $LOG_FILE"
echo ""
