#!/bin/bash
# GGUF 端到端测试 - 使用 Kylin Backend

MODEL_PATH="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"
TOKENIZER_DIR="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B"
PORT=8080

echo "=========================================="
echo "GGUF 格式端到端测试 (Kylin Backend)"
echo "=========================================="
echo "模型: $MODEL_PATH"
echo "Tokenizer: $TOKENIZER_DIR"
echo "端口: $PORT"
echo ""

# 查找可执行文件
if [ -f "./build/bin/cllm_server" ]; then
    SERVER_BIN="./build/bin/cllm_server"
elif [ -f "./bin/cllm_server" ]; then
    SERVER_BIN="./bin/cllm_server"
else
    echo "❌ 错误: 找不到 cllm_server 可执行文件"
    echo "请先编译: cd build && cmake .. && make"
    exit 1
fi

echo "✅ 使用服务器: $SERVER_BIN"
echo ""

# 启动服务器
echo "启动服务器..."
$SERVER_BIN \
    --model-path "$MODEL_PATH" \
    --port $PORT \
    --host 0.0.0.0 \
    --log-level info \
    > /tmp/cllm_gguf_test.log 2>&1 &

SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"
echo "日志: /tmp/cllm_gguf_test.log"
echo ""

# 等待服务器启动
echo "等待服务器启动..."
for i in {1..15}; do
    sleep 1
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器已启动 (${i}秒)"
        break
    fi
    if [ $i -eq 15 ]; then
        echo "❌ 服务器启动超时"
        echo "查看日志:"
        tail -30 /tmp/cllm_gguf_test.log
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

echo ""
echo "=========================================="
echo "测试 /generate 接口"
echo "=========================================="
echo "输入: hello"
echo ""

# 测试生成
RESPONSE=$(curl -s -X POST http://localhost:$PORT/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "hello",
        "max_tokens": 50,
        "temperature": 0.7
    }')

echo "响应:"
echo "$RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$RESPONSE"
echo ""

# 停止服务器
echo "停止服务器..."
kill $SERVER_PID 2>/dev/null
sleep 2

echo "✅ 测试完成"
echo "完整日志: /tmp/cllm_gguf_test.log"
