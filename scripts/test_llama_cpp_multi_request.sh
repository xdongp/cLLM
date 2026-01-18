#!/bin/bash
# 测试 llama.cpp 后端的多次请求处理
# 验证内存泄漏、并发安全和位置管理的修复

set -e

MODEL_PATH="${1:-model/Qwen/qwen3-0.6b-q4_k_m.gguf}"
PORT="${2:-18081}"
BUILD_DIR="${BUILD_DIR:-build}"

echo "========================================="
echo "测试 LlamaCppBackend 多次请求处理"
echo "========================================="
echo "模型路径: $MODEL_PATH"
echo "端口: $PORT"
echo "构建目录: $BUILD_DIR"
echo ""

# 检查模型文件
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    echo "请提供正确的模型路径作为第一个参数"
    exit 1
fi

# 检查可执行文件
SERVER_BIN="$BUILD_DIR/bin/cllm_server"
if [ ! -f "$SERVER_BIN" ]; then
    echo "❌ 错误: 服务器可执行文件不存在: $SERVER_BIN"
    echo "请先编译: cd $BUILD_DIR && make cllm_server"
    exit 1
fi

echo "✅ 准备就绪，启动服务器..."
echo ""

# 启动服务器（后台运行）
CLLM_LOG_LEVEL=info $SERVER_BIN --model-path "$MODEL_PATH" --port "$PORT" > /tmp/cllm_server_test.log 2>&1 &
SERVER_PID=$!

# 等待服务器启动
echo "等待服务器启动..."
for i in {1..10}; do
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" | grep -q "200"; then
        echo "✅ 服务器已启动 (PID: $SERVER_PID)"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ 服务器启动超时"
        kill $SERVER_PID 2>/dev/null || true
        cat /tmp/cllm_server_test.log
        exit 1
    fi
    sleep 1
done

echo ""
echo "========================================="
echo "开始多次请求测试"
echo "========================================="

# 测试函数
test_request() {
    local request_num=$1
    local prompt="$2"
    echo "[请求 #$request_num] 发送: \"$prompt\""
    
    local response=$(curl -s -X POST "http://localhost:$PORT/generate" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$prompt\", \"max_tokens\": 10}" \
        -w "\nHTTP_CODE:%{http_code}")
    
    local http_code=$(echo "$response" | grep "HTTP_CODE:" | cut -d: -f2)
    local body=$(echo "$response" | sed '/HTTP_CODE:/d')
    
    if [ "$http_code" = "200" ]; then
        echo "  ✅ 成功 (HTTP $http_code)"
        echo "  响应: $(echo "$body" | head -c 100)..."
        return 0
    else
        echo "  ❌ 失败 (HTTP $http_code)"
        echo "  响应: $body"
        return 1
    fi
}

# 执行多次请求测试
FAILED=0
TOTAL=5

test_request 1 "Hello, how are you?" || FAILED=$((FAILED + 1))
sleep 1

test_request 2 "What is artificial intelligence?" || FAILED=$((FAILED + 1))
sleep 1

test_request 3 "Tell me a joke." || FAILED=$((FAILED + 1))
sleep 1

test_request 4 "Explain quantum computing." || FAILED=$((FAILED + 1))
sleep 1

test_request 5 "What is machine learning?" || FAILED=$((FAILED + 1))

echo ""
echo "========================================="
echo "测试结果"
echo "========================================="
echo "总请求数: $TOTAL"
echo "成功: $((TOTAL - FAILED))"
echo "失败: $FAILED"

# 检查服务器是否仍在运行
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo ""
    echo "❌ 服务器已崩溃!"
    echo "查看日志:"
    tail -50 /tmp/cllm_server_test.log
    exit 1
fi

echo ""
echo "✅ 服务器仍在运行 (PID: $SERVER_PID)"

# 停止服务器
echo ""
echo "停止服务器..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo ""
if [ $FAILED -eq 0 ]; then
    echo "🎉 所有测试通过!"
    exit 0
else
    echo "⚠️  部分测试失败 ($FAILED/$TOTAL)"
    exit 1
fi
