#!/bin/bash
# GGUF 格式端到端测试脚本
# 测试模型: qwen3-0.6b-q8_0.gguf
# 后端: libtorch (如果支持) 或 kylin
# Tokenizer: tokenizers-cpp (HFTokenizer)

set -e

MODEL_PATH="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf"
TOKENIZER_DIR="/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B"
SERVER_PORT=8080
SERVER_HOST="0.0.0.0"
USE_LIBTORCH=false

echo "=========================================="
echo "GGUF 格式端到端测试"
echo "=========================================="
echo "模型路径: $MODEL_PATH"
echo "Tokenizer 目录: $TOKENIZER_DIR"
echo "后端: $([ "$USE_LIBTORCH" = "true" ] && echo "LibTorch" || echo "Kylin")"
echo "服务器地址: http://$SERVER_HOST:$SERVER_PORT"
echo ""

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型文件不存在: $MODEL_PATH"
    exit 1
fi

# 检查 tokenizer 目录是否存在
if [ ! -d "$TOKENIZER_DIR" ]; then
    echo "❌ 错误: Tokenizer 目录不存在: $TOKENIZER_DIR"
    exit 1
fi

# 检查可执行文件
BIN_PATH="./build/bin/cllm_server"
if [ ! -f "$BIN_PATH" ]; then
    BIN_PATH="./bin/cllm_server"
    if [ ! -f "$BIN_PATH" ]; then
        echo "❌ 错误: 找不到可执行文件 cllm_server"
        echo "请先编译项目: cd build && cmake .. && make"
        exit 1
    fi
fi

echo "✅ 找到可执行文件: $BIN_PATH"
echo ""

# 启动服务器（后台运行）
echo "启动服务器..."
if [ "$USE_LIBTORCH" = "true" ]; then
    $BIN_PATH \
        --model-path "$MODEL_PATH" \
        --use-libtorch \
        --port $SERVER_PORT \
        --host "$SERVER_HOST" \
        --log-level info \
        > /tmp/cllm_server_test.log 2>&1 &
else
    $BIN_PATH \
        --model-path "$MODEL_PATH" \
        --port $SERVER_PORT \
        --host "$SERVER_HOST" \
        --log-level info \
        > /tmp/cllm_server_test.log 2>&1 &
fi

SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"
echo "日志文件: /tmp/cllm_server_test.log"
echo ""

# 等待服务器启动
echo "等待服务器启动..."
for i in {1..10}; do
    sleep 1
    if curl -s http://localhost:$SERVER_PORT/health > /dev/null 2>&1; then
        echo "✅ 服务器已启动 (等待了 ${i} 秒)"
        break
    fi
    if [ $i -eq 10 ]; then
        echo "❌ 服务器启动超时"
        echo "查看日志:"
        tail -30 /tmp/cllm_server_test.log
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

# 测试健康检查端点
echo ""
echo "测试健康检查端点..."
HEALTH_RESPONSE=$(curl -s http://localhost:$SERVER_PORT/health || echo "FAILED")
if [ "$HEALTH_RESPONSE" = "FAILED" ]; then
    echo "❌ 健康检查失败"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo "✅ 健康检查通过: $HEALTH_RESPONSE"
echo ""

# 测试 /generate 接口
echo "=========================================="
echo "测试 /generate 接口"
echo "=========================================="
echo "输入: hello"
echo ""

GENERATE_RESPONSE=$(curl -s -X POST http://localhost:$SERVER_PORT/generate \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "hello",
        "max_tokens": 50,
        "temperature": 0.7
    }' || echo "FAILED")

if [ "$GENERATE_RESPONSE" = "FAILED" ]; then
    echo "❌ /generate 请求失败"
    echo "查看服务器日志:"
    tail -30 /tmp/cllm_server_test.log
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

echo "响应:"
echo "$GENERATE_RESPONSE" | python3 -m json.tool 2>/dev/null || echo "$GENERATE_RESPONSE"
echo ""

# 提取生成的文本
GENERATED_TEXT=$(echo "$GENERATE_RESPONSE" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('text', data.get('generated_text', 'N/A')))" 2>/dev/null || echo "N/A")

if [ "$GENERATED_TEXT" != "N/A" ] && [ -n "$GENERATED_TEXT" ]; then
    echo "✅ 生成成功!"
    echo "生成的文本: $GENERATED_TEXT"
else
    echo "⚠️  警告: 无法解析生成的文本"
    echo "完整响应: $GENERATE_RESPONSE"
fi

echo ""

# 停止服务器
echo "停止服务器..."
kill $SERVER_PID 2>/dev/null || true
sleep 2

if kill -0 $SERVER_PID 2>/dev/null; then
    echo "⚠️  服务器未正常退出，强制终止..."
    kill -9 $SERVER_PID 2>/dev/null || true
fi

echo "✅ 测试完成"
echo ""
echo "服务器日志保存在: /tmp/cllm_server_test.log"
echo ""
echo "查看完整日志: cat /tmp/cllm_server_test.log"
