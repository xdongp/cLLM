#!/bin/bash

cd /Users/dannypan/PycharmProjects/xllm/cpp/cLLM

# 清理旧进程
pkill -9 cllm_server 2>/dev/null
sleep 1

# 启动服务器并在后台运行
./build/bin/cllm_server --model-path model/Qwen/qwen3-0.6b-q4_k_m.gguf --port 8080 --log-level debug > /tmp/debug_server.log 2>&1 &
SERVER_PID=$!

echo "服务器 PID: $SERVER_PID"
echo "等待服务器启动..."
sleep 10

# 测试健康检查
echo ""
echo "=== Health Check ==="
curl -s http://localhost:8080/health | python3 -m json.tool

# 测试生成接口（简短prompt）
echo ""
echo "=== Test Generate with 'hi' ==="
curl -v -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hi", "max_tokens": 5, "temperature": 0.7}' 2>&1

echo ""
echo ""
echo "=== Server Log (last 100 lines) ==="
tail -100 /tmp/debug_server.log

echo ""
echo "清理服务器进程..."
kill $SERVER_PID 2>/dev/null
