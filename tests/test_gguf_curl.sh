#!/bin/bash
# 简单的 curl 测试脚本
# 用于测试 /generate 接口

SERVER_URL="http://localhost:8080"

echo "测试 /generate 接口"
echo "输入: hello"
echo ""

curl -X POST "$SERVER_URL/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "prompt": "hello",
        "max_tokens": 50,
        "temperature": 0.7
    }' | python3 -m json.tool

echo ""
