#!/bin/bash
# 功能测试脚本 - 验证模型输出内容

BASE_URL="http://localhost:18080"

echo "=========================================="
echo "cLLM 功能测试 - 模型输出验证"
echo "=========================================="

# 测试1: 简单的 hello
echo -e "\n【测试1】简单问候: hello"
echo "请求: max_tokens=20"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "max_tokens": 20}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# 测试2: 中文问候
echo -e "\n【测试2】中文问候: 你好"
echo "请求: max_tokens=30"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "max_tokens": 30}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# 测试3: 介绍人工智能
echo -e "\n【测试3】介绍人工智能"
echo "请求: max_tokens=100"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "请介绍人工智能", "max_tokens": 100}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# 测试4: 更详细的AI发展历程
echo -e "\n【测试4】AI发展历程"
echo "请求: max_tokens=150"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "请介绍一下人工智能的发展历程，包括重要的里程碑事件", "max_tokens": 150}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# 测试5: 代码相关问题
echo -e "\n【测试5】编程问题"
echo "请求: max_tokens=80"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "什么是Python编程语言？", "max_tokens": 80}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

# 测试6: 数学问题
echo -e "\n【测试6】数学问题"
echo "请求: max_tokens=60"
RESULT=$(curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "1+1等于几？请解释原因。", "max_tokens": 60}')
echo "响应:"
echo "$RESULT" | python3 -m json.tool 2>/dev/null || echo "$RESULT"

echo -e "\n=========================================="
echo "功能测试完成!"
echo "=========================================="
