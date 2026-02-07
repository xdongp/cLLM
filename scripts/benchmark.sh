#!/bin/bash
# 简单性能测试脚本

BASE_URL="http://localhost:18080"

echo "=========================================="
echo "cLLM Performance Benchmark"
echo "=========================================="

# 测试健康检查
echo -e "\n1. Health Check:"
curl -s "$BASE_URL/health" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/health"

# 测试模型信息
echo -e "\n2. Model Info:"
curl -s "$BASE_URL/model/info" | python3 -m json.tool 2>/dev/null || curl -s "$BASE_URL/model/info"

# 测试编码
echo -e "\n3. Encode Test:"
time curl -s -X POST "$BASE_URL/encode" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!"}' | python3 -m json.tool 2>/dev/null

# 短文本生成测试
echo -e "\n4. Short Generation (10 tokens):"
time curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "hello", "max_tokens": 10}' | python3 -m json.tool 2>/dev/null

# 中等文本生成测试
echo -e "\n5. Medium Generation (50 tokens):"
time curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "请介绍一下人工智能", "max_tokens": 50}' | python3 -m json.tool 2>/dev/null

# 长文本生成测试
echo -e "\n6. Long Generation (100 tokens):"
time curl -s -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "请介绍一下人工智能的发展历程", "max_tokens": 100}' | python3 -m json.tool 2>/dev/null

echo -e "\n=========================================="
echo "Benchmark completed!"
echo "=========================================="
