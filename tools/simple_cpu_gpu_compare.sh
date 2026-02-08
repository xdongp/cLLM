#!/bin/bash
# 简单的 CPU vs GPU 对比脚本

echo "========================================="
echo "CPU vs GPU 简单对比测试"
echo "========================================="

cd /Users/dannypan/PycharmProjects/cLLM/build

echo ""
echo "【测试1】输入 'hello' - CPU"
echo "----------------------------------------"
./bin/show_model_output --input "hello" --device cpu --max_tokens 3 2>&1 | grep -E "(Step|Token|解码结果)"

echo ""
echo "【测试1】输入 'hello' - GPU"
echo "----------------------------------------"
./bin/show_model_output --input "hello" --device gpu --max_tokens 3 2>&1 | grep -E "(Step|Token|解码结果)"

echo ""
echo "【测试2】输入 '你好' - CPU"
echo "----------------------------------------"
./bin/show_model_output --input "你好" --device cpu --max_tokens 3 2>&1 | grep -E "(Step|Token|解码结果)"

echo ""
echo "【测试2】输入 '你好' - GPU"
echo "----------------------------------------"
./bin/show_model_output --input "你好" --device gpu --max_tokens 3 2>&1 | grep -E "(Step|Token|解码结果)"

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
