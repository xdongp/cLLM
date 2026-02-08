#!/bin/bash

# Layer 1 CPU vs GPU 对比脚本

echo "========================================="
echo "Layer 1 CPU vs GPU 对比"
echo "========================================="

# 运行 CPU 并提取 Layer 1 信息
echo ""
echo "【CPU 输出】"
cd /Users/dannypan/PycharmProjects/cLLM/build
./bin/show_model_output --model /Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B --device cpu --prompt "你好" --max-tokens 1 --temperature 0.8 2>&1 | grep -E "\[CPU DEBUG\] Layer 1" | head -5

# 运行 GPU 并提取 Layer 1 信息
echo ""
echo "【GPU 输出】"
./bin/show_model_output --model /Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B --device gpu --prompt "你好" --max-tokens 1 --temperature 0.8 2>&1 | grep -E "\[GPU DEBUG\] Layer 1" | head -10

echo ""
echo "========================================="
echo "对比完成"
echo "========================================="
