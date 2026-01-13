#!/bin/bash
# 运行 GGUF 文件格式验证脚本

MODEL_PATH="${1:-/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q8_0.gguf}"

echo "=========================================="
echo "GGUF 文件格式验证"
echo "=========================================="
echo "文件: $MODEL_PATH"
echo ""

if [ ! -f "$MODEL_PATH" ]; then
    echo "❌ 错误: 文件不存在: $MODEL_PATH"
    exit 1
fi

echo "运行详细分析脚本..."
python3 /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/analyze_gguf_detailed.py "$MODEL_PATH"

echo ""
echo "运行简单验证脚本..."
python3 /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/verify_gguf.py "$MODEL_PATH"
