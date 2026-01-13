#!/bin/bash
# GGUF 文件格式验证脚本

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
echo ""
python3 /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/analyze_gguf_detailed.py "$MODEL_PATH"

echo ""
echo "运行快速验证脚本..."
echo ""
python3 /Users/dannypan/PycharmProjects/xllm/cpp/cLLM/test_gguf_format.py

echo ""
echo "=========================================="
echo "验证完成"
echo "=========================================="
