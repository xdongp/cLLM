#!/bin/bash

# 中文输入生成测试脚本

echo "========================================="
echo "中文输入生成测试 - CPU vs GPU"
echo "========================================="

MODEL_PATH="/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B"
TEST_INPUTS=("你好" "今天天气怎么样" "请介绍一下自己" "什么是人工智能")
MAX_TOKENS=20
TEMPERATURE=0.8

cd /Users/dannypan/PycharmProjects/cLLM/build

for input in "${TEST_INPUTS[@]}"; do
    echo ""
    echo "========================================="
    echo "测试输入: $input"
    echo "========================================="

    # CPU 生成
    echo ""
    echo "【CPU 生成】"
    CPU_OUTPUT=$(./bin/show_model_output --model "$MODEL_PATH" --device cpu --prompt "$input" --max-tokens $MAX_TOKENS --temperature $TEMPERATURE 2>&1)
    CPU_TOKENS=$(echo "$CPU_OUTPUT" | grep "Tokens:" | tail -1)
    CPU_DECODED=$(echo "$CPU_OUTPUT" | grep "解码结果 (skipSpecial=true):" -A1 | tail -1)
    echo "Tokens: $CPU_TOKENS"
    echo "生成结果: $CPU_DECODED"

    # GPU 生成
    echo ""
    echo "【GPU 生成】"
    GPU_OUTPUT=$(./bin/show_model_output --model "$MODEL_PATH" --device gpu --prompt "$input" --max-tokens $MAX_TOKENS --temperature $TEMPERATURE 2>&1)
    GPU_TOKENS=$(echo "$GPU_OUTPUT" | grep "Tokens:" | tail -1)
    GPU_DECODED=$(echo "$GPU_OUTPUT" | grep "解码结果 (skipSpecial=true):" -A1 | tail -1)
    echo "Tokens: $GPU_TOKENS"
    echo "生成结果: $GPU_DECODED"

    # 对比结果
    echo ""
    echo "【对比结果】"
    if [ "$CPU_TOKENS" = "$GPU_TOKENS" ]; then
        echo "✅ Tokens 完全一致"
    else
        echo "❌ Tokens 不一致"
        echo "  CPU: $CPU_TOKENS"
        echo "  GPU: $GPU_TOKENS"
    fi

done

echo ""
echo "========================================="
echo "中文生成测试完成"
echo "========================================="
