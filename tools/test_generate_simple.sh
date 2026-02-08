#!/bin/bash

# 简化的端到端 /generate 接口测试
# 直接测试 CPU 和 GPU 的生成结果

set -e

CWD="/Users/dannypan/PycharmProjects/cLLM"
BUILD_DIR="$CWD/build"
MODEL_PATH="$CWD/model/Qwen/Qwen3-0.6B"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "端到端 /generate 接口测试 (简化版)"
echo "========================================="

# 检查可执行文件
if [ ! -f "$BUILD_DIR/bin/show_model_output" ]; then
    echo -e "${RED}错误: show_model_output 不存在${NC}"
    exit 1
fi

# 测试用例
TEST_PROMPTS=("你好" "今天天气怎么样" "请介绍一下自己")
MAX_TOKENS=15

echo ""
echo "【测试配置】"
echo "模型路径: $MODEL_PATH"
echo "最大生成 tokens: $MAX_TOKENS"
echo "测试用例数: ${#TEST_PROMPTS[@]}"
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for prompt in "${TEST_PROMPTS[@]}"; do
    echo "----------------------------------------"
    echo "测试输入: $prompt"
    echo "----------------------------------------"
    
    # CPU 生成
    echo "CPU 生成中..."
    CPU_OUTPUT=$($BUILD_DIR/bin/show_model_output --model "$MODEL_PATH" --device cpu --prompt "$prompt" --max-tokens $MAX_TOKENS --temperature 0.8 2>&1)
    CPU_TOKENS=$(echo "$CPU_OUTPUT" | grep "Tokens:" | tail -1 | sed 's/.*Tokens: \[\(.*\)\].*/\1/')
    CPU_DECODED=$(echo "$CPU_OUTPUT" | grep "解码结果 (skipSpecial=true):" -A1 | tail -1 | sed 's/.*"\(.*\)".*/\1/')
    echo "CPU Tokens: [$CPU_TOKENS]"
    echo "CPU 结果: $CPU_DECODED"
    
    # GPU 生成
    echo ""
    echo "GPU 生成中..."
    GPU_OUTPUT=$($BUILD_DIR/bin/show_model_output --model "$MODEL_PATH" --device gpu --prompt "$prompt" --max-tokens $MAX_TOKENS --temperature 0.8 2>&1)
    GPU_TOKENS=$(echo "$GPU_OUTPUT" | grep "Tokens:" | tail -1 | sed 's/.*Tokens: \[\(.*\)\].*/\1/')
    GPU_DECODED=$(echo "$GPU_OUTPUT" | grep "解码结果 (skipSpecial=true):" -A1 | tail -1 | sed 's/.*"\(.*\)".*/\1/')
    echo "GPU Tokens: [$GPU_TOKENS]"
    echo "GPU 结果: $GPU_DECODED"
    
    # 对比结果
    echo ""
    if [ "$CPU_TOKENS" = "$GPU_TOKENS" ]; then
        echo -e "${GREEN}✅ Tokens 完全一致${NC}"
        ((PASS_COUNT++))
    else
        echo -e "${RED}❌ Tokens 不一致${NC}"
        echo "  CPU: [$CPU_TOKENS]"
        echo "  GPU: [$GPU_TOKENS]"
        ((FAIL_COUNT++))
    fi
    echo ""
done

echo "========================================="
echo "测试完成"
echo "========================================="
echo -e "通过: ${GREEN}$PASS_COUNT${NC}"
echo -e "失败: ${RED}$FAIL_COUNT${NC}"
echo ""

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "${GREEN}所有测试通过！${NC}"
    exit 0
else
    echo -e "${RED}有测试失败${NC}"
    exit 1
fi
