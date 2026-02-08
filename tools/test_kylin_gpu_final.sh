#!/bin/bash

# Kylin GPU 最终测试脚本
# 参考 docs/testing/cllm_kylin_hf_gpu_test_plan.md

set -e

CWD="/Users/dannypan/PycharmProjects/cLLM"
BUILD_DIR="$CWD/build"
MODEL_PATH="$CWD/model/Qwen/Qwen3-0.6B"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "Kylin GPU 最终测试"
echo "========================================="
echo ""

# 检查环境
echo "【1】检查测试环境..."
if [ ! -f "$BUILD_DIR/bin/show_model_output" ]; then
    echo -e "${RED}错误: show_model_output 不存在${NC}"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"
echo ""

# 显示测试配置
echo "【2】测试配置"
echo "----------------------------------------"
echo "模型: Qwen3-0.6B"
echo "后端: Kylin + GPU (Metal)"
echo "模型路径: $MODEL_PATH"
echo "----------------------------------------"
echo ""

# 执行测试
echo "【3】执行性能测试"
echo ""

TEST_PROMPTS=("你好" "今天天气怎么样" "什么是机器学习" "请写一段Python代码")
MAX_TOKENS=30

echo "测试参数:"
echo "  - 测试用例数: ${#TEST_PROMPTS[@]}"
echo "  - 最大生成 tokens: $MAX_TOKENS"
echo "  - 温度: 0.8"
echo ""

TOTAL_TOKENS=0
TOTAL_TIME=0

for i in "${!TEST_PROMPTS[@]}"; do
    prompt="${TEST_PROMPTS[$i]}"
    echo "----------------------------------------"
    echo "测试 $((i+1))/${#TEST_PROMPTS[@]}: $prompt"
    echo "----------------------------------------"
    
    # 使用 show_model_output 测试
    START_TIME=$(date +%s.%N)
    
    OUTPUT=$($BUILD_DIR/bin/show_model_output \
        --model "$MODEL_PATH" \
        --device gpu \
        --prompt "$prompt" \
        --max-tokens $MAX_TOKENS \
        --temperature 0.8 2>&1)
    
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    
    # 解析结果
    TOKENS_LINE=$(echo "$OUTPUT" | grep "生成的 Token 数量:" | tail -1)
    TOKENS=$(echo "$TOKENS_LINE" | sed 's/.*生成的 Token 数量: \([0-9]*\).*/\1/')
    
    RESULT_LINE=$(echo "$OUTPUT" | grep "解码结果 (skipSpecial=true):" -A1 | tail -1)
    RESULT=$(echo "$RESULT_LINE" | sed 's/.*"\(.*\)".*/\1/')
    
    echo "生成结果: $RESULT"
    echo "生成 tokens: $TOKENS"
    echo "耗时: ${DURATION}s"
    
    if [ -n "$TOKENS" ] && [ "$TOKENS" -gt 0 ] && [ -n "$DURATION" ]; then
        TPS=$(echo "scale=2; $TOKENS / $DURATION" | bc)
        echo -e "吞吐量: ${BLUE}${TPS} tokens/s${NC}"
        
        TOTAL_TOKENS=$((TOTAL_TOKENS + TOKENS))
        TOTAL_TIME=$(echo "$TOTAL_TIME + $DURATION" | bc)
    fi
    echo ""
done

# 计算平均性能
echo "========================================="
echo "性能统计"
echo "========================================="
if [ $(echo "$TOTAL_TIME > 0" | bc) -eq 1 ] && [ $TOTAL_TOKENS -gt 0 ]; then
    AVG_TPS=$(echo "scale=2; $TOTAL_TOKENS / $TOTAL_TIME" | bc)
    echo "总生成 tokens: $TOTAL_TOKENS"
    echo "总耗时: ${TOTAL_TIME}s"
    echo -e "平均吞吐量: ${BLUE}${AVG_TPS} tokens/s${NC}"
    echo ""
    
    # 评估等级
    if [ $(echo "$AVG_TPS >= 35" | bc) -eq 1 ]; then
        echo -e "性能等级: ${GREEN}⭐优秀${NC}"
    elif [ $(echo "$AVG_TPS >= 25" | bc) -eq 1 ]; then
        echo -e "性能等级: ${GREEN}✅良好${NC}"
    elif [ $(echo "$AVG_TPS >= 15" | bc) -eq 1 ]; then
        echo -e "性能等级: ${YELLOW}⚠️一般${NC}"
    else
        echo -e "性能等级: ${RED}❌较差${NC}"
    fi
fi
echo ""

echo "========================================="
echo "测试完成"
echo "========================================="
echo ""
