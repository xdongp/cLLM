#!/bin/bash

# Kylin GPU 简单测试脚本
# 参考 docs/testing/cllm_kylin_hf_gpu_test_plan.md

set -e

CWD="/Users/dannypan/PycharmProjects/cLLM"
BUILD_DIR="$CWD/build"
CONFIG_FILE="$CWD/config/config_kylin_gpu.yaml"
MODEL_PATH="$CWD/model/Qwen/Qwen3-0.6B"

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo "========================================="
echo "Kylin GPU 简单测试"
echo "========================================="
echo ""

# 检查环境
echo "【1】检查测试环境..."
if [ ! -f "$BUILD_DIR/bin/cllm_server" ]; then
    echo -e "${RED}错误: cllm_server 不存在${NC}"
    exit 1
fi

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型路径不存在: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 环境检查通过${NC}"
echo ""

# 检查 unified_benchmark.py
echo "【2】检查测试工具..."
if [ ! -f "$CWD/tools/unified_benchmark.py" ]; then
    echo -e "${YELLOW}警告: unified_benchmark.py 不存在，使用简化测试${NC}"
    USE_SIMPLE_TEST=1
else
    echo -e "${GREEN}✓ unified_benchmark.py 存在${NC}"
    USE_SIMPLE_TEST=0
fi
echo ""

# 启动服务器
echo "【3】启动 cLLM 服务器 (Kylin + GPU)..."
cd $BUILD_DIR

# 修改配置文件中的模型路径
sed -i.bak "s|path: \".*Qwen/Qwen3-0.6B\"|path: \"$MODEL_PATH\"|" $CONFIG_FILE

echo "启动服务器..."
./bin/cllm_server --config $CONFIG_FILE > /tmp/kylin_server.log 2>&1 &
SERVER_PID=$!
echo "服务器 PID: $SERVER_PID"
echo ""

# 等待服务器启动
echo "【4】等待服务器就绪..."
sleep 3

# 检查健康状态
echo "检查健康状态..."
for i in {1..10}; do
    HEALTH=$(curl -s http://localhost:8080/health 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "")
    if [ "$HEALTH" = "healthy" ]; then
        echo -e "${GREEN}✓ 服务器就绪${NC}"
        break
    fi
    echo "等待服务器... ($i/10)"
    sleep 1
done

if [ "$HEALTH" != "healthy" ]; then
    echo -e "${RED}错误: 服务器启动失败${NC}"
    echo "日志:"
    tail -30 /tmp/kylin_server.log
    kill $SERVER_PID 2>/dev/null
    exit 1
fi
echo ""

# 执行简单测试
echo "【5】执行简单生成测试..."
echo ""

TEST_PROMPTS=("你好" "今天天气怎么样" "什么是机器学习")
MAX_TOKENS=20

for i in "${!TEST_PROMPTS[@]}"; do
    prompt="${TEST_PROMPTS[$i]}"
    echo "----------------------------------------"
    echo "测试 $((i+1))/${#TEST_PROMPTS[@]}: $prompt"
    echo "----------------------------------------"
    
    # 使用 curl 测试 /generate 端点
    START_TIME=$(date +%s.%N)
    
    RESULT=$(curl -s -X POST http://localhost:8080/generate \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": $MAX_TOKENS,
            \"temperature\": 0.8,
            \"top_p\": 0.95
        }" 2>/dev/null)
    
    END_TIME=$(date +%s.%N)
    DURATION=$(echo "$END_TIME - $START_TIME" | bc)
    
    # 解析结果
    TEXT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('text',''))" 2>/dev/null || echo "解析失败")
    TOKENS=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('tokens_generated',0))" 2>/dev/null || echo "0")
    
    echo "生成文本: $TEXT"
    echo "生成 tokens: $TOKENS"
    echo "耗时: ${DURATION}s"
    
    if [ "$TOKENS" -gt 0 ]; then
        TPS=$(echo "scale=2; $TOKENS / $DURATION" | bc)
        echo -e "吞吐量: ${BLUE}${TPS} tokens/s${NC}"
    fi
    echo ""
done

# 停止服务器
echo "【6】停止服务器..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# 恢复配置文件
mv $CONFIG_FILE.bak $CONFIG_FILE

echo ""
echo "========================================="
echo "测试完成"
echo "========================================="
echo ""
echo "服务器日志: /tmp/kylin_server.log"
echo ""
