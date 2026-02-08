#!/bin/bash

# 端到端 /generate 接口测试脚本
# 测试 CPU 和 GPU 后端的生成结果是否一致

set -e

CWD="/Users/dannypan/PycharmProjects/cLLM"
BUILD_DIR="$CWD/build"
MODEL_PATH="$CWD/model/Qwen/Qwen3-0.6B"
CPU_PORT=8080
GPU_PORT=8081

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================="
echo "端到端 /generate 接口测试"
echo "========================================="

# 检查可执行文件是否存在
if [ ! -f "$BUILD_DIR/bin/cllm_server" ]; then
    echo -e "${RED}错误: cllm_server 不存在，请先编译${NC}"
    exit 1
fi

# 测试函数
test_generate() {
    local port=$1
    local backend=$2
    local prompt=$3
    local max_tokens=$4
    
    curl -s -X POST http://localhost:$port/generate \
        -H "Content-Type: application/json" \
        -d "{
            \"prompt\": \"$prompt\",
            \"max_tokens\": $max_tokens,
            \"temperature\": 0.8,
            \"top_p\": 0.95
        }" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('text',''))"
}

# 启动 CPU 服务器
echo ""
echo "【1】启动 CPU 服务器 (端口: $CPU_PORT)..."
cd $BUILD_DIR
./bin/cllm_server --config $CWD/config/config_kylin_cpu.yaml --port $CPU_PORT > /tmp/cpu_server.log 2>&1 &
CPU_PID=$!
echo "CPU 服务器 PID: $CPU_PID"

# 启动 GPU 服务器
echo ""
echo "【2】启动 GPU 服务器 (端口: $GPU_PORT)..."
./bin/cllm_server --config $CWD/config/config_kylin_gpu.yaml --port $GPU_PORT > /tmp/gpu_server.log 2>&1 &
GPU_PID=$!
echo "GPU 服务器 PID: $GPU_PID"

# 等待服务器启动
echo ""
echo "【3】等待服务器启动 (5秒)..."
sleep 5

# 检查服务器是否正常运行
echo ""
echo "【4】检查服务器健康状态..."
CPU_HEALTH=$(curl -s http://localhost:$CPU_PORT/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "failed")
GPU_HEALTH=$(curl -s http://localhost:$GPU_PORT/health | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',''))" 2>/dev/null || echo "failed")

if [ "$CPU_HEALTH" != "healthy" ]; then
    echo -e "${RED}CPU 服务器启动失败${NC}"
    echo "日志:"
    tail -20 /tmp/cpu_server.log
    kill $CPU_PID $GPU_PID 2>/dev/null
    exit 1
fi

if [ "$GPU_HEALTH" != "healthy" ]; then
    echo -e "${RED}GPU 服务器启动失败${NC}"
    echo "日志:"
    tail -20 /tmp/gpu_server.log
    kill $CPU_PID $GPU_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}CPU 服务器: 健康${NC}"
echo -e "${GREEN}GPU 服务器: 健康${NC}"

# 测试用例
TEST_PROMPTS=("你好" "今天天气怎么样" "请介绍一下自己" "什么是人工智能")
MAX_TOKENS=20

echo ""
echo "【5】执行生成测试..."
echo ""

PASS_COUNT=0
FAIL_COUNT=0

for prompt in "${TEST_PROMPTS[@]}"; do
    echo "----------------------------------------"
    echo "测试输入: $prompt"
    echo "----------------------------------------"
    
    # CPU 生成
    echo "CPU 生成中..."
    CPU_RESULT=$(test_generate $CPU_PORT "cpu" "$prompt" $MAX_TOKENS)
    echo "CPU 结果: $CPU_RESULT"
    
    # GPU 生成
    echo "GPU 生成中..."
    GPU_RESULT=$(test_generate $GPU_PORT "gpu" "$prompt" $MAX_TOKENS)
    echo "GPU 结果: $GPU_RESULT"
    
    # 对比结果
    if [ "$CPU_RESULT" = "$GPU_RESULT" ]; then
        echo -e "${GREEN}✅ 结果一致${NC}"
        ((PASS_COUNT++))
    else
        echo -e "${RED}❌ 结果不一致${NC}"
        ((FAIL_COUNT++))
    fi
    echo ""
done

# 停止服务器
echo "【6】停止服务器..."
kill $CPU_PID $GPU_PID 2>/dev/null
wait $CPU_PID $GPU_PID 2>/dev/null

echo ""
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
    echo -e "${RED}有测试失败，请检查日志${NC}"
    exit 1
fi
