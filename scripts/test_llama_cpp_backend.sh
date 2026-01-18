#!/bin/bash
# 测试 llama.cpp 后端 (GGUF 格式)

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=== 测试 llama.cpp 后端 (GGUF 格式) ==="

# 1. 检查测试程序是否存在
TEST_EXEC="build/bin/test_llama_cpp_backend_gguf"
if [ ! -f "$TEST_EXEC" ]; then
    echo -e "${RED}错误: 测试程序不存在: $TEST_EXEC${NC}"
    echo "请先编译测试程序:"
    echo "  cd build && cmake .. && make test_llama_cpp_backend_gguf -j\$(sysctl -n hw.ncpu)"
    exit 1
fi

echo -e "${GREEN}✓ 测试程序存在${NC}"

# 2. 查找模型文件
MODEL_PATHS=(
    "$CLLM_MODEL_PATH"
    "model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf"
    "../model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf"
    "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf"
)

MODEL_PATH=""
for path in "${MODEL_PATHS[@]}"; do
    if [ -n "$path" ] && [ -f "$path" ]; then
        MODEL_PATH="$path"
        break
    fi
done

if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 找不到 GGUF 模型文件${NC}"
    echo "请设置 CLLM_MODEL_PATH 环境变量，或确保模型文件存在于以下位置之一:"
    for path in "${MODEL_PATHS[@]}"; do
        if [ -n "$path" ]; then
            echo "  - $path"
        fi
    done
    exit 1
fi

echo -e "${GREEN}✓ 找到模型文件: $MODEL_PATH${NC}"

# 3. 设置环境变量
export CLLM_MODEL_PATH="$MODEL_PATH"
export CLLM_LOG_LEVEL="${CLLM_LOG_LEVEL:-info}"
export CLLM_MAX_NEW_TOKENS="${CLLM_MAX_NEW_TOKENS:-16}"
export CLLM_TEMPERATURE="${CLLM_TEMPERATURE:-0.7}"

echo "环境变量:"
echo "  CLLM_MODEL_PATH=$CLLM_MODEL_PATH"
echo "  CLLM_LOG_LEVEL=$CLLM_LOG_LEVEL"
echo "  CLLM_MAX_NEW_TOKENS=$CLLM_MAX_NEW_TOKENS"
echo "  CLLM_TEMPERATURE=$CLLM_TEMPERATURE"
echo ""

# 4. 运行测试
echo -e "${YELLOW}开始运行测试...${NC}"
echo ""

cd "$(dirname "$0")/.."
"$TEST_EXEC" "$@"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ 测试通过${NC}"
else
    echo ""
    echo -e "${RED}✗ 测试失败 (退出码: $EXIT_CODE)${NC}"
fi

exit $EXIT_CODE
