#!/bin/bash
# 运行Kylin引擎GGUF Q4_K_M推理测试

set -e

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Kylin引擎GGUF Q4_K_M推理测试 ===${NC}"

# 检查模型文件是否存在
MODEL_PATH="model/Qwen/qwen3-0.6b-q4_k_m.gguf"
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${RED}错误: 模型文件不存在: $MODEL_PATH${NC}"
    echo -e "${YELLOW}请确保模型文件存在于该路径${NC}"
    exit 1
fi

echo -e "${GREEN}✓ 模型文件存在: $MODEL_PATH${NC}"

# 检查构建目录
if [ ! -d "build" ]; then
    echo -e "${YELLOW}构建目录不存在，正在创建...${NC}"
    mkdir -p build
    cd build
    cmake .. -DBUILD_TESTS=ON
    cd ..
fi

# 进入构建目录
cd build

# 编译测试
echo -e "${GREEN}正在编译测试...${NC}"
cmake .. -DBUILD_TESTS=ON
make test_kylin_gguf_q4k -j$(nproc 2>/dev/null || echo 4)

# 运行测试
echo -e "${GREEN}正在运行测试...${NC}"
echo ""
./bin/test_kylin_gguf_q4k --gtest_color=yes

echo ""
echo -e "${GREEN}=== 测试完成 ===${NC}"
