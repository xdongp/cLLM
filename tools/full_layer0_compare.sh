#!/bin/bash
# CPU vs GPU Layer 0 完整对比脚本

echo "========================================="
echo "CPU vs GPU Layer 0 完整对比测试"
echo "========================================="

cd /Users/dannypan/PycharmProjects/cLLM/build

# 定义提取函数
extract_value() {
    local output="$1"
    local pattern="$2"
    echo "$output" | grep -oE "$pattern" | head -1
}

echo ""
echo "【步骤 1/2】运行 CPU 测试..."
echo "----------------------------------------"
CPU_OUTPUT=$(./bin/show_model_output --input "hello" --device cpu --max_tokens 1 2>&1)
echo "$CPU_OUTPUT" | grep "CPU DEBUG" | head -15

echo ""
echo "【步骤 2/2】运行 GPU 测试..."
echo "----------------------------------------"
GPU_OUTPUT=$(./bin/show_model_output --input "hello" --device gpu --max_tokens 1 2>&1)
echo "$GPU_OUTPUT" | grep "GPU DEBUG" | grep -E "(Embedding|Attention|FFN|Layer 0)" | head -15

echo ""
echo "========================================="
echo "Layer 0 逐层对比结果"
echo "========================================="

# 提取并对比每个步骤
echo ""
echo "【1. Embedding】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "Embedding" | grep "CPU DEBUG" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "Embedding" | grep "GPU DEBUG" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【2. Attention Input (RMS Norm 后)】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "Attention Input" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "Attention Input" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【3. Attention Output】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "Attention Output" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "Attention Output" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【4. FFN Input (Residual)】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "FFN Input" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "FFN Input" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【5. Post Attention RMS Norm】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "Post Attention RMS Norm" | head -1 | grep -oE '\[.*\]')"
echo "GPU (Raw): $(echo "$GPU_OUTPUT" | grep "Post Attention RMS Norm (Raw)" | head -1 | grep -oE '\[.*\]')"
echo "GPU (Weighted): $(echo "$GPU_OUTPUT" | grep "Post Attention RMS Norm (Weighted)" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【6. FFN Gate】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "FFN Gate - first" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "FFN Gate - first" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【7. FFN Up】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "FFN Up - first" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "FFN Up - first" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【8. FFN Down】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "FFN Down - first" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "FFN Down - first" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "【9. Layer 0 Output】"
echo "CPU: $(echo "$CPU_OUTPUT" | grep "Layer 0 Output" | head -1 | grep -oE '\[.*\]')"
echo "GPU: $(echo "$GPU_OUTPUT" | grep "Layer 0 Output" | head -1 | grep -oE '\[.*\]')"

echo ""
echo "========================================="
echo "对比完成"
echo "========================================="
