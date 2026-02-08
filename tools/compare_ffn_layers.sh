#!/bin/bash
# CPU vs GPU FFN 层逐层对比脚本

echo "========================================="
echo "CPU vs GPU FFN 层逐层对比测试"
echo "========================================="

cd /Users/dannypan/PycharmProjects/cLLM/build

# 获取 CPU 的 FFN 调试输出
echo ""
echo "【CPU 路径】输入 'hello'"
echo "----------------------------------------"
./bin/show_model_output --input "hello" --device cpu --max_tokens 1 2>&1 | grep -E "(FFN Input|Post Attention|FFN Gate|FFN Up|FFN Down|Layer 0 Output)" | head -20

# 获取 GPU 的 FFN 调试输出
echo ""
echo "【GPU 路径】输入 'hello'"
echo "----------------------------------------"
./bin/show_model_output --input "hello" --device gpu --max_tokens 1 2>&1 | grep -E "(FFN Input|Post Attention|FFN Gate|FFN Up|FFN Down|Layer 0 Output)" | head -20

echo ""
echo "========================================="
echo "对比分析："
echo "1. FFN Input - Attention + Residual 后的输入"
echo "2. Post Attention RMS Norm (Raw) - RMSNorm 原始输出"
echo "3. Post Attention RMS Norm (Weighted) - RMSNorm 乘以权重后的输出"
echo "4. FFN Gate - gate_proj 的输出"
echo "5. FFN Up - up_proj 的输出"
echo "6. FFN Down - down_proj 的输出"
echo "7. Layer 0 Output - Layer 0 最终输出"
echo "========================================="
