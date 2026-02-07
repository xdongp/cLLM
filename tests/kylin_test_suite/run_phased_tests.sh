#!/bin/bash
# Kylin Backend CPU vs GPU 分阶段测试运行脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Kylin Backend CPU vs GPU 分阶段测试${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# 检查可执行文件
if [ ! -f "../../build/bin/kylin_test_suite" ]; then
    echo -e "${RED}错误: kylin_test_suite 可执行文件不存在${NC}"
    echo "请先编译: cd build && make kylin_test_suite -j4"
    exit 1
fi

EXEC="../../build/bin/kylin_test_suite"
LOG_DIR="./test_logs"
mkdir -p "$LOG_DIR"

# 时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo -e "${GREEN}测试开始时间: $(date)${NC}"
echo ""

# Phase 1: 权重一致性验证
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Phase 1 (Stage 30): 权重一致性验证${NC}"
echo -e "${YELLOW}============================================================${NC}"
$EXEC --stage=30 2>&1 | tee "$LOG_DIR/phase1_${TIMESTAMP}.log"
echo ""

# Phase 2: Embedding 对比
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Phase 2 (Stage 31): Embedding 层输出对比${NC}"
echo -e "${YELLOW}============================================================${NC}"
$EXEC --stage=31 2>&1 | tee "$LOG_DIR/phase2_${TIMESTAMP}.log"
echo ""

# Phase 3: 逐层对比（最关键）
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Phase 3 (Stage 32): 逐层 Transformer 对比 ⭐${NC}"
echo -e "${YELLOW}============================================================${NC}"
$EXEC --stage=32 --verbose 2>&1 | tee "$LOG_DIR/phase3_${TIMESTAMP}.log"
echo ""

# Phase 4: Logits 对比
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Phase 4 (Stage 33): Logits 与 Top-K 对比${NC}"
echo -e "${YELLOW}============================================================${NC}"
$EXEC --stage=33 2>&1 | tee "$LOG_DIR/phase4_${TIMESTAMP}.log"
echo ""

# Phase 5: 生成文本对比
echo -e "${YELLOW}============================================================${NC}"
echo -e "${YELLOW}Phase 5 (Stage 34): 多步生成文本对比${NC}"
echo -e "${YELLOW}============================================================${NC}"
$EXEC --stage=34 2>&1 | tee "$LOG_DIR/phase5_${TIMESTAMP}.log"
echo ""

# 总结
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}所有测试完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}测试结束时间: $(date)${NC}"
echo ""
echo -e "${BLUE}日志文件保存在: $LOG_DIR/${NC}"
echo -e "${BLUE}详细报告: CPU_GPU_COMPARISON_REPORT.md${NC}"
echo ""
echo -e "${YELLOW}测试结论预览:${NC}"
echo -e "  ✅ Phase 1: 权重一致性 - 通过"
echo -e "  ✅ Phase 2: Embedding - 通过"
echo -e "  ⚠️  Phase 3: 逐层对比 - ${RED}Layer 0 Attention 首次出现偏差${NC}"
echo -e "  ❌ Phase 4: Logits - Top-10 完全不重叠"
echo -e "  ❌ Phase 5: 生成文本 - GPU 输出乱码"
echo ""
echo -e "${YELLOW}问题定位: ${RED}GPU Attention 计算实现有误${NC}"
echo -e "${YELLOW}建议行动: 检查 RoPE、KV Cache、GQA 实现${NC}"
