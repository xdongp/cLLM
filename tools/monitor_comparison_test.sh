#!/bin/bash
# 监控对比测试进度

CONCURRENCIES=(8 16 24 32 40)
ROUNDS=3
TOTAL_TESTS=$((ROUNDS * ${#CONCURRENCIES[@]} * 2))  # 3轮 × 5并发 × 2服务器

echo "监控对比测试进度..."
echo "总测试数: $TOTAL_TESTS"
echo ""

while true; do
    completed=$(ls -1 /tmp/comparison_results/*.json 2>/dev/null | wc -l | tr -d ' ')
    progress=$((completed * 100 / TOTAL_TESTS))
    
    echo "[$(date '+%H:%M:%S')] 进度: $completed/$TOTAL_TESTS ($progress%)"
    
    if [ $completed -ge $TOTAL_TESTS ]; then
        echo ""
        echo "✓ 所有测试完成！"
        break
    fi
    
    # 检查是否有进程在运行
    if ! ps aux | grep -E "run_comparison|unified_benchmark" | grep -v grep > /dev/null; then
        echo ""
        echo "⚠️ 测试进程已停止，但未完成所有测试"
        echo "已完成: $completed/$TOTAL_TESTS"
        break
    fi
    
    sleep 30
done

echo ""
echo "生成汇总报告..."
