#!/bin/bash
# 等待测试完成并分析结果

cd "$(dirname "$0")/.."

echo "等待所有测试完成..."
TOTAL=30

while true; do
    completed=$(ls -1 /tmp/comparison_results/*.json 2>/dev/null | wc -l | tr -d ' ')
    progress=$((completed * 100 / TOTAL))
    
    echo "[$(date '+%H:%M:%S')] 进度: $completed/$TOTAL ($progress%)"
    
    if [ $completed -ge $TOTAL ]; then
        echo ""
        echo "✓ 所有测试完成！"
        break
    fi
    
    # 检查测试进程是否还在运行
    if ! ps aux | grep -E "run_comparison|unified_benchmark" | grep -v grep > /dev/null 2>&1; then
        echo ""
        echo "⚠️ 测试进程已停止，但未完成所有测试"
        echo "已完成: $completed/$TOTAL"
        echo "继续等待30秒后再次检查..."
        sleep 30
        completed=$(ls -1 /tmp/comparison_results/*.json 2>/dev/null | wc -l | tr -d ' ')
        if [ $completed -ge $TOTAL ]; then
            echo "✓ 所有测试完成！"
            break
        fi
    fi
    
    sleep 60
done

echo ""
echo "=========================================="
echo "分析测试结果"
echo "=========================================="
echo ""

# 检查cLLM失败情况
echo "检查cLLM失败情况..."
python3 -c "
import json, glob, os
files = glob.glob('/tmp/comparison_results/cllm_*.json')
failed_tests = []
for f in files:
    try:
        with open(f) as j:
            d = json.load(j)
            c = d.get('concurrent', {})
            failed = c.get('failed_requests', 0)
            if failed > 0:
                failed_tests.append((os.path.basename(f), c.get('successful_requests', 0), failed))
    except Exception as e:
        pass
if failed_tests:
    print('发现cLLM失败测试:')
    for name, success, failed in failed_tests:
        print(f'  {name}: {success}/72, 失败: {failed}')
    print(f'\\n总失败数: {len(failed_tests)}')
else:
    print('✓ 所有cLLM测试都成功')
"

echo ""
echo "生成最终报告..."
python3 tools/generate_comparison_report.py

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
