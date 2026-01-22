#!/bin/bash

# 批处理大小优化测试脚本
# 测试不同的n_batch值以找到最优配置

BATCH_SIZES=(256 512 768 1024 1280)
CONCURRENCY=24
REQUESTS=72
MAX_TOKENS=50

echo "=========================================="
echo "批处理大小优化测试"
echo "=========================================="
echo "测试配置:"
echo "  并发数: $CONCURRENCY"
echo "  请求数: $REQUESTS"
echo "  最大tokens: $MAX_TOKENS"
echo ""

for batch_size in "${BATCH_SIZES[@]}"; do
    echo "=========================================="
    echo "测试 n_batch = $batch_size"
    echo "=========================================="
    
    # 更新配置文件
    sed -i '' "s/n_batch: [0-9]*/n_batch: $batch_size/" config/config.yaml
    
    # 停止现有服务器
    lsof -ti:8080 | xargs kill -9 2>/dev/null || true
    sleep 2
    
    # 启动服务器
    ./build/bin/cllm_server --config config/config.yaml > /tmp/cllm_server_${batch_size}.log 2>&1 &
    SERVER_PID=$!
    echo "服务器启动 (PID: $SERVER_PID)"
    
    # 等待服务器启动
    sleep 5
    
    # 运行测试
    python3 tools/unified_benchmark.py \
        --server-type cllm \
        --concurrency $CONCURRENCY \
        --requests $REQUESTS \
        --max-tokens $MAX_TOKENS \
        > /tmp/benchmark_${batch_size}.log 2>&1
    
    # 提取关键指标
    THROUGHPUT=$(grep "Avg throughput" /tmp/benchmark_${batch_size}.log | awk '{print $4}')
    AVG_TIME=$(grep "Avg response time" /tmp/benchmark_${batch_size}.log | awk '{print $4}')
    
    echo ""
    echo "结果:"
    echo "  吞吐量: $THROUGHPUT tokens/sec"
    echo "  平均响应时间: $AVG_TIME s"
    echo ""
    
    # 停止服务器
    kill $SERVER_PID 2>/dev/null || true
    sleep 2
done

echo "=========================================="
echo "测试完成"
echo "=========================================="

# 恢复默认配置
sed -i '' 's/n_batch: [0-9]*/n_batch: 512/' config/config.yaml
