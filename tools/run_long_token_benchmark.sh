#!/bin/bash
# Long Token Benchmark Script for llama_cpp + GGUF + GPU
# 测试 512, 1024, 2048, 4096, 8192 tokens
# 2 并发，10 请求

DATE=$(date +%Y%m%d)
RESULT_DIR="results/llama_cpp_long_tokens_${DATE}"
SERVER_URL="http://localhost:8085"
CONCURRENCY=2
REQUESTS=10

mkdir -p "$RESULT_DIR"

# 记录系统信息
echo "=== System Information ===" > "$RESULT_DIR/system_info.txt"
echo "Date: $(date)" >> "$RESULT_DIR/system_info.txt"
echo "OS: $(uname -a)" >> "$RESULT_DIR/system_info.txt"
echo "CPU: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Unknown')" >> "$RESULT_DIR/system_info.txt"
echo "Memory: $(sysctl -n hw.memsize 2>/dev/null | awk '{print $0/1024/1024/1024 " GB"}' || echo 'Unknown')" >> "$RESULT_DIR/system_info.txt"
echo "" >> "$RESULT_DIR/system_info.txt"

# 检查服务器健康
echo "Checking server health..."
HEALTH=$(curl -s "$SERVER_URL/health")
if [[ "$HEALTH" != *"healthy"* ]]; then
    echo "Error: Server is not healthy"
    echo "Response: $HEALTH"
    exit 1
fi
echo "Server is healthy"

# 测试函数
run_test() {
    local tokens=$1
    echo ""
    echo "=========================================="
    echo "Testing $tokens tokens ($CONCURRENCY concurrent, $REQUESTS requests)"
    echo "=========================================="
    
    # 记录测试前的系统资源
    echo "--- Pre-test resources ($tokens tokens) ---" >> "$RESULT_DIR/resource_log.txt"
    echo "Time: $(date)" >> "$RESULT_DIR/resource_log.txt"
    
    # 获取 cllm_server 进程信息
    CLLM_PID=$(pgrep -f "cllm_server")
    if [ -n "$CLLM_PID" ]; then
        echo "cLLM PID: $CLLM_PID" >> "$RESULT_DIR/resource_log.txt"
        ps -p $CLLM_PID -o pid,ppid,%cpu,%mem,rss,vsz,command >> "$RESULT_DIR/resource_log.txt"
    fi
    
    # 运行测试
    python3 tools/unified_benchmark.py \
        --server-type cllm \
        --server-url "$SERVER_URL" \
        --test-type api-concurrent \
        --requests $REQUESTS \
        --concurrency $CONCURRENCY \
        --max-tokens $tokens \
        --output-file "$RESULT_DIR/tokens_${tokens}.json"
    
    TEST_EXIT_CODE=$?
    
    # 记录测试后的系统资源
    echo "--- Post-test resources ($tokens tokens) ---" >> "$RESULT_DIR/resource_log.txt"
    echo "Time: $(date)" >> "$RESULT_DIR/resource_log.txt"
    echo "Test exit code: $TEST_EXIT_CODE" >> "$RESULT_DIR/resource_log.txt"
    
    if [ -n "$CLLM_PID" ]; then
        ps -p $CLLM_PID -o pid,ppid,%cpu,%mem,rss,vsz,command >> "$RESULT_DIR/resource_log.txt" 2>/dev/null || echo "Process ended" >> "$RESULT_DIR/resource_log.txt"
    fi
    echo "" >> "$RESULT_DIR/resource_log.txt"
    
    # 等待一下让系统稳定
    sleep 3
    
    # 检查服务器是否还活着
    HEALTH=$(curl -s "$SERVER_URL/health" 2>/dev/null)
    if [[ "$HEALTH" != *"healthy"* ]]; then
        echo "Warning: Server may have crashed after $tokens tokens test"
        echo "Server status: $HEALTH"
        return 1
    fi
    
    return 0
}

# 运行所有测试
TOKEN_SIZES=(512 1024 2048 4096 8192)

for tokens in "${TOKEN_SIZES[@]}"; do
    run_test $tokens
    if [ $? -ne 0 ]; then
        echo "Test failed at $tokens tokens, stopping..."
        break
    fi
done

echo ""
echo "=========================================="
echo "All tests completed!"
echo "Results saved to: $RESULT_DIR"
echo "=========================================="

# 生成汇总报告
echo ""
echo "Generating summary report..."
python3 -c "
import json
import os
import glob

result_dir = '$RESULT_DIR'
files = sorted(glob.glob(f'{result_dir}/tokens_*.json'))

print()
print('=' * 70)
print('Long Token Benchmark Summary - llama_cpp + GGUF + GPU')
print('=' * 70)
print()
print(f'Configuration: {$CONCURRENCY} concurrency, {$REQUESTS} requests per test')
print()
print('| Tokens | Success | Failed | Avg Time (s) | Throughput (t/s) | Avg t/s |')
print('|--------|---------|--------|--------------|------------------|---------|')

for f in files:
    try:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        tokens = data.get('max_tokens', 0)
        conc = data.get('concurrent', {})
        
        if conc:
            success = conc.get('successful_requests', 0)
            failed = conc.get('failed_requests', 0)
            avg_time = conc.get('avg_response_time', 0)
            throughput = conc.get('avg_throughput', 0)
            avg_tps = conc.get('avg_tokens_per_second', 0)
            
            print(f'| {tokens:6d} | {success:7d} | {failed:6d} | {avg_time:12.2f} | {throughput:16.2f} | {avg_tps:7.2f} |')
    except Exception as e:
        print(f'Error reading {f}: {e}')

print()
"
