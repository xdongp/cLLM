#!/bin/bash

API_URL="http://0.0.0.0:18081"
NUM_REQUESTS=20
CONCURRENCY=5
MAX_TOKENS=30

PROMPTS=(
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    "机器学习是人工智能的一个分支，它使计算机能够在不被明确编程的情况下从数据中学习。"
    "深度学习是机器学习的一个子集，它模仿人脑的工作方式来学习数据中的模式。"
    "自然语言处理是人工智能领域中的一个重要方向，致力于让计算机理解和生成人类语言。"
    "计算机视觉是人工智能的一个重要应用领域，旨在让计算机能够像人类一样理解和解释图像和视频。"
)

echo "========================================"
echo "xLLM API Concurrent Test"
echo "========================================"
echo "API URL: $API_URL"
echo "Number of requests: $NUM_REQUESTS"
echo "Concurrency: $CONCURRENCY"
echo "Max tokens: $MAX_TOKENS"
echo "========================================"

function send_request() {
    local prompt="${PROMPTS[$RANDOM % ${#PROMPTS[@]}]}"
    local start_time=$(date +%s.%N)
    local response=$(curl -s -X POST "$API_URL/generate" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$prompt\",\"temperature\":0.7,\"max_tokens\":$MAX_TOKENS,\"stream\":false}")
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    
    if echo "$response" | grep -q '"success":true'; then
        echo "✓ Success - ${duration}s"
    else
        echo "✗ Failed - ${duration}s - $response"
    fi
}

echo ""
echo "Running concurrent test..."
echo ""

for ((i=0; i<NUM_REQUESTS; i++)); do
    send_request &
    
    if (( (i + 1) % CONCURRENCY == 0 )); then
        wait
    fi
done

wait

echo ""
echo "========================================"
echo "Test completed"
echo "========================================"
