#!/bin/bash
# Test script for cLLM server endpoints
# Usage: ./test_server.sh [host] [port]

HOST=${1:-127.0.0.1}
PORT=${2:-8080}
BASE_URL="http://${HOST}:${PORT}"

echo "========================================"
echo "Testing cLLM Server at ${BASE_URL}"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
PASSED=0
FAILED=0

# Function to test an endpoint
test_endpoint() {
    local name=$1
    local method=$2
    local endpoint=$3
    local data=$4
    local expected_code=$5
    
    echo -n "Testing ${name}... "
    
    if [ "$method" == "GET" ]; then
        response=$(curl -s -w "\n%{http_code}" -X GET "${BASE_URL}${endpoint}")
    else
        response=$(curl -s -w "\n%{http_code}" -X POST "${BASE_URL}${endpoint}" \
            -H "Content-Type: application/json" \
            -d "$data")
    fi
    
    # Extract status code (last line)
    status_code=$(echo "$response" | tail -n1)
    # Extract body (all but last line)
    body=$(echo "$response" | head -n-1)
    
    if [ "$status_code" == "$expected_code" ]; then
        echo -e "${GREEN}✓ PASSED${NC} (HTTP $status_code)"
        if [ -n "$body" ]; then
            echo "  Response: $body"
        fi
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC} (Expected HTTP $expected_code, got $status_code)"
        if [ -n "$body" ]; then
            echo "  Response: $body"
        fi
        ((FAILED++))
    fi
    echo ""
}

# Test 1: Health Check
echo "=== Test 1: Health Check ==="
test_endpoint "Health Check" "GET" "/health" "" "200"

# Test 2: Encode Endpoint
echo "=== Test 2: Encode Endpoint ==="
test_endpoint "Encode Simple Text" "POST" "/encode" \
    '{"text": "Hello, world!"}' "200"

test_endpoint "Encode Empty Text" "POST" "/encode" \
    '{"text": ""}' "400"

test_endpoint "Encode Missing Text Field" "POST" "/encode" \
    '{}' "400"

# Test 3: Generate Endpoint
echo "=== Test 3: Generate Endpoint ==="
test_endpoint "Generate Simple" "POST" "/generate" \
    '{"prompt": "Hello", "max_tokens": 5}' "200"

test_endpoint "Generate with Parameters" "POST" "/generate" \
    '{"prompt": "Once upon a time", "max_tokens": 10, "temperature": 0.8, "top_p": 0.95}' "200"

# Test 4: Invalid Endpoint
echo "=== Test 4: Invalid Endpoint ==="
test_endpoint "404 Not Found" "GET" "/invalid" "" "404"

# Summary
echo "========================================"
echo "Test Summary"
echo "========================================"
echo -e "Passed: ${GREEN}${PASSED}${NC}"
echo -e "Failed: ${RED}${FAILED}${NC}"
echo "Total: $((PASSED + FAILED))"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}Some tests failed!${NC}"
    exit 1
fi
