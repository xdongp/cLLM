#!/bin/bash

# 项目根目录
export CLLM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 模型路径 (使用本地已有的 Qwen3-0.6B 模型)
export CLLM_TEST_MODEL_PATH="${CLLM_ROOT}/model/Qwen/Qwen3-0.6B"

# 测试数据路径
export CLLM_TEST_DATA_PATH="${CLLM_ROOT}/tests/data"

# 测试报告路径
export CLLM_TEST_REPORTS="${CLLM_ROOT}/test_reports"

# 日志路径
export CLLM_LOG_DIR="${CLLM_ROOT}/logs"

# 线程数
export CLLM_NUM_THREADS=8

# 设备
export CLLM_DEVICE="cpu"  # 或 "cuda:0"

# 日志级别
export CLLM_LOG_LEVEL="INFO"

echo "✅ Environment configured:"
echo "  MODEL_PATH: ${CLLM_TEST_MODEL_PATH}"
echo "  DATA_PATH: ${CLLM_TEST_DATA_PATH}"
echo "  REPORTS: ${CLLM_TEST_REPORTS}"
echo "  LOG_DIR: ${CLLM_LOG_DIR}"
