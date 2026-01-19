#!/bin/bash

# cLLM vs Ollama 测试示例脚本
# 展示不同的测试场景

echo "========================================"
echo "cLLM vs Ollama 测试示例"
echo "========================================"
echo ""

# 场景1: 快速验证测试 (10请求, 2并发, 50 tokens)
echo "[场景1] 快速验证测试"
echo "  参数: 10请求, 2并发, 50 tokens"
echo "  用途: 快速验证系统功能"
echo ""
python3 tools/run_cllm_ollama_comparison.py \
    --requests 10 \
    --concurrency 2 \
    --max-tokens 50
echo ""
echo "按Enter继续下一个场景..."
read
echo ""

# 场景2: 标准性能测试 (160请求, 5并发, 50 tokens)
echo "[场景2] 标准性能测试"
echo "  参数: 160请求, 5并发, 50 tokens"
echo "  用途: 全面评估系统性能"
echo ""
python3 tools/run_cllm_ollama_comparison.py \
    --requests 160 \
    --concurrency 5 \
    --max-tokens 50 \
    --save-results
echo ""
echo "按Enter继续下一个场景..."
read
echo ""

# 场景3: 高并发测试 (100请求, 20并发, 50 tokens)
echo "[场景3] 高并发测试"
echo "  参数: 100请求, 20并发, 50 tokens"
echo "  用途: 评估系统在高并发下的表现"
echo ""
python3 tools/run_cllm_ollama_comparison.py \
    --requests 100 \
    --concurrency 20 \
    --max-tokens 50 \
    --save-results
echo ""
echo "按Enter继续下一个场景..."
read
echo ""

# 场景4: 长文本生成测试 (50请求, 5并发, 200 tokens)
echo "[场景4] 长文本生成测试"
echo "  参数: 50请求, 5并发, 200 tokens"
echo "  用途: 评估系统生成长文本的能力"
echo ""
python3 tools/run_cllm_ollama_comparison.py \
    --requests 50 \
    --concurrency 5 \
    --max-tokens 200 \
    --save-results
echo ""

echo "========================================"
echo "所有测试场景完成！"
echo "========================================"
echo ""
echo "测试结果已保存到:"
echo "  - JSON结果: results/"
echo "  - 测试报告: docs/testing/"
echo ""
