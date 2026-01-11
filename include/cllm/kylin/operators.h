/**
 * @file operators.h
 * @brief Kylin 推理引擎核心算子库
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 编程规范：C++编程规范.md
 */
#pragma once

#include "tensor.h"
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {

/**
 * @brief 核心算子命名空间
 */
namespace ops {

// ============================================================================
// 基础算术运算
// ============================================================================

/**
 * @brief 矩阵乘法 C = A @ B
 * @param A 输入矩阵 [M, K]
 * @param B 输入矩阵 [K, N]
 * @return 输出矩阵 [M, N]
 */
Tensor matmul(const Tensor& A, const Tensor& B);

/**
 * @brief 元素级加法 C = A + B
 */
Tensor add(const Tensor& A, const Tensor& B);

/**
 * @brief 元素级乘法 C = A * B
 */
Tensor multiply(const Tensor& A, const Tensor& B);

// ============================================================================
// 激活函数
// ============================================================================

/**
 * @brief SiLU 激活函数 (x * sigmoid(x))
 */
Tensor silu(const Tensor& input);

/**
 * @brief Softmax (沿最后一个维度)
 * @param input 输入张量 [..., N]
 * @return 输出张量 [..., N]
 */
Tensor softmax(const Tensor& input);

// ============================================================================
// Normalization
// ============================================================================

/**
 * @brief RMS Normalization
 * @param input 输入张量 [batch, seq, hidden]
 * @param weight 权重 [hidden]
 * @param eps 防止除零的小常数
 * @return 输出张量 [batch, seq, hidden]
 */
Tensor rmsNorm(const Tensor& input, const Tensor& weight, float eps = 1e-6f);

// ============================================================================
// 张量变换
// ============================================================================

/**
 * @brief 张量重塑 (reshape)
 */
Tensor reshape(const Tensor& input, const std::vector<size_t>& newShape);

/**
 * @brief 张量转置 (transpose)
 * @param dim0 第一个维度
 * @param dim1 第二个维度
 */
Tensor transpose(const Tensor& input, size_t dim0, size_t dim1);

/**
 * @brief Embedding 查找
 * @param embeddings 嵌入表 [vocab_size, hidden_size]
 * @param indices 索引 [batch, seq_len]
 * @return 输出张量 [batch, seq_len, hidden_size]
 */
Tensor embedding(const Tensor& embeddings, const std::vector<int>& indices);

} // namespace ops

} // namespace kylin
} // namespace cllm
