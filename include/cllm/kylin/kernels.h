/**
 * @file kernels.h
 * @brief 自研推理引擎MVP阶段使用的基础算子声明
 */
#pragma once

#include <cstddef>

namespace cllm {
namespace kylin {
namespace kernels {

/**
 * @brief 通用矩阵乘法 C = A @ B
 *
 * 所有矩阵均采用 row-major 存储：
 * - A: [M, K]
 * - B: [K, N]
 * - C: [M, N]
 *
 * MVP 阶段实现为简单三层 for 循环，后续可替换为 SIMD 优化版本。
 */
void matmul(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K,
    bool transposeA = false,
    bool transposeB = false
);

/**
 * @brief 数值稳定的 softmax 实现（按最后一维归一化）
 *
 * 输入视为一个二维矩阵：outerDim x innerDim，其中 innerDim 是归一化维度。
 */
void softmax_stable(
    const float* input,
    float* output,
    size_t outerDim,
    size_t innerDim
);

/**
 * @brief RMSNorm：按最后一维做 Root Mean Square 归一化
 *
 * input 视为 [rows, cols]，weight 形状为 [cols]。
 */
void rmsnorm(
    const float* input,
    float* output,
    const float* weight,
    size_t rows,
    size_t cols,
    float eps
);

/**
 * @brief SiLU 激活函数: y = x * sigmoid(x)
 */
void silu(
    const float* input,
    float* output,
    size_t size
);

}  // namespace kernels
}  // namespace kylin
}  // namespace cllm
