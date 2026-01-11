/**
 * @file kernels.cpp
 * @brief 自研推理引擎基础算子实现（使用 Eigen 优化 matmul）
 */

#include "cllm/kylin/kernels.h"

#include <algorithm>
#include <cmath>
#include <limits>

// Eigen 高性能线性代数库（header-only）
#include <Eigen/Dense>

namespace cllm {
namespace kylin {
namespace kernels {

void matmul(
    const float* A,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K,
    bool transposeA,
    bool transposeB
) {
    // 使用 Eigen 进行高性能矩阵乘法（自动 SIMD 优化 + cache blocking）
    using MatrixXfRM = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    
    // 包装输入矩阵（避免拷贝）
    Eigen::Map<const MatrixXfRM> matA(A, transposeA ? K : M, transposeA ? M : K);
    Eigen::Map<const MatrixXfRM> matB(B, transposeB ? N : K, transposeB ? K : N);
    Eigen::Map<MatrixXfRM> matC(C, M, N);
    
    // 执行矩阵乘法（Eigen 会自动选择最优算法）
    if (transposeA && transposeB) {
        matC.noalias() = matA.transpose() * matB.transpose();
    } else if (transposeA) {
        matC.noalias() = matA.transpose() * matB;
    } else if (transposeB) {
        matC.noalias() = matA * matB.transpose();
    } else {
        matC.noalias() = matA * matB;
    }
}

void softmax_stable(
    const float* input,
    float* output,
    size_t outerDim,
    size_t innerDim
) {
    for (size_t i = 0; i < outerDim; ++i) {
        const float* rowIn = input + i * innerDim;
        float* rowOut = output + i * innerDim;

        // 1. 找最大值
        float maxVal = -std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < innerDim; ++j) {
            maxVal = std::max(maxVal, rowIn[j]);
        }

        // 2. 计算 exp(x - max) 和和
        float sumExp = 0.0f;
        for (size_t j = 0; j < innerDim; ++j) {
            float v = std::exp(rowIn[j] - maxVal);
            rowOut[j] = v;
            sumExp += v;
        }

        // 3. 归一化
        if (sumExp > 0.0f) {
            float invSum = 1.0f / sumExp;
            for (size_t j = 0; j < innerDim; ++j) {
                rowOut[j] *= invSum;
            }
        }
    }
}

void rmsnorm(
    const float* input,
    float* output,
    const float* weight,
    size_t rows,
    size_t cols,
    float eps
) {
    for (size_t i = 0; i < rows; ++i) {
        const float* rowIn = input + i * cols;
        float* rowOut = output + i * cols;

        // 1. 计算 mean(x^2)
        float meanSq = 0.0f;
        for (size_t j = 0; j < cols; ++j) {
            meanSq += rowIn[j] * rowIn[j];
        }
        meanSq /= static_cast<float>(cols);

        // 2. 计算 1 / sqrt(mean(x^2) + eps)
        float invRms = 1.0f / std::sqrt(meanSq + eps);

        // 3. 归一化并乘以权重
        for (size_t j = 0; j < cols; ++j) {
            float v = rowIn[j] * invRms;
            rowOut[j] = v * (weight ? weight[j] : 1.0f);
        }
    }
}

void silu(
    const float* input,
    float* output,
    size_t size
) {
    for (size_t i = 0; i < size; ++i) {
        float x = input[i];
        float sig = 1.0f / (1.0f + std::exp(-x));
        output[i] = x * sig;
    }
}

}  // namespace kernels
}  // namespace kylin
}  // namespace cllm
