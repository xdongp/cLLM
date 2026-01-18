/**
 * @file kernels.cpp
 * @brief 自研推理引擎基础算子实现（使用 Eigen 优化 matmul）
 */

#include "cllm/kylin/kernels.h"
#include "cllm/kylin/quantization.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <cstring>

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
    // 验证输入指针
    if (A == nullptr || B == nullptr || C == nullptr) {
        throw std::runtime_error("matmul: null pointer detected");
    }
    
    // 简单的三重循环实现（避免 Eigen 可能的问题）
    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // 计算 A[m][k] 的索引
                size_t a_idx = transposeA ? (k * M + m) : (m * K + k);
                // 计算 B[k][n] 的索引
                size_t b_idx = transposeB ? (n * K + k) : (k * N + n);
                
                sum += A[a_idx] * B[b_idx];
            }
            C[m * N + n] = sum;
        }
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
            // 避免数值溢出：裁剪输入值
            float x = rowIn[j];
            if (x > 1.0e18f) x = 1.0e18f;
            if (x < -1.0e18f) x = -1.0e18f;
            meanSq += x * x;
        }
        meanSq /= static_cast<float>(cols);

        // 2. 计算 1 / sqrt(mean(x^2) + eps)
        float invRms = 1.0f / std::sqrt(meanSq + eps);

        // 3. 归一化并乘以权重
        for (size_t j = 0; j < cols; ++j) {
            // 再次裁剪输入值
            float x = rowIn[j];
            if (x > 1.0e18f) x = 1.0e18f;
            if (x < -1.0e18f) x = -1.0e18f;
            float v = x * invRms;
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

void matmul_q4_K_f32(
    const void* A_quantized,
    const float* B,
    float* C,
    size_t M,
    size_t N,
    size_t K
) {
    using namespace quantization;
    
    const block_q4_K* A_blocks = static_cast<const block_q4_K*>(A_quantized);
    
    // 计算每行需要的块数
    const size_t blocksPerRow = (K + QK_K - 1) / QK_K;
    
    // 对每一行进行矩阵乘法
    for (size_t m = 0; m < M; ++m) {
        // 获取当前行的块
        const block_q4_K* rowBlocks = A_blocks + m * blocksPerRow;
        
        // 对每一列计算点积
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            
            // 处理当前行的所有块
            for (size_t blockIdx = 0; blockIdx < blocksPerRow; ++blockIdx) {
                const block_q4_K* block = &rowBlocks[blockIdx];
                const uint8_t* q = block->qs;
                
                // 获取超级块scale
                const float d = quantization::fp16_to_fp32(block->d);
                const float min = quantization::fp16_to_fp32(block->dmin);
                
                int is = 0;
                uint8_t sc, m_val;
                
                // 处理块内的每个64元素组
                for (int j = 0; j < QK_K && (blockIdx * QK_K + j) < K; j += 64) {
                    // 获取两个子块的scale和min
                    quantization::get_scale_min_k4(is + 0, block->scales, &sc, &m_val);
                    const float d1 = d * sc;
                    const float m1 = min * m_val;
                    
                    quantization::get_scale_min_k4(is + 1, block->scales, &sc, &m_val);
                    const float d2 = d * sc;
                    const float m2 = min * m_val;
                    
                    // 处理前32个元素
                    for (int l = 0; l < 32 && (blockIdx * QK_K + j + l) < K; ++l) {
                        size_t k_idx = blockIdx * QK_K + j + l;
                        float a_val = d1 * (q[l] & 0xF) - m1;
                        sum += a_val * B[k_idx * N + n];
                    }
                    
                    // 处理后32个元素
                    for (int l = 0; l < 32 && (blockIdx * QK_K + j + 32 + l) < K; ++l) {
                        size_t k_idx = blockIdx * QK_K + j + 32 + l;
                        float a_val = d2 * (q[l] >> 4) - m2;
                        sum += a_val * B[k_idx * N + n];
                    }
                    
                    q += 32;
                    is += 2;
                }
            }
            
            C[m * N + n] = sum;
        }
    }
}

}  // namespace kernels
}  // namespace kylin
}  // namespace cllm
