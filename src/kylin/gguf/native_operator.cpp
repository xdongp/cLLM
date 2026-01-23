/**
 * @file native_operator.cpp
 * @brief 自研 Native 算子实现
 * 
 * 复用 kernels.cpp 中的实现
 */

#include "cllm/kylin/gguf/native_operator.h"
#include "cllm/kylin/core/kernels.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <stdexcept>

namespace cllm {
namespace kylin {

void NativeOperator::matmul(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    bool transposeA,
    bool transposeB
) {
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    
    if (shapeA.size() < 2 || shapeB.size() < 2) {
        throw std::runtime_error("NativeOperator::matmul: tensors must be at least 2D");
    }
    
    // 获取矩阵维度
    size_t M = transposeA ? shapeA.back() : shapeA[shapeA.size() - 2];
    size_t K_A = transposeA ? shapeA[shapeA.size() - 2] : shapeA.back();
    size_t K_B = transposeB ? shapeB.back() : shapeB[shapeB.size() - 2];
    size_t N = transposeB ? shapeB[shapeB.size() - 2] : shapeB.back();
    
    if (K_A != K_B) {
        CLLM_ERROR("[NativeOperator] matmul dimension mismatch: K_A=%zu, K_B=%zu", K_A, K_B);
        throw std::runtime_error("NativeOperator::matmul: dimension mismatch");
    }
    
    // 处理批次维度
    size_t batchA = 1, batchB = 1;
    for (size_t i = 0; i < shapeA.size() - 2; ++i) {
        batchA *= shapeA[i];
    }
    for (size_t i = 0; i < shapeB.size() - 2; ++i) {
        batchB *= shapeB[i];
    }
    
    size_t batch = std::max(batchA, batchB);
    
    // 设置输出形状
    std::vector<size_t> outShape;
    if (shapeA.size() > 2) {
        for (size_t i = 0; i < shapeA.size() - 2; ++i) {
            outShape.push_back(shapeA[i]);
        }
    }
    outShape.push_back(M);
    outShape.push_back(N);
    C.resize(outShape);
    
    // 执行批量矩阵乘法
    size_t strideA = M * K_A;
    size_t strideB = K_B * N;
    size_t strideC = M * N;
    
    for (size_t b = 0; b < batch; ++b) {
        const float* pA = A.data() + (b % batchA) * strideA;
        const float* pB = B.data() + (b % batchB) * strideB;
        float* pC = C.data() + b * strideC;
        
        kernels::matmul(pA, pB, pC, M, N, K_A, transposeA, transposeB);
    }
}

void NativeOperator::add(const Tensor& A, const Tensor& B, Tensor& C) {
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    
    // 简单的逐元素加法（支持广播）
    size_t sizeA = A.size();
    size_t sizeB = B.size();
    
    if (sizeA >= sizeB) {
        C.resize(shapeA);
        for (size_t i = 0; i < sizeA; ++i) {
            C.data()[i] = A.data()[i] + B.data()[i % sizeB];
        }
    } else {
        C.resize(shapeB);
        for (size_t i = 0; i < sizeB; ++i) {
            C.data()[i] = A.data()[i % sizeA] + B.data()[i];
        }
    }
}

void NativeOperator::mul(const Tensor& A, const Tensor& B, Tensor& C) {
    const auto& shapeA = A.shape();
    const auto& shapeB = B.shape();
    
    // 简单的逐元素乘法（支持广播）
    size_t sizeA = A.size();
    size_t sizeB = B.size();
    
    if (sizeA >= sizeB) {
        C.resize(shapeA);
        for (size_t i = 0; i < sizeA; ++i) {
            C.data()[i] = A.data()[i] * B.data()[i % sizeB];
        }
    } else {
        C.resize(shapeB);
        for (size_t i = 0; i < sizeB; ++i) {
            C.data()[i] = A.data()[i % sizeA] * B.data()[i];
        }
    }
}

void NativeOperator::silu(const Tensor& input, Tensor& output) {
    output.resize(input.shape());
    kernels::silu(input.data(), output.data(), input.size());
}

void NativeOperator::softmax(const Tensor& input, Tensor& output, int axis) {
    const auto& shape = input.shape();
    
    if (shape.empty()) {
        throw std::runtime_error("NativeOperator::softmax: empty tensor");
    }
    
    // 处理负数 axis
    int ndim = static_cast<int>(shape.size());
    if (axis < 0) {
        axis = ndim + axis;
    }
    if (axis < 0 || axis >= ndim) {
        throw std::runtime_error("NativeOperator::softmax: invalid axis");
    }
    
    output.resize(shape);
    
    // 计算外层和内层维度
    size_t outerDim = 1;
    size_t innerDim = shape[axis];
    size_t stride = 1;
    
    for (int i = 0; i < axis; ++i) {
        outerDim *= shape[i];
    }
    for (int i = axis + 1; i < ndim; ++i) {
        stride *= shape[i];
    }
    
    // 如果 stride == 1（最后一维），直接调用优化的实现
    if (stride == 1) {
        kernels::softmax_stable(input.data(), output.data(), outerDim, innerDim);
    } else {
        // 通用实现（非连续内存）
        for (size_t outer = 0; outer < outerDim; ++outer) {
            for (size_t s = 0; s < stride; ++s) {
                // 找最大值
                float maxVal = -std::numeric_limits<float>::infinity();
                for (size_t i = 0; i < innerDim; ++i) {
                    size_t idx = outer * innerDim * stride + i * stride + s;
                    maxVal = std::max(maxVal, input.data()[idx]);
                }
                
                // 计算 exp 和 sum
                float sumExp = 0.0f;
                for (size_t i = 0; i < innerDim; ++i) {
                    size_t idx = outer * innerDim * stride + i * stride + s;
                    float v = std::exp(input.data()[idx] - maxVal);
                    output.data()[idx] = v;
                    sumExp += v;
                }
                
                // 归一化
                if (sumExp > 0.0f) {
                    float invSum = 1.0f / sumExp;
                    for (size_t i = 0; i < innerDim; ++i) {
                        size_t idx = outer * innerDim * stride + i * stride + s;
                        output.data()[idx] *= invSum;
                    }
                }
            }
        }
    }
}

void NativeOperator::rmsNorm(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    const auto& shape = input.shape();
    
    if (shape.empty()) {
        throw std::runtime_error("NativeOperator::rmsNorm: empty tensor");
    }
    
    size_t cols = shape.back();
    size_t rows = input.size() / cols;
    
    if (weight.size() != cols) {
        throw std::runtime_error("NativeOperator::rmsNorm: weight size mismatch");
    }
    
    output.resize(shape);
    kernels::rmsnorm(input.data(), output.data(), weight.data(), rows, cols, eps);
}

void NativeOperator::rope(
    Tensor& q,
    Tensor& k,
    size_t startPos,
    float freqBase
) {
    const auto& qShape = q.shape();
    const auto& kShape = k.shape();
    
    if (qShape.size() != 4 || kShape.size() != 4) {
        throw std::runtime_error("NativeOperator::rope: expected 4D tensors [batch, heads, seq, dim]");
    }
    
    size_t batch = qShape[0];
    size_t numHeads = qShape[1];
    size_t numKVHeads = kShape[1];
    size_t seqLen = qShape[2];
    size_t headDim = qShape[3];
    
    // 预计算频率
    std::vector<float> freqs(headDim / 2);
    for (size_t i = 0; i < headDim / 2; ++i) {
        freqs[i] = 1.0f / std::pow(freqBase, static_cast<float>(2 * i) / static_cast<float>(headDim));
    }
    
    // 应用 RoPE 到 Q
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numHeads; ++h) {
            for (size_t s = 0; s < seqLen; ++s) {
                size_t pos = startPos + s;
                size_t offset = ((b * numHeads + h) * seqLen + s) * headDim;
                float* qData = q.data() + offset;
                
                for (size_t i = 0; i < headDim / 2; ++i) {
                    float theta = static_cast<float>(pos) * freqs[i];
                    float cos_theta = std::cos(theta);
                    float sin_theta = std::sin(theta);
                    
                    float x0 = qData[2 * i];
                    float x1 = qData[2 * i + 1];
                    
                    qData[2 * i] = x0 * cos_theta - x1 * sin_theta;
                    qData[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
    
    // 应用 RoPE 到 K
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numKVHeads; ++h) {
            for (size_t s = 0; s < seqLen; ++s) {
                size_t pos = startPos + s;
                size_t offset = ((b * numKVHeads + h) * seqLen + s) * headDim;
                float* kData = k.data() + offset;
                
                for (size_t i = 0; i < headDim / 2; ++i) {
                    float theta = static_cast<float>(pos) * freqs[i];
                    float cos_theta = std::cos(theta);
                    float sin_theta = std::sin(theta);
                    
                    float x0 = kData[2 * i];
                    float x1 = kData[2 * i + 1];
                    
                    kData[2 * i] = x0 * cos_theta - x1 * sin_theta;
                    kData[2 * i + 1] = x0 * sin_theta + x1 * cos_theta;
                }
            }
        }
    }
}

} // namespace kylin
} // namespace cllm
