/**
 * @file feed_forward.h
 * @brief Feed-Forward Network (SwiGLU) 的简化实现
 */
#pragma once

#include "cllm/kylin/tensor.h"
#include "cllm/kylin/kernels.h"

namespace cllm {
namespace kylin {

/**
 * @brief 前馈网络（SwiGLU），用于 Transformer Block 中
 */
class FeedForwardNetwork {
public:
    FeedForwardNetwork(size_t hiddenSize, size_t intermediateSize);

    /// 设置权重（生命周期由上层管理）
    void setWeights(
        const Tensor& wGate,   // [hiddenSize, intermediateSize]
        const Tensor& wUp,     // [hiddenSize, intermediateSize]
        const Tensor& wDown    // [intermediateSize, hiddenSize]
    );

    /// 前向传播
    /// 输入: [batch, seq_len, hiddenSize]
    /// 输出: [batch, seq_len, hiddenSize]
    Tensor forward(const Tensor& input) const;

private:
    size_t hiddenSize_;
    size_t intermediateSize_;

    Tensor wGate_;
    Tensor wUp_;
    Tensor wDown_;
};

}  // namespace kylin
}  // namespace cllm
