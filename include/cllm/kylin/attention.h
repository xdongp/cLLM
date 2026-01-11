/**
 * @file attention.h
 * @brief Multi-Head Attention 的简化实现（MVP，无 KV Cache）
 */
#pragma once

#include "cllm/kylin/tensor.h"
#include "cllm/kylin/kernels.h"
#include "cllm/kylin/rope.h"

namespace cllm {
namespace kylin {

/**
 * @brief 多头自注意力（不含KV缓存，MVP阶段）
 *
 * 假设输入形状为 [batch, seq_len, hidden_size]。
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(
        size_t hiddenSize,
        size_t numHeads,
        float ropeTheta = 10000.0f
    );

    /// 设置权重（MVP阶段简单按引用保存，生命周期由上层保证）
    void setWeights(
        const Tensor& wq,
        const Tensor& wk,
        const Tensor& wv,
        const Tensor& wo
    );

    /// 无 KV 的前向传播
    /// 输入: [batch, seq_len, hidden_size]
    /// 输出: [batch, seq_len, hidden_size]
    Tensor forwardNoKV(const Tensor& input) const;

private:
    size_t hiddenSize_;
    size_t numHeads_;
    size_t headDim_;

    const Tensor* wq_;
    const Tensor* wk_;
    const Tensor* wv_;
    const Tensor* wo_;

    RoPE rope_;
};

}  // namespace kylin
}  // namespace cllm
