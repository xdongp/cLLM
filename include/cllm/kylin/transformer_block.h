/**
 * @file transformer_block.h
 * @brief Transformer Block 的简化实现（MVP，无 KV Cache）
 */
#pragma once

#include "cllm/kylin/tensor.h"
#include "cllm/kylin/attention.h"
#include "cllm/kylin/feed_forward.h"
#include "cllm/kylin/kernels.h"

namespace cllm {
namespace kylin {

/**
 * @brief 单个 Transformer Block：RMSNorm + MHA + 残差 + RMSNorm + FFN + 残差
 */
class TransformerBlock {
public:
    TransformerBlock(
        size_t hiddenSize,
        size_t numQHeads,
        size_t numKVHeads,
        size_t intermediateSize,
        float rmsNormEps,
        float ropeTheta = 10000.0f,
        // P3修复：RoPE扩展参数
        size_t maxSequenceLength = 2048,
        size_t ropeNctxOrig = 0,
        float ropeFreqScale = 1.0f,
        int ropeType = 0,
        float ropeExtFactor = 1.0f
    );

    /// 设置注意力权重
    void setAttentionWeights(
        const Tensor& wq,
        const Tensor& wk,
        const Tensor& wv,
        const Tensor& wo
    );

    /// 设置前馈网络权重
    void setFFNWeights(
        const Tensor& wGate,
        const Tensor& wUp,
        const Tensor& wDown
    );

    /// 设置两层RMSNorm的权重
    void setNormWeights(
        const Tensor& norm1Weight,
        const Tensor& norm2Weight
    );
    
    /// 设置Q/K的独立归一化权重（Qwen3等模型需要，可选）
    void setAttnQKNormWeights(
        const Tensor& attnQNormWeight,
        const Tensor& attnKNormWeight
    );

    /// 前向传播（无 KV Cache）
    /// 输入: [batch, seq_len, hiddenSize]
    /// 输出: [batch, seq_len, hiddenSize]
    Tensor forward(const Tensor& input) const;

private:
    size_t hiddenSize_;
    float rmsEps_;

    MultiHeadAttention attention_;
    FeedForwardNetwork ffn_;

    Tensor norm1Weight_;
    Tensor norm2Weight_;
    Tensor attnQNormWeight_;  // Q的独立归一化权重（可选）
    Tensor attnKNormWeight_;  // K的独立归一化权重（可选）
    bool hasAttnQKNorm_;      // 是否已设置Q/K归一化权重
};

}  // namespace kylin
}  // namespace cllm
