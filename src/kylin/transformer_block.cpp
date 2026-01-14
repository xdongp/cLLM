/**
 * @file transformer_block.cpp
 * @brief Transformer Block 的简化实现（MVP，无 KV Cache）
 */

#include "cllm/kylin/transformer_block.h"

#include "cllm/common/logger.h"

#include <stdexcept>

namespace cllm {
namespace kylin {

TransformerBlock::TransformerBlock(
    size_t hiddenSize,
    size_t numHeads,
    size_t intermediateSize,
    float rmsNormEps,
    float ropeTheta
)
    : hiddenSize_(hiddenSize)
    , rmsEps_(rmsNormEps)
    , attention_(hiddenSize, numHeads, ropeTheta)
    , ffn_(hiddenSize, intermediateSize)
    , norm1Weight_(nullptr)
    , norm2Weight_(nullptr) {}

void TransformerBlock::setAttentionWeights(
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo
) {
    attention_.setWeights(wq, wk, wv, wo);
}

void TransformerBlock::setFFNWeights(
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown
) {
    ffn_.setWeights(wGate, wUp, wDown);
}

void TransformerBlock::setNormWeights(
    const Tensor& norm1Weight,
    const Tensor& norm2Weight
) {
    norm1Weight_ = &norm1Weight;
    norm2Weight_ = &norm2Weight;
}

Tensor TransformerBlock::forward(const Tensor& input) const {
    CLLM_DEBUG("TransformerBlock::forward: 开始执行");
    
    if (!norm1Weight_ || !norm2Weight_) {
        throw std::runtime_error("TransformerBlock norm weights not set");
    }

    const auto& inShape = input.shape();
    if (inShape.size() != 3) {
        throw std::invalid_argument("TransformerBlock::forward expects [batch, seq, hidden]");
    }

    size_t batch = inShape[0];
    size_t seqLen = inShape[1];
    size_t hidden = inShape[2];
    if (hidden != hiddenSize_) {
        throw std::invalid_argument("TransformerBlock: input hidden size mismatch");
    }

    CLLM_DEBUG("TransformerBlock::forward: 输入形状 [batch=%zu, seqLen=%zu, hidden=%zu]", batch, seqLen, hidden);

    using namespace kernels;

    // 1. Pre-Norm + Attention
    CLLM_DEBUG("TransformerBlock::forward: 开始第一层 RMSNorm");
    Tensor norm1({batch, seqLen, hiddenSize_});
    rmsnorm(input.data(), norm1.data(), norm1Weight_->data(), batch * seqLen, hiddenSize_, rmsEps_);
    CLLM_DEBUG("TransformerBlock::forward: 第一层 RMSNorm 完成");

    CLLM_DEBUG("TransformerBlock::forward: 开始 Attention 计算");
    Tensor attnOut = attention_.forwardNoKV(norm1);
    CLLM_DEBUG("TransformerBlock::forward: Attention 计算完成");

    // 残差1: x1 = attnOut + input
    CLLM_DEBUG("TransformerBlock::forward: 开始第一层残差连接");
    Tensor x1({batch, seqLen, hiddenSize_});
    for (size_t i = 0; i < batch * seqLen * hiddenSize_; ++i) {
        x1[i] = attnOut[i] + input[i];
    }
    CLLM_DEBUG("TransformerBlock::forward: 第一层残差连接完成");

    // 2. Pre-Norm + FFN
    CLLM_DEBUG("TransformerBlock::forward: 开始第二层 RMSNorm");
    Tensor norm2({batch, seqLen, hiddenSize_});
    rmsnorm(x1.data(), norm2.data(), norm2Weight_->data(), batch * seqLen, hiddenSize_, rmsEps_);
    CLLM_DEBUG("TransformerBlock::forward: 第二层 RMSNorm 完成");

    CLLM_DEBUG("TransformerBlock::forward: 开始 FFN 计算");
    Tensor ffnOut = ffn_.forward(norm2);
    CLLM_DEBUG("TransformerBlock::forward: FFN 计算完成");

    // 残差2: out = ffnOut + x1
    CLLM_DEBUG("TransformerBlock::forward: 开始第二层残差连接");
    Tensor output({batch, seqLen, hiddenSize_});
    for (size_t i = 0; i < batch * seqLen * hiddenSize_; ++i) {
        output[i] = ffnOut[i] + x1[i];
    }
    CLLM_DEBUG("TransformerBlock::forward: 第二层残差连接完成");

    CLLM_DEBUG("TransformerBlock::forward: 执行完成");
    return output;
}

}  // namespace kylin
}  // namespace cllm
