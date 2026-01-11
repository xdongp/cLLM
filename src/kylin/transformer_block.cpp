/**
 * @file transformer_block.cpp
 * @brief Transformer Block 的简化实现（MVP，无 KV Cache）
 */

#include "cllm/kylin/transformer_block.h"

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

    using namespace kernels;

    // 1. Pre-Norm + Attention
    Tensor norm1({batch, seqLen, hiddenSize_});
    rmsnorm(input.data(), norm1.data(), norm1Weight_->data(), batch * seqLen, hiddenSize_, rmsEps_);

    Tensor attnOut = attention_.forwardNoKV(norm1);

    // 残差1: x1 = attnOut + input
    Tensor x1({batch, seqLen, hiddenSize_});
    for (size_t i = 0; i < batch * seqLen * hiddenSize_; ++i) {
        x1[i] = attnOut[i] + input[i];
    }

    // 2. Pre-Norm + FFN
    Tensor norm2({batch, seqLen, hiddenSize_});
    rmsnorm(x1.data(), norm2.data(), norm2Weight_->data(), batch * seqLen, hiddenSize_, rmsEps_);

    Tensor ffnOut = ffn_.forward(norm2);

    // 残差2: out = ffnOut + x1
    Tensor output({batch, seqLen, hiddenSize_});
    for (size_t i = 0; i < batch * seqLen * hiddenSize_; ++i) {
        output[i] = ffnOut[i] + x1[i];
    }

    return output;
}

}  // namespace kylin
}  // namespace cllm
