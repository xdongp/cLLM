/**
 * @file transformer_model.cpp
 * @brief 简化版 Transformer 模型（MVP，用于自研推理引擎）
 */

#include "cllm/kylin/transformer_model.h"

#include "cllm/kylin/kernels.h"

#include <stdexcept>

namespace cllm {
namespace kylin {

TransformerModel::TransformerModel(const ModelConfig& config)
    : config_(config)
    , embedding_(nullptr)
    , lmHead_(nullptr)
    , finalNormWeight_(nullptr)
    , rmsEps_(1e-5f) {  // 默认一个较小的 eps
    layers_.reserve(config_.numLayers);
    for (size_t i = 0; i < config_.numLayers; ++i) {
        layers_.emplace_back(
            config_.hiddenSize,
            config_.numAttentionHeads,
            config_.intermediateSize,
            rmsEps_
        );
    }
}

void TransformerModel::setEmbeddingWeight(const Tensor& embedding) {
    embedding_ = &embedding;
}

void TransformerModel::setLmHeadWeight(const Tensor& lmHead) {
    lmHead_ = &lmHead;
}

void TransformerModel::setBlockWeights(
    size_t layerIndex,
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo,
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown,
    const Tensor& norm1Weight,
    const Tensor& norm2Weight
) {
    if (layerIndex >= layers_.size()) {
        throw std::out_of_range("TransformerModel::setBlockWeights: layerIndex out of range");
    }

    auto& block = layers_[layerIndex];
    block.setAttentionWeights(wq, wk, wv, wo);
    block.setFFNWeights(wGate, wUp, wDown);
    block.setNormWeights(norm1Weight, norm2Weight);
}

void TransformerModel::setFinalNormWeight(const Tensor& normWeight) {
    finalNormWeight_ = &normWeight;
}

Tensor TransformerModel::forward(const std::vector<int>& inputIds) const {
    if (!embedding_ || !lmHead_) {
        throw std::runtime_error("TransformerModel embedding or lm_head not set");
    }

    size_t seqLen = inputIds.size();
    if (seqLen == 0) {
        throw std::invalid_argument("TransformerModel::forward: empty inputIds");
    }

    size_t hidden = config_.hiddenSize;
    size_t vocab = config_.vocabSize;

    // 1. embedding 查表: [1, seqLen, hidden]
    Tensor hiddenStates({1, seqLen, hidden});
    for (size_t t = 0; t < seqLen; ++t) {
        int tokenId = inputIds[t];
        if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocab) {
            throw std::out_of_range("TransformerModel::forward: token id out of vocab range");
        }

        // embedding_ 形状假设为 [vocab, hidden]
        const float* embRow = embedding_->data() + static_cast<size_t>(tokenId) * hidden;
        float* dst = hiddenStates.data() + t * hidden;
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] = embRow[h];
        }
    }

    // 2. 通过 N 层 TransformerBlock
    for (const auto& block : layers_) {
        hiddenStates = block.forward(hiddenStates);
    }

    // 3. 最终 RMSNorm（如果提供了权重）
    if (finalNormWeight_) {
        using namespace kernels;
        Tensor normOut({1, seqLen, hidden});
        rmsnorm(hiddenStates.data(), normOut.data(), finalNormWeight_->data(), 1 * seqLen, hidden, rmsEps_);
        hiddenStates = std::move(normOut);
    }

    // 4. 投影到 vocab 维度: [seqLen, hidden] @ [hidden, vocab] -> [seqLen, vocab]
    Tensor logits({seqLen, vocab});

    using namespace kernels;
    // 把 hiddenStates 视为 [seqLen, hidden]
    matmul(hiddenStates.data(), lmHead_->data(), logits.data(), seqLen, vocab, hidden);

    return logits;
}

}  // namespace kylin
}  // namespace cllm
