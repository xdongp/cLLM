/**
 * @file transformer_model.cpp
 * @brief 简化版 Transformer 模型（MVP，用于自研推理引擎）
 */

#include "cllm/kylin/transformer_model.h"

#include "cllm/kylin/kernels.h"
#include "cllm/common/logger.h"

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

    CLLM_INFO("TransformerModel::forward: seqLen=%zu, hidden=%zu, vocab=%zu", seqLen, hidden, vocab);

    // 1. embedding 查表: [1, seqLen, hidden]
    CLLM_INFO("TransformerModel::forward: 开始 embedding 查表...");
    
    // 验证 embedding 指针和形状
    if (embedding_ == nullptr) {
        throw std::runtime_error("TransformerModel::forward: embedding_ is null");
    }
    
    const auto& embShape = embedding_->shape();
    if (embShape.size() != 2) {
        throw std::runtime_error("TransformerModel::forward: embedding shape must be 2D");
    }
    
    size_t embVocab = embShape[0];
    size_t embHidden = embShape[1];
    CLLM_INFO("TransformerModel::forward: embedding 形状 [vocab=%zu, hidden=%zu]", embVocab, embHidden);
    
    if (embHidden != hidden) {
        throw std::runtime_error("TransformerModel::forward: embedding hidden size mismatch");
    }
    
    Tensor hiddenStates({1, seqLen, hidden});
    for (size_t t = 0; t < seqLen; ++t) {
        int tokenId = inputIds[t];
        if (tokenId < 0 || static_cast<size_t>(tokenId) >= vocab) {
            throw std::out_of_range("TransformerModel::forward: token id out of vocab range");
        }
        
        // 检查 tokenId 是否在 embedding vocab 范围内
        if (static_cast<size_t>(tokenId) >= embVocab) {
            throw std::out_of_range("TransformerModel::forward: token id out of embedding vocab range");
        }

        // embedding_ 形状假设为 [vocab, hidden]
        const float* embData = embedding_->data();
        if (embData == nullptr) {
            throw std::runtime_error("TransformerModel::forward: embedding data is null");
        }
        
        const float* embRow = embData + static_cast<size_t>(tokenId) * embHidden;
        float* dst = hiddenStates.data() + t * hidden;
        
        // 检查指针是否有效
        if (dst == nullptr) {
            throw std::runtime_error("TransformerModel::forward: hiddenStates data is null");
        }
        
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] = embRow[h];
        }
    }
    CLLM_INFO("TransformerModel::forward: embedding 查表完成");

    // 2. 通过 N 层 TransformerBlock
    CLLM_INFO("TransformerModel::forward: 开始通过 %zu 层 TransformerBlock...", layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        CLLM_INFO("TransformerModel::forward: 处理第 %zu 层...", i);
        hiddenStates = layers_[i].forward(hiddenStates);
        CLLM_INFO("TransformerModel::forward: 第 %zu 层处理完成", i);
    }
    CLLM_INFO("TransformerModel::forward: 所有 TransformerBlock 处理完成");

    // 3. 最终 RMSNorm（如果提供了权重）
    if (finalNormWeight_) {
        CLLM_INFO("TransformerModel::forward: 开始最终 RMSNorm...");
        using namespace kernels;
        Tensor normOut({1, seqLen, hidden});
        rmsnorm(hiddenStates.data(), normOut.data(), finalNormWeight_->data(), 1 * seqLen, hidden, rmsEps_);
        hiddenStates = std::move(normOut);
        CLLM_INFO("TransformerModel::forward: 最终 RMSNorm 完成");
    }

    // 4. 投影到 vocab 维度: [seqLen, hidden] @ [hidden, vocab] -> [seqLen, vocab]
    CLLM_INFO("TransformerModel::forward: 开始投影到 vocab 维度...");
    Tensor logits({seqLen, vocab});

    using namespace kernels;
    // 把 hiddenStates 从 [1, seqLen, hidden] 重塑为 [seqLen, hidden]
    matmul(hiddenStates.data() + 0, lmHead_->data(), logits.data(), seqLen, vocab, hidden);
    CLLM_INFO("TransformerModel::forward: 投影到 vocab 维度完成");

    return logits;
}

}  // namespace kylin
}  // namespace cllm
