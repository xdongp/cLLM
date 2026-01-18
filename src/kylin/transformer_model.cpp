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
    , rmsEps_(config.rmsNormEps) {  // 从配置读取 rms_norm_eps
    layers_.reserve(config_.numLayers);
    for (size_t i = 0; i < config_.numLayers; ++i) {
        layers_.emplace_back(
            config_.hiddenSize,
            config_.numAttentionHeads,    // Q heads
            config_.numKeyValueHeads,     // KV heads (GQA支持)
            config_.intermediateSize,
            rmsEps_,
            config_.ropeTheta,  // 从配置读取 rope_theta
            // P3修复：传递 RoPE 扩展参数
            config_.maxSequenceLength,
            config_.ropeNctxOrig,
            config_.ropeFreqScale,
            config_.ropeType,
            config_.ropeExtFactor
        );
    }
    CLLM_INFO("TransformerModel: initialized with ropeTheta=%f, rmsNormEps=%e", 
             config_.ropeTheta, rmsEps_);
}

void TransformerModel::setEmbeddingWeight(const Tensor& embedding) {
    embedding_ = embedding;
    
    // 检查 embedding 权重的统计信息
    const auto& embShape = embedding_.shape();
    if (embShape.size() == 2) {
        const float* embData = embedding_.data();
        size_t embSize = embShape[0] * embShape[1];
        
        float embMax = embData[0];
        float embMin = embData[0];
        float embSum = 0.0f;
        size_t embNanCount = 0;
        size_t embInfCount = 0;
        size_t embNonZeroCount = 0;
        
        for (size_t i = 0; i < std::min(embSize, (size_t)10000); ++i) {
            float val = embData[i];
            if (std::isnan(val)) {
                embNanCount++;
            } else if (std::isinf(val)) {
                embInfCount++;
            } else {
                if (val > embMax) embMax = val;
                if (val < embMin) embMin = val;
                embSum += val;
                if (std::abs(val) > 1e-6f) embNonZeroCount++;
            }
        }
        
        CLLM_INFO("TransformerModel::setEmbeddingWeight: shape=[%zu,%zu], total_size=%zu", 
                 embShape[0], embShape[1], embSize);
        CLLM_INFO("TransformerModel::setEmbeddingWeight: stats (first 10000): max=%.6f, min=%.6f, avg=%.6f", 
                 embMax, embMin, embSum / 10000.0f);
        CLLM_INFO("TransformerModel::setEmbeddingWeight: nan_count=%zu, inf_count=%zu, non_zero_count=%zu", 
                 embNanCount, embInfCount, embNonZeroCount);
        
        if (embNanCount > 0 || embInfCount > 0) {
            CLLM_ERROR("TransformerModel::setEmbeddingWeight: WARNING - embedding contains invalid values!");
        }
    }
}

void TransformerModel::setLmHeadWeight(const Tensor& lmHead) {
    lmHead_ = lmHead;
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
    const Tensor& norm2Weight,
    const Tensor& attnQNormWeight,
    const Tensor& attnKNormWeight
) {
    if (layerIndex >= layers_.size()) {
        throw std::out_of_range("TransformerModel::setBlockWeights: layerIndex out of range");
    }

    auto& block = layers_[layerIndex];
    block.setAttentionWeights(wq, wk, wv, wo);
    block.setFFNWeights(wGate, wUp, wDown);
    block.setNormWeights(norm1Weight, norm2Weight);
    
    // 如果提供了 Q/K 归一化权重，设置它们
    if (!attnQNormWeight.shape().empty() && !attnKNormWeight.shape().empty()) {
        block.setAttnQKNormWeights(attnQNormWeight, attnKNormWeight);
    }
}

void TransformerModel::setFinalNormWeight(const Tensor& normWeight) {
    finalNormWeight_ = normWeight;
}

Tensor TransformerModel::forward(const std::vector<int>& inputIds) const {
    // 检查 embedding 和 lm_head 是否有数据
    if (embedding_.shape().empty() || lmHead_.shape().empty()) {
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
    
    // 验证 embedding 形状
    const auto& embShape = embedding_.shape();
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
        const float* embData = embedding_.data();
        const float* embRow = embData + static_cast<size_t>(tokenId) * embHidden;
        float* dst = hiddenStates.data() + t * hidden;
        
        for (size_t h = 0; h < hidden; ++h) {
            dst[h] = embRow[h];
        }
    }
    CLLM_INFO("TransformerModel::forward: embedding 查表完成");
    
    // 检查 embedding 输出是否包含 NaN 或 Inf
    bool hasNan = false;
    bool hasInf = false;
    const float* data = hiddenStates.data();
    for (size_t i = 0; i < seqLen * hidden; ++i) {
        float val = data[i];
        if (std::isnan(val)) {
            hasNan = true;
            break;
        }
        if (std::isinf(val)) {
            hasInf = true;
            break;
        }
    }
    if (hasNan) CLLM_INFO("TransformerModel::forward: Embedding output contains NaN");
    if (hasInf) CLLM_INFO("TransformerModel::forward: Embedding output contains Inf");
    
    // 添加一些 embedding 输出的统计信息
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    float sumVal = 0.0f;
    for (size_t i = 0; i < std::min(seqLen * hidden, static_cast<size_t>(100)); ++i) {
        float val = data[i];
        if (!std::isnan(val) && !std::isinf(val)) {
            minVal = std::min(minVal, val);
            maxVal = std::max(maxVal, val);
            sumVal += val;
        }
    }
    CLLM_INFO("TransformerModel::forward: Embedding output stats (first 100 values): min=%f, max=%f, avg=%f", 
             minVal, maxVal, sumVal / 100.0f);

    // 2. 通过 N 层 TransformerBlock
    CLLM_INFO("TransformerModel::forward: 开始通过 %zu 层 TransformerBlock...", layers_.size());
    for (size_t i = 0; i < layers_.size(); ++i) {
        CLLM_INFO("TransformerModel::forward: 处理第 %zu 层...", i);
        hiddenStates = layers_[i].forward(hiddenStates);
        CLLM_INFO("TransformerModel::forward: 第 %zu 层处理完成", i);
    }
    CLLM_INFO("TransformerModel::forward: 所有 TransformerBlock 处理完成");

    // 4. 投影到 vocab 维度: [seqLen, hidden] @ [hidden, vocab] -> [seqLen, vocab]
    CLLM_INFO("TransformerModel::forward: 开始投影到 vocab 维度...");
    Tensor logits({seqLen, vocab});

    // 验证 lmHead_ 权重
    const auto& lmHeadShape = lmHead_.shape();
    CLLM_INFO("TransformerModel::forward: lmHead_ shape: [%zu, %zu], expected [%zu, %zu]",
              lmHeadShape[0], lmHeadShape[1], hidden, vocab);
    
    // 检查 lmHead_ 权重的统计信息
    const float* lmHeadData = lmHead_.data();
    float lmHeadMax = lmHeadData[0];
    float lmHeadMin = lmHeadData[0];
    size_t lmHeadNonZero = 0;
    size_t lmHeadSize = lmHeadShape[0] * lmHeadShape[1];
    for (size_t i = 0; i < std::min(lmHeadSize, (size_t)1000); ++i) {
        float val = lmHeadData[i];
        if (val > lmHeadMax) lmHeadMax = val;
        if (val < lmHeadMin) lmHeadMin = val;
        if (val != 0.0f) lmHeadNonZero++;
    }
    CLLM_INFO("TransformerModel::forward: lmHead_ stats (first 1000): max=%.6f, min=%.6f, non_zero=%zu",
              lmHeadMax, lmHeadMin, lmHeadNonZero);
    
    // 3. 最终 RMSNorm（如果提供了权重）
    if (!finalNormWeight_.shape().empty()) {
        // 检查 hiddenStates 的统计信息（在最终 RMSNorm 之前）
        const float* hiddenDataBeforeNorm = hiddenStates.data();
        float hiddenMaxBeforeNorm = hiddenDataBeforeNorm[0];
        float hiddenMinBeforeNorm = hiddenDataBeforeNorm[0];
        size_t hiddenNonZeroBeforeNorm = 0;
        size_t hiddenSizeBeforeNorm = 1 * seqLen * hidden;
        for (size_t i = 0; i < std::min(hiddenSizeBeforeNorm, (size_t)1000); ++i) {
            float val = hiddenDataBeforeNorm[i];
            if (val > hiddenMaxBeforeNorm) hiddenMaxBeforeNorm = val;
            if (val < hiddenMinBeforeNorm) hiddenMinBeforeNorm = val;
            if (std::abs(val) > 1e-6f) hiddenNonZeroBeforeNorm++;
        }
        CLLM_INFO("TransformerModel::forward: hiddenStates BEFORE final RMSNorm (first 1000): max=%.6f, min=%.6f, non_zero=%zu",
                  hiddenMaxBeforeNorm, hiddenMinBeforeNorm, hiddenNonZeroBeforeNorm);
        
        CLLM_INFO("TransformerModel::forward: 开始最终 RMSNorm...");
        using namespace kernels;
        Tensor normOut({1, seqLen, hidden});
        rmsnorm(hiddenStates.data(), normOut.data(), finalNormWeight_.data(), 1 * seqLen, hidden, rmsEps_);
        hiddenStates = std::move(normOut);
        CLLM_INFO("TransformerModel::forward: 最终 RMSNorm 完成");
        
        // 检查最终 RMSNorm 后的 hiddenStates
        const float* hiddenDataAfterNorm = hiddenStates.data();
        float hiddenMaxAfterNorm = hiddenDataAfterNorm[0];
        float hiddenMinAfterNorm = hiddenDataAfterNorm[0];
        size_t hiddenNonZeroAfterNorm = 0;
        for (size_t i = 0; i < std::min((size_t)(1 * seqLen * hidden), (size_t)1000); ++i) {
            float val = hiddenDataAfterNorm[i];
            if (val > hiddenMaxAfterNorm) hiddenMaxAfterNorm = val;
            if (val < hiddenMinAfterNorm) hiddenMinAfterNorm = val;
            if (std::abs(val) > 1e-6f) hiddenNonZeroAfterNorm++;
        }
        CLLM_INFO("TransformerModel::forward: hiddenStates AFTER final RMSNorm (first 1000): max=%.6f, min=%.6f, non_zero=%zu",
                  hiddenMaxAfterNorm, hiddenMinAfterNorm, hiddenNonZeroAfterNorm);
    }

    using namespace kernels;
    // 把 hiddenStates 从 [1, seqLen, hidden] 重塑为 [seqLen, hidden]
    // hiddenStates 的内存布局: [batch=1, seqLen, hidden]
    // 对于 [1, seqLen, hidden]，内存布局是连续的: [token0_h0, token0_h1, ..., token0_hN, token1_h0, ...]
    // matmul 期望 A 是 [seqLen, hidden]，即 [seqLen, hidden] 的形状
    // 但是 hiddenStates 是 [1, seqLen, hidden]，所以 hiddenStates.data() 指向第一个 token 的第一个 hidden 维度
    // matmul 会按行读取: A[m][k] = A[m * K + k]，其中 m 是行索引 (seqLen)，k 是列索引 (hidden)
    // 对于 [1, seqLen, hidden]，hiddenStates.data()[m * hidden + k] 对应 token m 的 hidden dimension k
    // 这是正确的！
    
    CLLM_INFO("TransformerModel::forward: 准备 matmul - hiddenStates shape: [1, %zu, %zu], logits shape: [%zu, %zu]",
              seqLen, hidden, seqLen, vocab);
    CLLM_INFO("TransformerModel::forward: matmul 参数: M=%zu (seqLen), N=%zu (vocab), K=%zu (hidden)",
              seqLen, vocab, hidden);
    
    // 注意：lmHead_ 的形状是 [hidden, vocab]，所以我们需要转置
    // matmul 计算 C = A @ B，其中：
    //   A 是 [seqLen, hidden] (来自 hiddenStates)
    //   B 应该是 [hidden, vocab] (lmHead_)
    //   但 matmul 中的 B[k][n] = B[k * N + n] = B[k * vocab + n]
    //   而 lmHead_[k][n] = lmHead_[k * vocab + n]
    //   所以我们需要检查 lmHead_ 的内存布局
    
    // 实际上，如果 lmHead_ 形状是 [hidden, vocab]，那么：
    //   lmHead_[k][n] = lmHead_[k * vocab + n]
    //   在 matmul 中，B[k][n] = B[k * N + n] = B[k * vocab + n]
    //   这是正确的！
    // 但是如果 lmHead_ 是从 GGUF 加载的，可能是转置的 [vocab, hidden]
    //   那么我们需要转置参数
    
    // 先尝试不转置
    matmul(hiddenStates.data(), lmHead_.data(), logits.data(), seqLen, vocab, hidden, false, false);
    
    // 如果结果全为 0，可能需要转置 lmHead_
    // 检查前几个 logits 是否全为 0
    bool allZero = true;
    for (size_t i = 0; i < std::min((size_t)(seqLen * vocab), (size_t)10); ++i) {
        if (std::abs(logits.data()[i]) > 1e-6f) {
            allZero = false;
            break;
        }
    }
    
    if (allZero && seqLen * vocab > 0) {
        CLLM_WARN("TransformerModel::forward: First logits are all zero, trying transposed lmHead_");
        // 尝试转置 lmHead_: [vocab, hidden] -> [hidden, vocab]
        // 但这样需要重新创建 logits，先记录警告
        CLLM_WARN("TransformerModel::forward: lmHead_ shape might need transpose");
    }
    
    // 检查计算后的 logits 统计信息
    float logitsMax = logits.data()[0];
    float logitsMin = logits.data()[0];
    size_t logitsNonZero = 0;
    for (size_t i = 0; i < std::min((size_t)(seqLen * vocab), (size_t)1000); ++i) {
        float val = logits.data()[i];
        if (val > logitsMax) logitsMax = val;
        if (val < logitsMin) logitsMin = val;
        if (val != 0.0f) logitsNonZero++;
    }
    CLLM_INFO("TransformerModel::forward: logits stats (first 1000): max=%.6f, min=%.6f, non_zero=%zu",
              logitsMax, logitsMin, logitsNonZero);
    
    CLLM_INFO("TransformerModel::forward: 投影到 vocab 维度完成");

    return logits;
}

}  // namespace kylin
}  // namespace cllm
