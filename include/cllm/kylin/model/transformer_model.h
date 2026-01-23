/**
 * @file transformer_model.h
 * @brief 简化版 Transformer 模型（MVP，用于自研推理引擎）
 */
#pragma once

#include "cllm/kylin/core/tensor.h"
#include "cllm/kylin/model/transformer_block.h"
#include "cllm/model/config.h"

#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief 简化版 Transformer Model
 *
 * 负责：
 * - embedding 查表
 * - N 层 TransformerBlock 前向
 * - 最终 RMSNorm + lm_head 投影到 vocab
 */
class TransformerModel {
public:
    explicit TransformerModel(const ModelConfig& config);

    const ModelConfig& getConfig() const { return config_; }

    /// 设置 embedding 权重（[vocabSize, hiddenSize]）
    void setEmbeddingWeight(const Tensor& embedding);

    /// 设置 lm_head 权重（[hiddenSize, vocabSize]）
    void setLmHeadWeight(const Tensor& lmHead);

    /// 设置某一层 Block 的权重
    void setBlockWeights(
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
        const Tensor& attnQNormWeight = Tensor(),  // 可选：Q的独立归一化权重
        const Tensor& attnKNormWeight = Tensor()   // 可选：K的独立归一化权重
    );

    /// 设置最终 RMSNorm 的权重
    void setFinalNormWeight(const Tensor& normWeight);

    /// 前向传播：输入 token id 序列，输出 logits
    /// 输出形状：[seq_len, vocabSize]
    Tensor forward(const std::vector<int>& inputIds) const;

private:
    ModelConfig config_;

    std::vector<TransformerBlock> layers_;

    Tensor embedding_;     // [vocab, hidden]
    Tensor lmHead_;        // [hidden, vocab]
    Tensor finalNormWeight_; // [hidden]

    float rmsEps_;
};

}  // namespace kylin
}  // namespace cllm
