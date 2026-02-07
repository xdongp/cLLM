/**
 * @file transformer_model.cpp
 * @brief Transformer 模型实现
 */

#include "cllm/kylin/model/transformer_model.h"
#include "cllm/common/logger.h"
#include "cllm/model/config.h"

namespace cllm {
namespace kylin {

// 默认构造函数
TransformerModel::TransformerModel()
    : HFTransformerModel("", DeviceType::CPU, QuantType::FP32) {
    CLLM_INFO("[TransformerModel] Default constructor");
}

// 构造函数 - 从 ModelConfig 创建
TransformerModel::TransformerModel(const ModelConfig& config) 
    : HFTransformerModel("", DeviceType::CPU, QuantType::FP32) {
    CLLM_INFO("[TransformerModel] Creating from ModelConfig");
    // TODO: 从 ModelConfig 初始化
}

// 构造函数 - 从模型目录加载
TransformerModel::TransformerModel(const std::string& modelDir, 
                                   DeviceType device,
                                   QuantType quantType)
    : HFTransformerModel(modelDir, device, quantType) {
    CLLM_INFO("[TransformerModel] Loading from %s", modelDir.c_str());
}

// 设置 Embedding 权重
void TransformerModel::setEmbeddingWeight(const Tensor& embedding) {
    CLLM_INFO("[TransformerModel] setEmbeddingWeight called, shape=[%zu, %zu]", 
              embedding.shape().empty() ? 0 : embedding.shape()[0],
              embedding.shape().size() < 2 ? 0 : embedding.shape()[1]);
    // TODO: 实现权重设置
}

// 设置 LM Head 权重
void TransformerModel::setLmHeadWeight(const Tensor& lmHead) {
    CLLM_INFO("[TransformerModel] setLmHeadWeight called, shape=[%zu, %zu]",
              lmHead.shape().empty() ? 0 : lmHead.shape()[0],
              lmHead.shape().size() < 2 ? 0 : lmHead.shape()[1]);
    // TODO: 实现权重设置
}

// 设置 Final Norm 权重
void TransformerModel::setFinalNormWeight(const Tensor& weight) {
    CLLM_INFO("[TransformerModel] setFinalNormWeight called, shape=[%zu]",
              weight.shape().empty() ? 0 : weight.shape()[0]);
    // TODO: 实现权重设置
}

// 设置 Block 权重
void TransformerModel::setBlockWeights(
    size_t layerIndex,
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo,
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown,
    const Tensor& norm1,
    const Tensor& norm2,
    const Tensor& qNorm,
    const Tensor& kNorm) {
    
    CLLM_INFO("[TransformerModel] setBlockWeights called for layer %zu", layerIndex);
    // TODO: 实现权重设置
}

// 前向推理（返回 Tensor）
Tensor TransformerModel::forward(const std::vector<int>& inputIds) {
    CLLM_INFO("[TransformerModel] forward called with %zu tokens", inputIds.size());
    
    // 调用父类的 forward
    std::vector<int32_t> inputIds32(inputIds.begin(), inputIds.end());
    std::vector<float> logits = HFTransformerModel::forward(inputIds32);
    
    // 转换为 Tensor
    if (logits.empty()) {
        return Tensor();
    }
    
    Tensor result({logits.size()});
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = logits[i];
    }
    return result;
}

// 前向推理（单 token）
Tensor TransformerModel::forward(int tokenId, int position) {
    CLLM_INFO("[TransformerModel] forward called for token %d at position %d", tokenId, position);
    
    // 调用父类的 forwardWithRequestId
    std::vector<int32_t> inputIds = {tokenId};
    std::vector<float> logits = HFTransformerModel::forwardWithRequestId(inputIds, 0);
    
    // 转换为 Tensor
    if (logits.empty()) {
        return Tensor();
    }
    
    Tensor result({logits.size()});
    for (size_t i = 0; i < logits.size(); ++i) {
        result[i] = logits[i];
    }
    return result;
}

} // namespace kylin
} // namespace cllm
