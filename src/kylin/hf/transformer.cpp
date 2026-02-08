/**
 * @file hf_transformer.cpp
 * @brief HuggingFace Transformer 模型实现 - 100%后端化版本
 * 
 * 完全使用 IComputeBackend 接口，模型类只负责协调
 * CPU/GPU 计算逻辑完全分离到各自的后端实现
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

// 量化类型名称
static const char* quantTypeName(QuantType type) {
    switch (type) {
        case QuantType::FP32: return "FP32";
        case QuantType::FP16: return "FP16";
        case QuantType::BF16: return "BF16";
        case QuantType::INT8: return "INT8";
        case QuantType::Q4_K: return "Q4_K";
        case QuantType::Q8_0: return "Q8_0";
        default: return "Unknown";
    }
}

HFTransformerModel::HFTransformerModel(const std::string& modelDir, DeviceType device, QuantType quantType)
    : deviceType_(device), quantType_(quantType) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    CLLM_INFO("[HFTransformer] Requested device: %s, quantization: %s", 
              device == DeviceType::Metal ? "Metal GPU" : 
              device == DeviceType::CUDA ? "CUDA GPU" : "CPU",
              quantTypeName(quantType));
    
    // 使用后端工厂创建后端
    backend_ = BackendFactory::create(device);
    if (!backend_) {
        CLLM_ERROR("[HFTransformer] Failed to create backend for device %d", static_cast<int>(device));
        return;
    }
    CLLM_INFO("[HFTransformer] Backend created: %s", backend_->getName().c_str());
    
    // 加载配置
    config_ = loadHFConfigFromDir(modelDir);
    if (!config_.isValid()) {
        CLLM_ERROR("[HFTransformer] Invalid model config");
        return;
    }
    config_.print();
    
    // 加载 safetensors
    std::string safetensorsPath = modelDir;
    if (safetensorsPath.back() != '/') safetensorsPath += '/';
    safetensorsPath += "model.safetensors";
    
    loader_ = std::make_unique<SafetensorsLoader>(safetensorsPath);
    if (!loader_->isValid()) {
        CLLM_ERROR("[HFTransformer] Failed to load safetensors");
        return;
    }
    
    // 加载权重
    if (!loadWeights()) {
        CLLM_ERROR("[HFTransformer] Failed to load weights");
        return;
    }
    
    // 初始化后端
    if (!backend_->initialize(config_)) {
        CLLM_ERROR("[HFTransformer] Failed to initialize backend");
        return;
    }
    
    // 加载权重到后端
    if (!backend_->loadWeights(modelWeights_)) {
        CLLM_ERROR("[HFTransformer] Failed to load weights to backend");
        return;
    }
    
    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully using %s backend", backend_->getName().c_str());
}

HFTransformerModel::~HFTransformerModel() {
    if (backend_) {
        backend_->shutdown();
    }
}

bool HFTransformerModel::loadWeights() {
    // 填充 ModelWeights 结构
    modelWeights_.embedTokens = loader_->getTensorData("model.embed_tokens.weight");
    modelWeights_.finalNormWeight = loader_->getTensorData("model.norm.weight");
    
    if (config_.tieWordEmbeddings) {
        modelWeights_.lmHeadWeight = modelWeights_.embedTokens;
    } else {
        modelWeights_.lmHeadWeight = loader_->getTensorData("lm_head.weight");
    }
    
    if (!modelWeights_.embedTokens || !modelWeights_.finalNormWeight) {
        CLLM_ERROR("[HFTransformer] Missing required weights");
        return false;
    }
    
    // 加载每层权重
    modelWeights_.layers.resize(config_.numHiddenLayers);
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& layer = modelWeights_.layers[i];
        
        layer.inputLayernorm = loader_->getTensorData(prefix + ".input_layernorm.weight");
        layer.qProj = loader_->getTensorData(prefix + ".self_attn.q_proj.weight");
        layer.kProj = loader_->getTensorData(prefix + ".self_attn.k_proj.weight");
        layer.vProj = loader_->getTensorData(prefix + ".self_attn.v_proj.weight");
        layer.oProj = loader_->getTensorData(prefix + ".self_attn.o_proj.weight");
        layer.qNorm = loader_->getTensorData(prefix + ".self_attn.q_norm.weight");
        layer.kNorm = loader_->getTensorData(prefix + ".self_attn.k_norm.weight");
        layer.postAttentionLayernorm = loader_->getTensorData(prefix + ".post_attention_layernorm.weight");
        layer.gateProj = loader_->getTensorData(prefix + ".mlp.gate_proj.weight");
        layer.upProj = loader_->getTensorData(prefix + ".mlp.up_proj.weight");
        layer.downProj = loader_->getTensorData(prefix + ".mlp.down_proj.weight");
        
        if (!layer.inputLayernorm || !layer.qProj || !layer.kProj || 
            !layer.vProj || !layer.oProj || !layer.postAttentionLayernorm ||
            !layer.gateProj || !layer.upProj || !layer.downProj) {
            CLLM_ERROR("[HFTransformer] Missing weights for layer %d", i);
            return false;
        }
    }
    
    modelWeights_.weightType = quantType_;
    return true;
}

std::vector<float> HFTransformerModel::forward(const std::vector<int32_t>& inputIds) {
    return forwardWithRequestId(inputIds, 0);
}

std::vector<float> HFTransformerModel::forwardWithRequestId(
    const std::vector<int32_t>& inputIds, 
    size_t requestId
) {
    if (!loaded_ || !backend_) {
        CLLM_ERROR("[HFTransformer] Model not loaded or backend not available");
        return {};
    }
    
    return backend_->forward(inputIds, static_cast<int>(requestId));
}

std::vector<std::vector<float>> HFTransformerModel::forwardBatch(
    const std::vector<std::vector<int32_t>>& batchInputIds,
    const std::vector<size_t>& requestIds
) {
    if (!loaded_ || !backend_) {
        CLLM_ERROR("[HFTransformer] Model not loaded or backend not available");
        return {};
    }
    
    std::vector<int> intRequestIds(requestIds.begin(), requestIds.end());
    return backend_->forwardBatch(batchInputIds, intRequestIds);
}

void HFTransformerModel::releaseKVCache(size_t requestId) {
    if (backend_) {
        backend_->releaseKVCache(static_cast<int>(requestId));
    }
}

int HFTransformerModel::getKVCacheCurrentLength(size_t requestId) const {
    if (backend_) {
        return backend_->getKVCacheCurrentLength(static_cast<int>(requestId));
    }
    return 0;
}

} // namespace kylin
} // namespace cllm
