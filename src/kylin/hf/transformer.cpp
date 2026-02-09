/**
 * @file hf_transformer.cpp
 * @brief HuggingFace Transformer 模型实现 - 100%后端化版本
 * 
 * 完全使用 IComputeBackend 接口，模型类只负责协调
 * CPU/GPU 计算逻辑完全分离到各自的后端实现
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/hf/safetensors_loader.h"
#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/kylin/core/quantization.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

// 使用 cllm::kylin::quantTypeName from quantization.h

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
    
    // 如果是量化模式，释放原始F32权重的内存映射
    if (quantType_ == QuantType::INT8 || quantType_ == QuantType::FP16) {
        loader_->releaseMappedData();
        CLLM_INFO("[HFTransformer] Released original F32 weights to save memory");
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
    // 根据量化类型决定加载方式
    if (quantType_ == QuantType::FP32) {
        // FP32 模式：直接使用内存映射的数据
        return loadWeightsFP32();
    } else if (quantType_ == QuantType::FP16) {
        // FP16 模式：转换权重为 FP16
        return loadWeightsQuantized(QuantType::FP16);
    } else if (quantType_ == QuantType::INT8) {
        // INT8 模式：转换权重为 INT8
        return loadWeightsQuantized(QuantType::INT8);
    } else {
        CLLM_ERROR("[HFTransformer] Unsupported quantization type: %s", quantTypeName(quantType_));
        return false;
    }
}

bool HFTransformerModel::loadWeightsFP32() {
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

bool HFTransformerModel::loadWeightsQuantized(QuantType qType) {
    CLLM_INFO("[HFTransformer] Loading weights with %s quantization...",
              qType == QuantType::INT8 ? "INT8" : "FP16");

    // 检测原始数据类型
    std::string embedDtype = loader_->getTensorDtype("model.embed_tokens.weight");
    bool isSourceBF16 = (embedDtype == "BF16" || embedDtype == "bfloat16");
    CLLM_INFO("[HFTransformer] Source dtype: %s, Target: %s", 
              embedDtype.empty() ? "unknown" : embedDtype.c_str(),
              quantTypeName(qType));

    // 为量化参数预留空间 (3个全局权重: embed, norm, lm_head + 每层11个权重)
    size_t totalWeights = 3 + config_.numHiddenLayers * 11;
    modelWeights_.scales.resize(totalWeights, 1.0f);
    modelWeights_.zeroPoints.resize(totalWeights, 0);

    // 量化嵌入层
    size_t embedCount = loader_->getTensorNumElements("model.embed_tokens.weight");
    if (isSourceBF16 && qType == QuantType::FP16) {
        // BF16 -> FP16 直接转换（避免中间 F32）
        const uint16_t* embedBF16 = static_cast<const uint16_t*>(loader_->getTensorData("model.embed_tokens.weight"));
        quantizedEmbedTokens_ = QuantizedWeight::fromBF16(embedBF16, embedCount, qType);
    } else {
        // 其他情况：先转 F32，再量化
        auto embedF32 = loader_->getTensorAsF32("model.embed_tokens.weight");
        quantizedEmbedTokens_ = QuantizedWeight::fromFP32(embedF32.data(), embedCount, qType);
    }
    modelWeights_.embedTokens = quantizedEmbedTokens_.data();
    modelWeights_.scales[0] = quantizedEmbedTokens_.scale();
    modelWeights_.zeroPoints[0] = quantizedEmbedTokens_.zeroPoint();

    // final norm 保持 F32 (LayerNorm权重不量化)
    size_t normCount = loader_->getTensorNumElements("model.norm.weight");
    auto normF32 = loader_->getTensorAsF32("model.norm.weight");
    f32FinalNormWeight_.resize(normCount);
    std::memcpy(f32FinalNormWeight_.data(), normF32.data(), normCount * sizeof(float));
    modelWeights_.finalNormWeight = f32FinalNormWeight_.data();
    modelWeights_.scales[1] = 1.0f;
    modelWeights_.zeroPoints[1] = 0;

    // 量化 lm_head (索引2)
    if (!config_.tieWordEmbeddings) {
        size_t lmHeadCount = loader_->getTensorNumElements("lm_head.weight");
        if (isSourceBF16 && qType == QuantType::FP16) {
            const uint16_t* lmHeadBF16 = static_cast<const uint16_t*>(loader_->getTensorData("lm_head.weight"));
            quantizedLmHeadWeight_ = QuantizedWeight::fromBF16(lmHeadBF16, lmHeadCount, qType);
        } else {
            auto lmHeadF32 = loader_->getTensorAsF32("lm_head.weight");
            quantizedLmHeadWeight_ = QuantizedWeight::fromFP32(lmHeadF32.data(), lmHeadCount, qType);
        }
        modelWeights_.lmHeadWeight = quantizedLmHeadWeight_.data();
        modelWeights_.scales[2] = quantizedLmHeadWeight_.scale();
        modelWeights_.zeroPoints[2] = quantizedLmHeadWeight_.zeroPoint();
    } else {
        modelWeights_.lmHeadWeight = modelWeights_.embedTokens;
        modelWeights_.scales[2] = modelWeights_.scales[0];
        modelWeights_.zeroPoints[2] = modelWeights_.zeroPoints[0];
    }

    // 量化每层权重
    modelWeights_.layers.resize(config_.numHiddenLayers);
    quantizedLayerWeights_.resize(config_.numHiddenLayers * 11);  // 11 weights per layer
    f32LayerWeights_.resize(config_.numHiddenLayers * 4);  // 4个F32权重 per layer (LayerNorm)

    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        auto& layer = modelWeights_.layers[i];
        size_t qwIdx = i * 11;
        size_t scaleIdx = 3 + qwIdx; // 从索引3开始

        // 定义权重列表：线性层权重需要量化，LayerNorm权重保持F32
        struct WeightInfo {
            const char* suffix;
            const void** target;
            bool shouldQuantize;  // 是否量化
        };

        WeightInfo weights[] = {
            {".input_layernorm.weight", &layer.inputLayernorm, false},      // LayerNorm - F32
            {".self_attn.q_proj.weight", &layer.qProj, true},               // 线性层 - 量化
            {".self_attn.k_proj.weight", &layer.kProj, true},               // 线性层 - 量化
            {".self_attn.v_proj.weight", &layer.vProj, true},               // 线性层 - 量化
            {".self_attn.o_proj.weight", &layer.oProj, true},               // 线性层 - 量化
            {".self_attn.q_norm.weight", &layer.qNorm, false},              // LayerNorm - F32
            {".self_attn.k_norm.weight", &layer.kNorm, false},              // LayerNorm - F32
            {".post_attention_layernorm.weight", &layer.postAttentionLayernorm, false}, // LayerNorm - F32
            {".mlp.gate_proj.weight", &layer.gateProj, true},               // 线性层 - 量化
            {".mlp.up_proj.weight", &layer.upProj, true},                   // 线性层 - 量化
            {".mlp.down_proj.weight", &layer.downProj, true},               // 线性层 - 量化
        };

        for (int w = 0; w < 11; ++w) {
            std::string name = prefix + weights[w].suffix;
            size_t count = loader_->getTensorNumElements(name);

            if (weights[w].shouldQuantize) {
                // 量化线性层权重
                if (isSourceBF16 && qType == QuantType::FP16) {
                    // BF16 -> FP16 直接转换
                    const uint16_t* bf16Data = static_cast<const uint16_t*>(loader_->getTensorData(name));
                    quantizedLayerWeights_[qwIdx + w] = QuantizedWeight::fromBF16(bf16Data, count, qType);
                } else {
                    // 其他情况：先转 F32，再量化
                    auto f32 = loader_->getTensorAsF32(name);
                    if (f32.empty()) {
                        CLLM_ERROR("[HFTransformer] Missing weight: %s", name.c_str());
                        return false;
                    }
                    quantizedLayerWeights_[qwIdx + w] = QuantizedWeight::fromFP32(f32.data(), count, qType);
                }
                *weights[w].target = quantizedLayerWeights_[qwIdx + w].data();
                // 保存量化参数
                modelWeights_.scales[scaleIdx + w] = quantizedLayerWeights_[qwIdx + w].scale();
                modelWeights_.zeroPoints[scaleIdx + w] = quantizedLayerWeights_[qwIdx + w].zeroPoint();
            } else {
                // LayerNorm权重保持F32，存储在f32LayerWeights_中
                auto f32 = loader_->getTensorAsF32(name);
                if (f32.empty()) {
                    CLLM_ERROR("[HFTransformer] Missing weight: %s", name.c_str());
                    return false;
                }
                size_t f32Idx = i * 4 + (w == 0 ? 0 : w == 5 ? 1 : w == 6 ? 2 : 3); // 映射到4个F32权重
                f32LayerWeights_[f32Idx].resize(count);
                std::memcpy(f32LayerWeights_[f32Idx].data(), f32.data(), count * sizeof(float));
                *weights[w].target = f32LayerWeights_[f32Idx].data();
                // F32权重的scale=1.0, zeroPoint=0
                modelWeights_.scales[scaleIdx + w] = 1.0f;
                modelWeights_.zeroPoints[scaleIdx + w] = 0;
            }
        }
    }

    modelWeights_.weightType = quantType_;
    CLLM_INFO("[HFTransformer] Quantized weights loaded successfully");
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
