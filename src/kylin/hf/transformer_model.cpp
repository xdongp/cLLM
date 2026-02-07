/**
 * @file transformer_model.cpp
 * @brief Transformer模型核心实现 - 构造函数和主流程
 */

#include "cllm/kylin/hf/hf_transformer_model.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

HFTransformerModel::HFTransformerModel(const std::string& modelDir, DeviceType device, QuantType quantType)
    : deviceType_(device), useGPU_(false), quantType_(quantType) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    CLLM_INFO("[HFTransformer] Requested device: %s, quantization: %s", 
              device == DeviceType::Metal ? "Metal GPU" : 
              device == DeviceType::CUDA ? "CUDA GPU" : "CPU",
              quantTypeName(quantType));
    
    ggml_kernels::initialize(device);
    
    if (device == DeviceType::Metal) {
#ifdef GGML_USE_METAL
        gpuBackend_ = std::make_unique<GGMLGPUBackend>();
        CLLM_INFO("[HFTransformer] GPU backend created");
#else
        CLLM_WARN("[HFTransformer] Metal not compiled, falling back to CPU");
        deviceType_ = DeviceType::CPU;
#endif
    }
    
    config_ = loadHFConfigFromDir(modelDir);
    if (!config_.isValid()) {
        CLLM_ERROR("[HFTransformer] Invalid model config");
        return;
    }
    config_.print();
    
    std::string safetensorsPath = modelDir;
    if (safetensorsPath.back() != '/') safetensorsPath += '/';
    safetensorsPath += "model.safetensors";
    
    loader_ = std::make_unique<SafetensorsLoader>(safetensorsPath);
    if (!loader_->isValid()) {
        CLLM_ERROR("[HFTransformer] Failed to load safetensors");
        return;
    }
    
    if (!loadWeights()) {
        CLLM_ERROR("[HFTransformer] Failed to load weights");
        return;
    }
    
    initializeBuffers();
    initializeRoPE();
    initializeKVCache();
    
    if (usePreconvertedWeights_) {
        preconvertWeightsForQuantType();
    }
    
    if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
        initializeGPUBackend();
    }
    
    initializeKVCachePool();
    initializeWorkBufferPool();
    
    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully");
}

HFTransformerModel::~HFTransformerModel() = default;

void HFTransformerModel::initializeBuffers() {
    hiddenStates_.resize(config_.hiddenSize);
    residual_.resize(config_.hiddenSize);
    normOutput_.resize(config_.hiddenSize);
    attnOutput_.resize(config_.hiddenSize);
    ffnOutput_.resize(config_.hiddenSize);
    
    int headDim = config_.getHeadDim();
    int qSize = config_.numAttentionHeads * headDim;
    int kvSize = config_.getNumKVHeads() * headDim;
    
    qkvBuffer_.resize(qSize + 2 * kvSize);
    qBuffer_.resize(qSize);
    kBuffer_.resize(kvSize);
    vBuffer_.resize(kvSize);
    attnScores_.resize(config_.numAttentionHeads * kMaxSeqLen);
    attnOutBuffer_.resize(qSize);
    
    gateBuffer_.resize(config_.intermediateSize);
    upBuffer_.resize(config_.intermediateSize);
    gateUpBuffer_.resize(config_.intermediateSize * 2);
    
    normWeightBuffer_.resize(config_.hiddenSize);
    qkNormBuffer_.resize(headDim);
}

void HFTransformerModel::initializeRoPE() {
    int headDim = config_.getHeadDim();
    ropeFreqsCos_.resize(kMaxSeqLen * headDim / 2);
    ropeFreqsSin_.resize(kMaxSeqLen * headDim / 2);
    
    for (int pos = 0; pos < kMaxSeqLen; ++pos) {
        for (int i = 0; i < headDim / 2; ++i) {
            float freq = 1.0f / std::pow(config_.ropeTheta, 2.0f * i / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * headDim / 2 + i] = std::cos(angle);
            ropeFreqsSin_[pos * headDim / 2 + i] = std::sin(angle);
        }
    }
}

void HFTransformerModel::initializeKVCache() {
    int headDim = config_.getHeadDim();
    int kvHeads = config_.getNumKVHeads();
    size_t kvSize = static_cast<size_t>(config_.numHiddenLayers) * kMaxSeqLen * kvHeads * headDim;
    kCache_.resize(kvSize, 0.0f);
    vCache_.resize(kvSize, 0.0f);
}

void HFTransformerModel::preconvertWeightsForQuantType() {
    if (quantType_ == QuantType::FP16) {
        convertWeightsToFP16();
        if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
            preconvertWeights();
        }
    } else if (quantType_ == QuantType::INT8) {
        convertWeightsToINT8();
        if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
            preconvertWeights();
        }
    } else {
        preconvertWeights();
    }
}

void HFTransformerModel::initializeGPUBackend() {
    if (!gpuBackend_->initialize(config_)) {
        CLLM_WARN("[HFTransformer] GPU backend initialization failed");
        useGPU_ = false;
        return;
    }
    
    std::vector<LayerWeightsGPU> layerWeights(config_.numHiddenLayers);
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        layerWeights[i].inputLayernorm = layersF32_[i].inputLayernorm.data();
        layerWeights[i].postAttentionLayernorm = layersF32_[i].postAttentionLayernorm.data();
        layerWeights[i].qNorm = layersF32_[i].qNorm.empty() ? nullptr : layersF32_[i].qNorm.data();
        layerWeights[i].kNorm = layersF32_[i].kNorm.empty() ? nullptr : layersF32_[i].kNorm.data();
        layerWeights[i].qProj = layersF32_[i].qProj.data();
        layerWeights[i].kProj = layersF32_[i].kProj.data();
        layerWeights[i].vProj = layersF32_[i].vProj.data();
        layerWeights[i].oProj = layersF32_[i].oProj.data();
        layerWeights[i].gateProj = layersF32_[i].gateProj.data();
        layerWeights[i].upProj = layersF32_[i].upProj.data();
        layerWeights[i].downProj = layersF32_[i].downProj.data();
    }
    
    if (gpuBackend_->uploadWeights(
            embedTokensF32_.data(),
            layerWeights,
            finalNormWeightF32_.data(),
            config_.tieWordEmbeddings ? nullptr : lmHeadWeightF32_.data())) {
        useGPU_ = true;
        CLLM_INFO("[HFTransformer] GPU backend ready");
    } else {
        CLLM_WARN("[HFTransformer] Failed to upload weights to GPU");
        useGPU_ = false;
    }
}

void HFTransformerModel::initializeKVCachePool() {
    kvCachePool_ = std::make_unique<KVCachePool>(
        kMaxConcurrentRequests,
        config_.numHiddenLayers,
        kMaxSeqLen,
        config_.getNumKVHeads(),
        config_.getHeadDim()
    );
}

void HFTransformerModel::initializeWorkBufferPool() {
    workBufferPool_ = std::make_unique<WorkBufferPool>(
        kMaxConcurrentRequests,
        config_.hiddenSize,
        config_.intermediateSize,
        config_.vocabSize,
        config_.numAttentionHeads,
        config_.getNumKVHeads(),
        config_.getHeadDim(),
        kMaxSeqLen
    );
}

} // namespace kylin
} // namespace cllm
