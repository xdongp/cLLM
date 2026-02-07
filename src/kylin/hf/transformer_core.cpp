/**
 * @file transformer_core.cpp
 * @brief Transformer 模型核心 - 构造函数和基础接口
 *
 * 包含模型初始化、配置加载、基础接口实现。
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

// ============================================================================
// 构造函数和析构函数
// ============================================================================

HFTransformerModel::HFTransformerModel(const std::string& modelDir, DeviceType device, QuantType quantType)
    : deviceType_(device), useGPU_(false), quantType_(quantType) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    CLLM_INFO("[HFTransformer] Requested device: %s, quantization: %s",
              device == DeviceType::Metal ? "Metal GPU" :
              device == DeviceType::CUDA ? "CUDA GPU" : "CPU",
              quantTypeName(quantType));

    // 初始化计算内核
    ggml_kernels::initialize(device);

    // 对于 Metal，尝试初始化 GPU 后端
    if (device == DeviceType::Metal) {
#ifdef GGML_USE_METAL
        gpuBackend_ = std::make_unique<GGMLGPUBackend>();
        CLLM_INFO("[HFTransformer] GPU backend created, will initialize after loading weights");
#else
        CLLM_WARN("[HFTransformer] Metal not compiled, falling back to CPU");
        deviceType_ = DeviceType::CPU;
#endif
    }

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

    // 预计算 RoPE 频率
    initializeRoPE();

    // 初始化 KV Cache
    initializeKVCache();

    // 分配工作缓冲区
    initializeBuffers();

    // 根据量化类型预转换权重
    if (usePreconvertedWeights_) {
        preconvertWeightsForQuantType();
    }

    // 初始化 GPU 后端
    if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
        initializeGPUBackend();
    }

    // 初始化 KV Cache Pool
    initializeKVCachePool();

    // 初始化工作缓冲区池
    initializeWorkBufferPool();

    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully");
}

HFTransformerModel::~HFTransformerModel() = default;

// ============================================================================
// 初始化辅助函数
// ============================================================================

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
    kvCacheLen_ = 0;
}

void HFTransformerModel::initializeBuffers() {
    hiddenStates_.resize(config_.hiddenSize);
    residual_.resize(config_.hiddenSize);
    normOutput_.resize(config_.hiddenSize);
    attnOutput_.resize(config_.hiddenSize);
    ffnOutput_.resize(config_.hiddenSize);
    normWeightBuffer_.resize(config_.hiddenSize);

    int headDim = config_.getHeadDim();
    int nHeads = config_.numAttentionHeads;
    int nKVHeads = config_.getNumKVHeads();

    qBuffer_.resize(nHeads * headDim);
    kBuffer_.resize(nKVHeads * headDim);
    vBuffer_.resize(nKVHeads * headDim);
    attnOutBuffer_.resize(nHeads * headDim);
    attnScores_.resize(nHeads * kMaxSeqLen);
}

void HFTransformerModel::initializeKVCachePool() {
    int headDim = config_.getHeadDim();
    int nKVHeads = config_.getNumKVHeads();
    size_t perLayerSize = static_cast<size_t>(kMaxSeqLen) * nKVHeads * headDim;

    kvCachePool_ = std::make_unique<KVCachePool>(
        config_.numHiddenLayers,
        perLayerSize,
        32  // 最大并发请求数
    );
    CLLM_INFO("[HFTransformer] KV Cache Pool initialized");
}

void HFTransformerModel::initializeWorkBufferPool() {
    workBufferPool_ = std::make_unique<WorkBufferPool>(
        config_.hiddenSize,
        config_.numAttentionHeads,
        config_.getNumKVHeads(),
        config_.getHeadDim(),
        kMaxSeqLen,
        32  // 最大并发请求数
    );
    CLLM_INFO("[HFTransformer] Work Buffer Pool initialized");
}

void HFTransformerModel::initializeGPUBackend() {
#ifdef GGML_USE_METAL
    if (!gpuBackend_->initialize(config_)) {
        CLLM_ERROR("[HFTransformer] Failed to initialize GPU backend");
        return;
    }

    // 上传权重到 GPU
    std::vector<LayerWeightsGPU> layerWeightsGPU;
    layerWeightsGPU.reserve(config_.numHiddenLayers);

    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        LayerWeightsGPU gpuLayer;
        // 转换并上传权重...
        // TODO: 实现权重上传
        layerWeightsGPU.push_back(gpuLayer);
    }

    useGPU_ = true;
    CLLM_INFO("[HFTransformer] GPU backend initialized");
#else
    CLLM_WARN("[HFTransformer] Metal not available, using CPU");
#endif
}

void HFTransformerModel::preconvertWeightsForQuantType() {
    switch (quantType_) {
        case QuantType::FP16:
            convertWeightsToFP16();
            break;
        case QuantType::INT8:
            convertWeightsToINT8();
            break;
        default:
            // FP32 不需要预转换
            break;
    }
}

// ============================================================================
// 基础工具函数
// ============================================================================

void HFTransformerModel::rmsNorm(const float* input, const float* weight,
                                  float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

void HFTransformerModel::applyRoPE(float* q, float* k, int headDim,
                                    int nHeads, int nKVHeads, int seqLen, int startPos) {
    for (int pos = 0; pos < seqLen; ++pos) {
        int actualPos = startPos + pos;
        const float* cosPtr = ropeFreqsCos_.data() + actualPos * headDim / 2;
        const float* sinPtr = ropeFreqsSin_.data() + actualPos * headDim / 2;

        // Q heads
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q + pos * nHeads * headDim + h * headDim;
            for (int i = 0; i < headDim / 2; ++i) {
                float x0 = qHead[i];
                float x1 = qHead[i + headDim / 2];
                qHead[i] = x0 * cosPtr[i] - x1 * sinPtr[i];
                qHead[i + headDim / 2] = x0 * sinPtr[i] + x1 * cosPtr[i];
            }
        }

        // K heads
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k + pos * nKVHeads * headDim + h * headDim;
            for (int i = 0; i < headDim / 2; ++i) {
                float x0 = kHead[i];
                float x1 = kHead[i + headDim / 2];
                kHead[i] = x0 * cosPtr[i] - x1 * sinPtr[i];
                kHead[i + headDim / 2] = x0 * sinPtr[i] + x1 * cosPtr[i];
            }
        }
    }
}

void HFTransformerModel::resetKVCache() {
    std::fill(kCache_.begin(), kCache_.end(), 0.0f);
    std::fill(vCache_.begin(), vCache_.end(), 0.0f);
    kvCacheLen_ = 0;
    CLLM_INFO("[HFTransformer] KV Cache reset");
}

} // namespace kylin
} // namespace cllm
