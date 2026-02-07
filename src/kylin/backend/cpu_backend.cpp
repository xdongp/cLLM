/**
 * @file cpu_backend.cpp
 * @brief CPU 计算后端实现
 */

#include "cllm/kylin/backend/cpu_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD 支持
#if defined(__ARM_NEON) || defined(__aarch64__)
    #define USE_NEON 1
    #include <arm_neon.h>
#endif

namespace cllm {
namespace kylin {

// 内部实现结构
struct CPUBackendImpl {
    // 模型配置
    HFModelConfig config;
    
    // 权重数据
    struct LayerWeights {
        std::vector<float> inputLayernorm;
        std::vector<float> qProj;
        std::vector<float> kProj;
        std::vector<float> vProj;
        std::vector<float> oProj;
        std::vector<float> qNorm;
        std::vector<float> kNorm;
        std::vector<float> postAttentionLayernorm;
        std::vector<float> gateProj;
        std::vector<float> upProj;
        std::vector<float> downProj;
    };
    std::vector<LayerWeights> layers;
    
    // 全局权重
    std::vector<float> embedTokens;
    std::vector<float> lmHeadWeight;
    std::vector<float> finalNormWeight;
    
    // KV Cache 管理
    struct KVCache {
        std::vector<float> kCache;
        std::vector<float> vCache;
        int currentLen = 0;
    };
    std::unordered_map<int, KVCache> kvCaches;
    
    // 工作缓冲区（预分配）
    std::vector<float> hiddenStates;
    std::vector<float> residual;
    std::vector<float> normOutput;
    std::vector<float> attnOutput;
    std::vector<float> ffnOutput;
    std::vector<float> qkvBuffer;
    std::vector<float> qBuffer;
    std::vector<float> kBuffer;
    std::vector<float> vBuffer;
    std::vector<float> attnScores;
    std::vector<float> attnOutBuffer;
    std::vector<float> gateBuffer;
    std::vector<float> upBuffer;
    std::vector<float> logitsBuffer;
    
    // RoPE 频率
    std::vector<float> ropeFreqsCos;
    std::vector<float> ropeFreqsSin;
    
    // 常量
    static constexpr int kMaxSeqLen = 4096;
    
    void allocateBuffers() {
        int hiddenSize = config.hiddenSize;
        int intermediateSize = config.intermediateSize;
        int numHeads = config.numAttentionHeads;
        int numKVHeads = config.getNumKVHeads();
        int headDim = config.getHeadDim();
        int maxSeqLen = kMaxSeqLen;
        
        hiddenStates.resize(hiddenSize);
        residual.resize(hiddenSize);
        normOutput.resize(hiddenSize);
        attnOutput.resize(hiddenSize);
        ffnOutput.resize(hiddenSize);
        
        int qSize = numHeads * headDim;
        int kvSize = numKVHeads * headDim;
        qkvBuffer.resize(qSize + 2 * kvSize);
        qBuffer.resize(qSize);
        kBuffer.resize(kvSize);
        vBuffer.resize(kvSize);
        attnScores.resize(numHeads * maxSeqLen);
        attnOutBuffer.resize(qSize);
        
        gateBuffer.resize(intermediateSize);
        upBuffer.resize(intermediateSize);
        logitsBuffer.resize(config.vocabSize);
    }
    
    void precomputeRoPE() {
        int headDim = config.getHeadDim();
        ropeFreqsCos.resize(kMaxSeqLen * headDim / 2);
        ropeFreqsSin.resize(kMaxSeqLen * headDim / 2);
        
        for (int pos = 0; pos < kMaxSeqLen; ++pos) {
            for (int i = 0; i < headDim / 2; ++i) {
                float freq = 1.0f / std::pow(config.ropeTheta, 2.0f * i / headDim);
                float angle = pos * freq;
                ropeFreqsCos[pos * headDim / 2 + i] = std::cos(angle);
                ropeFreqsSin[pos * headDim / 2 + i] = std::sin(angle);
            }
        }
    }
    
    // 获取或创建 KV Cache
    KVCache& getOrCreateKVCache(int requestId) {
        auto it = kvCaches.find(requestId);
        if (it == kvCaches.end()) {
            // 创建新的 KV Cache
            int numLayers = config.numHiddenLayers;
            int numKVHeads = config.getNumKVHeads();
            int headDim = config.getHeadDim();
            size_t kvSize = static_cast<size_t>(numLayers) * kMaxSeqLen * numKVHeads * headDim;
            
            KVCache cache;
            cache.kCache.resize(kvSize, 0.0f);
            cache.vCache.resize(kvSize, 0.0f);
            cache.currentLen = 0;
            kvCaches[requestId] = std::move(cache);
            return kvCaches[requestId];
        }
        return it->second;
    }
    
    // RMS Norm
    void rmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
        ggml_kernels::rms_norm(input, weight, output, size, eps);
    }
    
    // 矩阵乘法 F32
    void matmulF32(const float* weight, const float* input, float* output, int outFeatures, int inFeatures) {
        ggml_kernels::matmul_f32(weight, input, output, outFeatures, inFeatures);
    }
    
    // 应用 RoPE
    void applyRoPE(float* q, float* k, int headDim, int nHeads, int nKVHeads, int seqLen, int startPos) {
        const int halfDim = headDim / 2;
        
        for (int pos = 0; pos < seqLen; ++pos) {
            const int actualPos = startPos + pos;
            const float* cosPtr = ropeFreqsCos.data() + actualPos * halfDim;
            const float* sinPtr = ropeFreqsSin.data() + actualPos * halfDim;
            
            // Q 头
            for (int h = 0; h < nHeads; ++h) {
                float* head = q + h * headDim;
                for (int i = 0; i < halfDim; ++i) {
                    float x0 = head[i];
                    float x1 = head[i + halfDim];
                    float cos = cosPtr[i];
                    float sin = sinPtr[i];
                    head[i] = x0 * cos - x1 * sin;
                    head[i + halfDim] = x0 * sin + x1 * cos;
                }
            }
            
            // K 头
            for (int h = 0; h < nKVHeads; ++h) {
                float* head = k + h * headDim;
                for (int i = 0; i < halfDim; ++i) {
                    float x0 = head[i];
                    float x1 = head[i + halfDim];
                    float cos = cosPtr[i];
                    float sin = sinPtr[i];
                    head[i] = x0 * cos - x1 * sin;
                    head[i + halfDim] = x0 * sin + x1 * cos;
                }
            }
        }
    }
    
    // Softmax
    void softmax(float* scores, int size) {
        float maxVal = scores[0];
        for (int i = 1; i < size; ++i) {
            if (scores[i] > maxVal) maxVal = scores[i];
        }
        
        float sum = 0.0f;
        for (int i = 0; i < size; ++i) {
            scores[i] = std::exp(scores[i] - maxVal);
            sum += scores[i];
        }
        
        for (int i = 0; i < size; ++i) {
            scores[i] /= sum;
        }
    }
    
    // SiLU 激活函数
    float silu(float x) {
        return x / (1.0f + std::exp(-x));
    }
};

CPUBackend::CPUBackend() = default;

CPUBackend::~CPUBackend() {
    shutdown();
}

bool CPUBackend::initialize(const HFModelConfig& config) {
    config_ = config;
    
    // 分配实现
    impl_ = std::make_unique<CPUBackendImpl>();
    impl_->config = config;
    impl_->allocateBuffers();
    impl_->precomputeRoPE();
    
    initialized_ = true;
    CLLM_INFO("[CPUBackend] Initialized with hidden_size=%d, num_layers=%d",
              config.hiddenSize, config.numHiddenLayers);
    return true;
}

void CPUBackend::shutdown() {
    impl_.reset();
    initialized_ = false;
    weightsLoaded_ = false;
    CLLM_INFO("[CPUBackend] Shutdown");
}

bool CPUBackend::loadWeights(const ModelWeights& weights) {
    weights_ = weights;
    
    if (!impl_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
        return false;
    }
    
    // TODO: 从 weights 加载到 impl_->layers 等
    // 这里需要根据 weights.weightType 进行相应的转换
    
    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] Weights loaded (placeholder)");
    return true;
}

std::vector<float> CPUBackend::forward(
    const std::vector<int32_t>& inputIds,
    int requestId
) {
    if (!initialized_ || !impl_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
        return {};
    }
    
    if (!weightsLoaded_) {
        CLLM_ERROR("[CPUBackend] Weights not loaded");
        return {};
    }
    
    // 检查权重是否已实际加载到 layers
    if (impl_->layers.empty()) {
        CLLM_WARN("[CPUBackend] Weights not loaded into layers, forward not implemented yet");
        return {};
    }
    
    // TODO: 实现完整的前向推理
    // 1. Embedding
    // 2. 对于每一层：
    //    - RMS Norm
    //    - QKV Projection
    //    - RoPE
    //    - Attention
    //    - FFN
    // 3. Final RMS Norm
    // 4. LM Head
    
    CLLM_WARN("[CPUBackend] forward() not fully implemented yet");
    return {};
}

std::vector<std::vector<float>> CPUBackend::forwardBatch(
    const std::vector<std::vector<int32_t>>& batchInputIds,
    const std::vector<int>& requestIds
) {
    if (!initialized_ || !impl_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
        return {};
    }
    
    std::vector<std::vector<float>> results;
    results.reserve(batchInputIds.size());
    
    for (size_t i = 0; i < batchInputIds.size(); ++i) {
        results.push_back(forward(batchInputIds[i], requestIds[i]));
    }
    
    return results;
}

void CPUBackend::resetKVCache(int requestId) {
    if (!impl_) return;
    
    auto it = impl_->kvCaches.find(requestId);
    if (it != impl_->kvCaches.end()) {
        it->second.currentLen = 0;
        std::fill(it->second.kCache.begin(), it->second.kCache.end(), 0.0f);
        std::fill(it->second.vCache.begin(), it->second.vCache.end(), 0.0f);
    }
}

void CPUBackend::releaseKVCache(int requestId) {
    if (!impl_) return;
    impl_->kvCaches.erase(requestId);
}

int CPUBackend::getKVCacheCurrentLength(int requestId) const {
    if (!impl_) return 0;
    
    auto it = impl_->kvCaches.find(requestId);
    if (it != impl_->kvCaches.end()) {
        return it->second.currentLen;
    }
    return 0;
}

} // namespace kylin
} // namespace cllm
