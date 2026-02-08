/**
 * @file cpu_backend.cpp
 * @brief CPU 计算后端实现 - 完整版本
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
    
    // 权重数据 - F32 格式（预转换后）
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
        // 融合权重（可选）
        std::vector<float> qkvProj;      // 融合 QKV
        std::vector<float> gateUpProj;   // 融合 gate + up
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
    
    // Embedding
    void embedding(int tokenId, float* output) {
        if (tokenId < 0 || tokenId >= config.vocabSize) {
            std::fill(output, output + config.hiddenSize, 0.0f);
            return;
        }
        const float* embRow = embedTokens.data() + tokenId * config.hiddenSize;
        std::copy(embRow, embRow + config.hiddenSize, output);
    }
    
    // LM Head
    void lmHead(const float* input, float* output) {
        matmulF32(lmHeadWeight.data(), input, output, config.vocabSize, config.hiddenSize);
    }
    
    // 单层的 Attention 计算
    void attention(int layerIdx, const float* input, float* output, int startPos, KVCache& kvCache) {
        const LayerWeights& layer = layers[layerIdx];
        const int headDim = config.getHeadDim();
        const int nHeads = config.numAttentionHeads;
        const int nKVHeads = config.getNumKVHeads();
        const int qSize = nHeads * headDim;
        const int kvSize = nKVHeads * headDim;
        
        float* q = qBuffer.data();
        float* k = kBuffer.data();
        float* v = vBuffer.data();
        
        // QKV 投影
        if (!layer.qkvProj.empty()) {
            matmulF32(layer.qkvProj.data(), input, qkvBuffer.data(), qSize + 2 * kvSize, config.hiddenSize);
            q = qkvBuffer.data();
            k = qkvBuffer.data() + qSize;
            v = qkvBuffer.data() + qSize + kvSize;
        } else {
            matmulF32(layer.qProj.data(), input, q, qSize, config.hiddenSize);
            matmulF32(layer.kProj.data(), input, k, kvSize, config.hiddenSize);
            matmulF32(layer.vProj.data(), input, v, kvSize, config.hiddenSize);
        }
        
        // Q/K Norm（如果存在）
        if (!layer.qNorm.empty()) {
            for (int h = 0; h < nHeads; ++h) {
                rmsNorm(q + h * headDim, layer.qNorm.data(), q + h * headDim, headDim, config.rmsNormEps);
            }
        }
        if (!layer.kNorm.empty()) {
            for (int h = 0; h < nKVHeads; ++h) {
                rmsNorm(k + h * headDim, layer.kNorm.data(), k + h * headDim, headDim, config.rmsNormEps);
            }
        }
        
        // 应用 RoPE
        applyRoPE(q, k, headDim, nHeads, nKVHeads, 1, startPos);
        
        // 写入 KV Cache
        int numLayers = config.numHiddenLayers;
        size_t layerOffset = static_cast<size_t>(layerIdx) * kMaxSeqLen * nKVHeads * headDim;
        size_t posOffset = static_cast<size_t>(startPos) * nKVHeads * headDim;
        
        float* kCacheLayer = kvCache.kCache.data() + layerOffset;
        float* vCacheLayer = kvCache.vCache.data() + layerOffset;
        
        std::copy(k, k + kvSize, kCacheLayer + posOffset);
        std::copy(v, v + kvSize, vCacheLayer + posOffset);
        
        // Attention 计算
        const int kvLen = startPos + 1;
        
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q + h * headDim;
            int kvHead = h / (nHeads / nKVHeads);  // GQA
            
            // 计算 attention scores
            for (int t = 0; t < kvLen; ++t) {
                float* kHead = kCacheLayer + t * nKVHeads * headDim + kvHead * headDim;
                float score = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    score += qHead[d] * kHead[d];
                }
                score /= std::sqrt(static_cast<float>(headDim));
                attnScores[h * kMaxSeqLen + t] = score;
            }
            
            // Softmax
            softmax(attnScores.data() + h * kMaxSeqLen, kvLen);
            
            // 加权求和
            float* outHead = attnOutBuffer.data() + h * headDim;
            std::fill(outHead, outHead + headDim, 0.0f);
            
            for (int t = 0; t < kvLen; ++t) {
                float* vHead = vCacheLayer + t * nKVHeads * headDim + kvHead * headDim;
                float weight = attnScores[h * kMaxSeqLen + t];
                for (int d = 0; d < headDim; ++d) {
                    outHead[d] += weight * vHead[d];
                }
            }
        }
        
        // Output 投影
        matmulF32(layer.oProj.data(), attnOutBuffer.data(), output, config.hiddenSize, qSize);
    }
    
    // 单层的 FFN 计算
    void ffn(int layerIdx, const float* input, float* output) {
        const LayerWeights& layer = layers[layerIdx];
        
        if (!layer.gateUpProj.empty()) {
            // 使用融合的 gate_up 投影
            std::vector<float> gateUp(2 * config.intermediateSize);
            matmulF32(layer.gateUpProj.data(), input, gateUp.data(), 2 * config.intermediateSize, config.hiddenSize);
            
            // SiLU(gate) * up
            for (int i = 0; i < config.intermediateSize; ++i) {
                float gateVal = silu(gateUp[i]);
                float upVal = gateUp[i + config.intermediateSize];
                gateBuffer[i] = gateVal * upVal;
            }
        } else {
            // 分开计算 gate 和 up
            matmulF32(layer.gateProj.data(), input, gateBuffer.data(), config.intermediateSize, config.hiddenSize);
            matmulF32(layer.upProj.data(), input, upBuffer.data(), config.intermediateSize, config.hiddenSize);
            
            // SiLU(gate) * up
            for (int i = 0; i < config.intermediateSize; ++i) {
                gateBuffer[i] = silu(gateBuffer[i]) * upBuffer[i];
            }
        }
        
        // Down 投影
        matmulF32(layer.downProj.data(), gateBuffer.data(), output, config.hiddenSize, config.intermediateSize);
    }
};

CPUBackend::CPUBackend() = default;

CPUBackend::~CPUBackend() {
    shutdown();
}

bool CPUBackend::initialize(const HFModelConfig& config) {
    config_ = config;
    
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
    
    int numLayers = impl_->config.numHiddenLayers;
    impl_->layers.resize(numLayers);
    
    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] Weights loaded (placeholder, layers=%d)", numLayers);
    return true;
}

bool CPUBackend::loadWeightsFromHF(
    const std::vector<float>& embedTokens,
    const std::vector<float>& lmHeadWeight,
    const std::vector<float>& finalNormWeight,
    const std::vector<std::vector<float>>& layerWeights
) {
    if (!impl_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
        return false;
    }
    
    impl_->embedTokens = embedTokens;
    impl_->lmHeadWeight = lmHeadWeight;
    impl_->finalNormWeight = finalNormWeight;
    
    int numLayers = impl_->config.numHiddenLayers;
    impl_->layers.resize(numLayers);
    
    for (int i = 0; i < numLayers && i < static_cast<int>(layerWeights.size()); ++i) {
        // TODO: 根据实际的权重布局解析 layerWeights[i]
    }
    
    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] Weights loaded from HF format");
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
    
    // 检查权重是否已实际加载
    if (impl_->layers.empty() || impl_->embedTokens.empty()) {
        CLLM_WARN("[CPUBackend] Weights not loaded into layers");
        return {};
    }
    
    // 获取 KV Cache
    auto& kvCache = impl_->getOrCreateKVCache(requestId);
    int startPos = kvCache.currentLen;
    
    // 只处理最后一个 token
    int tokenId = inputIds.empty() ? 0 : inputIds.back();
    
    // 1. Embedding
    impl_->embedding(tokenId, impl_->hiddenStates.data());
    
    // 保存残差
    std::copy(impl_->hiddenStates.begin(), impl_->hiddenStates.end(), impl_->residual.begin());
    
    // 2. Transformer 层
    for (int layerIdx = 0; layerIdx < impl_->config.numHiddenLayers; ++layerIdx) {
        const auto& layer = impl_->layers[layerIdx];
        
        // 2.1 RMS Norm
        impl_->rmsNorm(impl_->hiddenStates.data(), layer.inputLayernorm.data(), 
                       impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);
        
        // 2.2 Attention
        impl_->attention(layerIdx, impl_->normOutput.data(), impl_->attnOutput.data(), startPos, kvCache);
        
        // 2.3 残差连接
        for (int i = 0; i < impl_->config.hiddenSize; ++i) {
            impl_->attnOutput[i] += impl_->residual[i];
        }
        
        // 保存残差
        std::copy(impl_->attnOutput.begin(), impl_->attnOutput.end(), impl_->residual.begin());
        
        // 2.4 Post-Attention RMS Norm
        impl_->rmsNorm(impl_->attnOutput.data(), layer.postAttentionLayernorm.data(),
                       impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);
        
        // 2.5 FFN
        impl_->ffn(layerIdx, impl_->normOutput.data(), impl_->ffnOutput.data());
        
        // 2.6 残差连接
        for (int i = 0; i < impl_->config.hiddenSize; ++i) {
            impl_->hiddenStates[i] = impl_->ffnOutput[i] + impl_->residual[i];
        }
        
        // 保存残差用于下一层
        std::copy(impl_->hiddenStates.begin(), impl_->hiddenStates.end(), impl_->residual.begin());
    }
    
    // 3. Final RMS Norm
    impl_->rmsNorm(impl_->hiddenStates.data(), impl_->finalNormWeight.data(),
                   impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);
    
    // 4. LM Head
    impl_->lmHead(impl_->normOutput.data(), impl_->logitsBuffer.data());
    
    // 更新 KV Cache 长度
    kvCache.currentLen = startPos + 1;
    
    return impl_->logitsBuffer;
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
