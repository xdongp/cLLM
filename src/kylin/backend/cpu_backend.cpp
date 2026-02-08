/**
 * @file cpu_backend.cpp
 * @brief CPU 计算后端实现 - 完整版本
 * 
 * 从 transformer.cpp 迁移的 CPU 计算逻辑
 * 支持 BF16/F32/FP16/INT8 权重
 */

#include "cllm/kylin/backend/cpu_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/kylin/core/quantization.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <functional>

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

// 默认最大序列长度
static constexpr int DEFAULT_MAX_SEQ_LEN = 4096;

// 辅助函数：获取最大序列长度
static inline int getMaxSeqLen(const HFModelConfig& config) {
    return config.maxPositionEmbeddings > 0 ? config.maxPositionEmbeddings : DEFAULT_MAX_SEQ_LEN;
}

// INT8 量化参数
struct QuantParams {
    float scale = 1.0f;
    int32_t zeroPoint = 0;
};

// 内部实现结构
struct CPUBackendImpl {
    // 模型配置
    HFModelConfig config;
    
    // 权重类型
    QuantType weightType = QuantType::FP32;
    
    // 权重数据 - F32 格式（用于 FP32/BF16/FP16）
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
        
        // INT8 格式权重（用于 INT8 量化模式）
        std::vector<int8_t> qProjInt8;
        std::vector<int8_t> kProjInt8;
        std::vector<int8_t> vProjInt8;
        std::vector<int8_t> oProjInt8;
        std::vector<int8_t> gateProjInt8;
        std::vector<int8_t> upProjInt8;
        std::vector<int8_t> downProjInt8;
        
        // INT8 量化参数
        QuantParams qProjQParams;
        QuantParams kProjQParams;
        QuantParams vProjQParams;
        QuantParams oProjQParams;
        QuantParams gateProjQParams;
        QuantParams upProjQParams;
        QuantParams downProjQParams;
    };
    std::vector<LayerWeights> layers;
    
    // 全局权重 - F32 格式
    std::vector<float> embedTokens;
    std::vector<float> lmHeadWeight;
    std::vector<float> finalNormWeight;
    
    // 全局权重 - INT8 格式
    std::vector<int8_t> embedTokensInt8;
    std::vector<int8_t> lmHeadWeightInt8;
    
    // INT8 量化参数
    QuantParams embedTokensQParams;
    QuantParams lmHeadWeightQParams;
    
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
    std::vector<float> gateUpBuffer;
    std::vector<float> logitsBuffer;
    std::vector<float> normWeightBuffer;
    std::vector<float> qkNormBuffer;
    
    // RoPE 频率
    std::vector<float> ropeFreqsCos;
    std::vector<float> ropeFreqsSin;
    
    void allocateBuffers() {
        int hiddenSize = config.hiddenSize;
        int intermediateSize = config.intermediateSize;
        int numHeads = config.numAttentionHeads;
        int numKVHeads = config.getNumKVHeads();
        int headDim = config.getHeadDim();
        int maxSeqLen = getMaxSeqLen(config);
        
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
        gateUpBuffer.resize(intermediateSize * 2);
        logitsBuffer.resize(config.vocabSize);
        normWeightBuffer.resize(hiddenSize);
        qkNormBuffer.resize(headDim);
    }
    
    void precomputeRoPE() {
        int headDim = config.getHeadDim();
        int maxSeqLen = getMaxSeqLen(config);
        ropeFreqsCos.resize(maxSeqLen * headDim / 2);
        ropeFreqsSin.resize(maxSeqLen * headDim / 2);
        
        for (int pos = 0; pos < maxSeqLen; ++pos) {
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
            int maxSeqLen = getMaxSeqLen(config);
            size_t kvSize = static_cast<size_t>(numLayers) * maxSeqLen * numKVHeads * headDim;
            
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

    // 矩阵乘法 BF16
    void matmulBF16(const uint16_t* weight, const float* input, float* output, int outFeatures, int inFeatures) {
        // 先将 BF16 权重转换为 F32，然后做矩阵乘法
        std::vector<float> weightF32(outFeatures * inFeatures);
        ggml_kernels::convert_bf16_to_f32(weight, weightF32.data(), outFeatures * inFeatures);
        matmulF32(weightF32.data(), input, output, outFeatures, inFeatures);
    }

    // 矩阵乘法 INT8
    void matmulINT8(const int8_t* weight, const float* input, float* output,
                    int outFeatures, int inFeatures, const QuantParams& qparams) {
        quant_kernels::matmul_int8_f32(weight, input, output, outFeatures, inFeatures,
                                       qparams.scale, qparams.zeroPoint);
    }

    // 通用矩阵乘法 - 根据权重类型自动选择
    void matmul(const float* f32Weight, const int8_t* int8Weight, const QuantParams& qparams,
                const float* input, float* output, int outFeatures, int inFeatures) {
        if (weightType == QuantType::INT8 && int8Weight != nullptr) {
            matmulINT8(int8Weight, input, output, outFeatures, inFeatures, qparams);
        } else {
            matmulF32(f32Weight, input, output, outFeatures, inFeatures);
        }
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
        if (weightType == QuantType::INT8) {
            // INT8 embedding: 反量化到 F32
            const int8_t* embRow = embedTokensInt8.data() + tokenId * config.hiddenSize;
            for (int i = 0; i < config.hiddenSize; ++i) {
                output[i] = (static_cast<float>(embRow[i]) - embedTokensQParams.zeroPoint) * embedTokensQParams.scale;
            }
        } else {
            const float* embRow = embedTokens.data() + tokenId * config.hiddenSize;
            std::copy(embRow, embRow + config.hiddenSize, output);
        }
    }
    
    // LM Head
    void lmHead(const float* input, float* output) {
        // 根据权重类型选择矩阵乘法
        matmul(lmHeadWeight.data(), lmHeadWeightInt8.data(), lmHeadWeightQParams,
               input, output, config.vocabSize, config.hiddenSize);
    }

    // 单层的 Attention 计算
    void attention(int layerIdx, const float* input, float* output, int seqLen, int startPos, KVCache& kvCache) {
        const LayerWeights& layer = layers[layerIdx];
        const int headDim = config.getHeadDim();
        const int nHeads = config.numAttentionHeads;
        const int nKVHeads = config.getNumKVHeads();
        const int qSize = nHeads * headDim;
        const int kvSize = nKVHeads * headDim;

        float* q = qBuffer.data();
        float* k = kBuffer.data();
        float* v = vBuffer.data();

        // QKV 投影 - 根据权重类型自动选择
        matmul(layer.qProj.data(), layer.qProjInt8.data(), layer.qProjQParams,
               input, q, qSize, config.hiddenSize);
        matmul(layer.kProj.data(), layer.kProjInt8.data(), layer.kProjQParams,
               input, k, kvSize, config.hiddenSize);
        matmul(layer.vProj.data(), layer.vProjInt8.data(), layer.vProjQParams,
               input, v, kvSize, config.hiddenSize);
        
        // Q/K Norm（如果存在）
        if (!layer.qNorm.empty()) {
            for (int h = 0; h < nHeads; ++h) {
                float* qHead = q + h * headDim;
                float sumSq = ggml_kernels::dot_product(qHead, qHead, headDim);
                float invRms = 1.0f / std::sqrt(sumSq / headDim + config.rmsNormEps);
                for (int i = 0; i < headDim; ++i) {
                    qHead[i] = qHead[i] * invRms * layer.qNorm[i];
                }
            }
        }
        if (!layer.kNorm.empty()) {
            for (int h = 0; h < nKVHeads; ++h) {
                float* kHead = k + h * headDim;
                float sumSq = ggml_kernels::dot_product(kHead, kHead, headDim);
                float invRms = 1.0f / std::sqrt(sumSq / headDim + config.rmsNormEps);
                for (int i = 0; i < headDim; ++i) {
                    kHead[i] = kHead[i] * invRms * layer.kNorm[i];
                }
            }
        }
        
        // 应用 RoPE
        applyRoPE(q, k, headDim, nHeads, nKVHeads, seqLen, startPos);
        
        // 写入 KV Cache
        int numLayers = config.numHiddenLayers;
        int maxSeqLen = getMaxSeqLen(config);
        size_t layerOffset = static_cast<size_t>(layerIdx) * maxSeqLen * nKVHeads * headDim;
        size_t posOffset = static_cast<size_t>(startPos) * nKVHeads * headDim;
        
        float* kCacheLayer = kvCache.kCache.data() + layerOffset;
        float* vCacheLayer = kvCache.vCache.data() + layerOffset;
        
        std::memcpy(kCacheLayer + posOffset, k, kvSize * sizeof(float));
        std::memcpy(vCacheLayer + posOffset, v, kvSize * sizeof(float));
        
        // Attention 计算
        const int kvLen = startPos + seqLen;
        const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
        const int gqa = nHeads / nKVHeads;
        const int vStride = nKVHeads * headDim;
        
        float* attnOut = attnOutBuffer.data();
        std::fill(attnOut, attnOut + qSize, 0.0f);
        
        // 并行处理每个 attention head
        #pragma omp parallel for schedule(static) if(nHeads >= 4)
        for (int h = 0; h < nHeads; ++h) {
            const int kvHead = h / gqa;
            const float* qHead = q + h * headDim;
            float* localScores = attnScores.data() + h * maxSeqLen;
            
            // 计算 attention scores + softmax
            float maxScore = -1e30f;
            for (int t = 0; t < kvLen; ++t) {
                const float* kRow = kCacheLayer + t * vStride + kvHead * headDim;
                float dot = ggml_kernels::dot_product(qHead, kRow, headDim) * scale;
                localScores[t] = dot;
                maxScore = (dot > maxScore) ? dot : maxScore;
            }
            
            float sumExp = 0.0f;
            for (int t = 0; t < kvLen; ++t) {
                float e = std::exp(localScores[t] - maxScore);
                localScores[t] = e;
                sumExp += e;
            }
            const float invSum = 1.0f / sumExp;
            for (int t = 0; t < kvLen; ++t) {
                localScores[t] *= invSum;
            }
            
            // Weighted sum of V
            float* outHead = attnOut + h * headDim;
            std::memset(outHead, 0, headDim * sizeof(float));
            
            for (int t = 0; t < kvLen; ++t) {
                const float* vRow = vCacheLayer + t * vStride + kvHead * headDim;
                const float weight = localScores[t];
                for (int d = 0; d < headDim; ++d) {
                    outHead[d] += weight * vRow[d];
                }
            }
        }
        
        // Output 投影
        matmul(layer.oProj.data(), layer.oProjInt8.data(), layer.oProjQParams,
               attnOut, output, config.hiddenSize, qSize);
    }

    // 单层的 FFN 计算
    void ffn(int layerIdx, const float* input, float* output) {
        const LayerWeights& layer = layers[layerIdx];

        // Gate 和 Up 投影
        matmul(layer.gateProj.data(), layer.gateProjInt8.data(), layer.gateProjQParams,
               input, gateBuffer.data(), config.intermediateSize, config.hiddenSize);
        matmul(layer.upProj.data(), layer.upProjInt8.data(), layer.upProjQParams,
               input, upBuffer.data(), config.intermediateSize, config.hiddenSize);
        
        // DEBUG: 打印 Gate 和 Up 值
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] FFN Gate - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      gateBuffer[0], gateBuffer[1], gateBuffer[2], gateBuffer[3], gateBuffer[4]);
            CLLM_DEBUG_CPU("[CPU DEBUG] FFN Up - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      upBuffer[0], upBuffer[1], upBuffer[2], upBuffer[3], upBuffer[4]);
        }
        
        // SiLU(gate) * up
        for (int i = 0; i < config.intermediateSize; ++i) {
            gateBuffer[i] = silu(gateBuffer[i]) * upBuffer[i];
        }
        
        // DEBUG: 打印 SiLU(gate) * up 值
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] FFN Gate*Up - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      gateBuffer[0], gateBuffer[1], gateBuffer[2], gateBuffer[3], gateBuffer[4]);
        }
        
        // Down 投影
        matmul(layer.downProj.data(), layer.downProjInt8.data(), layer.downProjQParams,
               gateBuffer.data(), output, config.hiddenSize, config.intermediateSize);
        
        // DEBUG: 打印 Down 输出值
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] FFN Down - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      output[0], output[1], output[2], output[3], output[4]);
        }
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
    if (!impl_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
        return false;
    }

    weights_ = weights;
    impl_->weightType = weights.weightType;

    // 根据权重类型选择加载方式
    if (weights.weightType == QuantType::INT8) {
        CLLM_INFO("[CPUBackend] Loading INT8 weights");
        return loadWeightsINT8(weights);
    } else if (weights.weightType == QuantType::FP16) {
        CLLM_INFO("[CPUBackend] Loading FP16 weights");
        return loadWeightsFP16(weights);
    } else {
        // BF16/FP32
        CLLM_INFO("[CPUBackend] Loading BF16/FP32 weights");
        return loadWeightsBF16(weights);
    }
}

bool CPUBackend::loadWeightsINT8(const ModelWeights& weights) {
    // 加载嵌入权重 (INT8)
    if (weights.embedTokens) {
        size_t count = config_.vocabSize * config_.hiddenSize;
        impl_->embedTokensInt8.resize(count);
        const int8_t* src = static_cast<const int8_t*>(weights.embedTokens);
        std::copy(src, src + count, impl_->embedTokensInt8.begin());
        // 从 weights 获取量化参数
        impl_->embedTokensQParams.scale = weights.scales.empty() ? 1.0f : weights.scales[0];
        impl_->embedTokensQParams.zeroPoint = weights.zeroPoints.empty() ? 0 : weights.zeroPoints[0];
    }

    // 加载 LM Head 权重 (INT8)
    if (weights.lmHeadWeight) {
        size_t count = config_.vocabSize * config_.hiddenSize;
        impl_->lmHeadWeightInt8.resize(count);
        const int8_t* src = static_cast<const int8_t*>(weights.lmHeadWeight);
        std::copy(src, src + count, impl_->lmHeadWeightInt8.begin());
        impl_->lmHeadWeightQParams.scale = weights.scales.size() > 2 ? weights.scales[2] : weights.scales[0];
        impl_->lmHeadWeightQParams.zeroPoint = weights.zeroPoints.size() > 2 ? weights.zeroPoints[2] : weights.zeroPoints[0];
    }

    // Final Norm 保持 F32 (LayerNorm 权重通常不量化)
    if (weights.finalNormWeight) {
        impl_->finalNormWeight.resize(config_.hiddenSize);
        std::copy(static_cast<const float*>(weights.finalNormWeight),
                  static_cast<const float*>(weights.finalNormWeight) + config_.hiddenSize,
                  impl_->finalNormWeight.begin());
    }

    return loadLayerWeightsINT8(weights);
}

bool CPUBackend::loadWeightsFP16(const ModelWeights& weights) {
    // FP16: 转换为 F32 存储
    if (weights.embedTokens) {
        impl_->embedTokens.resize(config_.vocabSize * config_.hiddenSize);
        quant_kernels::convert_fp16_to_f32(
            static_cast<const uint16_t*>(weights.embedTokens),
            impl_->embedTokens.data(),
            config_.vocabSize * config_.hiddenSize
        );
    }

    if (weights.lmHeadWeight) {
        impl_->lmHeadWeight.resize(config_.vocabSize * config_.hiddenSize);
        quant_kernels::convert_fp16_to_f32(
            static_cast<const uint16_t*>(weights.lmHeadWeight),
            impl_->lmHeadWeight.data(),
            config_.vocabSize * config_.hiddenSize
        );
    }

    // LayerNorm 权重保持 F32，直接复制
    if (weights.finalNormWeight) {
        impl_->finalNormWeight.resize(config_.hiddenSize);
        std::copy(static_cast<const float*>(weights.finalNormWeight),
                  static_cast<const float*>(weights.finalNormWeight) + config_.hiddenSize,
                  impl_->finalNormWeight.begin());
    }

    return loadLayerWeightsFP16(weights);
}

bool CPUBackend::loadWeightsBF16(const ModelWeights& weights) {
    // 转换并加载嵌入权重
    if (weights.embedTokens) {
        impl_->embedTokens.resize(config_.vocabSize * config_.hiddenSize);
        ggml_kernels::convert_bf16_to_f32(
            static_cast<const uint16_t*>(weights.embedTokens),
            impl_->embedTokens.data(),
            config_.vocabSize * config_.hiddenSize
        );
    }

    // 转换并加载 LM Head 权重
    if (weights.lmHeadWeight) {
        impl_->lmHeadWeight.resize(config_.vocabSize * config_.hiddenSize);
        ggml_kernels::convert_bf16_to_f32(
            static_cast<const uint16_t*>(weights.lmHeadWeight),
            impl_->lmHeadWeight.data(),
            config_.vocabSize * config_.hiddenSize
        );
    }

    // 转换并加载 Final Norm 权重
    if (weights.finalNormWeight) {
        impl_->finalNormWeight.resize(config_.hiddenSize);
        ggml_kernels::convert_bf16_to_f32(
            static_cast<const uint16_t*>(weights.finalNormWeight),
            impl_->finalNormWeight.data(),
            config_.hiddenSize
        );
    }
    
    // 转换并加载每层权重
    int numLayers = config_.numHiddenLayers;
    impl_->layers.resize(numLayers);
    
    for (int i = 0; i < numLayers && i < static_cast<int>(weights.layers.size()); ++i) {
        const auto& srcLayer = weights.layers[i];
        auto& dstLayer = impl_->layers[i];
        
        int hiddenSize = config_.hiddenSize;
        int intermediateSize = config_.intermediateSize;
        int numHeads = config_.numAttentionHeads;
        int numKVHeads = config_.getNumKVHeads();
        int headDim = config_.getHeadDim();
        
        // Input LayerNorm
        if (srcLayer.inputLayernorm) {
            dstLayer.inputLayernorm.resize(hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.inputLayernorm),
                dstLayer.inputLayernorm.data(),
                hiddenSize
            );
        }
        
        // Q, K, V, O Proj
        if (srcLayer.qProj) {
            dstLayer.qProj.resize(numHeads * headDim * hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.qProj),
                dstLayer.qProj.data(),
                numHeads * headDim * hiddenSize
            );
        }
        
        if (srcLayer.kProj) {
            dstLayer.kProj.resize(numKVHeads * headDim * hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.kProj),
                dstLayer.kProj.data(),
                numKVHeads * headDim * hiddenSize
            );
        }
        
        if (srcLayer.vProj) {
            dstLayer.vProj.resize(numKVHeads * headDim * hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.vProj),
                dstLayer.vProj.data(),
                numKVHeads * headDim * hiddenSize
            );
        }
        
        if (srcLayer.oProj) {
            dstLayer.oProj.resize(hiddenSize * numHeads * headDim);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.oProj),
                dstLayer.oProj.data(),
                hiddenSize * numHeads * headDim
            );
        }
        
        // Q/K Norm (optional)
        if (srcLayer.qNorm) {
            dstLayer.qNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.qNorm),
                dstLayer.qNorm.data(),
                headDim
            );
        }
        
        if (srcLayer.kNorm) {
            dstLayer.kNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.kNorm),
                dstLayer.kNorm.data(),
                headDim
            );
        }
        
        // Post-Attention LayerNorm
        if (srcLayer.postAttentionLayernorm) {
            dstLayer.postAttentionLayernorm.resize(hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.postAttentionLayernorm),
                dstLayer.postAttentionLayernorm.data(),
                hiddenSize
            );
        }
        
        // FFN weights
        if (srcLayer.gateProj) {
            dstLayer.gateProj.resize(intermediateSize * hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.gateProj),
                dstLayer.gateProj.data(),
                intermediateSize * hiddenSize
            );
        }
        
        if (srcLayer.upProj) {
            dstLayer.upProj.resize(intermediateSize * hiddenSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.upProj),
                dstLayer.upProj.data(),
                intermediateSize * hiddenSize
            );
        }
        
        if (srcLayer.downProj) {
            dstLayer.downProj.resize(hiddenSize * intermediateSize);
            ggml_kernels::convert_bf16_to_f32(
                static_cast<const uint16_t*>(srcLayer.downProj),
                dstLayer.downProj.data(),
                hiddenSize * intermediateSize
            );
        }
    }
    
    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] Weights loaded successfully (%d layers)", numLayers);
    return true;
}

bool CPUBackend::loadLayerWeightsINT8(const ModelWeights& weights) {
    int numLayers = config_.numHiddenLayers;
    impl_->layers.resize(numLayers);

    // 量化参数索引: 3个全局权重(embed, norm, lm_head) + 每层11个权重
    // 每层权重顺序: input_layernorm, q_proj, k_proj, v_proj, o_proj, q_norm, k_norm, post_layernorm, gate_proj, up_proj, down_proj
    const size_t layerWeightsStartIdx = 3;

    for (int i = 0; i < numLayers && i < static_cast<int>(weights.layers.size()); ++i) {
        const auto& srcLayer = weights.layers[i];
        auto& dstLayer = impl_->layers[i];

        int hiddenSize = config_.hiddenSize;
        int intermediateSize = config_.intermediateSize;
        int numHeads = config_.numAttentionHeads;
        int numKVHeads = config_.getNumKVHeads();
        int headDim = config_.getHeadDim();

        // 计算该层权重的起始索引 (跳过3个全局权重)
        size_t baseIdx = layerWeightsStartIdx + i * 11;

        // Input LayerNorm - 保持 F32 (索引0)
        if (srcLayer.inputLayernorm) {
            dstLayer.inputLayernorm.resize(hiddenSize);
            std::copy(static_cast<const float*>(srcLayer.inputLayernorm),
                      static_cast<const float*>(srcLayer.inputLayernorm) + hiddenSize,
                      dstLayer.inputLayernorm.begin());
        }

        // Q, K, V, O Proj - INT8
        if (srcLayer.qProj) {
            size_t count = static_cast<size_t>(numHeads) * headDim * hiddenSize;
            dstLayer.qProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.qProj),
                      static_cast<const int8_t*>(srcLayer.qProj) + count,
                      dstLayer.qProjInt8.begin());
            // q_proj 是每层第2个权重 (索引1)
            dstLayer.qProjQParams.scale = weights.scales.size() > baseIdx + 1 ? weights.scales[baseIdx + 1] : 1.0f;
            dstLayer.qProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 1 ? weights.zeroPoints[baseIdx + 1] : 0;
        }

        if (srcLayer.kProj) {
            size_t count = static_cast<size_t>(numKVHeads) * headDim * hiddenSize;
            dstLayer.kProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.kProj),
                      static_cast<const int8_t*>(srcLayer.kProj) + count,
                      dstLayer.kProjInt8.begin());
            dstLayer.kProjQParams.scale = weights.scales.size() > baseIdx + 2 ? weights.scales[baseIdx + 2] : 1.0f;
            dstLayer.kProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 2 ? weights.zeroPoints[baseIdx + 2] : 0;
        }

        if (srcLayer.vProj) {
            size_t count = static_cast<size_t>(numKVHeads) * headDim * hiddenSize;
            dstLayer.vProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.vProj),
                      static_cast<const int8_t*>(srcLayer.vProj) + count,
                      dstLayer.vProjInt8.begin());
            dstLayer.vProjQParams.scale = weights.scales.size() > baseIdx + 3 ? weights.scales[baseIdx + 3] : 1.0f;
            dstLayer.vProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 3 ? weights.zeroPoints[baseIdx + 3] : 0;
        }

        if (srcLayer.oProj) {
            size_t count = static_cast<size_t>(hiddenSize) * numHeads * headDim;
            dstLayer.oProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.oProj),
                      static_cast<const int8_t*>(srcLayer.oProj) + count,
                      dstLayer.oProjInt8.begin());
            dstLayer.oProjQParams.scale = weights.scales.size() > baseIdx + 4 ? weights.scales[baseIdx + 4] : 1.0f;
            dstLayer.oProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 4 ? weights.zeroPoints[baseIdx + 4] : 0;
        }

        // Q/K Norm - F32
        if (srcLayer.qNorm) {
            dstLayer.qNorm.resize(headDim);
            std::copy(static_cast<const float*>(srcLayer.qNorm),
                      static_cast<const float*>(srcLayer.qNorm) + headDim,
                      dstLayer.qNorm.begin());
        }

        if (srcLayer.kNorm) {
            dstLayer.kNorm.resize(headDim);
            std::copy(static_cast<const float*>(srcLayer.kNorm),
                      static_cast<const float*>(srcLayer.kNorm) + headDim,
                      dstLayer.kNorm.begin());
        }

        // Post-Attention LayerNorm - F32
        if (srcLayer.postAttentionLayernorm) {
            dstLayer.postAttentionLayernorm.resize(hiddenSize);
            std::copy(static_cast<const float*>(srcLayer.postAttentionLayernorm),
                      static_cast<const float*>(srcLayer.postAttentionLayernorm) + hiddenSize,
                      dstLayer.postAttentionLayernorm.begin());
        }

        // FFN weights - INT8 (gate=idx8, up=idx9, down=idx10)
        if (srcLayer.gateProj) {
            size_t count = static_cast<size_t>(intermediateSize) * hiddenSize;
            dstLayer.gateProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.gateProj),
                      static_cast<const int8_t*>(srcLayer.gateProj) + count,
                      dstLayer.gateProjInt8.begin());
            dstLayer.gateProjQParams.scale = weights.scales.size() > baseIdx + 8 ? weights.scales[baseIdx + 8] : 1.0f;
            dstLayer.gateProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 8 ? weights.zeroPoints[baseIdx + 8] : 0;
        }

        if (srcLayer.upProj) {
            size_t count = static_cast<size_t>(intermediateSize) * hiddenSize;
            dstLayer.upProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.upProj),
                      static_cast<const int8_t*>(srcLayer.upProj) + count,
                      dstLayer.upProjInt8.begin());
            dstLayer.upProjQParams.scale = weights.scales.size() > baseIdx + 9 ? weights.scales[baseIdx + 9] : 1.0f;
            dstLayer.upProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 9 ? weights.zeroPoints[baseIdx + 9] : 0;
        }

        if (srcLayer.downProj) {
            size_t count = static_cast<size_t>(hiddenSize) * intermediateSize;
            dstLayer.downProjInt8.resize(count);
            std::copy(static_cast<const int8_t*>(srcLayer.downProj),
                      static_cast<const int8_t*>(srcLayer.downProj) + count,
                      dstLayer.downProjInt8.begin());
            dstLayer.downProjQParams.scale = weights.scales.size() > baseIdx + 10 ? weights.scales[baseIdx + 10] : 1.0f;
            dstLayer.downProjQParams.zeroPoint = weights.zeroPoints.size() > baseIdx + 10 ? weights.zeroPoints[baseIdx + 10] : 0;
        }
    }

    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] INT8 weights loaded successfully (%d layers)", numLayers);
    return true;
}

bool CPUBackend::loadLayerWeightsFP16(const ModelWeights& weights) {
    int numLayers = config_.numHiddenLayers;
    impl_->layers.resize(numLayers);

    for (int i = 0; i < numLayers && i < static_cast<int>(weights.layers.size()); ++i) {
        const auto& srcLayer = weights.layers[i];
        auto& dstLayer = impl_->layers[i];

        int hiddenSize = config_.hiddenSize;
        int intermediateSize = config_.intermediateSize;
        int numHeads = config_.numAttentionHeads;
        int numKVHeads = config_.getNumKVHeads();
        int headDim = config_.getHeadDim();

        // Input LayerNorm - 保持 F32
        if (srcLayer.inputLayernorm) {
            dstLayer.inputLayernorm.resize(hiddenSize);
            std::copy(static_cast<const float*>(srcLayer.inputLayernorm),
                      static_cast<const float*>(srcLayer.inputLayernorm) + hiddenSize,
                      dstLayer.inputLayernorm.begin());
        }

        // Q, K, V, O Proj - FP16 转 F32
        if (srcLayer.qProj) {
            dstLayer.qProj.resize(static_cast<size_t>(numHeads) * headDim * hiddenSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.qProj),
                dstLayer.qProj.data(), static_cast<size_t>(numHeads) * headDim * hiddenSize);
        }

        if (srcLayer.kProj) {
            dstLayer.kProj.resize(static_cast<size_t>(numKVHeads) * headDim * hiddenSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.kProj),
                dstLayer.kProj.data(), static_cast<size_t>(numKVHeads) * headDim * hiddenSize);
        }

        if (srcLayer.vProj) {
            dstLayer.vProj.resize(static_cast<size_t>(numKVHeads) * headDim * hiddenSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.vProj),
                dstLayer.vProj.data(), static_cast<size_t>(numKVHeads) * headDim * hiddenSize);
        }

        if (srcLayer.oProj) {
            dstLayer.oProj.resize(static_cast<size_t>(hiddenSize) * numHeads * headDim);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.oProj),
                dstLayer.oProj.data(), static_cast<size_t>(hiddenSize) * numHeads * headDim);
        }

        // Q/K Norm - 保持 F32
        if (srcLayer.qNorm) {
            dstLayer.qNorm.resize(headDim);
            std::copy(static_cast<const float*>(srcLayer.qNorm),
                      static_cast<const float*>(srcLayer.qNorm) + headDim,
                      dstLayer.qNorm.begin());
        }

        if (srcLayer.kNorm) {
            dstLayer.kNorm.resize(headDim);
            std::copy(static_cast<const float*>(srcLayer.kNorm),
                      static_cast<const float*>(srcLayer.kNorm) + headDim,
                      dstLayer.kNorm.begin());
        }

        // Post-Attention LayerNorm - 保持 F32
        if (srcLayer.postAttentionLayernorm) {
            dstLayer.postAttentionLayernorm.resize(hiddenSize);
            std::copy(static_cast<const float*>(srcLayer.postAttentionLayernorm),
                      static_cast<const float*>(srcLayer.postAttentionLayernorm) + hiddenSize,
                      dstLayer.postAttentionLayernorm.begin());
        }

        // FFN weights - FP16 转 F32
        if (srcLayer.gateProj) {
            dstLayer.gateProj.resize(static_cast<size_t>(intermediateSize) * hiddenSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.gateProj),
                dstLayer.gateProj.data(), static_cast<size_t>(intermediateSize) * hiddenSize);
        }

        if (srcLayer.upProj) {
            dstLayer.upProj.resize(static_cast<size_t>(intermediateSize) * hiddenSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.upProj),
                dstLayer.upProj.data(), static_cast<size_t>(intermediateSize) * hiddenSize);
        }

        if (srcLayer.downProj) {
            dstLayer.downProj.resize(static_cast<size_t>(hiddenSize) * intermediateSize);
            quant_kernels::convert_fp16_to_f32(
                static_cast<const uint16_t*>(srcLayer.downProj),
                dstLayer.downProj.data(), static_cast<size_t>(hiddenSize) * intermediateSize);
        }
    }

    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] FP16 weights loaded successfully (%d layers)", numLayers);
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
    
    // 获取 KV Cache
    auto& kvCache = impl_->getOrCreateKVCache(requestId);
    int startPos = kvCache.currentLen;
    
    // 🔥 关键优化：支持多 token 输入（批量 prefill）
    // 当 inputIds.size() > 1 时，逐个处理所有 tokens
    // 但只返回最后一个 token 的 logits
    if (inputIds.empty()) {
        CLLM_ERROR("[CPUBackend] Empty input");
        return {};
    }
    
    // 逐个处理所有 input tokens
    for (size_t tokenIdx = 0; tokenIdx < inputIds.size(); ++tokenIdx) {
        int tokenId = inputIds[tokenIdx];
        int currentPos = startPos + static_cast<int>(tokenIdx);
        
        // 1. Embedding
        impl_->embedding(tokenId, impl_->hiddenStates.data());

    // DEBUG: 打印 Embedding 输出
    CLLM_DEBUG_CPU("[CPU DEBUG] Embedding - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
              impl_->hiddenStates[0], impl_->hiddenStates[1], impl_->hiddenStates[2],
              impl_->hiddenStates[3], impl_->hiddenStates[4]);

    // 保存残差
    std::copy(impl_->hiddenStates.begin(), impl_->hiddenStates.end(), impl_->residual.begin());
    
    // 2. Transformer 层
    for (int layerIdx = 0; layerIdx < impl_->config.numHiddenLayers; ++layerIdx) {
        const auto& layer = impl_->layers[layerIdx];
        
        // 2.1 RMS Norm
        impl_->rmsNorm(impl_->hiddenStates.data(), layer.inputLayernorm.data(),
                       impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);

        // DEBUG: 打印 Attention Input (RMS Norm 后的输出)
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] Attention Input - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->normOutput[0], impl_->normOutput[1], impl_->normOutput[2],
                      impl_->normOutput[3], impl_->normOutput[4]);
        }

        // 2.2 Attention - 使用 currentPos 作为当前位置
        impl_->attention(layerIdx, impl_->normOutput.data(), impl_->attnOutput.data(), 1, currentPos, kvCache);

        // DEBUG: 打印 Attention 输出
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] Attention Output - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->attnOutput[0], impl_->attnOutput[1], impl_->attnOutput[2],
                      impl_->attnOutput[3], impl_->attnOutput[4]);
        }

        // 2.3 残差连接
        ggml_kernels::vector_add(impl_->residual.data(), impl_->attnOutput.data(),
                                 impl_->hiddenStates.data(), impl_->config.hiddenSize);

        // DEBUG: 打印 FFN Input (Attention + Residual 后的输出)
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] FFN Input - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->hiddenStates[0], impl_->hiddenStates[1], impl_->hiddenStates[2],
                      impl_->hiddenStates[3], impl_->hiddenStates[4]);
        }

        // 保存残差
        std::copy(impl_->hiddenStates.begin(), impl_->hiddenStates.end(), impl_->residual.begin());

        // 2.4 Post-Attention RMS Norm
        impl_->rmsNorm(impl_->hiddenStates.data(), layer.postAttentionLayernorm.data(),
                       impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);

        // DEBUG: 打印 Post-Attention RMS Norm 输出
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] Post Attention RMS Norm - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->normOutput[0], impl_->normOutput[1], impl_->normOutput[2],
                      impl_->normOutput[3], impl_->normOutput[4]);
        }

        // 2.5 FFN
        impl_->ffn(layerIdx, impl_->normOutput.data(), impl_->ffnOutput.data());
        
        // 2.6 残差连接
        ggml_kernels::vector_add(impl_->residual.data(), impl_->ffnOutput.data(),
                                 impl_->hiddenStates.data(), impl_->config.hiddenSize);

        // DEBUG: 打印 Layer 0 Output
        if (layerIdx == 0) {
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 0 Output - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->hiddenStates[0], impl_->hiddenStates[1], impl_->hiddenStates[2],
                      impl_->hiddenStates[3], impl_->hiddenStates[4]);
        }

        // DEBUG: 打印 Layer 1 关键步骤
        if (layerIdx == 1) {
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 1 Attention Input - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->normOutput[0], impl_->normOutput[1], impl_->normOutput[2],
                      impl_->normOutput[3], impl_->normOutput[4]);
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 1 Attention Output - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->attnOutput[0], impl_->attnOutput[1], impl_->attnOutput[2],
                      impl_->attnOutput[3], impl_->attnOutput[4]);
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 1 FFN Input - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->hiddenStates[0], impl_->hiddenStates[1], impl_->hiddenStates[2],
                      impl_->hiddenStates[3], impl_->hiddenStates[4]);
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 1 FFN Output - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->ffnOutput[0], impl_->ffnOutput[1], impl_->ffnOutput[2],
                      impl_->ffnOutput[3], impl_->ffnOutput[4]);
            CLLM_DEBUG_CPU("[CPU DEBUG] Layer 1 Output - first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      impl_->hiddenStates[0], impl_->hiddenStates[1], impl_->hiddenStates[2],
                      impl_->hiddenStates[3], impl_->hiddenStates[4]);
        }

        // 保存残差用于下一层
        std::copy(impl_->hiddenStates.begin(), impl_->hiddenStates.end(), impl_->residual.begin());
    }
    
        // 3. Final RMS Norm
        impl_->rmsNorm(impl_->hiddenStates.data(), impl_->finalNormWeight.data(),
                       impl_->normOutput.data(), impl_->config.hiddenSize, impl_->config.rmsNormEps);
        
        // 4. LM Head - 只在最后一个 token 时计算 logits
        if (tokenIdx == inputIds.size() - 1) {
            impl_->lmHead(impl_->normOutput.data(), impl_->logitsBuffer.data());
        }
        
        // 更新 KV Cache 长度
        kvCache.currentLen = currentPos + 1;
    }
    
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
