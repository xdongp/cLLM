/**
 * @file ggml_backend.cpp
 * @brief GGML GPU 后端完整实现
 * 
 * 包含完整的 Transformer 计算：
 * - Q/K/V Projection
 * - Q/K RMS Norm (Qwen3)
 * - RoPE (旋转位置编码)
 * - KV Cache
 * - Attention (QK^T + softmax + V)
 * - GQA (Grouped Query Attention)
 * - O Projection
 * - FFN (SwiGLU)
 */

#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {

static constexpr int MAX_SEQ_LEN = 2048;

GGMLGPUBackend::GGMLGPUBackend() = default;

GGMLGPUBackend::~GGMLGPUBackend() {
    if (computeBuffer_) {
        ggml_backend_buffer_free(computeBuffer_);
        computeBuffer_ = nullptr;
    }
    if (weightBuffer_) {
        ggml_backend_buffer_free(weightBuffer_);
        weightBuffer_ = nullptr;
    }
    if (computeCtx_) {
        ggml_free(computeCtx_);
        computeCtx_ = nullptr;
    }
    if (weightCtx_) {
        ggml_free(weightCtx_);
        weightCtx_ = nullptr;
    }
    if (backend_) {
        ggml_backend_free(backend_);
        backend_ = nullptr;
    }
    CLLM_INFO("[GGMLGPUBackend] Resources released");
}

bool GGMLGPUBackend::initialize(const HFModelConfig& config) {
    if (initialized_) return true;
    
    config_ = config;
    
#ifdef GGML_USE_METAL
    CLLM_INFO("[GGMLGPUBackend] Initializing Metal backend...");
    backend_ = ggml_backend_metal_init();
    if (!backend_) {
        CLLM_ERROR("[GGMLGPUBackend] Failed to initialize Metal backend");
        return false;
    }
    CLLM_INFO("[GGMLGPUBackend] ✅ Metal backend initialized");
#else
    CLLM_ERROR("[GGMLGPUBackend] Metal not compiled");
    return false;
#endif
    
    if (!createWeightTensors()) {
        CLLM_ERROR("[GGMLGPUBackend] Failed to create weight tensors");
        return false;
    }
    
    // 预计算 RoPE 频率
    precomputeRoPE();
    
    // 初始化 CPU 端 KV Cache（用于调试和数据传输）
    const int headDim = config_.getHeadDim();
    const int nKVHeads = config_.getNumKVHeads();
    const int numLayers = config_.numHiddenLayers;
    kCacheCPU_.resize(numLayers);
    vCacheCPU_.resize(numLayers);
    for (int l = 0; l < numLayers; ++l) {
        kCacheCPU_[l].resize(MAX_SEQ_LEN * nKVHeads * headDim, 0.0f);
        vCacheCPU_[l].resize(MAX_SEQ_LEN * nKVHeads * headDim, 0.0f);
    }
    
    kvCacheLen_ = 0;
    initialized_ = true;
    CLLM_INFO("[GGMLGPUBackend] ✅ GPU Backend initialization complete");
    return true;
}

void GGMLGPUBackend::precomputeRoPE() {
    const int headDim = config_.getHeadDim();
    const int halfDim = headDim / 2;
    const float ropeTheta = config_.ropeTheta;
    
    ropeFreqsCos_.resize(MAX_SEQ_LEN * halfDim);
    ropeFreqsSin_.resize(MAX_SEQ_LEN * halfDim);
    
    for (int pos = 0; pos < MAX_SEQ_LEN; ++pos) {
        for (int i = 0; i < halfDim; ++i) {
            float freq = 1.0f / std::pow(ropeTheta, 2.0f * i / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * halfDim + i] = std::cos(angle);
            ropeFreqsSin_[pos * halfDim + i] = std::sin(angle);
        }
    }
    
    CLLM_INFO("[GGMLGPUBackend] RoPE frequencies precomputed (theta=%.0f)", ropeTheta);
}

bool GGMLGPUBackend::createWeightTensors() {
    const int vocabSize = config_.vocabSize;
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int numLayers = config_.numHiddenLayers;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // 张量数量估算
    const size_t numTensors = 3 + numLayers * 11 + 128;
    const size_t ctxSize = ggml_tensor_overhead() * numTensors + 
                           ggml_graph_overhead() * 2 + 32 * 1024 * 1024;
    
    CLLM_INFO("[GGMLGPUBackend] Creating tensors (layers=%d)", numLayers);
    
    struct ggml_init_params params = {
        .mem_size   = ctxSize,
        .mem_buffer = nullptr,
        .no_alloc   = true,
    };
    
    weightCtx_ = ggml_init(params);
    if (!weightCtx_) {
        CLLM_ERROR("[GGMLGPUBackend] Failed to create weight context");
        return false;
    }
    
    // 嵌入层 [hiddenSize, vocabSize]（GGML 列主序）
    embedTokens_ = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, vocabSize);
    ggml_set_name(embedTokens_, "embed_tokens");
    
    finalNorm_ = ggml_new_tensor_1d(weightCtx_, GGML_TYPE_F32, hiddenSize);
    ggml_set_name(finalNorm_, "final_norm");
    
    lmHead_ = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, vocabSize);
    ggml_set_name(lmHead_, "lm_head");
    
    // 每层权重
    layers_.resize(numLayers);
    
    for (int i = 0; i < numLayers; ++i) {
        char name[64];
        LayerTensors& layer = layers_[i];
        
        snprintf(name, sizeof(name), "layer.%d.input_layernorm", i);
        layer.inputLayernorm = ggml_new_tensor_1d(weightCtx_, GGML_TYPE_F32, hiddenSize);
        ggml_set_name(layer.inputLayernorm, name);
        
        // 投影矩阵 [in_features, out_features] for GGML
        snprintf(name, sizeof(name), "layer.%d.q_proj", i);
        layer.qProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, qSize);
        ggml_set_name(layer.qProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.k_proj", i);
        layer.kProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, kvSize);
        ggml_set_name(layer.kProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.v_proj", i);
        layer.vProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, kvSize);
        ggml_set_name(layer.vProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.o_proj", i);
        layer.oProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, qSize, hiddenSize);
        ggml_set_name(layer.oProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.q_norm", i);
        layer.qNorm = ggml_new_tensor_1d(weightCtx_, GGML_TYPE_F32, headDim);
        ggml_set_name(layer.qNorm, name);
        
        snprintf(name, sizeof(name), "layer.%d.k_norm", i);
        layer.kNorm = ggml_new_tensor_1d(weightCtx_, GGML_TYPE_F32, headDim);
        ggml_set_name(layer.kNorm, name);
        
        snprintf(name, sizeof(name), "layer.%d.post_attention_layernorm", i);
        layer.postAttentionLayernorm = ggml_new_tensor_1d(weightCtx_, GGML_TYPE_F32, hiddenSize);
        ggml_set_name(layer.postAttentionLayernorm, name);
        
        snprintf(name, sizeof(name), "layer.%d.gate_proj", i);
        layer.gateProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, intermediateSize);
        ggml_set_name(layer.gateProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.up_proj", i);
        layer.upProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, intermediateSize);
        ggml_set_name(layer.upProj, name);
        
        snprintf(name, sizeof(name), "layer.%d.down_proj", i);
        layer.downProj = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, intermediateSize, hiddenSize);
        ggml_set_name(layer.downProj, name);
    }
    
    // 分配 GPU buffer
    weightBuffer_ = ggml_backend_alloc_ctx_tensors(weightCtx_, backend_);
    if (!weightBuffer_) {
        CLLM_ERROR("[GGMLGPUBackend] Failed to allocate GPU buffer");
        return false;
    }
    
    size_t bufferSize = ggml_backend_buffer_get_size(weightBuffer_);
    CLLM_INFO("[GGMLGPUBackend] ✅ GPU buffer: %.2f MB", bufferSize / (1024.0 * 1024.0));
    
    return true;
}

bool GGMLGPUBackend::uploadWeights(
    const float* embedTokens,
    const std::vector<LayerWeightsGPU>& layerWeights,
    const float* finalNorm,
    const float* lmHead
) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return false;
    }
    
    const int vocabSize = config_.vocabSize;
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    CLLM_INFO("[GGMLGPUBackend] Uploading weights to GPU...");
    
    // 直接上传（行主序 -> 列主序隐式转置）
    ggml_backend_tensor_set(embedTokens_, embedTokens, 0, 
                            (size_t)vocabSize * hiddenSize * sizeof(float));
    
    ggml_backend_tensor_set(finalNorm_, finalNorm, 0, hiddenSize * sizeof(float));
    
    if (!config_.tieWordEmbeddings && lmHead) {
        ggml_backend_tensor_set(lmHead_, lmHead, 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    } else {
        ggml_backend_tensor_set(lmHead_, embedTokens, 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    }
    
    // 每层权重
    for (size_t i = 0; i < layerWeights.size() && i < layers_.size(); ++i) {
        const LayerWeightsGPU& src = layerWeights[i];
        LayerTensors& dst = layers_[i];
        
        // 1D 权重
        ggml_backend_tensor_set(dst.inputLayernorm, src.inputLayernorm, 0,
                                hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.postAttentionLayernorm, src.postAttentionLayernorm, 0,
                                hiddenSize * sizeof(float));
        
        if (src.qNorm) {
            ggml_backend_tensor_set(dst.qNorm, src.qNorm, 0, headDim * sizeof(float));
        }
        if (src.kNorm) {
            ggml_backend_tensor_set(dst.kNorm, src.kNorm, 0, headDim * sizeof(float));
        }
        
        // 2D 权重 - 直接上传
        ggml_backend_tensor_set(dst.qProj, src.qProj, 0, (size_t)qSize * hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.kProj, src.kProj, 0, (size_t)kvSize * hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.vProj, src.vProj, 0, (size_t)kvSize * hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.oProj, src.oProj, 0, (size_t)hiddenSize * qSize * sizeof(float));
        ggml_backend_tensor_set(dst.gateProj, src.gateProj, 0, (size_t)intermediateSize * hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.upProj, src.upProj, 0, (size_t)intermediateSize * hiddenSize * sizeof(float));
        ggml_backend_tensor_set(dst.downProj, src.downProj, 0, (size_t)hiddenSize * intermediateSize * sizeof(float));
    }
    
    CLLM_INFO("[GGMLGPUBackend] ✅ Weights uploaded");
    return true;
}

// 使用 BLAS 优化的函数
static inline void cpuRmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

static inline void cpuMatmul(const float* weight, const float* input, float* output, int M, int K) {
    ggml_kernels::matmul_f32(weight, input, output, M, K);
}

// CPU 辅助函数：应用 RoPE
static void cpuApplyRoPE(float* x, int nHeads, int headDim, int position,
                          const float* freqsCos, const float* freqsSin) {
    const int halfDim = headDim / 2;
    const float* cosPtr = freqsCos + position * halfDim;
    const float* sinPtr = freqsSin + position * halfDim;
    
    for (int h = 0; h < nHeads; ++h) {
        float* head = x + h * headDim;
        for (int i = 0; i < halfDim; ++i) {
            float x0 = head[i];
            float x1 = head[i + halfDim];
            head[i] = x0 * cosPtr[i] - x1 * sinPtr[i];
            head[i + halfDim] = x0 * sinPtr[i] + x1 * cosPtr[i];
        }
    }
}

// CPU 辅助函数：SiLU
static float cpuSilu(float x) {
    return x / (1.0f + std::exp(-x));
}

std::vector<float> GGMLGPUBackend::forward(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return {};
    }
    
    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const int numLayers = config_.numHiddenLayers;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int intermediateSize = config_.intermediateSize;
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    const int gqa = nHeads / nKVHeads;
    const int totalLen = position + 1;
    const float eps = config_.rmsNormEps;
    
    // ===== 首次调用时缓存权重到 CPU =====
    if (weightsCached_.empty()) {
        cacheWeightsToCPU();
    }
    
    // 1. 获取 Embedding
    std::vector<float> hidden(hiddenSize);
    const float* embedData = weightsCached_["embed_tokens"].data();
    std::copy(embedData + tokenId * hiddenSize, 
              embedData + (tokenId + 1) * hiddenSize, 
              hidden.begin());
    
    // 临时缓冲区
    std::vector<float> residual(hiddenSize);
    std::vector<float> normOut(hiddenSize);
    std::vector<float> q(qSize), k(kvSize), v(kvSize);
    std::vector<float> attnOut(qSize);
    std::vector<float> oOut(hiddenSize);
    std::vector<float> gate(intermediateSize), up(intermediateSize);
    std::vector<float> ffnOut(hiddenSize);
    
    // 2. Transformer Layers
    for (int l = 0; l < numLayers; ++l) {
        // 从缓存获取权重
        const float* inputNormW = weightsCached_["layer." + std::to_string(l) + ".input_layernorm"].data();
        const float* qProjW = weightsCached_["layer." + std::to_string(l) + ".q_proj"].data();
        const float* kProjW = weightsCached_["layer." + std::to_string(l) + ".k_proj"].data();
        const float* vProjW = weightsCached_["layer." + std::to_string(l) + ".v_proj"].data();
        const float* oProjW = weightsCached_["layer." + std::to_string(l) + ".o_proj"].data();
        const float* qNormW = weightsCached_["layer." + std::to_string(l) + ".q_norm"].data();
        const float* kNormW = weightsCached_["layer." + std::to_string(l) + ".k_norm"].data();
        const float* postNormW = weightsCached_["layer." + std::to_string(l) + ".post_attention_layernorm"].data();
        const float* gateProjW = weightsCached_["layer." + std::to_string(l) + ".gate_proj"].data();
        const float* upProjW = weightsCached_["layer." + std::to_string(l) + ".up_proj"].data();
        const float* downProjW = weightsCached_["layer." + std::to_string(l) + ".down_proj"].data();
        
        // 保存残差
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        
        // Input LayerNorm
        cpuRmsNorm(hidden.data(), inputNormW, normOut.data(), hiddenSize, eps);
        
        // Q, K, V Projections
        cpuMatmul(qProjW, normOut.data(), q.data(), qSize, hiddenSize);
        cpuMatmul(kProjW, normOut.data(), k.data(), kvSize, hiddenSize);
        cpuMatmul(vProjW, normOut.data(), v.data(), kvSize, hiddenSize);
        
        // Q/K RMS Norm (per head)
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q.data() + h * headDim;
            float sumSq = 0.0f;
            for (int i = 0; i < headDim; ++i) sumSq += qHead[i] * qHead[i];
            float scale = 1.0f / std::sqrt(sumSq / headDim + eps);
            for (int i = 0; i < headDim; ++i) qHead[i] = qHead[i] * scale * qNormW[i];
        }
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k.data() + h * headDim;
            float sumSq = 0.0f;
            for (int i = 0; i < headDim; ++i) sumSq += kHead[i] * kHead[i];
            float scale = 1.0f / std::sqrt(sumSq / headDim + eps);
            for (int i = 0; i < headDim; ++i) kHead[i] = kHead[i] * scale * kNormW[i];
        }
        
        // RoPE
        cpuApplyRoPE(q.data(), nHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());
        cpuApplyRoPE(k.data(), nKVHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());
        
        // 更新 KV Cache
        float* kCacheLayer = kCacheCPU_[l].data();
        float* vCacheLayer = vCacheCPU_[l].data();
        std::copy(k.begin(), k.end(), kCacheLayer + position * kvSize);
        std::copy(v.begin(), v.end(), vCacheLayer + position * kvSize);
        
        // Attention
        std::fill(attnOut.begin(), attnOut.end(), 0.0f);
        const float attnScale = 1.0f / std::sqrt((float)headDim);
        
        for (int h = 0; h < nHeads; ++h) {
            const int kvHead = h / gqa;
            const float* qHead = q.data() + h * headDim;
            float* outHead = attnOut.data() + h * headDim;
            
            // 计算 attention scores
            std::vector<float> scores(totalLen);
            float maxScore = -1e30f;
            
            for (int t = 0; t < totalLen; ++t) {
                const float* kT = kCacheLayer + t * kvSize + kvHead * headDim;
                float score = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    score += qHead[d] * kT[d];
                }
                score *= attnScale;
                scores[t] = score;
                maxScore = std::max(maxScore, score);
            }
            
            // Softmax
            float sumExp = 0.0f;
            for (int t = 0; t < totalLen; ++t) {
                scores[t] = std::exp(scores[t] - maxScore);
                sumExp += scores[t];
            }
            for (int t = 0; t < totalLen; ++t) {
                scores[t] /= sumExp;
            }
            
            // Weighted sum of V
            for (int t = 0; t < totalLen; ++t) {
                const float* vT = vCacheLayer + t * kvSize + kvHead * headDim;
                const float w = scores[t];
                for (int d = 0; d < headDim; ++d) {
                    outHead[d] += w * vT[d];
                }
            }
        }
        
        // O Projection
        cpuMatmul(oProjW, attnOut.data(), oOut.data(), hiddenSize, qSize);
        
        // Residual
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + oOut[i];
        }
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        
        // Post Attention LayerNorm
        cpuRmsNorm(hidden.data(), postNormW, normOut.data(), hiddenSize, eps);
        
        // FFN: SwiGLU
        cpuMatmul(gateProjW, normOut.data(), gate.data(), intermediateSize, hiddenSize);
        cpuMatmul(upProjW, normOut.data(), up.data(), intermediateSize, hiddenSize);
        
        for (int i = 0; i < intermediateSize; ++i) {
            gate[i] = cpuSilu(gate[i]) * up[i];
        }
        
        cpuMatmul(downProjW, gate.data(), ffnOut.data(), hiddenSize, intermediateSize);
        
        // Residual
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnOut[i];
        }
    }
    
    // 3. Final LayerNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    cpuRmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);
    
    // 4. LM Head
    const float* lmHeadW = weightsCached_["lm_head"].data();
    std::vector<float> logits(vocabSize);
    cpuMatmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    
    kvCacheLen_ = totalLen;
    
    return logits;
}

void GGMLGPUBackend::cacheWeightsToCPU() {
    CLLM_INFO("[GGMLGPUBackend] Caching weights to CPU for fast access...");
    
    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const int intermediateSize = config_.intermediateSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    const int numLayers = config_.numHiddenLayers;
    
    // Embedding
    weightsCached_["embed_tokens"].resize((size_t)vocabSize * hiddenSize);
    ggml_backend_tensor_get(embedTokens_, weightsCached_["embed_tokens"].data(), 
                            0, (size_t)vocabSize * hiddenSize * sizeof(float));
    
    // Final norm
    weightsCached_["final_norm"].resize(hiddenSize);
    ggml_backend_tensor_get(finalNorm_, weightsCached_["final_norm"].data(), 
                            0, hiddenSize * sizeof(float));
    
    // LM Head
    weightsCached_["lm_head"].resize((size_t)vocabSize * hiddenSize);
    ggml_backend_tensor_get(lmHead_, weightsCached_["lm_head"].data(), 
                            0, (size_t)vocabSize * hiddenSize * sizeof(float));
    
    // Each layer
    for (int l = 0; l < numLayers; ++l) {
        LayerTensors& layer = layers_[l];
        std::string prefix = "layer." + std::to_string(l) + ".";
        
        weightsCached_[prefix + "input_layernorm"].resize(hiddenSize);
        ggml_backend_tensor_get(layer.inputLayernorm, 
                                weightsCached_[prefix + "input_layernorm"].data(), 
                                0, hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "q_proj"].resize(qSize * hiddenSize);
        ggml_backend_tensor_get(layer.qProj, 
                                weightsCached_[prefix + "q_proj"].data(), 
                                0, qSize * hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "k_proj"].resize(kvSize * hiddenSize);
        ggml_backend_tensor_get(layer.kProj, 
                                weightsCached_[prefix + "k_proj"].data(), 
                                0, kvSize * hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "v_proj"].resize(kvSize * hiddenSize);
        ggml_backend_tensor_get(layer.vProj, 
                                weightsCached_[prefix + "v_proj"].data(), 
                                0, kvSize * hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "o_proj"].resize(hiddenSize * qSize);
        ggml_backend_tensor_get(layer.oProj, 
                                weightsCached_[prefix + "o_proj"].data(), 
                                0, hiddenSize * qSize * sizeof(float));
        
        weightsCached_[prefix + "q_norm"].resize(headDim);
        ggml_backend_tensor_get(layer.qNorm, 
                                weightsCached_[prefix + "q_norm"].data(), 
                                0, headDim * sizeof(float));
        
        weightsCached_[prefix + "k_norm"].resize(headDim);
        ggml_backend_tensor_get(layer.kNorm, 
                                weightsCached_[prefix + "k_norm"].data(), 
                                0, headDim * sizeof(float));
        
        weightsCached_[prefix + "post_attention_layernorm"].resize(hiddenSize);
        ggml_backend_tensor_get(layer.postAttentionLayernorm, 
                                weightsCached_[prefix + "post_attention_layernorm"].data(), 
                                0, hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "gate_proj"].resize(intermediateSize * hiddenSize);
        ggml_backend_tensor_get(layer.gateProj, 
                                weightsCached_[prefix + "gate_proj"].data(), 
                                0, intermediateSize * hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "up_proj"].resize(intermediateSize * hiddenSize);
        ggml_backend_tensor_get(layer.upProj, 
                                weightsCached_[prefix + "up_proj"].data(), 
                                0, intermediateSize * hiddenSize * sizeof(float));
        
        weightsCached_[prefix + "down_proj"].resize(hiddenSize * intermediateSize);
        ggml_backend_tensor_get(layer.downProj, 
                                weightsCached_[prefix + "down_proj"].data(), 
                                0, hiddenSize * intermediateSize * sizeof(float));
    }
    
    CLLM_INFO("[GGMLGPUBackend] ✅ Weights cached to CPU");
}

void GGMLGPUBackend::resetKVCache() {
    kvCacheLen_ = 0;
    for (auto& cache : kCacheCPU_) {
        std::fill(cache.begin(), cache.end(), 0.0f);
    }
    for (auto& cache : vCacheCPU_) {
        std::fill(cache.begin(), cache.end(), 0.0f);
    }
}

const char* GGMLGPUBackend::getName() const {
#ifdef GGML_USE_METAL
    return "Metal (Apple GPU)";
#else
    return "CPU";
#endif
}

} // namespace kylin
} // namespace cllm
