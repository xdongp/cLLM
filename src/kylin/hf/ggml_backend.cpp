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

#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {

static constexpr int MAX_SEQ_LEN = 2048;

struct SchedStats {
    ggml_backend_sched_t sched = nullptr;
    ggml_backend_t gpu = nullptr;
    ggml_backend_t cpu = nullptr;
    int total = 0;
    int gpuNodes = 0;
    int cpuNodes = 0;
    int otherNodes = 0;
};

static bool sched_eval_cb(struct ggml_tensor* t, bool ask, void* user_data) {
    if (ask) {
        return true;
    }
    auto* stats = static_cast<SchedStats*>(user_data);
    if (!stats || !stats->sched || !t) {
        return true;
    }
    stats->total++;
    ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(stats->sched, t);
    if (backend == stats->gpu) {
        stats->gpuNodes++;
    } else if (backend == stats->cpu) {
        stats->cpuNodes++;
    } else {
        stats->otherNodes++;
    }
    return true;
}

GGMLGPUBackend::GGMLGPUBackend() = default;

GGMLGPUBackend::~GGMLGPUBackend() {
    if (graphSched_) {
        ggml_backend_sched_free(graphSched_);
        graphSched_ = nullptr;
    }
    if (graphBuffer_) {
        ggml_backend_buffer_free(graphBuffer_);
        graphBuffer_ = nullptr;
    }
    if (graphCtx_) {
        ggml_free(graphCtx_);
        graphCtx_ = nullptr;
    }
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
    if (backendCPU_) {
        ggml_backend_free(backendCPU_);
        backendCPU_ = nullptr;
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
    const char* graphEnv = std::getenv("CLLM_GGML_GRAPH_STAGE");
    graphStage_ = graphEnv ? std::atoi(graphEnv) : 0;
    
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

    backendCPU_ = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
    if (!backendCPU_) {
        CLLM_WARN("[GGMLGPUBackend] Failed to initialize CPU backend, mixed sched disabled");
    } else {
        CLLM_INFO("[GGMLGPUBackend] ✅ CPU backend initialized (for mixed sched)");
    }

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
    if (graphStage_ > 0) {
        CLLM_WARN("[GGMLGPUBackend] GGML graph stage=%d enabled", graphStage_);
    }
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

    if (graphStage_ > 0) {
        return forwardGraphMinimal(tokenId, position);
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

std::vector<float> GGMLGPUBackend::forwardGraphMinimal(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return {};
    }

    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const bool usePersistent = graphStage_ >= 7;

    auto build_graph = [&](ggml_context* ctx, bool record_handles,
                           ggml_tensor** out_token, ggml_tensor** out_pos,
                           ggml_cgraph** out_graph,
                           std::vector<ggml_tensor*>* out_k_cache_layers,
                           std::vector<ggml_tensor*>* out_v_cache_layers,
                           std::vector<ggml_tensor*>* out_k_cache_upd_layers,
                           std::vector<ggml_tensor*>* out_v_cache_upd_layers) -> ggml_tensor* {
        // token ids (I32)
        ggml_tensor* token = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        ggml_tensor* pos = nullptr;
        std::vector<ggml_tensor*> k_cache_layers;
        std::vector<ggml_tensor*> v_cache_layers;
        std::vector<ggml_tensor*> k_cache_upd_layers;
        std::vector<ggml_tensor*> v_cache_upd_layers;

        // Embedding lookup: [hiddenSize, 1]
        ggml_tensor* emb = ggml_get_rows(ctx, embedTokens_, token);
        ggml_tensor* x = emb;

        if (graphStage_ >= 2) {
            if (graphStage_ >= 3) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
            }

            auto build_layer = [&](int layerIdx, ggml_tensor* inp) -> ggml_tensor* {
                LayerTensors& layer = layers_[layerIdx];
                ggml_tensor* norm = ggml_rms_norm(ctx, inp, config_.rmsNormEps);
                ggml_tensor* norm_w = ggml_mul(ctx, norm, ggml_repeat(ctx, layer.inputLayernorm, norm));

                ggml_tensor* attn_out = nullptr;
                if (graphStage_ >= 3) {
                    const int headDim = config_.getHeadDim();
                    const int nHeads = config_.numAttentionHeads;
                    const int nKVHeads = config_.getNumKVHeads();
                    const float kq_scale = 1.0f / std::sqrt((float) headDim);
                    const int totalLen = position + 1;

                    ggml_tensor* q = ggml_mul_mat(ctx, layer.qProj, norm_w);
                    ggml_tensor* k = ggml_mul_mat(ctx, layer.kProj, norm_w);
                    ggml_tensor* v = ggml_mul_mat(ctx, layer.vProj, norm_w);

                    q = ggml_reshape_3d(ctx, q, headDim, nHeads, 1);
                    k = ggml_reshape_3d(ctx, k, headDim, nKVHeads, 1);
                    v = ggml_reshape_3d(ctx, v, headDim, nKVHeads, 1);

                    ggml_tensor* qn = ggml_rms_norm(ctx, q, config_.rmsNormEps);
                    q = ggml_mul(ctx, qn, ggml_repeat(ctx, layer.qNorm, qn));
                    ggml_tensor* kn = ggml_rms_norm(ctx, k, config_.rmsNormEps);
                    k = ggml_mul(ctx, kn, ggml_repeat(ctx, layer.kNorm, kn));

                    const int n_ctx_orig = std::max(1, config_.maxPositionEmbeddings);
                    const int rope_mode = 0;
                    q = ggml_rope_ext(ctx, q, pos, nullptr, headDim, rope_mode, n_ctx_orig,
                                      config_.ropeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    k = ggml_rope_ext(ctx, k, pos, nullptr, headDim, rope_mode, n_ctx_orig,
                                      config_.ropeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                    ggml_tensor* k_use = k;
                    ggml_tensor* v_use = v;

                    if (graphStage_ >= 4) {
                        const int kvSize = nKVHeads * headDim;
                        if ((int) kCacheGraphCPU_.size() != config_.numHiddenLayers) {
                            kCacheGraphCPU_.assign(config_.numHiddenLayers, {});
                            vCacheGraphCPU_.assign(config_.numHiddenLayers, {});
                        }
                        if (kCacheGraphCPU_[layerIdx].empty()) {
                            kCacheGraphCPU_[layerIdx].resize((size_t) MAX_SEQ_LEN * kvSize, 0.0f);
                            vCacheGraphCPU_[layerIdx].resize((size_t) MAX_SEQ_LEN * kvSize, 0.0f);
                        }

                        ggml_tensor* k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, nKVHeads, totalLen);
                        ggml_tensor* v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, nKVHeads, totalLen);
                        k_cache_layers.push_back(k_cache);
                        v_cache_layers.push_back(v_cache);

                        ggml_tensor* k_cache_upd = ggml_set(ctx, k_cache, k,
                                                            k_cache->nb[1], k_cache->nb[2], k_cache->nb[3],
                                                            (size_t) position * k_cache->nb[2]);
                        ggml_tensor* v_cache_upd = ggml_set(ctx, v_cache, v,
                                                            v_cache->nb[1], v_cache->nb[2], v_cache->nb[3],
                                                            (size_t) position * v_cache->nb[2]);

                        k_use = k_cache_upd;
                        v_use = v_cache_upd;
                        k_cache_upd_layers.push_back(k_cache_upd);
                        v_cache_upd_layers.push_back(v_cache_upd);
                    }

                    ggml_tensor* q4 = ggml_permute(ctx, ggml_reshape_4d(ctx, q, headDim, 1, nHeads, 1), 0, 2, 1, 3);
                    ggml_tensor* k4 = ggml_permute(ctx, ggml_reshape_4d(ctx, k_use, headDim, 1, nKVHeads, totalLen), 0, 2, 1, 3);
                    ggml_tensor* v4 = ggml_permute(ctx, ggml_reshape_4d(ctx, v_use, headDim, 1, nKVHeads, totalLen), 0, 2, 1, 3);

                    ggml_tensor* attn = ggml_flash_attn_ext(ctx, q4, k4, v4, nullptr, kq_scale, 0.0f, 0.0f);
                    ggml_flash_attn_ext_set_prec(attn, GGML_PREC_F32);

                    ggml_tensor* attn2d = ggml_reshape_2d(ctx, attn, headDim, nHeads);
                    attn_out = ggml_reshape_2d(ctx, attn2d, headDim * nHeads, 1);
                    attn_out = ggml_mul_mat(ctx, layer.oProj, attn_out);
                }

                if (attn_out) {
                    inp = ggml_add(ctx, inp, attn_out);
                }

                ggml_tensor* post = ggml_rms_norm(ctx, inp, config_.rmsNormEps);
                ggml_tensor* post_w = ggml_mul(ctx, post, ggml_repeat(ctx, layer.postAttentionLayernorm, post));

                ggml_tensor* gate = ggml_mul_mat(ctx, layer.gateProj, post_w);
                ggml_tensor* up = ggml_mul_mat(ctx, layer.upProj, post_w);
                ggml_tensor* gate_act = ggml_silu(ctx, gate);
                ggml_tensor* ffn = ggml_mul(ctx, gate_act, up);
                ggml_tensor* down = ggml_mul_mat(ctx, layer.downProj, ffn);

                return ggml_add(ctx, inp, down);
            };

            if (graphStage_ >= 5) {
                for (int l = 0; l < config_.numHiddenLayers; ++l) {
                    x = build_layer(l, x);
                }
            } else {
                x = build_layer(0, x);
            }

            ggml_tensor* fn = ggml_rms_norm(ctx, x, config_.rmsNormEps);
            x = ggml_mul(ctx, fn, ggml_repeat(ctx, finalNorm_, fn));
        }

        ggml_tensor* logits = ggml_mul_mat(ctx, lmHead_, x);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, logits);

        if (record_handles) {
            graphToken_ = token;
            graphPos_ = pos;
            graphLogits_ = logits;
            graph_ = graph;
            graphKCacheLayers_ = std::move(k_cache_layers);
            graphVCacheLayers_ = std::move(v_cache_layers);
            graphKCacheUpdLayers_ = std::move(k_cache_upd_layers);
            graphVCacheUpdLayers_ = std::move(v_cache_upd_layers);
        }

        if (out_k_cache_layers) *out_k_cache_layers = std::move(k_cache_layers);
        if (out_v_cache_layers) *out_v_cache_layers = std::move(v_cache_layers);
        if (out_k_cache_upd_layers) *out_k_cache_upd_layers = std::move(k_cache_upd_layers);
        if (out_v_cache_upd_layers) *out_v_cache_upd_layers = std::move(v_cache_upd_layers);
        if (out_token) *out_token = token;
        if (out_pos) *out_pos = pos;
        if (out_graph) *out_graph = graph;
        return logits;
    };

    ggml_context* ctx = nullptr;
    ggml_tensor* logits = nullptr;
    ggml_tensor* token = nullptr;
    ggml_tensor* pos = nullptr;
    ggml_cgraph* graph = nullptr;
    std::vector<ggml_tensor*> k_cache_layers;
    std::vector<ggml_tensor*> v_cache_layers;
    std::vector<ggml_tensor*> k_cache_upd_layers;
    std::vector<ggml_tensor*> v_cache_upd_layers;

    if (usePersistent) {
        const bool needRebuild = graphCtx_ == nullptr ||
                                 graphBuiltPosition_ != position ||
                                 graphBuiltStage_ != graphStage_;
        if (needRebuild) {
            if (graphSched_) { ggml_backend_sched_free(graphSched_); graphSched_ = nullptr; }
            if (graphBuffer_) { ggml_backend_buffer_free(graphBuffer_); graphBuffer_ = nullptr; }
            if (graphCtx_) { ggml_free(graphCtx_); graphCtx_ = nullptr; }
            graph_ = nullptr;
            graphToken_ = nullptr;
            graphPos_ = nullptr;
            graphLogits_ = nullptr;
            graphKCacheLayers_.clear();
            graphVCacheLayers_.clear();
            graphKCacheUpdLayers_.clear();
            graphVCacheUpdLayers_.clear();

            const size_t ctxSize = ggml_tensor_overhead() * 32 + ggml_graph_overhead() + 32 * 1024 * 1024;
            struct ggml_init_params params = {
                .mem_size   = ctxSize,
                .mem_buffer = nullptr,
                .no_alloc   = true,
            };
            graphCtx_ = ggml_init(params);
            if (!graphCtx_) {
                CLLM_ERROR("[GGMLGPUBackend] Failed to create persistent graph context");
                return {};
            }

            logits = build_graph(graphCtx_, true, &token, &pos, &graph,
                                 nullptr, nullptr, nullptr, nullptr);

            if (graphStage_ >= 6) {
                ggml_backend_t backends[2] = { backend_, nullptr };
                int n_backends = 1;
                if (graphStage_ >= 8 && backendCPU_) {
                    backends[1] = backendCPU_;
                    n_backends = 2;
                }
                graphSched_ = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);
                if (!graphSched_) {
                    CLLM_ERROR("[GGMLGPUBackend] Failed to create backend scheduler");
                    ggml_free(graphCtx_);
                    graphCtx_ = nullptr;
                    return {};
                }
                if (!ggml_backend_sched_alloc_graph(graphSched_, graph_)) {
                    CLLM_ERROR("[GGMLGPUBackend] Failed to alloc graph via scheduler");
                    ggml_backend_sched_free(graphSched_);
                    graphSched_ = nullptr;
                    ggml_free(graphCtx_);
                    graphCtx_ = nullptr;
                    return {};
                }
            } else {
                graphBuffer_ = ggml_backend_alloc_ctx_tensors(graphCtx_, backend_);
                if (!graphBuffer_) {
                    CLLM_ERROR("[GGMLGPUBackend] Failed to allocate compute buffer");
                    ggml_free(graphCtx_);
                    graphCtx_ = nullptr;
                    return {};
                }
            }

            graphBuiltPosition_ = position;
            graphBuiltStage_ = graphStage_;
        }
        ctx = graphCtx_;
        logits = graphLogits_;
    } else {
        const size_t ctxSize = ggml_tensor_overhead() * 16 + ggml_graph_overhead() + 16 * 1024 * 1024;
        struct ggml_init_params params = {
            .mem_size   = ctxSize,
            .mem_buffer = nullptr,
            .no_alloc   = true,
        };
        ctx = ggml_init(params);
        if (!ctx) {
            CLLM_ERROR("[GGMLGPUBackend] Failed to create compute context");
            return {};
        }
        logits = build_graph(ctx, false, &token, &pos, &graph,
                             &k_cache_layers, &v_cache_layers,
                             &k_cache_upd_layers, &v_cache_upd_layers);
    }

    ggml_backend_buffer_t buffer = nullptr;
    ggml_backend_sched_t sched = nullptr;
    if (!usePersistent) {
        if (graphStage_ >= 6) {
            ggml_backend_t backends[2] = { backend_, nullptr };
            int n_backends = 1;
            if (graphStage_ >= 8 && backendCPU_) {
                backends[1] = backendCPU_;
                n_backends = 2;
            }
            sched = ggml_backend_sched_new(backends, nullptr, n_backends, GGML_DEFAULT_GRAPH_SIZE, false, true);
            if (!sched) {
                CLLM_ERROR("[GGMLGPUBackend] Failed to create backend scheduler");
                ggml_free(ctx);
                return {};
            }
            if (!ggml_backend_sched_alloc_graph(sched, graph)) {
                CLLM_ERROR("[GGMLGPUBackend] Failed to alloc graph via scheduler");
                ggml_backend_sched_free(sched);
                ggml_free(ctx);
                return {};
            }
        } else {
            buffer = ggml_backend_alloc_ctx_tensors(ctx, backend_);
            if (!buffer) {
                CLLM_ERROR("[GGMLGPUBackend] Failed to allocate compute buffer");
                ggml_free(ctx);
                return {};
            }
        }
    }

    ggml_tensor* tokenTensor = usePersistent ? graphToken_ : token;
    if (!tokenTensor) {
        CLLM_ERROR("[GGMLGPUBackend] Token tensor missing");
        if (!usePersistent) {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (sched) ggml_backend_sched_free(sched);
            ggml_free(ctx);
        }
        return {};
    }
    const int32_t tokenVal = static_cast<int32_t>(tokenId);
    ggml_backend_tensor_set(tokenTensor, &tokenVal, 0, sizeof(int32_t));
    if (usePersistent ? graphPos_ : pos) {
        const int32_t posVal = static_cast<int32_t>(position);
        ggml_backend_tensor_set(usePersistent ? graphPos_ : pos, &posVal, 0, sizeof(int32_t));
    }

    const auto& kLayers = usePersistent ? graphKCacheLayers_ : k_cache_layers;
    const auto& vLayers = usePersistent ? graphVCacheLayers_ : v_cache_layers;
    if (!kLayers.empty()) {
        const int headDim = config_.getHeadDim();
        const int nKVHeads = config_.getNumKVHeads();
        const int kvSize = nKVHeads * headDim;
        const int totalLen = position + 1;
        for (size_t i = 0; i < kLayers.size(); ++i) {
            ggml_backend_tensor_set(kLayers[i], kCacheGraphCPU_[i].data(), 0,
                                    (size_t) totalLen * kvSize * sizeof(float));
            ggml_backend_tensor_set(vLayers[i], vCacheGraphCPU_[i].data(), 0,
                                    (size_t) totalLen * kvSize * sizeof(float));
        }
    }

    SchedStats stats;
    if (usePersistent && graphSched_) {
        stats.sched = graphSched_;
        stats.gpu = backend_;
        stats.cpu = backendCPU_;
        ggml_backend_sched_set_eval_callback(graphSched_, sched_eval_cb, &stats);
    } else if (!usePersistent && sched) {
        stats.sched = sched;
        stats.gpu = backend_;
        stats.cpu = backendCPU_;
        ggml_backend_sched_set_eval_callback(sched, sched_eval_cb, &stats);
    }

    const ggml_status status = usePersistent
        ? (graphSched_ ? ggml_backend_sched_graph_compute(graphSched_, graph_) :
                         ggml_backend_graph_compute(backend_, graph_))
        : (sched ? ggml_backend_sched_graph_compute(sched, graph) :
                   ggml_backend_graph_compute(backend_, graph));

    if (status != GGML_STATUS_SUCCESS) {
        CLLM_ERROR("[GGMLGPUBackend] ggml_backend_graph_compute failed: %d", status);
        if (!usePersistent) {
            if (buffer) ggml_backend_buffer_free(buffer);
            if (sched) ggml_backend_sched_free(sched);
            ggml_free(ctx);
        }
        return {};
    }

    std::vector<float> out(vocabSize);
    ggml_backend_tensor_get(logits, out.data(), 0, (size_t) vocabSize * sizeof(float));

    if (stats.sched) {
        const int splits = ggml_backend_sched_get_n_splits(stats.sched);
        CLLM_INFO("[GGMLGPUBackend] sched splits=%d, nodes=%d (gpu=%d cpu=%d other=%d)",
                  splits, stats.total, stats.gpuNodes, stats.cpuNodes, stats.otherNodes);
        ggml_backend_sched_set_eval_callback(stats.sched, nullptr, nullptr);
    }

    const auto& kUpdLayers = usePersistent ? graphKCacheUpdLayers_ : k_cache_upd_layers;
    const auto& vUpdLayers = usePersistent ? graphVCacheUpdLayers_ : v_cache_upd_layers;
    if (!kUpdLayers.empty()) {
        const int headDim = config_.getHeadDim();
        const int nKVHeads = config_.getNumKVHeads();
        const int kvSize = nKVHeads * headDim;
        const int totalLen = position + 1;
        for (size_t i = 0; i < kUpdLayers.size(); ++i) {
            ggml_backend_tensor_get(kUpdLayers[i], kCacheGraphCPU_[i].data(), 0,
                                    (size_t) totalLen * kvSize * sizeof(float));
            ggml_backend_tensor_get(vUpdLayers[i], vCacheGraphCPU_[i].data(), 0,
                                    (size_t) totalLen * kvSize * sizeof(float));
        }
    }

    if (!usePersistent) {
        if (buffer) ggml_backend_buffer_free(buffer);
        if (sched) ggml_backend_sched_free(sched);
        ggml_free(ctx);
    }
    return out;
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
    for (auto& layer : kCacheGraphCPU_) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }
    for (auto& layer : vCacheGraphCPU_) {
        std::fill(layer.begin(), layer.end(), 0.0f);
    }
    graphBuiltPosition_ = -1;
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
