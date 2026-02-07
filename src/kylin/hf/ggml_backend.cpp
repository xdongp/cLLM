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

GGMLGPUBackend::GGMLGPUBackend() {
    graphNormWeightedAllLayers_.resize(28, nullptr);  // 初始化28层的存储
}

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
    // 默认启用所有层 (stage 5) 以确保完整的模型计算
    graphStage_ = graphEnv ? std::atoi(graphEnv) : 5;
    
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
    
    // 输出 GGML 版本信息
    CLLM_INFO("[GGMLGPUBackend] GGML Version: %s", ggml_version());

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
    
    // 嵌入层 [hiddenSize, vocabSize]（GGML 列主序，用于矩阵乘法）
    embedTokens_ = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, vocabSize);
    ggml_set_name(embedTokens_, "embed_tokens");

    // 嵌入层 [hiddenSize, vocabSize]（用于 ggml_get_rows lookup）
    // 注意：ggml_get_rows 返回 [src->ne[0], num_indices]，所以需要 hiddenSize 作为第一维
    embedTokensLookup_ = ggml_new_tensor_2d(weightCtx_, GGML_TYPE_F32, hiddenSize, vocabSize);
    ggml_set_name(embedTokensLookup_, "embed_tokens_lookup");
    
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
        
        // 投影矩阵 for GGML
        // ggml_mul_mat(a, b): a=[n, k], b=[m, k], result=[m, n]
        // 对于 Q 投影: a=[qSize, hiddenSize], b=[1, hiddenSize], result=[1, qSize]
        // GGML 张量是列主序，所以 ggml_new_tensor_2d(ne0, ne1) 创建 [ne0, ne1] 的张量
        // 为了匹配行主序的 [qSize, hiddenSize] 权重，我们需要创建 [hiddenSize, qSize] 的列主序张量
        // 但这样会导致维度不匹配，所以我们需要重新考虑...
        
        // 实际上，PyTorch 权重是 [outFeatures, inFeatures] 行主序
        // GGML 的 ggml_mul_mat 期望 a=[n, k] 即 [outFeatures, inFeatures]
        // 在列主序中，这对应于 [inFeatures, outFeatures] 的存储
        // 所以 ggml_new_tensor_2d(hiddenSize, qSize) 是正确的
        
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

// 辅助函数：转置矩阵（行主序 -> 列主序）
// 输入: [rows, cols] 行主序 (PyTorch格式)
// 输出: [cols, rows] 列主序（GGML格式）
// GGML 使用列主序存储，对于 [ne0, ne1] 的张量，元素 (i, j) 存储在 data[i + j * ne0]
static std::vector<float> transposeMatrix(const float* src, int rows, int cols) {
    std::vector<float> dst(rows * cols);
    // src[i][j] (行主序) = src[i * cols + j]
    // dst 是 [cols, rows] 的列主序张量
    // dst[i][j] (列主序) = dst[i + j * cols]
    // 我们需要: dst[i][j] = src[j][i]
    for (int i = 0; i < cols; ++i) {
        for (int j = 0; j < rows; ++j) {
            // dst[i][j] = src[j][i]
            dst[i + j * cols] = src[j * cols + i];
        }
    }
    return dst;
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
    
    CLLM_INFO("[GGMLGPUBackend] Uploading weights to GPU (with transpose for GGML column-major)...");
    
    // 嵌入层 [hiddenSize, vocabSize]（用于矩阵乘法）
    // 原始数据: [vocabSize, hiddenSize] 行主序
    // GGML 列主序: [hiddenSize, vocabSize]
    auto embedTransposed = transposeMatrix(embedTokens, vocabSize, hiddenSize);
    ggml_backend_tensor_set(embedTokens_, embedTransposed.data(), 0, 
                            (size_t)vocabSize * hiddenSize * sizeof(float));
    
    // 嵌入层 [hiddenSize, vocabSize]（用于 ggml_get_rows lookup）
    // ggml_get_rows 的行为: 给定 indices [n], 返回 a[indices[i], :] 即第 indices[i] 行
    // 在 GGML 列主序 [ne0, ne1] 中，元素 (i, j) 存储在 data[i + j * ne0]
    // 第 j 行的数据是 data[i + j * ne0] (i 从 0 到 ne0-1)
    // 所以我们需要: embedTokensLookup_[i + j * hiddenSize] = embedTokens[j * hiddenSize + i]
    // 即直接复制原始嵌入（行主序），因为 embedTokens[j * hiddenSize + i] 正好对应列主序的 (i, j)
    // 等等，让我重新分析...
    // 
    // 原始嵌入: [vocabSize, hiddenSize] 行主序
    //   embedTokens[token_id * hiddenSize + dim] = token_id 的第 dim 维
    //
    // GGML 列主序 [hiddenSize, vocabSize]:
    //   data[dim + token_id * hiddenSize] = (dim, token_id) 位置的值
    //
    // 所以我们需要: data[dim + token_id * hiddenSize] = embedTokens[token_id * hiddenSize + dim]
    // 即直接复制原始数据！
    ggml_backend_tensor_set(embedTokensLookup_, embedTokens, 0, 
                            (size_t)vocabSize * hiddenSize * sizeof(float));
    
    ggml_backend_tensor_set(finalNorm_, finalNorm, 0, hiddenSize * sizeof(float));
    
    if (!config_.tieWordEmbeddings && lmHead) {
        auto lmHeadTransposed = transposeMatrix(lmHead, vocabSize, hiddenSize);
        ggml_backend_tensor_set(lmHead_, lmHeadTransposed.data(), 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    } else {
        // 共享权重，使用已转置的嵌入层权重
        ggml_backend_tensor_set(lmHead_, embedTransposed.data(), 0,
                                (size_t)vocabSize * hiddenSize * sizeof(float));
    }
    
    // 每层权重
    for (size_t i = 0; i < layerWeights.size() && i < layers_.size(); ++i) {
        const LayerWeightsGPU& src = layerWeights[i];
        LayerTensors& dst = layers_[i];
        
        // 1D 权重 - 不需要转置
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
        
        // 2D 权重 - 需要转置 (行主序 -> 列主序)
        // PyTorch: [outFeatures, inFeatures] 行主序
        // GGML: [inFeatures, outFeatures] 列主序 (即转置后的行主序)
        
        // qProj: [qSize, hiddenSize] -> [hiddenSize, qSize]
        auto qProjTransposed = transposeMatrix(src.qProj, qSize, hiddenSize);
        ggml_backend_tensor_set(dst.qProj, qProjTransposed.data(), 0, (size_t)qSize * hiddenSize * sizeof(float));
        
        // kProj: [kvSize, hiddenSize] -> [hiddenSize, kvSize]
        auto kProjTransposed = transposeMatrix(src.kProj, kvSize, hiddenSize);
        ggml_backend_tensor_set(dst.kProj, kProjTransposed.data(), 0, (size_t)kvSize * hiddenSize * sizeof(float));
        
        // vProj: [kvSize, hiddenSize] -> [hiddenSize, kvSize]
        auto vProjTransposed = transposeMatrix(src.vProj, kvSize, hiddenSize);
        ggml_backend_tensor_set(dst.vProj, vProjTransposed.data(), 0, (size_t)kvSize * hiddenSize * sizeof(float));
        
        // oProj: [hiddenSize, qSize] -> [qSize, hiddenSize]
        auto oProjTransposed = transposeMatrix(src.oProj, hiddenSize, qSize);
        ggml_backend_tensor_set(dst.oProj, oProjTransposed.data(), 0, (size_t)hiddenSize * qSize * sizeof(float));
        
        // gateProj: [intermediateSize, hiddenSize] -> [hiddenSize, intermediateSize]
        auto gateProjTransposed = transposeMatrix(src.gateProj, intermediateSize, hiddenSize);
        ggml_backend_tensor_set(dst.gateProj, gateProjTransposed.data(), 0, (size_t)intermediateSize * hiddenSize * sizeof(float));
        
        // upProj: [intermediateSize, hiddenSize] -> [hiddenSize, intermediateSize]
        auto upProjTransposed = transposeMatrix(src.upProj, intermediateSize, hiddenSize);
        ggml_backend_tensor_set(dst.upProj, upProjTransposed.data(), 0, (size_t)intermediateSize * hiddenSize * sizeof(float));
        
        // downProj: [hiddenSize, intermediateSize] -> [intermediateSize, hiddenSize]
        auto downProjTransposed = transposeMatrix(src.downProj, hiddenSize, intermediateSize);
        ggml_backend_tensor_set(dst.downProj, downProjTransposed.data(), 0, (size_t)hiddenSize * intermediateSize * sizeof(float));
    }
    
    CLLM_INFO("[GGMLGPUBackend] ✅ Weights uploaded (transposed for GGML)");
    return true;
}

// 使用 BLAS 优化的函数
static inline void cpuRmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

// 矩阵乘法：output = input * weight
// input: [K]
// weight: [K, M]  (因为已经被转置为 GGML 格式)
// output: [M]
// CPU 矩阵乘法 - 使用 ggml_kernels 的优化实现
// weight: [M, K] 行主序 (与 PyTorch 一致)
// input: [K]
// output: [M]
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
    // 如果启用了计算图，使用 GPU 路径
    if (graphStage_ > 0) {
        return forwardGraphMinimal(tokenId, position);
    }
    // 否则使用 CPU 路径
    return forwardCPU(tokenId, position);
}

// 保留原来的 CPU 实现作为参考（已移至 cpu_backend.cpp）
// 以下代码将在后续清理中移除
#if 0
std::vector<float> GGMLGPUBackend::forwardCPU_Old(int tokenId, int position) {
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
    
    // 注意：由于权重已经被转置为 GGML 格式 [hiddenSize, vocabSize]，
    // 所以需要使用正确的索引方式
    for (int i = 0; i < hiddenSize; ++i) {
        hidden[i] = embedData[i * vocabSize + tokenId];
    }
    if (position == 0 || position == 1) {
        float minVal = hidden[0], maxVal = hidden[0], sum = 0;
        for (int i = 0; i < hiddenSize; ++i) {
            if (hidden[i] < minVal) minVal = hidden[i];
            if (hidden[i] > maxVal) maxVal = hidden[i];
            sum += hidden[i];
        }

    }
    
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
        // 权重是 GGML 格式 [K, M] = [hiddenSize, qSize]
        // 输入 normOut: [K] = [hiddenSize]
        // 输出 q: [M] = [qSize]
        cpuMatmul(qProjW, normOut.data(), q.data(), qSize, hiddenSize);
        cpuMatmul(kProjW, normOut.data(), k.data(), kvSize, hiddenSize);
        cpuMatmul(vProjW, normOut.data(), v.data(), kvSize, hiddenSize);
        if ((l == 0 || l == numLayers - 1) && (position == 0 || position == 1)) {
            float qMin = q[0], qMax = q[0], qSum = 0;
            float kMin = k[0], kMax = k[0], kSum = 0;
            float vMin = v[0], vMax = v[0], vSum = 0;
            
            for (int i = 0; i < qSize; ++i) {
                if (q[i] < qMin) qMin = q[i];
                if (q[i] > qMax) qMax = q[i];
                qSum += q[i];
            }
            
            for (int i = 0; i < kvSize; ++i) {
                if (k[i] < kMin) kMin = k[i];
                if (k[i] > kMax) kMax = k[i];
                kSum += k[i];
                if (v[i] < vMin) vMin = v[i];
                if (v[i] > vMax) vMax = v[i];
                vSum += v[i];
            }
        }
        
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
                float score = ggml_kernels::dot_product(qHead, kT, headDim) * attnScale;
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
        // 权重是 GGML 格式 [K, M] = [qSize, hiddenSize]
        // 输入 attnOut: [K] = [qSize]
        // 输出 oOut: [M] = [hiddenSize]
        cpuMatmul(oProjW, attnOut.data(), oOut.data(), hiddenSize, qSize);
        if ((l == 0 || l == numLayers - 1) && position == 0) {
            float outMin = oOut[0], outMax = oOut[0], outSum = 0;
            for (int i = 0; i < hiddenSize; ++i) {
                if (oOut[i] < outMin) outMin = oOut[i];
                if (oOut[i] > outMax) outMax = oOut[i];
                outSum += oOut[i];
            }
        }

        // Residual
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + oOut[i];
        }
        std::copy(hidden.begin(), hidden.end(), residual.begin());

        // Post Attention LayerNorm
        cpuRmsNorm(hidden.data(), postNormW, normOut.data(), hiddenSize, eps);

        // FFN: SwiGLU
        // gateProj: [K, M] = [hiddenSize, intermediateSize]
        // 输入 normOut: [K] = [hiddenSize]
        // 输出 gate: [M] = [intermediateSize]
        cpuMatmul(gateProjW, normOut.data(), gate.data(), intermediateSize, hiddenSize);
        cpuMatmul(upProjW, normOut.data(), up.data(), intermediateSize, hiddenSize);

        for (int i = 0; i < intermediateSize; ++i) {
            gate[i] = cpuSilu(gate[i]) * up[i];
        }

        // downProj: [K, M] = [intermediateSize, hiddenSize]
        // 输入 gate: [K] = [intermediateSize]
        // 输出 ffnOut: [M] = [hiddenSize]
        cpuMatmul(downProjW, gate.data(), ffnOut.data(), hiddenSize, intermediateSize);
        if ((l == 0 || l == numLayers - 1) && (position == 0 || position == 1)) {
            float outMin = ffnOut[0], outMax = ffnOut[0], outSum = 0;
            for (int i = 0; i < hiddenSize; ++i) {
                if (ffnOut[i] < outMin) outMin = ffnOut[i];
                if (ffnOut[i] > outMax) outMax = ffnOut[i];
                outSum += ffnOut[i];
            }
        }
        
        // Residual
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnOut[i];
        }
    }
    
    // 3. Final LayerNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    cpuRmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);
    
    // 4. LM Head
    // lmHead: [K, M] = [hiddenSize, vocabSize]
    // 输入 normOut: [K] = [hiddenSize]
    // 输出 logits: [M] = [vocabSize]
    const float* lmHeadW = weightsCached_["lm_head"].data();
    std::vector<float> logits(vocabSize);
    cpuMatmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    
    kvCacheLen_ = totalLen;
    
    return logits;
}
#endif  // #if 0 - 旧CPU实现结束

std::vector<float> GGMLGPUBackend::forwardGraphMinimal(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return {};
    }

    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const bool usePersistent = graphStage_ >= 7;
    
    // 当 position = 0 时，重置 KV Cache（新序列开始）
    if (position == 0) {
        for (auto& kCache : kCacheGraphCPU_) {
            std::fill(kCache.begin(), kCache.end(), 0.0f);
        }
        for (auto& vCache : vCacheGraphCPU_) {
            std::fill(vCache.begin(), vCache.end(), 0.0f);
        }
    }

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
        std::vector<ggml_tensor*> k_cur_layers;  // Current K for each layer
        std::vector<ggml_tensor*> v_cur_layers;  // Current V for each layer

        // Embedding lookup: [hiddenSize, 1]
        // ggml_get_rows 返回 [src->ne[0], num_indices]，所以需要 src->ne[0]=hiddenSize
        // 使用 embedTokensLookup_ [hiddenSize, vocabSize]（注意维度顺序！）
        ggml_tensor* emb = ggml_get_rows(ctx, embedTokensLookup_, token);
        // ggml_get_rows 返回 4D [hiddenSize, 1, 1, 1]，已经是正确的了
        ggml_tensor* x = emb;
        
        // 保存 Embedding 用于调试
        graphEmbedding_ = emb;

        CLLM_INFO("[GPU] embedTokensLookup_->ne=[%d,%d]", (int)embedTokensLookup_->ne[0], (int)embedTokensLookup_->ne[1]);
        CLLM_INFO("[GPU] ggml_get_rows result: emb->ne=[%d,%d,%d,%d]", (int)emb->ne[0], (int)emb->ne[1], (int)emb->ne[2], (int)emb->ne[3]);

        if (graphStage_ >= 2) {
            if (graphStage_ >= 3) {
                pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
                ggml_set_input(pos);
            }

            auto build_layer = [&](int layerIdx, ggml_tensor* inp) -> ggml_tensor* {
                LayerTensors& layer = layers_[layerIdx];
                
                CLLM_INFO("[GPU Layer %d] input->ne=[%d,%d,%d,%d]", layerIdx,
                          (int)inp->ne[0], (int)inp->ne[1], (int)inp->ne[2], (int)inp->ne[3]);
                
                // RMSNorm: 需要处理单个 token 的情况
                // 保存 RMS Norm 前的输入用于调试 (所有 position 都保存)
                if (layerIdx == 0) {
                    graphNormInput_ = inp;
                } else if (layerIdx == 1) {
                    graphNormInputLayer1_ = inp;
                }

                ggml_tensor* norm = ggml_rms_norm(ctx, inp, config_.rmsNormEps);

                // 保存 RMS Norm 后的输出（乘以权重前）用于调试 (所有 position 都保存)
                if (layerIdx == 0) {
                    graphNormOutput_ = norm;
                } else if (layerIdx == 1) {
                    graphNormOutputLayer1_ = norm;
                }

                // ggml_repeat(a, b) 将 a 重复以匹配 b 的形状
                ggml_tensor* norm_w = ggml_mul(ctx, norm,
                    ggml_repeat(ctx, layer.inputLayernorm, norm));

                // 保存 RMS Norm 乘以权重后的输出用于调试 (所有 position 都保存)
                if (layerIdx == 0) {
                    graphNormWeighted_ = norm_w;
                } else if (layerIdx == 1) {
                    graphNormWeightedLayer1_ = norm_w;
                }

                CLLM_INFO("[GPU Layer %d] after RMSNorm: norm_w->ne=[%d,%d,%d,%d]", layerIdx,
                          (int)norm_w->ne[0], (int)norm_w->ne[1], (int)norm_w->ne[2], (int)norm_w->ne[3]);

                // 保存所有层的 RMS Norm 后输出用于逐层验证
                if (position == 0 && layerIdx < 28) {
                    graphNormWeightedAllLayers_[layerIdx] = norm_w;
                }

                ggml_tensor* attn_out = nullptr;
                if (graphStage_ >= 3) {
                    const int headDim = config_.getHeadDim();
                    const int nHeads = config_.numAttentionHeads;
                    const int nKVHeads = config_.getNumKVHeads();
                    const float kq_scale = 1.0f / std::sqrt((float) headDim);
                    const int totalLen = position + 1;
                    
                    // 保存 Attention 输入用于调试
                    if (layerIdx == 0) {
                        graphAttnInput_ = norm_w;
                    }

                    // Q/K/V Projection
                    ggml_tensor* q = ggml_mul_mat(ctx, layer.qProj, norm_w);
                    ggml_tensor* k = ggml_mul_mat(ctx, layer.kProj, norm_w);
                    ggml_tensor* v = ggml_mul_mat(ctx, layer.vProj, norm_w);

                    CLLM_INFO("[GPU Layer %d] QKV: q->ne=[%d,%d,%d,%d] k->ne=[%d,%d,%d,%d] v->ne=[%d,%d,%d,%d]", layerIdx,
                              (int)q->ne[0], (int)q->ne[1], (int)q->ne[2], (int)q->ne[3],
                              (int)k->ne[0], (int)k->ne[1], (int)k->ne[2], (int)k->ne[3],
                              (int)v->ne[0], (int)v->ne[1], (int)v->ne[2], (int)v->ne[3]);

                    // 保存Q/K投影后的张量用于调试
                    if (layerIdx == 0) {
                        graphQProj_ = q;
                        graphKProj_ = k;
                    } else if (layerIdx == 1) {
                        graphQProjLayer1_ = q;
                        graphKProjLayer1_ = k;
                    }

                    // Reshape to 3D: (headDim, nHeads, 1) or (headDim, nKVHeads, 1)
                    q = ggml_reshape_3d(ctx, q, headDim, nHeads, 1);
                    k = ggml_reshape_3d(ctx, k, headDim, nKVHeads, 1);
                    v = ggml_reshape_3d(ctx, v, headDim, nKVHeads, 1);

                    // Q/K RMS Norm (Qwen3 specific)
                    ggml_tensor* qn = ggml_rms_norm(ctx, q, config_.rmsNormEps);
                    ggml_tensor* q_norm_3d = ggml_reshape_3d(ctx, layer.qNorm, headDim, 1, 1);
                    q = ggml_mul(ctx, qn, q_norm_3d);
                    
                    ggml_tensor* kn = ggml_rms_norm(ctx, k, config_.rmsNormEps);
                    ggml_tensor* k_norm_3d = ggml_reshape_3d(ctx, layer.kNorm, headDim, 1, 1);
                    k = ggml_mul(ctx, kn, k_norm_3d);

                    // RoPE (Qwen3 uses NEOX style)
                    const int n_ctx_orig = std::max(1, config_.maxPositionEmbeddings);
                    const int rope_mode = 2;  // GGML_ROPE_TYPE_NEOX
                    if (layerIdx == 0 && position == 0) {

                    }
                    
                    // 保存RoPE前的Q/K用于调试
                    if (layerIdx == 0) {
                        graphQBeforeRoPE_ = q;
                        graphKBeforeRoPE_ = k;
                    }
                    
                    q = ggml_rope_ext(ctx, q, pos, nullptr, headDim, rope_mode, n_ctx_orig,
                                      config_.ropeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
                    k = ggml_rope_ext(ctx, k, pos, nullptr, headDim, rope_mode, n_ctx_orig,
                                      config_.ropeTheta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

                    // 保存RoPE后的Q/K用于调试 (所有position都保存)
                    // 注意：保存的是RoPE后的q/k，RoPE前的qn/kn在graph执行后可能无效
                    if (layerIdx == 0) {
                        graphQAfterRoPE_ = q;
                        graphKAfterRoPE_ = k;
                    } else if (layerIdx == 1) {
                        graphQAfterRoPELayer1_ = q;
                        graphKAfterRoPELayer1_ = k;
                    }
                    
                    if (layerIdx == 0 || layerIdx == 1) {
                        CLLM_INFO("[GPU Layer %d position=%d] After RoPE: q->ne=[%d,%d,%d,%d] k->ne=[%d,%d,%d,%d]", 
                                  layerIdx, position,
                                  (int)q->ne[0], (int)q->ne[1], (int)q->ne[2], (int)q->ne[3],
                                  (int)k->ne[0], (int)k->ne[1], (int)k->ne[2], (int)k->ne[3]);
                    }

                    // KV Cache handling
                    // Since Metal doesn't support ggml_set, we use CPU-side KV cache:
                    // 1. Initialize KV cache tensors with size (headDim, nKVHeads, totalLen)
                    // 2. Before graph execution, copy previous KV values from CPU to GPU
                    // 3. After graph execution, copy current K/V from GPU to CPU cache
                    
                    const int kvSize = nKVHeads * headDim;
                    if ((int) kCacheGraphCPU_.size() != config_.numHiddenLayers) {
                        kCacheGraphCPU_.assign(config_.numHiddenLayers, {});
                        vCacheGraphCPU_.assign(config_.numHiddenLayers, {});
                    }
                    if ((int) kCacheGraphCPU_.size() <= layerIdx) {
                        kCacheGraphCPU_.resize(config_.numHiddenLayers);
                        vCacheGraphCPU_.resize(config_.numHiddenLayers);
                    }
                    if (kCacheGraphCPU_[layerIdx].empty()) {
                        kCacheGraphCPU_[layerIdx].resize((size_t) MAX_SEQ_LEN * kvSize, 0.0f);
                        vCacheGraphCPU_[layerIdx].resize((size_t) MAX_SEQ_LEN * kvSize, 0.0f);
                    }

                    // Create KV cache tensors for historical data only
                    // When position=0, no historical data (size 0, but we create size 1 as minimum)
                    // When position>0, historical data is positions 0 to position-1
                    const int histLen = position;
                    ggml_tensor* k_cache = nullptr;
                    ggml_tensor* v_cache = nullptr;
                    
                    if (histLen > 0) {
                        k_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, nKVHeads, histLen);
                        v_cache = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, nKVHeads, histLen);
                        ggml_set_input(k_cache);
                        ggml_set_input(v_cache);
                    }
                    k_cache_layers.push_back(k_cache);
                    v_cache_layers.push_back(v_cache);
                    
                    // Note: We don't use ggml_set (Metal doesn't support it)
                    // Instead, we:
                    // 1. Before graph execution, copy previous K/V values from CPU to GPU (positions 0 to position-1)
                    // 2. After graph execution, copy current K/V values from GPU to CPU (position)
                    // 3. Next forward, the full cache (including current) will be copied to GPU
                    
                    // Save current K/V references for later retrieval
                    k_cur_layers.push_back(k);
                    v_cur_layers.push_back(v);
                    
                    // Concatenate historical KV cache with current K/V
                    // k_cache/v_cache: (headDim, nKVHeads, position) - historical K/V (nullptr when position=0)
                    // k/v: (headDim, nKVHeads, 1) - current K/V
                    // Result: (headDim, nKVHeads, totalLen) where totalLen = position + 1
                    ggml_tensor* k_full = nullptr;
                    ggml_tensor* v_full = nullptr;

                    if (position == 0 || k_cache == nullptr) {
                        // First token: just use current k/v
                        // 使用 ggml_cont 创建连续的副本，确保可以正确复制数据
                        k_full = ggml_cont(ctx, k);
                        v_full = ggml_cont(ctx, v);
                    } else {
                        // Subsequent tokens: concatenate historical cache with current k/v
                        // k_cache contains positions 0 to position-1
                        // k contains current position
                        k_full = ggml_concat(ctx, k_cache, k, 2);  // concat along dim 2 (sequence dim)
                        v_full = ggml_concat(ctx, v_cache, v, 2);
                    }

                    // Save references to full K/V for later retrieval
                    k_cache_upd_layers.push_back(k_full);
                    v_cache_upd_layers.push_back(v_full);

                    // 标准 Attention 实现（参考 llama.cpp）
                    // 输入维度: Q (headDim, nHeads, 1), K (headDim, nKVHeads, totalLen), V (headDim, nKVHeads, totalLen)
                    
                    // 确保张量是连续的
                    ggml_tensor* q_cont = ggml_cont(ctx, q);
                    ggml_tensor* k_cont = ggml_cont(ctx, k_full);
                    ggml_tensor* v_cont = ggml_cont(ctx, v_full);
                    
                    // GGML矩阵乘法规则 (ggml_can_mul_mat):
                    // - a->ne[0] == b->ne[0] (第一个维度必须匹配)
                    // - b->ne[2] % a->ne[2] == 0 (a的第三维必须能广播到b)
                    // - b->ne[3] % a->ne[3] == 0 (a的第四维必须能广播到b)
                    // 结果形状: [a->ne[1], b->ne[1], b->ne[2], b->ne[3]]
                    
                    // 方法：使用 permute 调整维度以匹配 llama.cpp 的实现
                    // 1. 将 Q 从 (headDim, nHeads, 1) permute 到 (headDim, 1, nHeads)
                    // 2. 将 K 从 (headDim, nKVHeads, totalLen) permute 到 (headDim, totalLen, nKVHeads)
                    // 3. 将 V 从 (headDim, nKVHeads, totalLen) permute 到 (headDim, totalLen, nKVHeads)
                    // 4. 执行矩阵乘法: kq = K @ Q, kqv = V @ kq
                    
                    // 步骤 1: Permute Q, K, V
                    // Q: (headDim, nHeads, 1) -> (headDim, 1, nHeads)
                    q_cont = ggml_permute(ctx, q_cont, 0, 2, 1, 3);
                    
                    // K: (headDim, nKVHeads, totalLen) -> (headDim, totalLen, nKVHeads)
                    k_cont = ggml_permute(ctx, k_cont, 0, 2, 1, 3);
                    
                    // V: (headDim, nKVHeads, totalLen) -> (headDim, totalLen, nKVHeads)
                    v_cont = ggml_permute(ctx, v_cont, 0, 2, 1, 3);
                    
                    // 步骤 2: 计算 KQ = K @ Q
                    // k_cont: (headDim, totalLen, nKVHeads)
                    // q_cont: (headDim, 1, nHeads)
                    // 检查维度:
                    // - k->ne[0] == q->ne[0]: headDim == headDim ✓
                    // - q->ne[2] % k->ne[2] == 0: nHeads % nKVHeads == 0 (对于GQA成立) ✓
                    // 结果: (totalLen, 1, nHeads)
                    ggml_tensor* kq = ggml_mul_mat(ctx, k_cont, q_cont);
                    
                    // 缩放
                    kq = ggml_scale(ctx, kq, kq_scale);
                    
                    // Softmax
                    ggml_tensor* kq_soft = ggml_soft_max(ctx, kq);
                    
                    // 步骤 3: 计算 KQV = V @ KQ
                    // v_cont: (headDim, totalLen, nKVHeads)
                    // kq_soft: (totalLen, 1, nHeads)
                    // 检查维度:
                    // - v->ne[0] == kq->ne[0]: headDim == totalLen? 不匹配！
                    
                    // 需要调整：KQ 应该是 (totalLen, nHeads, 1) 而不是 (totalLen, 1, nHeads)
                    // 让我们重新调整 KQ 的维度
                    
                    // 实际上，根据GGML的矩阵乘法规则，结果形状是 [a->ne[1], b->ne[1], b->ne[2], b->ne[3]]
                    // kq = ggml_mul_mat(k, q):
                    // - a=k: (headDim, totalLen, nKVHeads, 1)
                    // - b=q: (headDim, 1, nHeads, 1)
                    // - 结果: (totalLen, 1, nHeads, 1)
                    
                    // 对于 KQV = V @ KQ:
                    // - a=v: (headDim, totalLen, nKVHeads, 1)
                    // - b=kq: (totalLen, 1, nHeads, 1)
                    // - v->ne[0]=headDim, kq->ne[0]=totalLen，不匹配！
                    
                    // 正确的做法：
                    // 1. KQ = Q^T @ K (而不是 K @ Q)
                    // 2. KQV = V @ KQ^T
                    
                    // 重新实现：
                    // 首先，我们需要将 Q 和 K 转换为正确的维度
                    // Q: (headDim, nHeads, 1) -> 需要转换为 (nHeads, headDim, 1) 用于 Q^T @ K
                    // K: (headDim, nKVHeads, totalLen) -> 保持 (headDim, nKVHeads, totalLen)
                    
                    // 使用 permute 将 Q 转换为 (nHeads, headDim, 1)
                    ggml_tensor* q_for_kq = ggml_permute(ctx, q_cont, 1, 0, 2, 3);  // (nHeads, headDim, 1)
                    
                    // KQ = Q^T @ K
                    // q_for_kq: (nHeads, headDim, 1)
                    // k_cont: (headDim, totalLen, nKVHeads)
                    // 需要 k_cont 的维度为 (headDim, nKVHeads, totalLen) 才能与 q_for_kq 匹配
                    
                    // 让我重新思考：标准的 Attention 公式是:
                    // Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
                    
                    // 在 GGML 中，矩阵乘法是 A @ B，其中 A 的列数必须等于 B 的行数
                    // Q: (headDim, nHeads, 1) - 每个 head 有 headDim 个特征
                    // K: (headDim, nKVHeads, totalLen) - 每个 KV head 有 headDim 个特征，totalLen 个位置
                    // K^T: (totalLen, headDim, nKVHeads) - 转置后
                    
                    // Q @ K^T: (headDim, nHeads, 1) @ (totalLen, headDim, nKVHeads)
                    // 这需要 Q 的 headDim 与 K^T 的 headDim 匹配
                    
                    // 实际上，GGML 的矩阵乘法是按最后两个维度进行的
                    // 对于 3D 张量 (d0, d1, d2)，矩阵乘法在 (d0, d1) 上进行
                    
                    // 让我参考 llama.cpp 的实现:
                    // q = ggml_permute(ctx0, q, 0, 2, 1, 3);  // (headDim, n_tokens, nHeads)
                    // k = ggml_permute(ctx0, k, 0, 2, 1, 3);  // (headDim, n_kv, nKVHeads)
                    // kq = ggml_mul_mat(ctx0, k, q);  // K @ Q
                    
                    // 在 llama.cpp 中，k 和 q 的维度是:
                    // k: (headDim, n_kv, nKVHeads)
                    // q: (headDim, n_tokens, nHeads)
                    // kq = K @ Q: (n_kv, n_tokens, nHeads)
                    
                    // 然后 kqv = V @ kq:
                    // v: (headDim, n_kv, nKVHeads)
                    // kq: (n_kv, n_tokens, nHeads)
                    // kqv: (headDim, n_tokens, nHeads)
                    
                    // 在我们的情况下:
                    // q: (headDim, nHeads, 1) - 当前 token
                    // k: (headDim, nKVHeads, totalLen) - 所有历史 token
                    // v: (headDim, nKVHeads, totalLen) - 所有历史 token
                    
                    // 按照 llama.cpp 的方式:
                    // q_perm = permute(q, 0, 2, 1, 3): (headDim, 1, nHeads)
                    // k_perm = permute(k, 0, 2, 1, 3): (headDim, totalLen, nKVHeads)
                    // kq = mul_mat(k_perm, q_perm): (totalLen, 1, nHeads)
                    
                    // 但是 v_perm = permute(v, 0, 2, 1, 3): (headDim, totalLen, nKVHeads)
                    // kq: (totalLen, 1, nHeads)
                    // kqv = mul_mat(v_perm, kq): 需要 v_perm->ne[0] == kq->ne[0]，即 headDim == totalLen，不匹配！
                    
                    // 问题在于 llama.cpp 的实现中，v 的维度是 (headDim, n_kv, nKVHeads)
                    // 而 kq 的维度是 (n_kv, n_tokens, nHeads)
                    // 所以 V @ KQ 是 (headDim, n_kv, nKVHeads) @ (n_kv, n_tokens, nHeads) = (headDim, n_tokens, nHeads)
                    
                    // 但在我们的情况下，kq 是 (totalLen, 1, nHeads)，不是 (totalLen, nHeads, 1)
                    
                    // 让我重新理解 GGML 的矩阵乘法...
                    // 实际上，GGML 的 mul_mat 是按广播规则进行的
                    // 对于 3D 张量 A (d0, d1, d2) 和 B (d0, d1, d3):
                    // 结果 C (d1, d2, d3) 是 A[:, :, i] @ B[:, :, j] 对所有 i, j
                    
                    // 等等，让我重新阅读 ggml_can_mul_mat 的注释:
                    // return (t0->ne[0]           == t1->ne[0])  &&
                    //        (t1->ne[2]%t0->ne[2] == 0)          && // verify t0 is broadcastable
                    //        (t1->ne[3]%t0->ne[3] == 0);
                    
                    // 这意味着:
                    // - t0->ne[0] == t1->ne[0]: 第一个维度必须匹配（这是矩阵乘法的内部维度）
                    // - t1->ne[2] % t0->ne[2] == 0: t0 的第三维必须能广播到 t1
                    // - t1->ne[3] % t0->ne[3] == 0: t0 的第四维必须能广播到 t1
                    
                    // 结果形状: [t0->ne[1], t1->ne[1], t1->ne[2], t1->ne[3]]
                    
                    // 所以对于 kq = mul_mat(k, q):
                    // k: (headDim, totalLen, nKVHeads, 1) -> ne[0]=headDim, ne[1]=totalLen, ne[2]=nKVHeads, ne[3]=1
                    // q: (headDim, 1, nHeads, 1) -> ne[0]=headDim, ne[1]=1, ne[2]=nHeads, ne[3]=1
                    // 检查:
                    // - k->ne[0] == q->ne[0]: headDim == headDim ✓
                    // - q->ne[2] % k->ne[2]: nHeads % nKVHeads == 0 (对于GQA成立) ✓
                    // - q->ne[3] % k->ne[3]: 1 % 1 == 0 ✓
                    // 结果: (totalLen, 1, nHeads, 1)
                    
                    // 对于 kqv = mul_mat(v, kq):
                    // v: (headDim, totalLen, nKVHeads, 1) -> ne[0]=headDim, ne[1]=totalLen, ne[2]=nKVHeads, ne[3]=1
                    // kq: (totalLen, 1, nHeads, 1) -> ne[0]=totalLen, ne[1]=1, ne[2]=nHeads, ne[3]=1
                    // 检查:
                    // - v->ne[0] == kq->ne[0]: headDim == totalLen? 不匹配！
                    
                    // 问题在于 v 和 kq 的第一个维度不匹配
                    // 在标准的 Attention 中，应该是 V @ softmax(Q @ K^T)
                    // Q @ K^T: (nHeads, headDim) @ (headDim, totalLen) = (nHeads, totalLen)
                    // V: (totalLen, headDim) 或 (headDim, totalLen) 取决于布局
                    
                    // 在 llama.cpp 中，v 在 mul_mat 之前被转置了:
                    // if (!v_trans) { v = ggml_cont(ctx0, ggml_transpose(ctx0, v)); }
                    // 这意味着 v 的维度从 (headDim, n_kv, nKVHeads) 变为 (n_kv, headDim, nKVHeads)
                    
                    // 所以正确的流程是:
                    // 1. q = permute(q, 0, 2, 1, 3): (headDim, 1, nHeads)
                    // 2. k = permute(k, 0, 2, 1, 3): (headDim, totalLen, nKVHeads)
                    // 3. v = permute(v, 0, 2, 1, 3): (headDim, totalLen, nKVHeads)
                    // 4. v = transpose(v): (totalLen, headDim, nKVHeads)
                    // 5. kq = mul_mat(k, q): (totalLen, 1, nHeads)
                    // 6. kqv = mul_mat(v, kq): (headDim, 1, nHeads)
                    
                    // 让我按照这个流程重新实现:
                    
                    // 重新获取原始张量
                    q_cont = ggml_cont(ctx, q);
                    k_cont = ggml_cont(ctx, k_full);
                    v_cont = ggml_cont(ctx, v_full);
                    
                    // Permute 到正确的维度
                    // Q: (headDim, nHeads, 1) -> (headDim, 1, nHeads)
                    q_cont = ggml_permute(ctx, q_cont, 0, 2, 1, 3);
                    
                    // K: (headDim, nKVHeads, totalLen) -> (headDim, totalLen, nKVHeads)
                    k_cont = ggml_permute(ctx, k_cont, 0, 2, 1, 3);
                    
                    // V: (headDim, nKVHeads, totalLen) -> (totalLen, headDim, nKVHeads)
                    // 先 permute 到 (headDim, totalLen, nKVHeads)，然后 transpose
                    v_cont = ggml_permute(ctx, v_cont, 0, 2, 1, 3);  // (headDim, totalLen, nKVHeads)
                    v_cont = ggml_cont(ctx, ggml_transpose(ctx, v_cont));  // (totalLen, headDim, nKVHeads)
                    
                    // 计算 KQ = K @ Q
                    // k_cont: (headDim, totalLen, nKVHeads)
                    // q_cont: (headDim, 1, nHeads)
                    // 结果: (totalLen, 1, nHeads)
                    kq = ggml_mul_mat(ctx, k_cont, q_cont);
                    
                    // 缩放
                    kq = ggml_scale(ctx, kq, kq_scale);
                    
                    // Softmax
                    kq_soft = ggml_soft_max(ctx, kq);
                    
                    // 计算 KQV = V @ KQ
                    // v_cont: (totalLen, headDim, nKVHeads)
                    // kq_soft: (totalLen, 1, nHeads)
                    // 检查:
                    // - v->ne[0] == kq->ne[0]: totalLen == totalLen ✓
                    // - kq->ne[2] % v->ne[2]: nHeads % nKVHeads == 0 (对于GQA成立) ✓
                    // 结果: (headDim, 1, nHeads)
                    ggml_tensor* kqv = ggml_mul_mat(ctx, v_cont, kq_soft);
                    
                    // 重新调整维度: (headDim, 1, nHeads) -> (headDim, nHeads, 1)
                    kqv = ggml_permute(ctx, kqv, 0, 2, 1, 3);
                    
                    // reshape 到 (headDim * nHeads, 1) 以进行 O projection
                    attn_out = ggml_reshape_2d(ctx, kqv, headDim * nHeads, 1);
                    attn_out = ggml_mul_mat(ctx, layer.oProj, attn_out);
                }

                if (attn_out) {
                    inp = ggml_add(ctx, inp, attn_out);
                }
                
                // 保存 Attention 输出用于调试
                if (layerIdx == 0) {
                    graphAttnOutput_ = attn_out;
                }
                
                // 保存 Layer 0 Attention + Residual 后的输出用于调试
                if (layerIdx == 0) {
                    graphLayer0Output_ = inp;
                }

                ggml_tensor* post = ggml_rms_norm(ctx, inp, config_.rmsNormEps);
                ggml_tensor* post_w = ggml_mul(ctx, post, 
                    ggml_reshape_2d(ctx, layer.postAttentionLayernorm, hiddenSize, 1));

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
                    // 保存 Layer 0 的输出用于调试
                    if (l == 0) {
                        graphLayer0Output_ = x;
                    }
                }
            } else {
                x = build_layer(0, x);
                graphLayer0Output_ = x;
            }

            ggml_tensor* fn = ggml_rms_norm(ctx, x, config_.rmsNormEps);
            ggml_tensor* fn_w = ggml_mul(ctx, fn, 
                ggml_reshape_2d(ctx, finalNorm_, hiddenSize, 1));
            x = fn_w;
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
            graphKCurLayers_ = std::move(k_cur_layers);
            graphVCurLayers_ = std::move(v_cur_layers);
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
    std::vector<ggml_tensor*> k_cur_layers;
    std::vector<ggml_tensor*> v_cur_layers;

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
            graphKCurLayers_.clear();
            graphVCurLayers_.clear();

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
        // k_cache/v_cache only contain historical data (0 to position-1)
        // For position=0, they are empty (nullptr), for position>0, they have 'position' elements
        const int histLen = position;
        const size_t histSize = (size_t) histLen * kvSize * sizeof(float);
        for (size_t i = 0; i < kLayers.size(); ++i) {
            if (histLen > 0 && kLayers[i] != nullptr && vLayers[i] != nullptr) {
                // 获取 tensor 的实际大小
                size_t kLayerSize = ggml_nbytes(kLayers[i]);
                // 打印 Layer 0 和 Layer 1 的 KV Cache 前5个值用于调试
                if (i == 0 || i == 1) {

                    // 打印 position-1 位置的 KV 值（即最后一个历史位置）
                    size_t lastPosOffset = (size_t)(histLen - 1) * kvSize;
                }
                ggml_backend_tensor_set(kLayers[i], kCacheGraphCPU_[i].data(), 0, kLayerSize);
                ggml_backend_tensor_set(vLayers[i], vCacheGraphCPU_[i].data(), 0, kLayerSize);
            } else if (histLen > 0) {
                CLLM_WARN("[DEBUG] Layer %zu has nullptr kLayers or vLayers: kLayers[i]=%p, vLayers[i]=%p", 
                          i, (void*)kLayers[i], (void*)vLayers[i]);
            }
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
    // 在 position 0 和 1 时都打印 Layer 0 的 Q/K Projection 和 RoPE 结果
    if (position == 0 || position == 1) {
        const int headDim = config_.getHeadDim();
        const int nHeads = config_.numAttentionHeads;
        const int nKVHeads = config_.getNumKVHeads();
        const int qSize = nHeads * headDim;
        const int kvSize = nKVHeads * headDim;
        const int hiddenSize = config_.hiddenSize;
        // 打印 Embedding 输出
        if (graphEmbedding_) {
            std::vector<float> embedding(hiddenSize);
            ggml_backend_tensor_get(graphEmbedding_, embedding.data(), 0, embedding.size() * sizeof(float));
        }

        // 打印 Layer 0 最终输出
        if (graphLayer0Output_) {
            std::vector<float> layer0Output(hiddenSize);
            ggml_backend_tensor_get(graphLayer0Output_, layer0Output.data(), 0, layer0Output.size() * sizeof(float));
        }

        // 打印 Attention 输入
        if (graphAttnInput_) {
            std::vector<float> attnInput(hiddenSize);
            ggml_backend_tensor_get(graphAttnInput_, attnInput.data(), 0, attnInput.size() * sizeof(float));
        }

        // 打印 Attention 输出
        if (graphAttnOutput_) {
            std::vector<float> attnOutput(hiddenSize);
            ggml_backend_tensor_get(graphAttnOutput_, attnOutput.data(), 0, attnOutput.size() * sizeof(float));
        }

        // 在 position 0 和 1 时都打印 RMS Norm 信息
        // RMS Norm 前输入
        if (graphNormInput_) {
            std::vector<float> normInput(hiddenSize);
            ggml_backend_tensor_get(graphNormInput_, normInput.data(), 0, normInput.size() * sizeof(float));
        }

        // RMS Norm 后输出（乘以权重前）
        if (graphNormOutput_) {
            std::vector<float> normOutput(hiddenSize);
            ggml_backend_tensor_get(graphNormOutput_, normOutput.data(), 0, normOutput.size() * sizeof(float));
        }

        // RMS Norm 后输出（乘以权重后）
        if (graphNormWeighted_) {
            std::vector<float> normWeighted(hiddenSize);
            ggml_backend_tensor_get(graphNormWeighted_, normWeighted.data(), 0, normWeighted.size() * sizeof(float));
        }

        // Q/K 投影后（在 reshape 之前，是 2D 张量）- 所有 position 都打印
        if (graphQProj_ && graphKProj_) {
            std::vector<float> qProj(qSize);
            std::vector<float> kProj(kvSize);

            ggml_backend_tensor_get(graphQProj_, qProj.data(), 0, qProj.size() * sizeof(float));
            ggml_backend_tensor_get(graphKProj_, kProj.data(), 0, kProj.size() * sizeof(float));

        }

        // RoPE 前 - 所有 position 都打印
        if (graphQBeforeRoPE_ && graphKBeforeRoPE_) {
            std::vector<float> qBeforeRoPE(headDim * nHeads);
            std::vector<float> kBeforeRoPE(headDim * nKVHeads);

            ggml_backend_tensor_get(graphQBeforeRoPE_, qBeforeRoPE.data(), 0, qBeforeRoPE.size() * sizeof(float));
            ggml_backend_tensor_get(graphKBeforeRoPE_, kBeforeRoPE.data(), 0, kBeforeRoPE.size() * sizeof(float));

        }

        // RoPE 后 - 所有 position 都打印
        if (graphQAfterRoPE_ && graphKAfterRoPE_) {
            std::vector<float> qAfterRoPE(headDim * nHeads);
            std::vector<float> kAfterRoPE(headDim * nKVHeads);

            ggml_backend_tensor_get(graphQAfterRoPE_, qAfterRoPE.data(), 0, qAfterRoPE.size() * sizeof(float));
            ggml_backend_tensor_get(graphKAfterRoPE_, kAfterRoPE.data(), 0, kAfterRoPE.size() * sizeof(float));
            if (position == 1) {
                const int halfDim = headDim / 2;
                for (int i = 0; i < 5; ++i) {
                    float freq = 1.0f / std::pow(config_.ropeTheta, 2.0f * i / headDim);
                    float angle = position * freq;
                }
            }
        }

        // Layer 1 调试信息 - 在 position 0 和 1 时都打印 RMS Norm
        if (graphNormInputLayer1_) {
            std::vector<float> normInput(hiddenSize);
            ggml_backend_tensor_get(graphNormInputLayer1_, normInput.data(), 0, normInput.size() * sizeof(float));
        }
        if (graphNormOutputLayer1_) {
            std::vector<float> normOutput(hiddenSize);
            ggml_backend_tensor_get(graphNormOutputLayer1_, normOutput.data(), 0, normOutput.size() * sizeof(float));
        }
        if (graphNormWeightedLayer1_) {
            std::vector<float> normWeighted(hiddenSize);
            ggml_backend_tensor_get(graphNormWeightedLayer1_, normWeighted.data(), 0, normWeighted.size() * sizeof(float));
        }
        
        // Layer 1 Q/K Projection 和 RoPE - 所有 position 都打印
        if (graphQProjLayer1_ && graphKProjLayer1_) {
            std::vector<float> qProj(qSize);
            std::vector<float> kProj(kvSize);
            ggml_backend_tensor_get(graphQProjLayer1_, qProj.data(), 0, qProj.size() * sizeof(float));
            ggml_backend_tensor_get(graphKProjLayer1_, kProj.data(), 0, kProj.size() * sizeof(float));

        }
        if (graphQAfterRoPELayer1_ && graphKAfterRoPELayer1_) {
            std::vector<float> qAfterRoPE(headDim * nHeads);
            std::vector<float> kAfterRoPE(headDim * nKVHeads);
            ggml_backend_tensor_get(graphQAfterRoPELayer1_, qAfterRoPE.data(), 0, qAfterRoPE.size() * sizeof(float));
            ggml_backend_tensor_get(graphKAfterRoPELayer1_, kAfterRoPE.data(), 0, kAfterRoPE.size() * sizeof(float));

        }
    }
    // 只在 position == 0 时打印，因为 graphNormWeightedAllLayers_ 只在此时被设置
    if (position == 0) {
        for (int layerIdx = 0; layerIdx < 28; ++layerIdx) {
            if (graphNormWeightedAllLayers_[layerIdx]) {
                std::vector<float> normData(hiddenSize);
                ggml_backend_tensor_get(graphNormWeightedAllLayers_[layerIdx], normData.data(), 0, normData.size() * sizeof(float));

                float minVal = normData[0], maxVal = normData[0];
                double sum = 0;
                for (int i = 0; i < hiddenSize; ++i) {
                    if (normData[i] < minVal) minVal = normData[i];
                    if (normData[i] > maxVal) maxVal = normData[i];
                    sum += normData[i];
                }
                float mean = sum / hiddenSize;

                CLLM_INFO("[GPU Layer %2d] RMS Norm Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                          layerIdx, minVal, maxVal, mean, normData[0], normData[1], normData[2], normData[3], normData[4]);
            }
        }
    }
    {
        float minVal = out[0], maxVal = out[0];
        double sum = 0;
        size_t nanCount = 0, infCount = 0;

        for (size_t i = 0; i < static_cast<size_t>(vocabSize); ++i) {
            float v = out[i];
            if (std::isnan(v)) { nanCount++; continue; }
            if (std::isinf(v)) { infCount++; continue; }
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
            sum += v;
        }

        CLLM_INFO("[GPU FINAL] Logits: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  minVal, maxVal, sum / vocabSize, out[0], out[1], out[2], out[3], out[4]);
        // Top 10 tokens
        std::vector<std::pair<float, int>> topTokens;
        for (int i = 0; i < vocabSize && i < 200000; ++i) {
            topTokens.push_back({out[i], i});
        }
        std::partial_sort(topTokens.begin(), topTokens.begin() + std::min(10, (int)topTokens.size()),
                         topTokens.end(), std::greater<>());
        for (int i = 0; i < std::min(10, (int)topTokens.size()); ++i) {
        }

        // 关键 token 检查
        if (vocabSize > 151645) {
        }
    }

    if (stats.sched) {
        const int splits = ggml_backend_sched_get_n_splits(stats.sched);
        CLLM_INFO("[GGMLGPUBackend] sched splits=%d, nodes=%d (gpu=%d cpu=%d other=%d)",
                  splits, stats.total, stats.gpuNodes, stats.cpuNodes, stats.otherNodes);
        ggml_backend_sched_set_eval_callback(stats.sched, nullptr, nullptr);
    }

    // 从 GPU 复制完整的 K/V 序列到 CPU 缓存
    // 我们使用 k_cache_upd_layers/v_cache_upd_layers 获取完整的 K/V (历史 + 当前)
    const auto& kUpdLayers = usePersistent ? graphKCacheUpdLayers_ : k_cache_upd_layers;
    const auto& vUpdLayers = usePersistent ? graphVCacheUpdLayers_ : v_cache_upd_layers;
    if (!kUpdLayers.empty()) {
        const int headDim = config_.getHeadDim();
        const int nKVHeads = config_.getNumKVHeads();
        const int kvSize = nKVHeads * headDim;
        const int totalLen = position + 1;
        for (size_t i = 0; i < kUpdLayers.size(); ++i) {
            // 获取当前 tensor 的实际大小
            size_t tensorSize = ggml_nbytes(kUpdLayers[i]);
            // 从 GPU 复制完整的 K/V 序列 (包含历史和当前)
            ggml_backend_tensor_get(kUpdLayers[i], kCacheGraphCPU_[i].data(), 0, tensorSize);
            ggml_backend_tensor_get(vUpdLayers[i], vCacheGraphCPU_[i].data(), 0, tensorSize);
            
            // 打印 Layer 0 和 Layer 1 的 KV Cache 统计信息用于调试
            if (i == 0 || i == 1) {
                // 计算当前位置的 KV 值统计
                size_t kvOffset = (size_t) position * kvSize;
                float kMin = kCacheGraphCPU_[i][kvOffset], kMax = kCacheGraphCPU_[i][kvOffset];
                float vMin = vCacheGraphCPU_[i][kvOffset], vMax = vCacheGraphCPU_[i][kvOffset];
                double kSum = 0, vSum = 0;
                for (int j = 0; j < kvSize; ++j) {
                    float kVal = kCacheGraphCPU_[i][kvOffset + j];
                    float vVal = vCacheGraphCPU_[i][kvOffset + j];
                    if (kVal < kMin) kMin = kVal;
                    if (kVal > kMax) kMax = kVal;
                    if (vVal < vMin) vMin = vVal;
                    if (vVal > vMax) vMax = vVal;
                    kSum += kVal;
                    vSum += vVal;
                }

            }
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
    CLLM_INFO("[GGMLGPUBackend] Caching weights to CPU for fast access (transposing back to row-major)...");
    
    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const int intermediateSize = config_.intermediateSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    const int numLayers = config_.numHiddenLayers;
    
    // Helper lambda to get and transpose 2D weights
    auto getAndTranspose = [&](ggml_tensor* tensor, int rows, int cols, std::vector<float>& out) {
        out.resize(rows * cols);
        std::vector<float> temp(rows * cols);
        ggml_backend_tensor_get(tensor, temp.data(), 0, rows * cols * sizeof(float));
        // Transpose from GGML column-major [cols, rows] to row-major [rows, cols]
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                // GGML: temp[j * rows + i] (column j, row i)
                // Row-major: out[i * cols + j] (row i, column j)
                out[i * cols + j] = temp[j * rows + i];
            }
        }
    };
    
    // Embedding: GGML [hiddenSize, vocabSize] -> row-major [vocabSize, hiddenSize]
    getAndTranspose(embedTokens_, vocabSize, hiddenSize, weightsCached_["embed_tokens"]);
    
    // Final norm
    weightsCached_["final_norm"].resize(hiddenSize);
    ggml_backend_tensor_get(finalNorm_, weightsCached_["final_norm"].data(), 
                            0, hiddenSize * sizeof(float));
    
    // LM Head [vocabSize, hiddenSize] - transpose from [hiddenSize, vocabSize]
    weightsCached_["lm_head"].resize((size_t)vocabSize * hiddenSize);
    getAndTranspose(lmHead_, vocabSize, hiddenSize, weightsCached_["lm_head"]);
    
    // Each layer
    for (int l = 0; l < numLayers; ++l) {
        LayerTensors& layer = layers_[l];
        std::string prefix = "layer." + std::to_string(l) + ".";
        
        // 1D weights - no transpose needed
        weightsCached_[prefix + "input_layernorm"].resize(hiddenSize);
        ggml_backend_tensor_get(layer.inputLayernorm, 
                                weightsCached_[prefix + "input_layernorm"].data(), 
                                0, hiddenSize * sizeof(float));
        
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
        
        // 2D weights - transpose from GGML column-major to row-major
        // qProj: [qSize, hiddenSize] (GGML: [hiddenSize, qSize])
        getAndTranspose(layer.qProj, qSize, hiddenSize, weightsCached_[prefix + "q_proj"]);
        
        // kProj: [kvSize, hiddenSize] (GGML: [hiddenSize, kvSize])
        getAndTranspose(layer.kProj, kvSize, hiddenSize, weightsCached_[prefix + "k_proj"]);
        
        // vProj: [kvSize, hiddenSize] (GGML: [hiddenSize, kvSize])
        getAndTranspose(layer.vProj, kvSize, hiddenSize, weightsCached_[prefix + "v_proj"]);
        
        // oProj: [hiddenSize, qSize] (GGML: [qSize, hiddenSize])
        getAndTranspose(layer.oProj, hiddenSize, qSize, weightsCached_[prefix + "o_proj"]);
        
        // gateProj: [intermediateSize, hiddenSize] (GGML: [hiddenSize, intermediateSize])
        getAndTranspose(layer.gateProj, intermediateSize, hiddenSize, weightsCached_[prefix + "gate_proj"]);
        
        // upProj: [intermediateSize, hiddenSize] (GGML: [hiddenSize, intermediateSize])
        getAndTranspose(layer.upProj, intermediateSize, hiddenSize, weightsCached_[prefix + "up_proj"]);
        
        // downProj: [hiddenSize, intermediateSize] (GGML: [intermediateSize, hiddenSize])
        getAndTranspose(layer.downProj, hiddenSize, intermediateSize, weightsCached_[prefix + "down_proj"]);
    }
    
    CLLM_INFO("[GGMLGPUBackend] ✅ Weights cached to CPU (transposed to row-major)");
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

std::vector<std::vector<float>> GGMLGPUBackend::forwardBatch(
    const std::vector<int>& tokenIds,
    const std::vector<int>& positions) {
    std::vector<std::vector<float>> results;
    results.reserve(tokenIds.size());
    
    for (size_t i = 0; i < tokenIds.size(); ++i) {
        auto logits = forwardGraphMinimal(tokenIds[i], positions[i]);
        results.push_back(std::move(logits));
    }
    
    return results;
}

// ============================================================================
// 调试功能：导出中间层结果
// ============================================================================

std::vector<float> GGMLGPUBackend::forwardWithDebug(int tokenId, int position,
                                                    std::vector<LayerOutput>* layerOutputs,
                                                    std::vector<float>* embeddingOutput,
                                                    std::vector<float>* finalNormOutput) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend] Not initialized");
        return {};
    }

    const int hiddenSize = config_.hiddenSize;
    const int vocabSize = config_.vocabSize;
    const int numLayers = config_.numHiddenLayers;
    const int intermediateSize = config_.intermediateSize;
    const float eps = config_.rmsNormEps;

    // 初始化输出结构
    if (layerOutputs) {
        layerOutputs->clear();
        layerOutputs->resize(numLayers);
    }
    if (embeddingOutput) {
        embeddingOutput->resize(hiddenSize);
    }
    if (finalNormOutput) {
        finalNormOutput->resize(hiddenSize);
    }

    // 如果启用了 GPU 图执行，先执行 GPU 获取最终 logits
    std::vector<float> gpuLogits;
    int savedGraphStage = graphStage_;
    
    if (graphStage_ > 0) {
        gpuLogits = forwardGraphMinimal(tokenId, position);
        if (gpuLogits.empty()) {
            CLLM_ERROR("[DEBUG] GPU graph execution failed");
            return {};
        }
        // 注意：不要在图执行后访问 graphQAfterRoPE_，因为它可能指向已释放的张量
    }
    
    // 如果需要中间结果，临时禁用 GPU 图，使用 CPU 路径
    if (graphStage_ > 0 && layerOutputs) {
        graphStage_ = 0;
    } else {
    }

    // 使用 CPU 回退路径导出中间结果
    // 注意：这里复用 forwardCPU 的逻辑，但添加中间结果导出

    // ===== 首次调用时缓存权重到 CPU =====
    if (weightsCached_.empty()) {
        cacheWeightsToCPU();
    }

    // 1. Embedding
    // embed_tokens is row-major [vocabSize, hiddenSize]
    // lookup: embedData[tokenId * hiddenSize + i]
    std::vector<float> hidden(hiddenSize);
    const float* embedData = weightsCached_["embed_tokens"].data();
    for (int i = 0; i < hiddenSize; ++i) {
        hidden[i] = embedData[tokenId * hiddenSize + i];
    }

    if (embeddingOutput) {
        std::copy(hidden.begin(), hidden.end(), embeddingOutput->begin());
    }
    const float* qProjW0 = weightsCached_["layer.0.q_proj"].data();
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    const int gqa = nHeads / nKVHeads;
    const int totalLen = position + 1;

    // 临时缓冲区
    std::vector<float> residual(hiddenSize);
    std::vector<float> normOut(hiddenSize);
    std::vector<float> q(qSize), k(kvSize), v(kvSize);
    std::vector<float> attnOut(hiddenSize);
    std::vector<float> oOut(hiddenSize);
    std::vector<float> gate(intermediateSize), up(intermediateSize);
    std::vector<float> ffnOut(hiddenSize);

    // 2. Transformer Layers
    for (int l = 0; l < numLayers; ++l) {
        // 获取权重
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

        if (layerOutputs && position == 0) {
            layerOutputs->at(l).afterInputNorm.resize(hiddenSize);
            std::copy(normOut.begin(), normOut.end(), layerOutputs->at(l).afterInputNorm.begin());
        }

        // QKV Projection
        cpuMatmul(qProjW, normOut.data(), q.data(), qSize, hiddenSize);
        cpuMatmul(kProjW, normOut.data(), k.data(), kvSize, hiddenSize);
        cpuMatmul(vProjW, normOut.data(), v.data(), kvSize, hiddenSize);

        if (layerOutputs && position == 0) {
            // 保存 Q, K, V 拼接后的输出，与 HFTransformerModel 保持一致
            layerOutputs->at(l).afterQKV.resize(qSize + 2 * kvSize);
            std::copy(q.begin(), q.end(), layerOutputs->at(l).afterQKV.begin());
            std::copy(k.begin(), k.end(), layerOutputs->at(l).afterQKV.begin() + qSize);
            std::copy(v.begin(), v.end(), layerOutputs->at(l).afterQKV.begin() + qSize + kvSize);
        }

        // Q/K RMS Norm - 使用与 HFTransformerModel 相同的实现
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q.data() + h * headDim;
            float sumSq = ggml_kernels::dot_product(qHead, qHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + eps);
            for (int i = 0; i < headDim; ++i) qHead[i] = qHead[i] * invRms * qNormW[i];
        }
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k.data() + h * headDim;
            float sumSq = ggml_kernels::dot_product(kHead, kHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + eps);
            for (int i = 0; i < headDim; ++i) kHead[i] = kHead[i] * invRms * kNormW[i];
        }
        if (position == 0 && l == 0) {
        }

        // RoPE
        cpuApplyRoPE(q.data(), nHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());
        cpuApplyRoPE(k.data(), nKVHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());

        if (position == 0 && l == 0) {

        }

        // Update KV Cache
        float* kCacheLayer = kCacheCPU_[l].data();
        float* vCacheLayer = vCacheCPU_[l].data();
        std::copy(k.begin(), k.end(), kCacheLayer + position * kvSize);
        std::copy(v.begin(), v.end(), vCacheLayer + position * kvSize);

        // Attention - 使用与 HFTransformerModel 相同的实现
        std::fill(attnOut.begin(), attnOut.end(), 0.0f);
        const float attnScale = 1.0f / std::sqrt((float)headDim);

        for (int h = 0; h < nHeads; ++h) {
            const int kvHead = h / gqa;
            const float* qHead = q.data() + h * headDim;
            float* outHead = attnOut.data() + h * headDim;

            // 使用栈分配避免动态分配
            float scores[256];  // 假设最大序列长度不超过 256
            float maxScore = -1e30f;

            // 第一遍：计算 scores 并找 max
            for (int t = 0; t < totalLen; ++t) {
                const float* kRow = kCacheLayer + t * kvSize + kvHead * headDim;
                float dot = ggml_kernels::dot_product(qHead, kRow, headDim) * attnScale;
                scores[t] = dot;
                maxScore = (dot > maxScore) ? dot : maxScore;
            }

            // 第二遍：exp 和 sum
            float sumExp = 0.0f;
            for (int t = 0; t < totalLen; ++t) {
                float e = std::exp(scores[t] - maxScore);
                scores[t] = e;
                sumExp += e;
            }

            // 第三遍：归一化并计算 weighted sum
            const float invSum = 1.0f / sumExp;
            std::fill(outHead, outHead + headDim, 0.0f);
            
            for (int t = 0; t < totalLen; ++t) {
                const float* vRow = vCacheLayer + t * kvSize + kvHead * headDim;
                const float w = scores[t] * invSum;
                // 使用 SIMD 友好的循环
                for (int d = 0; d < headDim; ++d) {
                    outHead[d] += w * vRow[d];
                }
            }
        }

        if (layerOutputs && position == 0 && l == 0) {
        }

        // O Projection
        cpuMatmul(oProjW, attnOut.data(), oOut.data(), hiddenSize, qSize);

        if (layerOutputs && position == 0 && l == 0) {
        }

        // Residual
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + oOut[i];
        }
        std::copy(hidden.begin(), hidden.end(), residual.begin());

        if (layerOutputs && position == 0) {
            layerOutputs->at(l).afterAttention.resize(hiddenSize);
            std::copy(hidden.begin(), hidden.end(), layerOutputs->at(l).afterAttention.begin());
        }

        // Post Attention LayerNorm
        cpuRmsNorm(hidden.data(), postNormW, normOut.data(), hiddenSize, eps);

        if (layerOutputs && position == 0) {
            layerOutputs->at(l).afterPostNorm.resize(hiddenSize);
            std::copy(normOut.begin(), normOut.end(), layerOutputs->at(l).afterPostNorm.begin());
        }

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

        if (layerOutputs && position == 0) {
            layerOutputs->at(l).afterFFN.resize(hiddenSize);
            std::copy(hidden.begin(), hidden.end(), layerOutputs->at(l).afterFFN.begin());
        }
    }

    // 3. Final LayerNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    cpuRmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);

    if (finalNormOutput) {
        std::copy(normOut.begin(), normOut.end(), finalNormOutput->begin());
    }
    // 4. LM Head
    const float* lmHeadW = weightsCached_["lm_head"].data();
    std::vector<float> logits(vocabSize);
    cpuMatmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    kvCacheLen_ = totalLen;

    // 恢复原始的 graphStage_
    graphStage_ = savedGraphStage;
    
    // 如果执行了 GPU 图，返回 GPU 的结果；否则返回 CPU 的结果
    if (!gpuLogits.empty()) {
        // 对比 CPU 和 GPU 的 logits
        float maxDiff = 0.0f;
        float sumSqDiff = 0.0f;
        for (int i = 0; i < vocabSize; ++i) {
            float diff = std::abs(gpuLogits[i] - logits[i]);
            maxDiff = std::max(maxDiff, diff);
            sumSqDiff += diff * diff;
        }
        float rmse = std::sqrt(sumSqDiff / vocabSize);
        return gpuLogits;
    }

    return logits;
}

} // namespace kylin
} // namespace cllm
