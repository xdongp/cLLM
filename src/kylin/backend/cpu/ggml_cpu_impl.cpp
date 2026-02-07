/**
 * @file ggml_cpu_impl.cpp
 * @brief GGML CPU 后端完整实现
 *
 * 包含完整的 Transformer CPU 计算：
 * - Q/K/V Projection
 * - Q/K RMS Norm (Qwen3)
 * - RoPE (旋转位置编码)
 * - KV Cache
 * - Attention (QK^T + softmax + V)
 * - GQA (Grouped Query Attention)
 * - O Projection
 * - FFN (SwiGLU)
 */

#include "cllm/kylin/backend/cpu/cpu_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {
namespace backend {

static constexpr int MAX_SEQ_LEN = 2048;

// ============================================================================
// CPU 辅助函数
// ============================================================================

static inline void cpuRmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

static inline void cpuMatmul(const float* weight, const float* input, float* output, int M, int K) {
    ggml_kernels::matmul_f32(weight, input, output, M, K);
}

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

static float cpuSilu(float x) {
    return x / (1.0f + std::exp(-x));
}

static void cpuSoftmax(float* x, int size) {
    float maxVal = x[0];
    for (int i = 1; i < size; ++i) {
        if (x[i] > maxVal) maxVal = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        x[i] = std::exp(x[i] - maxVal);
        sum += x[i];
    }

    for (int i = 0; i < size; ++i) {
        x[i] /= sum;
    }
}

// ============================================================================
// 构造函数和析构函数
// ============================================================================

CPUBackend::CPUBackend() = default;

CPUBackend::~CPUBackend() {
    weightsCached_.clear();
}

// ============================================================================
// 初始化
// ============================================================================

bool CPUBackend::initialize(const HFModelConfig& config) {
    config_ = config;

    // 初始化 KV Cache
    const int numLayers = config_.numHiddenLayers;
    const int nKVHeads = config_.getNumKVHeads();
    const int headDim = config_.getHeadDim();
    const int maxSeqLen = MAX_SEQ_LEN;

    kCache_.resize(numLayers);
    vCache_.resize(numLayers);

    for (int l = 0; l < numLayers; ++l) {
        kCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
        vCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
    }

    // 初始化 RoPE
    initRoPE();

    initialized_ = true;
    CLLM_INFO("[CPUBackend] Initialized with %d layers", numLayers);
    return true;
}

void CPUBackend::initRoPE() {
    int headDim = config_.getHeadDim();
    ropeFreqsCos_.resize(MAX_SEQ_LEN * headDim / 2);
    ropeFreqsSin_.resize(MAX_SEQ_LEN * headDim / 2);

    for (int pos = 0; pos < MAX_SEQ_LEN; ++pos) {
        for (int i = 0; i < headDim / 2; ++i) {
            float freq = 1.0f / std::pow(config_.ropeTheta, 2.0f * i / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * headDim / 2 + i] = std::cos(angle);
            ropeFreqsSin_[pos * headDim / 2 + i] = std::sin(angle);
        }
    }
}

// ============================================================================
// 权重上传
// ============================================================================

bool CPUBackend::uploadWeights(const std::unordered_map<std::string, std::vector<float>>& weightsMap) {
    weightsCached_ = weightsMap;
    CLLM_INFO("[CPUBackend] Uploaded %zu weights", weightsCached_.size());
    return true;
}

// ============================================================================
// 前向传播
// ============================================================================

std::vector<float> CPUBackend::forward(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[CPUBackend::forward] Not initialized");
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

    // 工作缓冲区
    std::vector<float> hidden(hiddenSize);
    std::vector<float> residual(hiddenSize);
    std::vector<float> normOut(hiddenSize);
    std::vector<float> q(qSize);
    std::vector<float> k(kvSize);
    std::vector<float> v(kvSize);
    std::vector<float> attnOut(qSize);
    std::vector<float> oOut(hiddenSize);
    std::vector<float> gate(intermediateSize);
    std::vector<float> up(intermediateSize);
    std::vector<float> ffnOut(hiddenSize);
    std::vector<float> attnScores(MAX_SEQ_LEN);

    // 1. Embedding
    const float* embedData = weightsCached_["embed_tokens"].data();
    for (int i = 0; i < hiddenSize; ++i) {
        hidden[i] = embedData[i * vocabSize + tokenId];
    }

    // 2. Transformer Layers
    for (int l = 0; l < numLayers; ++l) {
        // 获取层权重
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

        // Q/K Norm (Qwen3)
        if (qNormW && kNormW) {
            for (int h = 0; h < nHeads; ++h) {
                cpuRmsNorm(q.data() + h * headDim, qNormW, q.data() + h * headDim, headDim, eps);
            }
            for (int h = 0; h < nKVHeads; ++h) {
                cpuRmsNorm(k.data() + h * headDim, kNormW, k.data() + h * headDim, headDim, eps);
            }
        }

        // RoPE
        cpuApplyRoPE(q.data(), nHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());
        cpuApplyRoPE(k.data(), nKVHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());

        // 更新 KV Cache
        float* kCachePtr = kCache_[l].data() + position * kvSize;
        float* vCachePtr = vCache_[l].data() + position * kvSize;
        std::copy(k.begin(), k.end(), kCachePtr);
        std::copy(v.begin(), v.end(), vCachePtr);

        // Attention
        for (int h = 0; h < nHeads; ++h) {
            int kvHead = h / gqa;

            // 计算 Q @ K^T
            for (int t = 0; t < totalLen; ++t) {
                float dot = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    float qVal = q[h * headDim + d];
                    float kVal = kCache_[l][t * kvSize + kvHead * headDim + d];
                    dot += qVal * kVal;
                }
                attnScores[t] = dot / std::sqrt(static_cast<float>(headDim));
            }

            // Softmax
            cpuSoftmax(attnScores.data(), totalLen);

            // 计算 Attention @ V
            for (int d = 0; d < headDim; ++d) {
                float sum = 0.0f;
                for (int t = 0; t < totalLen; ++t) {
                    float vVal = vCache_[l][t * kvSize + kvHead * headDim + d];
                    sum += attnScores[t] * vVal;
                }
                attnOut[h * headDim + d] = sum;
            }
        }

        // O Projection
        cpuMatmul(oProjW, attnOut.data(), oOut.data(), hiddenSize, qSize);

        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + oOut[i];
        }

        // 保存残差
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

        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnOut[i];
        }
    }

    // 3. Final LayerNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    cpuRmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);

    // 4. LM Head
    std::vector<float> logits(vocabSize);
    const float* lmHeadW = weightsCached_["lm_head"].data();
    cpuMatmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);

    cachePosition_ = totalLen;
    return logits;
}

// ============================================================================
// KV Cache 管理
// ============================================================================

void CPUBackend::resetKVCache() {
    const int numLayers = config_.numHiddenLayers;
    const int nKVHeads = config_.getNumKVHeads();
    const int headDim = config_.getHeadDim();
    const int maxSeqLen = MAX_SEQ_LEN;
    const int kvSize = maxSeqLen * nKVHeads * headDim;

    for (int l = 0; l < numLayers; ++l) {
        std::fill(kCache_[l].begin(), kCache_[l].end(), 0.0f);
        std::fill(vCache_[l].begin(), vCache_[l].end(), 0.0f);
    }

    cachePosition_ = 0;
    CLLM_INFO("[CPUBackend] KV Cache reset");
}

// ============================================================================
// 类成员函数实现（供内部使用）
// ============================================================================

void CPUBackend::rmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    cpuRmsNorm(input, weight, output, size, eps);
}

void CPUBackend::matmul(const float* weight, const float* input, float* output, int M, int K) {
    cpuMatmul(weight, input, output, M, K);
}

void CPUBackend::applyRoPE(float* x, int nHeads, int headDim, int position) {
    cpuApplyRoPE(x, nHeads, headDim, position, ropeFreqsCos_.data(), ropeFreqsSin_.data());
}

void CPUBackend::softmax(float* x, int size) {
    cpuSoftmax(x, size);
}

float CPUBackend::silu(float x) {
    return cpuSilu(x);
}

} // namespace backend
} // namespace kylin
} // namespace cllm
