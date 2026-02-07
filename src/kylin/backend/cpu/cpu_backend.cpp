/**
 * @file cpu_backend.cpp
 * @brief CPU 后端实现 - 纯 CPU 推理
 * 
 * 包含完整的 Transformer CPU 计算实现。
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

// ============================================================================
// 构造函数和析构函数
// ============================================================================

CPUBackend::CPUBackend() = default;

CPUBackend::~CPUBackend() {
    // 清理权重缓存
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
    const int maxSeqLen = 2048; // 最大序列长度
    
    kCache_.resize(numLayers);
    vCache_.resize(numLayers);
    
    for (int l = 0; l < numLayers; ++l) {
        kCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
        vCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
    }
    
    // 初始化 RoPE
    initRoPE();
    
    initialized_ = true;
    CLLM_INFO("[CPUBackend] Initialized successfully");
    return true;
}

void CPUBackend::initRoPE() {
    const int headDim = config_.getHeadDim();
    const int halfDim = headDim / 2;
    const int maxSeqLen = 2048;
    const float theta = 10000.0f;
    
    ropeFreqsCos_.resize(maxSeqLen * halfDim);
    ropeFreqsSin_.resize(maxSeqLen * halfDim);
    
    for (int pos = 0; pos < maxSeqLen; ++pos) {
        for (int i = 0; i < halfDim; ++i) {
            float freq = 1.0f / std::pow(theta, (2.0f * i) / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * halfDim + i] = std::cos(angle);
            ropeFreqsSin_[pos * halfDim + i] = std::sin(angle);
        }
    }
}

// ============================================================================
// 权重管理
// ============================================================================

bool CPUBackend::uploadWeights(const std::unordered_map<std::string, std::vector<float>>& weightsMap) {
    weightsCached_ = weightsMap;
    CLLM_INFO("[CPUBackend] Uploaded %zu weights to CPU", weightsMap.size());
    return true;
}

// ============================================================================
// 辅助函数
// ============================================================================

void CPUBackend::rmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

void CPUBackend::matmul(const float* weight, const float* input, float* output, int M, int K) {
    ggml_kernels::matmul_f32(weight, input, output, M, K);
}

void CPUBackend::applyRoPE(float* x, int nHeads, int headDim, int position) {
    const int halfDim = headDim / 2;
    const float* cosPtr = ropeFreqsCos_.data() + position * halfDim;
    const float* sinPtr = ropeFreqsSin_.data() + position * halfDim;
    
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

void CPUBackend::softmax(float* x, int size) {
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

float CPUBackend::silu(float x) {
    return x / (1.0f + std::exp(-x));
}

// ============================================================================
// 前向传播
// ============================================================================

std::vector<float> CPUBackend::forward(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[CPUBackend] Not initialized");
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
    const float eps = config_.rmsNormEps;
    
    // 工作缓冲区
    std::vector<float> hidden(hiddenSize);
    std::vector<float> residual(hiddenSize);
    std::vector<float> normOut(hiddenSize);
    std::vector<float> q(qSize), k(kvSize), v(kvSize);
    std::vector<float> attnOut(qSize);
    std::vector<float> oOut(hiddenSize);
    std::vector<float> ffnGate(intermediateSize);
    std::vector<float> ffnUp(intermediateSize);
    std::vector<float> ffnDown(hiddenSize);
    
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
        
        // Input RMSNorm
        rmsNorm(hidden.data(), inputNormW, normOut.data(), hiddenSize, eps);
        
        // Q, K, V Projections
        matmul(qProjW, normOut.data(), q.data(), qSize, hiddenSize);
        matmul(kProjW, normOut.data(), k.data(), kvSize, hiddenSize);
        matmul(vProjW, normOut.data(), v.data(), kvSize, hiddenSize);
        
        // Q/K Norm (Qwen3)
        if (qNormW && kNormW) {
            for (int h = 0; h < nHeads; ++h) {
                rmsNorm(q.data() + h * headDim, qNormW, q.data() + h * headDim, headDim, eps);
            }
            for (int h = 0; h < nKVHeads; ++h) {
                rmsNorm(k.data() + h * headDim, kNormW, k.data() + h * headDim, headDim, eps);
            }
        }
        
        // RoPE
        applyRoPE(q.data(), nHeads, headDim, position);
        applyRoPE(k.data(), nKVHeads, headDim, position);
        
        // 存储到 KV Cache
        float* kCachePtr = kCache_[l].data() + position * kvSize;
        float* vCachePtr = vCache_[l].data() + position * kvSize;
        std::memcpy(kCachePtr, k.data(), kvSize * sizeof(float));
        std::memcpy(vCachePtr, v.data(), kvSize * sizeof(float));
        
        // Attention
        for (int h = 0; h < nHeads; ++h) {
            int kvHead = h / gqa;
            
            // QK^T
            std::vector<float> scores(position + 1);
            for (int pos = 0; pos <= position; ++pos) {
                float* kCachePos = kCache_[l].data() + pos * kvSize + kvHead * headDim;
                float dot = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    dot += q[h * headDim + d] * kCachePos[d];
                }
                scores[pos] = dot / std::sqrt(static_cast<float>(headDim));
            }
            
            // Softmax
            softmax(scores.data(), position + 1);
            
            // V
            for (int d = 0; d < headDim; ++d) {
                float sum = 0.0f;
                for (int pos = 0; pos <= position; ++pos) {
                    float* vCachePos = vCache_[l].data() + pos * kvSize + kvHead * headDim;
                    sum += scores[pos] * vCachePos[d];
                }
                attnOut[h * headDim + d] = sum;
            }
        }
        
        // O Projection
        matmul(oProjW, attnOut.data(), oOut.data(), hiddenSize, qSize);
        
        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + oOut[i];
        }
        
        // 保存残差
        std::copy(hidden.begin(), hidden.end(), residual.begin());
        
        // Post Attention RMSNorm
        rmsNorm(hidden.data(), postNormW, normOut.data(), hiddenSize, eps);
        
        // FFN Gate & Up
        matmul(gateProjW, normOut.data(), ffnGate.data(), intermediateSize, hiddenSize);
        matmul(upProjW, normOut.data(), ffnUp.data(), intermediateSize, hiddenSize);
        
        // SwiGLU
        for (int i = 0; i < intermediateSize; ++i) {
            ffnGate[i] = silu(ffnGate[i]) * ffnUp[i];
        }
        
        // FFN Down
        matmul(downProjW, ffnGate.data(), ffnDown.data(), hiddenSize, intermediateSize);
        
        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnDown[i];
        }
    }
    
    // 3. Final RMSNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    rmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);
    
    // 4. LM Head
    std::vector<float> logits(vocabSize);
    const float* lmHeadW = weightsCached_["lm_head"].data();
    matmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    
    return logits;
}

// ============================================================================
// KV Cache 管理
// ============================================================================

void CPUBackend::resetKVCache() {
    cachePosition_ = 0;
    for (auto& cache : kCache_) {
        std::fill(cache.begin(), cache.end(), 0.0f);
    }
    for (auto& cache : vCache_) {
        std::fill(cache.begin(), cache.end(), 0.0f);
    }
    CLLM_INFO("[CPUBackend] KV Cache reset");
}

} // namespace backend
} // namespace kylin
} // namespace cllm
