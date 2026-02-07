/**
 * @file cpu_backend.cpp
 * @brief CPU 后端实现 - 包装 GGMLGPUBackend::forwardCPU
 * 
 * 这个文件作为 backend/cpu 目录的入口点，
 * 实际实现委托给 hf/ggml_backend.cpp 中的 GGMLGPUBackend::forwardCPU。
 */

#include "cllm/kylin/backend/cpu/cpu_backend.h"
#include "cllm/kylin/hf/ggml_backend.h"
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
    const int maxSeqLen = 2048;
    
    kCache_.resize(numLayers);
    vCache_.resize(numLayers);
    
    for (int l = 0; l < numLayers; ++l) {
        kCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
        vCache_[l].resize(maxSeqLen * nKVHeads * headDim, 0.0f);
    }
    
    // 预计算 RoPE
    initRoPE();
    
    initialized_ = true;
    CLLM_INFO("[CPUBackend] CPU Backend initialized (layers=%d)", numLayers);
    return true;
}

void CPUBackend::initRoPE() {
    const int headDim = config_.getHeadDim();
    const int halfDim = headDim / 2;
    const float ropeTheta = config_.ropeTheta;
    const int maxSeqLen = 2048;
    
    ropeFreqsCos_.resize(maxSeqLen * halfDim);
    ropeFreqsSin_.resize(maxSeqLen * halfDim);
    
    for (int pos = 0; pos < maxSeqLen; ++pos) {
        for (int i = 0; i < halfDim; ++i) {
            float freq = 1.0f / std::pow(ropeTheta, 2.0f * i / headDim);
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
    CLLM_INFO("[CPUBackend] Weights cached: %zu tensors", weightsCached_.size());
    return true;
}

// ============================================================================
// Forward 实现
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
    std::vector<float> q(qSize);
    std::vector<float> k(kvSize);
    std::vector<float> v(kvSize);
    std::vector<float> attnOut(qSize);
    std::vector<float> oOut(hiddenSize);
    std::vector<float> ffnGate(intermediateSize);
    std::vector<float> ffnUp(intermediateSize);
    std::vector<float> ffnDown(hiddenSize);
    
    const int maxSeqLen = 2048;
    std::vector<float> attnScores(maxSeqLen);
    
    // 1. Embedding
    auto embedIt = weightsCached_.find("embed_tokens");
    if (embedIt == weightsCached_.end()) {
        CLLM_ERROR("[CPUBackend] embed_tokens not found");
        return {};
    }
    const float* embedData = embedIt->second.data();
    for (int i = 0; i < hiddenSize; ++i) {
        hidden[i] = embedData[i * vocabSize + tokenId];
    }
    
    // 2. Transformer Layers
    for (int l = 0; l < numLayers; ++l) {
        // 获取层权重
        auto getWeight = [&](const std::string& name) -> const float* {
            auto it = weightsCached_.find(name);
            if (it == weightsCached_.end()) {
                CLLM_ERROR("[CPUBackend] Weight not found: %s", name.c_str());
                return nullptr;
            }
            return it->second.data();
        };
        
        std::string prefix = "layer." + std::to_string(l) + ".";
        const float* inputNormW = getWeight(prefix + "input_layernorm");
        const float* qProjW = getWeight(prefix + "q_proj");
        const float* kProjW = getWeight(prefix + "k_proj");
        const float* vProjW = getWeight(prefix + "v_proj");
        const float* oProjW = getWeight(prefix + "o_proj");
        const float* qNormW = getWeight(prefix + "q_norm");
        const float* kNormW = getWeight(prefix + "k_norm");
        const float* postNormW = getWeight(prefix + "post_attention_layernorm");
        const float* gateProjW = getWeight(prefix + "gate_proj");
        const float* upProjW = getWeight(prefix + "up_proj");
        const float* downProjW = getWeight(prefix + "down_proj");
        
        if (!inputNormW || !qProjW || !kProjW || !vProjW || !oProjW) {
            CLLM_ERROR("[CPUBackend] Missing weights for layer %d", l);
            return {};
        }
        
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
        
        // 更新 KV Cache
        std::copy(k.begin(), k.end(), kCache_[l].begin() + position * kvSize);
        std::copy(v.begin(), v.end(), vCache_[l].begin() + position * kvSize);
        
        // Attention
        const int totalLen = position + 1;
        std::fill(attnOut.begin(), attnOut.end(), 0.0f);
        
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q.data() + h * headDim;
            float* outHead = attnOut.data() + h * headDim;
            int kvHead = h / gqa;
            
            // 计算 attention scores
            for (int t = 0; t < totalLen; ++t) {
                float* kHead = kCache_[l].data() + t * kvSize + kvHead * headDim;
                float score = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    score += qHead[d] * kHead[d];
                }
                attnScores[t] = score / std::sqrt(static_cast<float>(headDim));
            }
            
            // Softmax
            softmax(attnScores.data(), totalLen);
            
            // 加权求和
            for (int d = 0; d < headDim; ++d) {
                float sum = 0.0f;
                for (int t = 0; t < totalLen; ++t) {
                    float* vHead = vCache_[l].data() + t * kvSize + kvHead * headDim;
                    sum += attnScores[t] * vHead[d];
                }
                outHead[d] = sum;
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
        
        // FFN (SwiGLU)
        matmul(gateProjW, normOut.data(), ffnGate.data(), intermediateSize, hiddenSize);
        matmul(upProjW, normOut.data(), ffnUp.data(), intermediateSize, hiddenSize);
        
        // SwiGLU: gate = silu(gate) * up
        for (int i = 0; i < intermediateSize; ++i) {
            ffnGate[i] = silu(ffnGate[i]) * ffnUp[i];
        }
        
        // Down projection
        matmul(downProjW, ffnGate.data(), ffnDown.data(), hiddenSize, intermediateSize);
        
        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnDown[i];
        }
    }
    
    // 3. Final RMSNorm
    auto finalNormIt = weightsCached_.find("final_norm");
    if (finalNormIt == weightsCached_.end()) {
        CLLM_ERROR("[CPUBackend] final_norm not found");
        return {};
    }
    const float* finalNormW = finalNormIt->second.data();
    rmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);
    
    // 4. LM Head
    auto lmHeadIt = weightsCached_.find("lm_head");
    if (lmHeadIt == weightsCached_.end()) {
        CLLM_ERROR("[CPUBackend] lm_head not found");
        return {};
    }
    const float* lmHeadW = lmHeadIt->second.data();
    std::vector<float> logits(vocabSize);
    matmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    
    cachePosition_ = position + 1;
    return logits;
}

// ============================================================================
// 辅助函数
// ============================================================================

void CPUBackend::rmsNorm(const float* input, const float* weight, float* output, int size, float eps) {
    float sum = 0.0f;
    for (int i = 0; i < size; ++i) {
        sum += input[i] * input[i];
    }
    float scale = 1.0f / std::sqrt(sum / size + eps);
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

void CPUBackend::matmul(const float* weight, const float* input, float* output, int M, int K) {
    for (int m = 0; m < M; ++m) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += weight[m * K + k] * input[k];
        }
        output[m] = sum;
    }
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
