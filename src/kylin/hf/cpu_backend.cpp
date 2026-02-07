/**
 * @file cpu_backend.cpp
 * @brief CPU 后端实现 - 纯 CPU 推理
 * 
 * 包含完整的 Transformer CPU 计算：
 * - Embedding Lookup
 * - RMS Norm
 * - Q/K/V Projection
 * - RoPE (旋转位置编码)
 * - KV Cache 管理
 * - Attention (QK^T + Softmax + V)
 * - GQA (Grouped Query Attention)
 * - O Projection
 * - FFN (SwiGLU)
 * - LM Head
 */

#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <algorithm>

namespace cllm {
namespace kylin {

// ============================================================================
// CPU 辅助函数
// ============================================================================

/**
 * @brief CPU RMS Normalization
 */
static inline void cpuRmsNorm(const float* input, const float* weight, 
                               float* output, int size, float eps) {
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

/**
 * @brief CPU 矩阵乘法: output = weight @ input
 * weight: [M, K] 行主序
 * input: [K]
 * output: [M]
 */
static inline void cpuMatmul(const float* weight, const float* input, 
                              float* output, int M, int K) {
    ggml_kernels::matmul_f32(weight, input, output, M, K);
}

/**
 * @brief CPU RoPE (旋转位置编码)
 */
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

/**
 * @brief CPU SiLU 激活函数
 */
static inline float cpuSilu(float x) {
    return x / (1.0f + std::exp(-x));
}

/**
 * @brief CPU Softmax
 */
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
// CPU 后端主实现
// ============================================================================

std::vector<float> GGMLGPUBackend::forwardCPU(int tokenId, int position) {
    if (!initialized_) {
        CLLM_ERROR("[GGMLGPUBackend::forwardCPU] Not initialized");
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
    
    // 缓存权重到 CPU（首次调用）
    if (weightsCached_.empty()) {
        cacheWeightsToCPU();
    }
    
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
    
    // Attention 缓冲区
    const int maxSeqLen = 2048;
    std::vector<float> attnScores(maxSeqLen);
    
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
        cpuApplyRoPE(q.data(), nHeads, headDim, position, 
                     ropeFreqsCos_.data(), ropeFreqsSin_.data());
        cpuApplyRoPE(k.data(), nKVHeads, headDim, position, 
                     ropeFreqsCos_.data(), ropeFreqsSin_.data());
        
        // 更新 KV Cache
        float* kCacheLayer = kCacheCPU_[l].data();
        float* vCacheLayer = vCacheCPU_[l].data();
        std::copy(k.begin(), k.end(), kCacheLayer + position * kvSize);
        std::copy(v.begin(), v.end(), vCacheLayer + position * kvSize);
        
        // Attention
        const int totalLen = position + 1;
        std::fill(attnOut.begin(), attnOut.end(), 0.0f);
        
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q.data() + h * headDim;
            float* outHead = attnOut.data() + h * headDim;
            int kvHead = h / gqa;
            
            // 计算 attention scores
            for (int t = 0; t < totalLen; ++t) {
                float* kHead = kCacheLayer + t * kvSize + kvHead * headDim;
                float score = 0.0f;
                for (int d = 0; d < headDim; ++d) {
                    score += qHead[d] * kHead[d];
                }
                attnScores[t] = score / std::sqrt(static_cast<float>(headDim));
            }
            
            // Softmax
            cpuSoftmax(attnScores.data(), totalLen);
            
            // 加权求和
            for (int d = 0; d < headDim; ++d) {
                float sum = 0.0f;
                for (int t = 0; t < totalLen; ++t) {
                    float* vHead = vCacheLayer + t * kvSize + kvHead * headDim;
                    sum += attnScores[t] * vHead[d];
                }
                outHead[d] = sum;
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
        
        // Post Attention RMSNorm
        cpuRmsNorm(hidden.data(), postNormW, normOut.data(), hiddenSize, eps);
        
        // FFN (SwiGLU)
        cpuMatmul(gateProjW, normOut.data(), ffnGate.data(), intermediateSize, hiddenSize);
        cpuMatmul(upProjW, normOut.data(), ffnUp.data(), intermediateSize, hiddenSize);
        
        // SwiGLU: gate = silu(gate) * up
        for (int i = 0; i < intermediateSize; ++i) {
            ffnGate[i] = cpuSilu(ffnGate[i]) * ffnUp[i];
        }
        
        // Down projection
        cpuMatmul(downProjW, ffnGate.data(), ffnDown.data(), hiddenSize, intermediateSize);
        
        // 残差连接
        for (int i = 0; i < hiddenSize; ++i) {
            hidden[i] = residual[i] + ffnDown[i];
        }
    }
    
    // 3. Final RMSNorm
    const float* finalNormW = weightsCached_["final_norm"].data();
    cpuRmsNorm(hidden.data(), finalNormW, normOut.data(), hiddenSize, eps);
    
    // 4. LM Head
    std::vector<float> logits(vocabSize);
    const float* lmHeadW = weightsCached_["lm_head"].data();
    cpuMatmul(lmHeadW, normOut.data(), logits.data(), vocabSize, hiddenSize);
    
    return logits;
}

// cacheWeightsToCPU() 实现在 ggml_backend.cpp 中

} // namespace kylin
} // namespace cllm
