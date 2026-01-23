/**
 * @file hf_transformer.cpp
 * @brief HuggingFace Transformer 模型实现
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>

namespace cllm {
namespace kylin {

HFTransformerModel::HFTransformerModel(const std::string& modelDir) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    
    // 加载配置
    config_ = loadHFConfigFromDir(modelDir);
    if (!config_.isValid()) {
        CLLM_ERROR("[HFTransformer] Invalid model config");
        return;
    }
    config_.print();
    
    // 加载 safetensors
    std::string safetensorsPath = modelDir;
    if (safetensorsPath.back() != '/') safetensorsPath += '/';
    safetensorsPath += "model.safetensors";
    
    loader_ = std::make_unique<SafetensorsLoader>(safetensorsPath);
    if (!loader_->isValid()) {
        CLLM_ERROR("[HFTransformer] Failed to load safetensors");
        return;
    }
    
    // 加载权重
    if (!loadWeights()) {
        CLLM_ERROR("[HFTransformer] Failed to load weights");
        return;
    }
    
    // 预计算 RoPE 频率
    int headDim = config_.getHeadDim();
    ropeFreqsCos_.resize(kMaxSeqLen * headDim / 2);
    ropeFreqsSin_.resize(kMaxSeqLen * headDim / 2);
    
    for (int pos = 0; pos < kMaxSeqLen; ++pos) {
        for (int i = 0; i < headDim / 2; ++i) {
            float freq = 1.0f / std::pow(config_.ropeTheta, 2.0f * i / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * headDim / 2 + i] = std::cos(angle);
            ropeFreqsSin_[pos * headDim / 2 + i] = std::sin(angle);
        }
    }
    
    // 初始化 KV Cache
    int kvHeads = config_.getNumKVHeads();
    size_t kvSize = static_cast<size_t>(config_.numHiddenLayers) * kMaxSeqLen * kvHeads * headDim;
    kCache_.resize(kvSize, 0.0f);
    vCache_.resize(kvSize, 0.0f);
    
    // 分配工作缓冲区（预分配避免运行时分配）
    hiddenStates_.resize(config_.hiddenSize);
    residual_.resize(config_.hiddenSize);
    normOutput_.resize(config_.hiddenSize);
    attnOutput_.resize(config_.hiddenSize);
    ffnOutput_.resize(config_.hiddenSize);
    
    // QKV 缓冲区
    int qSize = config_.numAttentionHeads * headDim;
    int kvSize2 = kvHeads * headDim;
    qkvBuffer_.resize(qSize + 2 * kvSize2);
    
    // Attention 工作缓冲区（预分配）
    qBuffer_.resize(qSize);
    kBuffer_.resize(kvSize2);
    vBuffer_.resize(kvSize2);
    attnScores_.resize(kMaxSeqLen);
    attnOutBuffer_.resize(qSize);
    
    // FFN 工作缓冲区
    gateBuffer_.resize(config_.intermediateSize);
    upBuffer_.resize(config_.intermediateSize);
    
    // Norm 权重缓冲区（避免每层重复分配）
    normWeightBuffer_.resize(config_.hiddenSize);
    qkNormBuffer_.resize(headDim);
    
    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully (buffers pre-allocated)");
}

HFTransformerModel::~HFTransformerModel() = default;

bool HFTransformerModel::loadWeights() {
    // 加载嵌入层
    embedTokens_ = static_cast<const uint16_t*>(
        loader_->getTensorData("model.embed_tokens.weight"));
    if (!embedTokens_) {
        CLLM_ERROR("[HFTransformer] Missing embed_tokens.weight");
        return false;
    }
    
    // 加载 LM Head（可能与 embed_tokens 共享）
    if (config_.tieWordEmbeddings) {
        lmHeadWeight_ = embedTokens_;
        CLLM_INFO("[HFTransformer] LM head tied with embed_tokens");
    } else {
        lmHeadWeight_ = static_cast<const uint16_t*>(
            loader_->getTensorData("lm_head.weight"));
        if (!lmHeadWeight_) {
            CLLM_ERROR("[HFTransformer] Missing lm_head.weight");
            return false;
        }
    }
    
    // 加载最终 norm
    finalNormWeight_ = static_cast<const uint16_t*>(
        loader_->getTensorData("model.norm.weight"));
    if (!finalNormWeight_) {
        CLLM_ERROR("[HFTransformer] Missing model.norm.weight");
        return false;
    }
    
    // 加载每一层
    layers_.resize(config_.numHiddenLayers);
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        LayerWeights& layer = layers_[i];
        
        layer.inputLayernorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".input_layernorm.weight"));
        layer.qProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.q_proj.weight"));
        layer.kProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.k_proj.weight"));
        layer.vProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.v_proj.weight"));
        layer.oProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.o_proj.weight"));
        layer.qNorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.q_norm.weight"));
        layer.kNorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.k_norm.weight"));
        layer.postAttentionLayernorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".post_attention_layernorm.weight"));
        layer.gateProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.gate_proj.weight"));
        layer.upProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.up_proj.weight"));
        layer.downProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.down_proj.weight"));
        
        // 验证必需的权重
        if (!layer.inputLayernorm || !layer.qProj || !layer.kProj || !layer.vProj ||
            !layer.oProj || !layer.postAttentionLayernorm ||
            !layer.gateProj || !layer.upProj || !layer.downProj) {
            CLLM_ERROR("[HFTransformer] Missing weights for layer %d", i);
            return false;
        }
    }
    
    CLLM_INFO("[HFTransformer] All weights loaded");
    return true;
}

void HFTransformerModel::resetKVCache() {
    kvCacheLen_ = 0;
    std::fill(kCache_.begin(), kCache_.end(), 0.0f);
    std::fill(vCache_.begin(), vCache_.end(), 0.0f);
}

std::vector<float> HFTransformerModel::forward(const std::vector<int32_t>& inputIds) {
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    int seqLen = static_cast<int>(inputIds.size());
    int startPos = kvCacheLen_;
    
    CLLM_DEBUG("[HFTransformer] Forward: seq_len=%d, start_pos=%d", seqLen, startPos);
    
    // 目前只支持单 token 推理
    if (seqLen != 1) {
        CLLM_WARN("[HFTransformer] Currently only supports single token inference");
        // 可以扩展支持多 token
    }
    
    // Embedding
    embedding(inputIds, hiddenStates_);
    
    // 保存残差
    std::copy(hiddenStates_.begin(), hiddenStates_.end(), residual_.begin());
    
    // Transformer 层
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        // 1. RMS Norm (input_layernorm) - 使用预分配的缓冲区
        bf16ToF32Array(layers_[i].inputLayernorm, normWeightBuffer_.data(), config_.hiddenSize);
        rmsNorm(hiddenStates_.data(), normWeightBuffer_.data(), normOutput_.data(), 
                config_.hiddenSize, config_.rmsNormEps);
        
        // 2. Self-Attention
        attention(i, normOutput_.data(), attnOutput_.data(), seqLen, startPos);
        
        // 3. Residual Add（使用指针避免边界检查开销）
        float* hs = hiddenStates_.data();
        const float* res = residual_.data();
        const float* attn = attnOutput_.data();
        for (int j = 0; j < config_.hiddenSize; ++j) {
            hs[j] = res[j] + attn[j];
        }
        std::copy(hiddenStates_.begin(), hiddenStates_.end(), residual_.begin());
        
        // 4. RMS Norm (post_attention_layernorm)
        bf16ToF32Array(layers_[i].postAttentionLayernorm, normWeightBuffer_.data(), config_.hiddenSize);
        rmsNorm(hiddenStates_.data(), normWeightBuffer_.data(), normOutput_.data(),
                config_.hiddenSize, config_.rmsNormEps);
        
        // 5. FFN
        ffn(i, normOutput_.data(), ffnOutput_.data());
        
        // 6. Residual Add
        const float* ffnOut = ffnOutput_.data();
        for (int j = 0; j < config_.hiddenSize; ++j) {
            hs[j] = res[j] + ffnOut[j];
        }
        std::copy(hiddenStates_.begin(), hiddenStates_.end(), residual_.begin());
    }
    
    // Final RMS Norm - 使用预分配缓冲区
    bf16ToF32Array(finalNormWeight_, normWeightBuffer_.data(), config_.hiddenSize);
    rmsNorm(hiddenStates_.data(), normWeightBuffer_.data(), normOutput_.data(),
            config_.hiddenSize, config_.rmsNormEps);
    
    // LM Head
    std::vector<float> logits(config_.vocabSize);
    lmHead(normOutput_.data(), logits.data());
    
    // 更新 KV Cache 长度
    kvCacheLen_ += seqLen;
    
    return logits;
}

void HFTransformerModel::embedding(const std::vector<int32_t>& inputIds, 
                                   std::vector<float>& output) {
    // 只处理最后一个 token（或第一个，对于单 token 输入）
    int tokenId = inputIds.back();
    if (tokenId < 0 || tokenId >= config_.vocabSize) {
        CLLM_ERROR("[HFTransformer] Invalid token ID: %d", tokenId);
        std::fill(output.begin(), output.end(), 0.0f);
        return;
    }
    
    // 从 BF16 嵌入表中查找
    const uint16_t* embRow = embedTokens_ + tokenId * config_.hiddenSize;
    bf16ToF32Array(embRow, output.data(), config_.hiddenSize);
}

void HFTransformerModel::rmsNorm(const float* input, const float* weight, 
                                  float* output, int size, float eps) {
    // 计算 RMS
    float sumSq = 0.0f;
    for (int i = 0; i < size; ++i) {
        sumSq += input[i] * input[i];
    }
    float rms = std::sqrt(sumSq / size + eps);
    float scale = 1.0f / rms;
    
    // 归一化并乘以权重
    for (int i = 0; i < size; ++i) {
        output[i] = input[i] * scale * weight[i];
    }
}

void HFTransformerModel::attention(int layerIdx, const float* input, 
                                    float* output, int seqLen, int startPos) {
    const LayerWeights& layer = layers_[layerIdx];
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // Q, K, V 投影 - 使用预分配缓冲区
    float* q = qBuffer_.data();
    float* k = kBuffer_.data();
    float* v = vBuffer_.data();
    matmulBF16(layer.qProj, input, q, qSize, config_.hiddenSize);
    matmulBF16(layer.kProj, input, k, kvSize, config_.hiddenSize);
    matmulBF16(layer.vProj, input, v, kvSize, config_.hiddenSize);
    
    // Q/K Norm (Qwen3 特有) - 使用预分配缓冲区
    if (layer.qNorm && layer.kNorm) {
        bf16ToF32Array(layer.qNorm, qkNormBuffer_.data(), headDim);
        
        // 对每个 Q 头进行 RMS Norm（原地操作）
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q + h * headDim;
            // 计算 RMS
            float sumSq = 0.0f;
            for (int i = 0; i < headDim; ++i) {
                sumSq += qHead[i] * qHead[i];
            }
            float rms = std::sqrt(sumSq / headDim + config_.rmsNormEps);
            float invRms = 1.0f / rms;
            // 归一化并乘以权重
            for (int i = 0; i < headDim; ++i) {
                qHead[i] = qHead[i] * invRms * qkNormBuffer_[i];
            }
        }
        
        bf16ToF32Array(layer.kNorm, qkNormBuffer_.data(), headDim);
        // 对每个 K 头进行 RMS Norm（原地操作）
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k + h * headDim;
            float sumSq = 0.0f;
            for (int i = 0; i < headDim; ++i) {
                sumSq += kHead[i] * kHead[i];
            }
            float rms = std::sqrt(sumSq / headDim + config_.rmsNormEps);
            float invRms = 1.0f / rms;
            for (int i = 0; i < headDim; ++i) {
                kHead[i] = kHead[i] * invRms * qkNormBuffer_[i];
            }
        }
    }
    
    // RoPE
    applyRoPE(q, k, headDim, nHeads, nKVHeads, seqLen, startPos);
    
    // 存储 K, V 到 cache
    const int cacheOffset = layerIdx * kMaxSeqLen * nKVHeads * headDim + startPos * nKVHeads * headDim;
    std::copy(k, k + kvSize, kCache_.begin() + cacheOffset);
    std::copy(v, v + kvSize, vCache_.begin() + cacheOffset);
    
    // Attention 计算
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    const int totalLen = startPos + seqLen;
    
    // 使用预分配缓冲区
    float* attnOut = attnOutBuffer_.data();
    std::fill(attnOut, attnOut + qSize, 0.0f);
    float* scores = attnScores_.data();
    
    // GQA: 每 gqa 个 Q 头共享一个 KV 头
    const int gqa = nHeads / nKVHeads;
    const float* kCacheBase = kCache_.data() + layerIdx * kMaxSeqLen * nKVHeads * headDim;
    const float* vCacheBase = vCache_.data() + layerIdx * kMaxSeqLen * nKVHeads * headDim;
    
    for (int h = 0; h < nHeads; ++h) {
        const int kvHead = h / gqa;
        const float* qHead = q + h * headDim;
        
        // 计算 attention scores
        for (int t = 0; t < totalLen; ++t) {
            const float* kRow = kCacheBase + t * nKVHeads * headDim + kvHead * headDim;
            float dot = 0.0f;
            for (int d = 0; d < headDim; ++d) {
                dot += qHead[d] * kRow[d];
            }
            scores[t] = dot * scale;
        }
        
        // Softmax（数值稳定版本）
        float maxScore = scores[0];
        for (int t = 1; t < totalLen; ++t) {
            if (scores[t] > maxScore) maxScore = scores[t];
        }
        float sumExp = 0.0f;
        for (int t = 0; t < totalLen; ++t) {
            scores[t] = std::exp(scores[t] - maxScore);
            sumExp += scores[t];
        }
        const float invSum = 1.0f / sumExp;
        for (int t = 0; t < totalLen; ++t) {
            scores[t] *= invSum;
        }
        
        // Weighted sum of V
        float* outHead = attnOut + h * headDim;
        for (int t = 0; t < totalLen; ++t) {
            const float* vRow = vCacheBase + t * nKVHeads * headDim + kvHead * headDim;
            const float weight = scores[t];
            for (int d = 0; d < headDim; ++d) {
                outHead[d] += weight * vRow[d];
            }
        }
    }
    
    // O 投影
    matmulBF16(layer.oProj, attnOut, output, config_.hiddenSize, qSize);
}

void HFTransformerModel::ffn(int layerIdx, const float* input, float* output) {
    const LayerWeights& layer = layers_[layerIdx];
    const int intermediateSize = config_.intermediateSize;
    
    // Gate 和 Up 投影 - 使用预分配缓冲区
    float* gate = gateBuffer_.data();
    float* up = upBuffer_.data();
    matmulBF16(layer.gateProj, input, gate, intermediateSize, config_.hiddenSize);
    matmulBF16(layer.upProj, input, up, intermediateSize, config_.hiddenSize);
    
    // SwiGLU: silu(gate) * up
    // silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
    for (int i = 0; i < intermediateSize; ++i) {
        const float x = gate[i];
        // 避免 exp 溢出：对于 x > 20, sigmoid(x) ≈ 1
        const float sigmoid = (x > 20.0f) ? 1.0f : (x < -20.0f) ? 0.0f : 1.0f / (1.0f + std::exp(-x));
        gate[i] = x * sigmoid * up[i];
    }
    
    // Down 投影
    matmulBF16(layer.downProj, gate, output, config_.hiddenSize, intermediateSize);
}

void HFTransformerModel::lmHead(const float* input, float* output) {
    matmulBF16(lmHeadWeight_, input, output, config_.vocabSize, config_.hiddenSize);
}

void HFTransformerModel::applyRoPE(float* q, float* k, int headDim, 
                                    int nHeads, int nKVHeads, int seqLen, int startPos) {
    // 对每个位置应用 RoPE
    for (int pos = 0; pos < seqLen; ++pos) {
        int actualPos = startPos + pos;
        
        // Q 头
        for (int h = 0; h < nHeads; ++h) {
            float* head = q + h * headDim;
            for (int i = 0; i < headDim / 2; ++i) {
                float cos = ropeFreqsCos_[actualPos * headDim / 2 + i];
                float sin = ropeFreqsSin_[actualPos * headDim / 2 + i];
                float x0 = head[i];
                float x1 = head[i + headDim / 2];
                head[i] = x0 * cos - x1 * sin;
                head[i + headDim / 2] = x0 * sin + x1 * cos;
            }
        }
        
        // K 头
        for (int h = 0; h < nKVHeads; ++h) {
            float* head = k + h * headDim;
            for (int i = 0; i < headDim / 2; ++i) {
                float cos = ropeFreqsCos_[actualPos * headDim / 2 + i];
                float sin = ropeFreqsSin_[actualPos * headDim / 2 + i];
                float x0 = head[i];
                float x1 = head[i + headDim / 2];
                head[i] = x0 * cos - x1 * sin;
                head[i + headDim / 2] = x0 * sin + x1 * cos;
            }
        }
    }
}

void HFTransformerModel::matmulBF16(const uint16_t* weight, const float* input, 
                                     float* output, int outFeatures, int inFeatures,
                                     int batchSize) {
    // weight: [outFeatures, inFeatures] in row-major (BF16)
    // input: [inFeatures] (F32)
    // output: [outFeatures] (F32)
    
    for (int i = 0; i < outFeatures; ++i) {
        float sum = 0.0f;
        const uint16_t* row = weight + i * inFeatures;
        for (int j = 0; j < inFeatures; ++j) {
            float w = bf16ToF32(row[j]);
            sum += w * input[j];
        }
        output[i] = sum;
    }
}

} // namespace kylin
} // namespace cllm
