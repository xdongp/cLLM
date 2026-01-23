/**
 * @file ggml_transformer.cpp
 * @brief 基于 GGML 的 Transformer 模型实现
 * 
 * 重构版本：使用独立的 tensor_stats、kv_cache_ops、attention_graph 模块
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <algorithm>

namespace cllm {
namespace kylin {

// ========== 构造与析构 ==========

GGMLTransformerModel::GGMLTransformerModel(BackendType backend)
    : loaded_(false)
    , tokEmbed_(nullptr)
    , outputNorm_(nullptr)
    , output_(nullptr)
    , kvCacheLen_(0)
    , maxKVCacheLen_(0)
    , pendingStartPos_(0)
    , pendingSeqLen_(0)
    , backendType_(backend)
    , debugEmbedding_(nullptr)
    , debugLayer0Output_(nullptr)
    , debugFinalNorm_(nullptr)
    , debugLogits_(nullptr) {
    
    debugLayerOutputs_.reserve(32);
    CLLM_DEBUG("[GGMLTransformerModel] Created with backend type: %d", static_cast<int>(backend));
}

GGMLTransformerModel::~GGMLTransformerModel() = default;

// ========== 模型加载 ==========

bool GGMLTransformerModel::loadFromGGUF(const std::string& path) {
    CLLM_INFO("[GGMLTransformerModel] Loading model from: %s", path.c_str());
    
    try {
        loader_ = std::make_unique<GGUFLoader>(path);
        if (!loader_->isValid()) {
            CLLM_ERROR("[GGMLTransformerModel] Invalid GGUF file");
            return false;
        }
        
        config_ = loader_->loadConfig();
        if (!config_.isValid()) {
            CLLM_ERROR("[GGMLTransformerModel] Invalid model config");
            return false;
        }
        
        CLLM_INFO("[GGMLTransformerModel] Config: arch=%s, layers=%u, hidden=%u, heads=%u/%u",
                  config_.architecture.c_str(), config_.blockCount, config_.embeddingLength,
                  config_.headCount, config_.headCountKV);
        
        // 计算内存需求
        size_t weightMemSize = std::max(static_cast<size_t>(loader_->getTensorCount()) * 
                               config_.embeddingLength * config_.embeddingLength * 4,
                               static_cast<size_t>(2ULL * 1024 * 1024 * 1024));
        
        size_t kvCacheMaxSeq = std::min(config_.contextLength, static_cast<uint32_t>(2048));
        size_t kvCacheMemSize = static_cast<size_t>(2 * config_.blockCount * config_.headCountKV *
                                kvCacheMaxSeq * config_.headDim() * sizeof(float) * 1.2);
        kvCacheMemSize = std::max(kvCacheMemSize, static_cast<size_t>(256 * 1024 * 1024));
        
        size_t computeMemSize = 512 * 1024 * 1024;
        
        CLLM_INFO("[GGMLTransformerModel] Memory: weight=%zuMB, kvcache=%zuMB, compute=%zuMB",
                  weightMemSize / (1024 * 1024), kvCacheMemSize / (1024 * 1024), 
                  computeMemSize / (1024 * 1024));
        
        weightCtx_ = std::make_unique<GGMLContext>(weightMemSize, backendType_);
        kvCacheCtx_ = std::make_unique<GGMLContext>(kvCacheMemSize, backendType_);
        computeCtx_ = std::make_unique<GGMLContext>(computeMemSize, backendType_);
        
        std::map<std::string, ggml_tensor*> tensors;
        loader_->loadTensors(weightCtx_.get(), tensors);
        CLLM_INFO("[GGMLTransformerModel] Loaded %zu tensors", tensors.size());
        
        mapWeights();
        allocateKVCache();
        
        loaded_ = true;
        CLLM_INFO("[GGMLTransformerModel] Model loaded successfully");
        return true;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("[GGMLTransformerModel] Load failed: %s", e.what());
        return false;
    }
}

void GGMLTransformerModel::mapWeights() {
    CLLM_INFO("[GGMLTransformerModel] Mapping weights...");
    
    // Token embedding
    tokEmbed_ = findWeight("token_embd.weight");
    if (!tokEmbed_) tokEmbed_ = findWeight("model.embed_tokens.weight");
    if (!tokEmbed_) throw std::runtime_error("Token embedding not found");
    
    // Output norm
    outputNorm_ = findWeight("output_norm.weight");
    if (!outputNorm_) outputNorm_ = findWeight("model.norm.weight");
    if (!outputNorm_) throw std::runtime_error("Output norm not found");
    
    // LM Head
    output_ = findWeight("output.weight");
    if (!output_) output_ = findWeight("lm_head.weight");
    if (!output_) {
        CLLM_INFO("[GGMLTransformerModel] Output not found, using tied embedding");
        output_ = tokEmbed_;
    }
    
    // 调试：打印关键张量信息
    CLLM_INFO("[GGMLTransformerModel] tokEmbed_: type=%s, shape=[%lld, %lld]",
              GGMLContext::typeToString(tokEmbed_->type).c_str(), tokEmbed_->ne[0], tokEmbed_->ne[1]);
    CLLM_INFO("[GGMLTransformerModel] outputNorm_: type=%s, shape=[%lld]",
              GGMLContext::typeToString(outputNorm_->type).c_str(), outputNorm_->ne[0]);
    CLLM_INFO("[GGMLTransformerModel] output_: type=%s, shape=[%lld, %lld], same_as_embed=%d",
              GGMLContext::typeToString(output_->type).c_str(), output_->ne[0], output_->ne[1],
              (output_ == tokEmbed_) ? 1 : 0);
    
    // 每层权重
    layers_.resize(config_.blockCount);
    
    for (uint32_t i = 0; i < config_.blockCount; ++i) {
        LayerWeights& l = layers_[i];
        std::string pfx = "blk." + std::to_string(i);
        std::string altPfx = "model.layers." + std::to_string(i);
        
        l.attnNorm = findWeight(pfx + ".attn_norm.weight");
        if (!l.attnNorm) l.attnNorm = findWeight(altPfx + ".input_layernorm.weight");
        
        l.ffnNorm = findWeight(pfx + ".ffn_norm.weight");
        if (!l.ffnNorm) l.ffnNorm = findWeight(altPfx + ".post_attention_layernorm.weight");
        
        l.wq = findWeight(pfx + ".attn_q.weight");
        if (!l.wq) l.wq = findWeight(altPfx + ".self_attn.q_proj.weight");
        
        l.wk = findWeight(pfx + ".attn_k.weight");
        if (!l.wk) l.wk = findWeight(altPfx + ".self_attn.k_proj.weight");
        
        l.wv = findWeight(pfx + ".attn_v.weight");
        if (!l.wv) l.wv = findWeight(altPfx + ".self_attn.v_proj.weight");
        
        l.wo = findWeight(pfx + ".attn_output.weight");
        if (!l.wo) l.wo = findWeight(altPfx + ".self_attn.o_proj.weight");
        
        l.attnQNorm = findWeight(pfx + ".attn_q_norm.weight");
        l.attnKNorm = findWeight(pfx + ".attn_k_norm.weight");
        
        l.wGate = findWeight(pfx + ".ffn_gate.weight");
        if (!l.wGate) l.wGate = findWeight(altPfx + ".mlp.gate_proj.weight");
        
        l.wUp = findWeight(pfx + ".ffn_up.weight");
        if (!l.wUp) l.wUp = findWeight(altPfx + ".mlp.up_proj.weight");
        
        l.wDown = findWeight(pfx + ".ffn_down.weight");
        if (!l.wDown) l.wDown = findWeight(altPfx + ".mlp.down_proj.weight");
        
        if (!l.attnNorm || !l.ffnNorm || !l.wq || !l.wk || !l.wv || !l.wo ||
            !l.wGate || !l.wUp || !l.wDown) {
            throw std::runtime_error("Missing weights for layer " + std::to_string(i));
        }
    }
    
    // 打印 Layer 0 权重维度
    if (!layers_.empty() && layers_[0].wq && layers_[0].wGate) {
        const auto& l0 = layers_[0];
        CLLM_INFO("[GGMLTransformerModel] Layer0 wq: shape=[%lld, %lld]", l0.wq->ne[0], l0.wq->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wk: shape=[%lld, %lld]", l0.wk->ne[0], l0.wk->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wv: shape=[%lld, %lld]", l0.wv->ne[0], l0.wv->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wo: shape=[%lld, %lld]", l0.wo->ne[0], l0.wo->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wGate: shape=[%lld, %lld]", l0.wGate->ne[0], l0.wGate->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wUp: shape=[%lld, %lld]", l0.wUp->ne[0], l0.wUp->ne[1]);
        CLLM_INFO("[GGMLTransformerModel] Layer0 wDown: shape=[%lld, %lld]", l0.wDown->ne[0], l0.wDown->ne[1]);
        if (l0.attnQNorm) {
            CLLM_INFO("[GGMLTransformerModel] Layer0 attnQNorm: shape=[%lld]", l0.attnQNorm->ne[0]);
        }
        if (l0.attnKNorm) {
            CLLM_INFO("[GGMLTransformerModel] Layer0 attnKNorm: shape=[%lld]", l0.attnKNorm->ne[0]);
        }
    }
    
    CLLM_INFO("[GGMLTransformerModel] All weights mapped");
}

ggml_tensor* GGMLTransformerModel::findWeight(const std::string& name) {
    return weightCtx_ ? ggml_get_tensor(weightCtx_->raw(), name.c_str()) : nullptr;
}

void GGMLTransformerModel::allocateKVCache() {
    ggml_context* ctx = kvCacheCtx_->raw();
    const size_t nLayers = config_.blockCount;
    const size_t nKVHeads = config_.headCountKV;
    const size_t headDim = config_.headDim();
    const size_t maxSeq = std::min(config_.contextLength, static_cast<uint32_t>(2048));
    
    kCaches_.resize(nLayers);
    vCaches_.resize(nLayers);
    pendingK_.resize(nLayers, nullptr);
    pendingV_.resize(nLayers, nullptr);
    
    for (size_t i = 0; i < nLayers; ++i) {
        kCaches_[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, maxSeq, nKVHeads);
        vCaches_[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, headDim, maxSeq, nKVHeads);
        ggml_set_name(kCaches_[i], ("k_cache_" + std::to_string(i)).c_str());
        ggml_set_name(vCaches_[i], ("v_cache_" + std::to_string(i)).c_str());
        ggml_set_zero(kCaches_[i]);
        ggml_set_zero(vCaches_[i]);
    }
    
    kvCacheLen_ = 0;
    maxKVCacheLen_ = maxSeq;
    CLLM_INFO("[GGMLTransformerModel] KV cache: %zu layers, %zu heads, %zu dim, %zu max_seq",
              nLayers, nKVHeads, headDim, maxSeq);
}

void GGMLTransformerModel::flushKVCache() {
    if (pendingSeqLen_ == 0) return;
    
    const size_t nLayers = config_.blockCount;
    const size_t headDim = config_.headDim();
    const size_t nKVHeads = config_.headCountKV;
    
    for (size_t i = 0; i < nLayers; ++i) {
        if (!pendingK_[i] || !pendingV_[i] || !pendingK_[i]->data || !pendingV_[i]->data)
            continue;
        
        const size_t srcNb1 = pendingK_[i]->nb[1];
        const size_t srcNb2 = pendingK_[i]->nb[2];
        const size_t dstNb1 = kCaches_[i]->nb[1];
        const size_t dstNb2 = kCaches_[i]->nb[2];
        
        const size_t bytesToCopy = pendingSeqLen_ * headDim * sizeof(float);
        
        for (size_t h = 0; h < nKVHeads; ++h) {
            const char* srcK = static_cast<const char*>(pendingK_[i]->data) + h * srcNb2;
            const char* srcV = static_cast<const char*>(pendingV_[i]->data) + h * srcNb2;
            char* dstK = static_cast<char*>(kCaches_[i]->data) + pendingStartPos_ * dstNb1 + h * dstNb2;
            char* dstV = static_cast<char*>(vCaches_[i]->data) + pendingStartPos_ * dstNb1 + h * dstNb2;
            std::memcpy(dstK, srcK, bytesToCopy);
            std::memcpy(dstV, srcV, bytesToCopy);
        }
        
        pendingK_[i] = nullptr;
        pendingV_[i] = nullptr;
    }
    
    pendingSeqLen_ = 0;
}

// ========== 推理接口 ==========

std::vector<float> GGMLTransformerModel::forward(const std::vector<int32_t>& inputIds) {
    if (!loaded_) throw std::runtime_error("Model not loaded");
    if (inputIds.empty()) throw std::invalid_argument("Empty input");
    
    const size_t seqLen = inputIds.size();
    const size_t vocabSize = config_.vocabSize;
    
    CLLM_DEBUG("[Forward] seq_len=%zu, start_pos=%zu", seqLen, kvCacheLen_);
    
    debugLayerOutputs_.clear();
    computeCtx_->reset();
    
    pendingStartPos_ = kvCacheLen_;
    pendingSeqLen_ = seqLen;
    
    ggml_tensor* logits = buildForwardGraph(inputIds, kvCacheLen_);
    ggml_cgraph* graph = computeCtx_->buildGraph(logits);
    computeCtx_->compute(graph);
    
    // 调试输出
    if (debugEmbedding_) {
        TensorStats stats;
        if (safeComputeTensorStats(debugEmbedding_, stats)) {
            printTensorStats("Embedding", debugEmbedding_, stats);
            CLLM_INFO("[Debug] Embedding: min=%.4f, max=%.4f, mean=%.4f, std=%.4f",
                      stats.minVal, stats.maxVal, stats.mean, stats.stddev);
        }
    }
    
    // 打印所有层输出的统计信息（用于追踪累积误差）
    float prevStd = 0.0f;
    for (size_t i = 0; i < debugLayerOutputs_.size(); ++i) {
        TensorStats stats;
        if (safeComputeTensorStats(debugLayerOutputs_[i], stats)) {
            float stdRatio = (prevStd > 0.01f) ? (stats.stddev / prevStd) : 1.0f;
            // 只打印前 5 层、最后 5 层、或 std 突然变化的层
            if (i < 5 || i >= debugLayerOutputs_.size() - 5 || stdRatio > 1.5f || stdRatio < 0.5f) {
                CLLM_INFO("[Debug] Layer %zu: min=%.2f, max=%.2f, mean=%.4f, std=%.4f (ratio=%.2f)",
                          i, stats.minVal, stats.maxVal, stats.mean, stats.stddev, stdRatio);
            }
            prevStd = stats.stddev;
        }
    }
    
    // Layer 27 详细调试
    if (debugLayer27AttnInput_) {
        TensorStats attnInStats, attnOutStats, ffnInStats, ffnOutStats;
        safeComputeTensorStats(debugLayer27AttnInput_, attnInStats);
        safeComputeTensorStats(debugLayer27AttnOutput_, attnOutStats);
        safeComputeTensorStats(debugLayer27FfnInput_, ffnInStats);
        safeComputeTensorStats(debugLayer27FfnOutput_, ffnOutStats);
        
        CLLM_INFO("[Layer 27 Detail] Attn Input:  min=%.2f, max=%.2f, std=%.2f", 
                  attnInStats.minVal, attnInStats.maxVal, attnInStats.stddev);
        CLLM_INFO("[Layer 27 Detail] Attn Output: min=%.2f, max=%.2f, std=%.2f", 
                  attnOutStats.minVal, attnOutStats.maxVal, attnOutStats.stddev);
        CLLM_INFO("[Layer 27 Detail] FFN Input:   min=%.2f, max=%.2f, std=%.2f", 
                  ffnInStats.minVal, ffnInStats.maxVal, ffnInStats.stddev);
        CLLM_INFO("[Layer 27 Detail] FFN Output:  min=%.2f, max=%.2f, std=%.2f", 
                  ffnOutStats.minVal, ffnOutStats.maxVal, ffnOutStats.stddev);
    }
    
    flushKVCache();
    kvCacheLen_ += seqLen;
    
    // 提取 logits
    // 注意：GGML 是 column-major 存储，需要正确处理内存布局
    std::vector<float> result(seqLen * vocabSize);
    const size_t dim0 = logits->ne[0];
    const size_t dim1 = logits->ne[1];
    const float* src = static_cast<const float*>(logits->data);
    
    if (seqLen == 1 && dim0 == vocabSize) {
        // 单 token：logits 形状 [vocab, 1]，直接复制
        memcpy(result.data(), logits->data, vocabSize * sizeof(float));
    } else if (dim0 == seqLen && dim1 == vocabSize) {
        // transpose 后的批量推理：logits 形状 [seqLen, vocab]
        // GGML column-major: element[s,v] 在 offset s + v*seqLen
        // 我们需要：result[s*vocab + v] = element[s,v]
        for (size_t s = 0; s < seqLen; ++s) {
            for (size_t v = 0; v < vocabSize; ++v) {
                result[s * vocabSize + v] = src[s + v * seqLen];
            }
        }
    } else if (dim0 == vocabSize && dim1 == seqLen) {
        // 原始格式：logits 形状 [vocab, seqLen]
        // GGML column-major: element[v,s] 在 offset v + s*vocab
        // 我们需要：result[s*vocab + v] = element[v,s]
        for (size_t s = 0; s < seqLen; ++s) {
            for (size_t v = 0; v < vocabSize; ++v) {
                result[s * vocabSize + v] = src[v + s * vocabSize];
            }
        }
    }
    
    CLLM_INFO("[Forward] Logits shape: [%lld, %lld]", logits->ne[0], logits->ne[1]);
    
    return result;
}

std::vector<float> GGMLTransformerModel::forwardOneToken(int32_t tokenId, size_t position) {
    if (position != kvCacheLen_) {
        CLLM_ERROR("[ForwardOneToken] position (%zu) != kvCacheLen_ (%zu)", position, kvCacheLen_);
    }
    
    auto result = forward({tokenId});
    return std::vector<float>(result.begin(), result.begin() + config_.vocabSize);
}

void GGMLTransformerModel::clearKVCache() {
    for (auto& k : kCaches_) if (k) ggml_set_zero(k);
    for (auto& v : vCaches_) if (v) ggml_set_zero(v);
    for (auto& k : pendingK_) k = nullptr;
    for (auto& v : pendingV_) v = nullptr;
    kvCacheLen_ = 0;
    pendingStartPos_ = 0;
    pendingSeqLen_ = 0;
    CLLM_DEBUG("[GGMLTransformerModel] KV cache cleared");
}

// ========== 调试接口 ==========

std::map<std::string, TensorStats> GGMLTransformerModel::getLayer0DebugStats() const {
    std::map<std::string, TensorStats> statsMap;
    
    auto addStats = [&](const char* name, ggml_tensor* tensor) {
        TensorStats stats;
        if (safeComputeTensorStats(tensor, stats)) {
            statsMap[name] = stats;
        }
    };
    
    addStats("attn_norm_output", debugLayer0Nodes_.attnNormOutput);
    addStats("qkv_output", debugLayer0Nodes_.qkvOutput);
    addStats("q_norm_output", debugLayer0Nodes_.qNormOutput);
    addStats("k_before_norm", debugLayer0Nodes_.kBeforeNorm);
    addStats("k_after_rms_norm", debugLayer0Nodes_.kAfterRmsNorm);
    addStats("k_norm_output", debugLayer0Nodes_.kNormOutput);
    addStats("rope_q_output", debugLayer0Nodes_.ropeQOutput);
    addStats("rope_k_output", debugLayer0Nodes_.ropeKOutput);
    addStats("attention_output", debugLayer0Nodes_.attentionOutput);
    addStats("ffn_norm_output", debugLayer0Nodes_.ffnNormOutput);
    addStats("ffn_gate_output", debugLayer0Nodes_.ffnGateOutput);
    addStats("ffn_up_output", debugLayer0Nodes_.ffnUpOutput);
    addStats("ffn_hidden_output", debugLayer0Nodes_.ffnHiddenOutput);
    addStats("ffn_output", debugLayer0Nodes_.ffnOutput);
    
    return statsMap;
}

std::vector<TensorStats> GGMLTransformerModel::getAllLayerStats() const {
    std::vector<TensorStats> statsVec;
    statsVec.reserve(debugLayerOutputs_.size());
    
    for (const auto* tensor : debugLayerOutputs_) {
        TensorStats stats;
        safeComputeTensorStats(tensor, stats);
        statsVec.push_back(stats);
    }
    
    return statsVec;
}

TensorStats GGMLTransformerModel::getFinalNormStats() const {
    TensorStats stats;
    safeComputeTensorStats(debugFinalNorm_, stats);
    return stats;
}

TensorStats GGMLTransformerModel::getLogitsStats() const {
    TensorStats stats;
    if (debugLogits_) {
        CLLM_INFO("[getLogitsStats] debugLogits_: type=%d, ne=[%lld, %lld], data=%p",
                  debugLogits_->type, debugLogits_->ne[0], debugLogits_->ne[1], debugLogits_->data);
    } else {
        CLLM_INFO("[getLogitsStats] debugLogits_ is NULL");
    }
    safeComputeTensorStats(debugLogits_, stats);
    return stats;
}

// ========== KV Cache 验证接口 ==========

bool GGMLTransformerModel::verifyKVCacheIntegrity(size_t layerIdx, size_t expectedLen) const {
#ifdef NDEBUG
    (void)layerIdx; (void)expectedLen;
    return true;
#else
    if (layerIdx >= kCaches_.size()) return false;
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    if (!kCache || !vCache || !kCache->data || !vCache->data) return false;
    
    const size_t headDim = kCache->ne[0];
    const size_t nKVHeads = kCache->ne[2];
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    
    const float* kData = static_cast<const float*>(kCache->data);
    const float* vData = static_cast<const float*>(vCache->data);
    
    for (size_t h = 0; h < nKVHeads; ++h) {
        for (size_t s = 0; s < expectedLen; ++s) {
            for (size_t d = 0; d < headDim; ++d) {
                const size_t idx = d + s * nb1 + h * nb2;
                if (std::isnan(kData[idx]) || std::isnan(vData[idx]) ||
                    std::isinf(kData[idx]) || std::isinf(vData[idx])) {
                    return false;
                }
            }
        }
    }
    return true;
#endif
}

KVCacheStats GGMLTransformerModel::getKVCacheStats(size_t layerIdx) const {
    KVCacheStats stats;
    stats.layerIdx = layerIdx;
    stats.currentLen = kvCacheLen_;
    stats.isValid = false;
    
    if (layerIdx >= kCaches_.size()) return stats;
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    if (!kCache || !vCache) return stats;
    
    stats.headDim = kCache->ne[0];
    stats.maxSeq = kCache->ne[1];
    stats.nKVHeads = kCache->ne[2];
    
    if (kvCacheLen_ == 0) {
        stats.isValid = true;
        return stats;
    }
    
    // 计算统计
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    const float* kData = static_cast<const float*>(kCache->data);
    const float* vData = static_cast<const float*>(vCache->data);
    
    std::vector<float> kValues, vValues;
    for (size_t h = 0; h < stats.nKVHeads; ++h) {
        for (size_t s = 0; s < kvCacheLen_; ++s) {
            for (size_t d = 0; d < stats.headDim; ++d) {
                const size_t idx = d + s * nb1 + h * nb2;
                kValues.push_back(kData[idx]);
                vValues.push_back(vData[idx]);
            }
        }
    }
    
    stats.kStats = computeTensorStats(kValues.data(), kValues.size());
    stats.vStats = computeTensorStats(vValues.data(), vValues.size());
    stats.isValid = stats.kStats.isValid() && stats.vStats.isValid();
    
    return stats;
}

std::vector<KVCacheStats> GGMLTransformerModel::getAllKVCacheStats() const {
    std::vector<KVCacheStats> allStats;
    allStats.reserve(config_.blockCount);
    for (size_t i = 0; i < config_.blockCount; ++i) {
        allStats.push_back(getKVCacheStats(i));
    }
    return allStats;
}

bool GGMLTransformerModel::validateKVCacheIntegrity(size_t expectedLen) const {
    for (size_t i = 0; i < config_.blockCount; ++i) {
        if (!verifyKVCacheIntegrity(i, expectedLen)) return false;
    }
    return true;
}

bool GGMLTransformerModel::getKVAtPosition(size_t layerIdx, size_t position,
                                            std::vector<float>& kData, 
                                            std::vector<float>& vData) const {
    if (layerIdx >= kCaches_.size() || position >= kvCacheLen_) return false;
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    if (!kCache || !vCache) return false;
    
    const size_t headDim = kCache->ne[0];
    const size_t nKVHeads = kCache->ne[2];
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    
    kData.resize(headDim * nKVHeads);
    vData.resize(headDim * nKVHeads);
    
    const float* kSrc = static_cast<const float*>(kCache->data);
    const float* vSrc = static_cast<const float*>(vCache->data);
    
    size_t outIdx = 0;
    for (size_t h = 0; h < nKVHeads; ++h) {
        for (size_t d = 0; d < headDim; ++d) {
            const size_t idx = d + position * nb1 + h * nb2;
            kData[outIdx] = kSrc[idx];
            vData[outIdx] = vSrc[idx];
            outIdx++;
        }
    }
    
    return true;
}

// ========== 计算图构建 ==========

ggml_tensor* GGMLTransformerModel::buildForwardGraph(
    const std::vector<int32_t>& inputIds,
    size_t startPos
) {
    ggml_context* ctx = computeCtx_->raw();
    const size_t seqLen = inputIds.size();
    
    // 输入张量
    ggml_tensor* inputTensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
    memcpy(inputTensor->data, inputIds.data(), seqLen * sizeof(int32_t));
    
    // Token embedding
    ggml_tensor* hidden_states = ggml_get_rows(ctx, tokEmbed_, inputTensor);
    debugEmbedding_ = hidden_states;
    
    // Transformer 层
    for (size_t i = 0; i < config_.blockCount; ++i) {
        hidden_states = buildLayerGraph(ctx, hidden_states, layers_[i], i, startPos, seqLen);
        debugLayerOutputs_.push_back(hidden_states);
        if (i == 0) debugLayer0Output_ = hidden_states;
    }
    
    // 最终归一化
    hidden_states = ggml_rms_norm(ctx, hidden_states, config_.rmsNormEps);
    hidden_states = ggml_mul(ctx, hidden_states, outputNorm_);
    debugFinalNorm_ = hidden_states;
    
    // LM Head
    ggml_tensor* logits = ggml_mul_mat(ctx, output_, hidden_states);
    debugLogits_ = logits;
    
    if (seqLen > 1) {
        logits = ggml_cont(ctx, ggml_transpose(ctx, logits));
    }
    
    return logits;
}

ggml_tensor* GGMLTransformerModel::buildLayerGraph(
    ggml_context* ctx,
    ggml_tensor* input,
    const LayerWeights& layer,
    size_t layerIdx,
    size_t startPos,
    size_t seqLen
) {
    ggml_tensor* residual = input;
    
    // Attention 归一化
    ggml_tensor* attnInput = ggml_rms_norm(ctx, input, config_.rmsNormEps);
    attnInput = ggml_mul(ctx, attnInput, layer.attnNorm);
    if (layerIdx == 0) debugLayer0Nodes_.attnNormOutput = attnInput;
    
    // 构建 Attention
    AttentionConfig attnConfig;
    attnConfig.nHeads = config_.headCount;
    attnConfig.nKVHeads = config_.headCountKV;
    attnConfig.headDim = config_.headDim();
    attnConfig.contextLength = config_.contextLength;
    attnConfig.rmsNormEps = config_.rmsNormEps;
    attnConfig.ropeFreqBase = config_.ropeFreqBase;
    attnConfig.ropeType = config_.ropeType;
    
    AttentionWeights attnWeights;
    attnWeights.wq = layer.wq;
    attnWeights.wk = layer.wk;
    attnWeights.wv = layer.wv;
    attnWeights.wo = layer.wo;
    attnWeights.attnQNorm = layer.attnQNorm;
    attnWeights.attnKNorm = layer.attnKNorm;
    
    KVCacheRef kvCache;
    kvCache.kCache = kCaches_[layerIdx];
    kvCache.vCache = vCaches_[layerIdx];
    kvCache.cacheLen = kvCacheLen_;
    kvCache.maxLen = maxKVCacheLen_;
    
    AttentionGraphBuilder attnBuilder(attnConfig);
    AttentionResult attnResult = attnBuilder.build(ctx, attnInput, attnWeights, kvCache,
                                                    startPos, seqLen, layerIdx);
    
    pendingK_[layerIdx] = attnResult.kNew;
    pendingV_[layerIdx] = attnResult.vNew;
    
    if (layerIdx == 0) {
        debugLayer0Nodes_.attentionOutput = attnResult.output;
        auto& dbgNodes = attnBuilder.getDebugNodes();
        debugLayer0Nodes_.qkvOutput = dbgNodes.qkvOutput;
        debugLayer0Nodes_.qNormOutput = dbgNodes.qNormOutput;
        debugLayer0Nodes_.kBeforeNorm = dbgNodes.kBeforeNorm;
        debugLayer0Nodes_.kAfterRmsNorm = dbgNodes.kAfterRmsNorm;
        debugLayer0Nodes_.kNormOutput = dbgNodes.kNormOutput;
        debugLayer0Nodes_.ropeQOutput = dbgNodes.ropeQOutput;
        debugLayer0Nodes_.ropeKOutput = dbgNodes.ropeKOutput;
    }
    
    // 残差连接
    ggml_tensor* x = ggml_add(ctx, residual, attnResult.output);
    residual = x;
    
    // FFN 归一化
    ggml_tensor* ffnInput = ggml_rms_norm(ctx, x, config_.rmsNormEps);
    ffnInput = ggml_mul(ctx, ffnInput, layer.ffnNorm);
    if (layerIdx == 0) debugLayer0Nodes_.ffnNormOutput = ffnInput;
    
    // 保存 Layer 27 的 FFN 输入用于调试
    ggml_tensor* layer27FfnInput = nullptr;
    if (layerIdx == 27) {
        layer27FfnInput = ffnInput;
    }
    
    // FFN
    ggml_tensor* ffnOutput = buildFFNGraph(ctx, ffnInput, layer);
    if (layerIdx == 0) debugLayer0Nodes_.ffnOutput = ffnOutput;
    
    // 保存 Layer 27 调试信息
    if (layerIdx == 27) {
        debugLayer27FfnInput_ = layer27FfnInput;
        debugLayer27FfnOutput_ = ffnOutput;
        debugLayer27AttnInput_ = input;
        debugLayer27AttnOutput_ = attnResult.output;
    }
    
    // 残差连接
    return ggml_add(ctx, residual, ffnOutput);
}

ggml_tensor* GGMLTransformerModel::buildFFNGraph(
    ggml_context* ctx,
    ggml_tensor* input,
    const LayerWeights& layer
) {
    // SwiGLU: down(swiglu(gate(x), up(x)))
    // 使用 ggml_swiglu_split 以与 llama.cpp 保持一致
    ggml_tensor* gate = ggml_mul_mat(ctx, layer.wGate, input);
    ggml_tensor* up = ggml_mul_mat(ctx, layer.wUp, input);
    
    // 保存 Layer 0 的 FFN Gate/Up 投影输出用于调试
    if (this->debugLayer0Nodes_.ffnGateOutput == nullptr) {
        this->debugLayer0Nodes_.ffnGateOutput = gate;
    }
    if (this->debugLayer0Nodes_.ffnUpOutput == nullptr) {
        this->debugLayer0Nodes_.ffnUpOutput = up;
    }
    
    // 使用 ggml_swiglu_split：等效于 silu(gate) * up
    // 这与 llama.cpp 的实现保持一致
    ggml_tensor* hidden = ggml_swiglu_split(ctx, gate, up);
    
    // 保存 Layer 0 的 FFN 隐藏层输出用于调试
    if (this->debugLayer0Nodes_.ffnHiddenOutput == nullptr) {
        this->debugLayer0Nodes_.ffnHiddenOutput = hidden;
    }
    
    return ggml_mul_mat(ctx, layer.wDown, hidden);
}

} // namespace kylin
} // namespace cllm
