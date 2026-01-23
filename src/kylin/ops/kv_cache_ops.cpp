/**
 * @file kv_cache_ops.cpp
 * @brief KV Cache 操作和管理实现
 */

#include "cllm/kylin/ops/kv_cache_ops.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

namespace cllm {
namespace kylin {

bool KVCacheManager::allocate(ggml_context* ctx, const KVCacheConfig& config) {
    if (!ctx) {
        CLLM_ERROR("[KVCacheManager] Invalid GGML context");
        return false;
    }
    
    config_ = config;
    kCaches_.resize(config.nLayers);
    vCaches_.resize(config.nLayers);
    
    for (size_t i = 0; i < config.nLayers; ++i) {
        // K cache: [head_dim, max_seq, n_kv_heads]
        kCaches_[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                          config.headDim, config.maxSeqLen, config.nKVHeads);
        ggml_set_name(kCaches_[i], ("k_cache_" + std::to_string(i)).c_str());
        
        // V cache: [head_dim, max_seq, n_kv_heads]
        vCaches_[i] = ggml_new_tensor_3d(ctx, GGML_TYPE_F32,
                                          config.headDim, config.maxSeqLen, config.nKVHeads);
        ggml_set_name(vCaches_[i], ("v_cache_" + std::to_string(i)).c_str());
        
        ggml_set_zero(kCaches_[i]);
        ggml_set_zero(vCaches_[i]);
    }
    
    currentLen_ = 0;
    
    CLLM_INFO("[KVCacheManager] Allocated: %zu layers, %zu heads, %zu head_dim, %zu max_seq",
              config.nLayers, config.nKVHeads, config.headDim, config.maxSeqLen);
    
    return true;
}

void KVCacheManager::clear() {
    for (auto& k : kCaches_) {
        if (k) ggml_set_zero(k);
    }
    for (auto& v : vCaches_) {
        if (v) ggml_set_zero(v);
    }
    currentLen_ = 0;
    CLLM_DEBUG("[KVCacheManager] Cache cleared");
}

bool KVCacheManager::writeToCache(size_t layerIdx, ggml_tensor* kNew, ggml_tensor* vNew,
                                   size_t startPos, size_t seqLen) {
    if (layerIdx >= kCaches_.size()) {
        CLLM_ERROR("[KVCacheManager] Invalid layer index: %zu", layerIdx);
        return false;
    }
    return writeSingleLayer(layerIdx, kNew, vNew, startPos, seqLen);
}

bool KVCacheManager::flushPending(const std::vector<ggml_tensor*>& pendingK,
                                   const std::vector<ggml_tensor*>& pendingV,
                                   size_t startPos, size_t seqLen) {
    if (seqLen == 0) return true;
    
    const size_t nLayers = kCaches_.size();
    
    for (size_t i = 0; i < nLayers; ++i) {
        if (i >= pendingK.size() || i >= pendingV.size()) continue;
        if (!pendingK[i] || !pendingV[i]) continue;
        if (!pendingK[i]->data || !pendingV[i]->data) continue;
        
        if (!writeSingleLayer(i, pendingK[i], pendingV[i], startPos, seqLen)) {
            CLLM_WARN("[KVCacheManager] Failed to write layer %zu", i);
        }
    }
    
    return true;
}

bool KVCacheManager::writeSingleLayer(size_t layerIdx, ggml_tensor* kNew, ggml_tensor* vNew,
                                       size_t startPos, size_t seqLen) {
    ggml_tensor* kCache = kCaches_[layerIdx];
    ggml_tensor* vCache = vCaches_[layerIdx];
    
    // 获取维度信息
    const size_t srcHeadDim = kNew->ne[0];
    const size_t srcSeqLen = kNew->ne[1];
    const size_t srcNHeads = kNew->ne[2];
    
    const size_t dstHeadDim = kCache->ne[0];
    const size_t dstMaxSeq = kCache->ne[1];
    const size_t dstNHeads = kCache->ne[2];
    
    // 验证维度
    if (srcHeadDim != dstHeadDim || srcNHeads != dstNHeads) {
        CLLM_ERROR("[KVCacheManager] Dimension mismatch in layer %zu: "
                   "src=[%zu,%zu,%zu] dst=[%zu,%zu,%zu]",
                   layerIdx, srcHeadDim, srcSeqLen, srcNHeads, 
                   dstHeadDim, dstMaxSeq, dstNHeads);
        return false;
    }
    
    // 获取 stride 信息
    const size_t srcNb1 = kNew->nb[1];
    const size_t srcNb2 = kNew->nb[2];
    const size_t dstNb1 = kCache->nb[1];
    const size_t dstNb2 = kCache->nb[2];
    
    // 检查连续性
    const bool srcContiguous = (kNew->nb[0] == sizeof(float) && 
                                srcNb1 == srcHeadDim * sizeof(float) &&
                                srcNb2 == srcHeadDim * srcSeqLen * sizeof(float));
    const bool dstContiguous = (kCache->nb[0] == sizeof(float) && 
                                dstNb1 == dstHeadDim * sizeof(float));
    
    if (srcContiguous && dstContiguous) {
        // 优化路径：按 head 批量复制
        const size_t bytesToCopy = seqLen * dstHeadDim * sizeof(float);
        
        for (size_t h = 0; h < srcNHeads; ++h) {
            const char* srcK = static_cast<const char*>(kNew->data) + h * srcNb2;
            const char* srcV = static_cast<const char*>(vNew->data) + h * srcNb2;
            
            char* dstK = static_cast<char*>(kCache->data) + startPos * dstNb1 + h * dstNb2;
            char* dstV = static_cast<char*>(vCache->data) + startPos * dstNb1 + h * dstNb2;
            
            std::memcpy(dstK, srcK, bytesToCopy);
            std::memcpy(dstV, srcV, bytesToCopy);
        }
        
        CLLM_DEBUG("[KVCacheManager] Layer %zu: batch copy %zu bytes/head", layerIdx, bytesToCopy);
    } else {
        // 安全路径：逐位置复制
        const size_t bytesPerPos = srcHeadDim * sizeof(float);
        
        for (size_t h = 0; h < srcNHeads; ++h) {
            for (size_t s = 0; s < seqLen; ++s) {
                const char* srcK = static_cast<const char*>(kNew->data) + s * srcNb1 + h * srcNb2;
                const char* srcV = static_cast<const char*>(vNew->data) + s * srcNb1 + h * srcNb2;
                
                char* dstK = static_cast<char*>(kCache->data) + (startPos + s) * dstNb1 + h * dstNb2;
                char* dstV = static_cast<char*>(vCache->data) + (startPos + s) * dstNb1 + h * dstNb2;
                
                std::memcpy(dstK, srcK, bytesPerPos);
                std::memcpy(dstV, srcV, bytesPerPos);
            }
        }
        
        CLLM_DEBUG("[KVCacheManager] Layer %zu: per-position copy (non-contiguous)", layerIdx);
    }
    
    return true;
}

ggml_tensor* KVCacheManager::getKCache(size_t layerIdx) const {
    return layerIdx < kCaches_.size() ? kCaches_[layerIdx] : nullptr;
}

ggml_tensor* KVCacheManager::getVCache(size_t layerIdx) const {
    return layerIdx < vCaches_.size() ? vCaches_[layerIdx] : nullptr;
}

bool KVCacheManager::verifyIntegrity(size_t layerIdx, size_t expectedLen) const {
#ifdef NDEBUG
    (void)layerIdx;
    (void)expectedLen;
    return true;
#else
    if (layerIdx >= kCaches_.size() || layerIdx >= vCaches_.size()) {
        CLLM_ERROR("[KVCacheManager] Invalid layer index: %zu", layerIdx);
        return false;
    }
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    
    if (!kCache || !vCache || !kCache->data || !vCache->data) {
        CLLM_ERROR("[KVCacheManager] Layer %zu: cache data is null", layerIdx);
        return false;
    }
    
    const size_t headDim = kCache->ne[0];
    const size_t maxSeq = kCache->ne[1];
    const size_t nKVHeads = kCache->ne[2];
    
    if (expectedLen > maxSeq) {
        CLLM_ERROR("[KVCacheManager] Layer %zu: expectedLen (%zu) > maxSeq (%zu)", 
                   layerIdx, expectedLen, maxSeq);
        return false;
    }
    
    // 检查 NaN/Inf
    size_t nanCount = 0, infCount = 0;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    double sum = 0.0;
    
    const float* kData = static_cast<const float*>(kCache->data);
    const float* vData = static_cast<const float*>(vCache->data);
    
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    
    for (size_t h = 0; h < nKVHeads; ++h) {
        for (size_t s = 0; s < expectedLen; ++s) {
            for (size_t d = 0; d < headDim; ++d) {
                const size_t idx = d + s * nb1 + h * nb2;
                float kVal = kData[idx];
                float vVal = vData[idx];
                
                if (std::isnan(kVal) || std::isnan(vVal)) {
                    nanCount++;
                } else if (std::isinf(kVal) || std::isinf(vVal)) {
                    infCount++;
                } else {
                    minVal = std::min(minVal, std::min(kVal, vVal));
                    maxVal = std::max(maxVal, std::max(kVal, vVal));
                    sum += static_cast<double>(kVal + vVal);
                }
            }
        }
    }
    
    if (nanCount > 0 || infCount > 0) {
        CLLM_WARN("[KVCacheManager] Layer %zu: found %zu NaN, %zu Inf", layerIdx, nanCount, infCount);
        return false;
    }
    
    const size_t totalElements = headDim * expectedLen * nKVHeads * 2;
    double mean = sum / static_cast<double>(totalElements);
    CLLM_DEBUG("[KVCacheManager] Layer %zu OK: len=%zu, range=[%.4f, %.4f], mean=%.6f",
               layerIdx, expectedLen, minVal, maxVal, mean);
    
    return true;
#endif
}

bool KVCacheManager::validateAllLayers(size_t expectedLen) const {
    if (expectedLen > config_.maxSeqLen) {
        CLLM_ERROR("[KVCacheManager] expectedLen (%zu) > maxSeqLen (%zu)", 
                   expectedLen, config_.maxSeqLen);
        return false;
    }
    
    bool allValid = true;
    size_t invalidCount = 0;
    
    for (size_t i = 0; i < kCaches_.size(); ++i) {
        if (!verifyIntegrity(i, expectedLen)) {
            allValid = false;
            invalidCount++;
        }
    }
    
    if (!allValid) {
        CLLM_ERROR("[KVCacheManager] %zu layers failed integrity check", invalidCount);
    }
    
    return allValid;
}

KVCacheStats KVCacheManager::getStats(size_t layerIdx) const {
    KVCacheStats stats;
    stats.layerIdx = layerIdx;
    stats.isValid = false;
    stats.currentLen = currentLen_;
    
    if (layerIdx >= kCaches_.size()) {
        CLLM_ERROR("[KVCacheManager] Invalid layer index: %zu", layerIdx);
        return stats;
    }
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    
    if (!kCache || !vCache || !kCache->data || !vCache->data) {
        return stats;
    }
    
    stats.headDim = kCache->ne[0];
    stats.maxSeq = kCache->ne[1];
    stats.nKVHeads = kCache->ne[2];
    
    if (currentLen_ == 0) {
        stats.isValid = true;
        return stats;
    }
    
    // 收集有效数据范围内的统计
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    
    const float* kData = static_cast<const float*>(kCache->data);
    const float* vData = static_cast<const float*>(vCache->data);
    
    std::vector<float> kValues, vValues;
    const size_t totalElements = stats.headDim * currentLen_ * stats.nKVHeads;
    kValues.reserve(totalElements);
    vValues.reserve(totalElements);
    
    for (size_t h = 0; h < stats.nKVHeads; ++h) {
        for (size_t s = 0; s < currentLen_; ++s) {
            for (size_t d = 0; d < stats.headDim; ++d) {
                const size_t idx = d + s * nb1 + h * nb2;
                kValues.push_back(kData[idx]);
                vValues.push_back(vData[idx]);
            }
        }
    }
    
    stats.kStats = computeTensorStats(kValues.data(), kValues.size(), kDefaultFirstN);
    stats.vStats = computeTensorStats(vValues.data(), vValues.size(), kDefaultFirstN);
    
    stats.isValid = stats.kStats.isValid() && stats.vStats.isValid();
    
    return stats;
}

std::vector<KVCacheStats> KVCacheManager::getAllStats() const {
    std::vector<KVCacheStats> allStats;
    allStats.reserve(kCaches_.size());
    
    for (size_t i = 0; i < kCaches_.size(); ++i) {
        allStats.push_back(getStats(i));
    }
    
    return allStats;
}

bool KVCacheManager::getDataAtPosition(size_t layerIdx, size_t position,
                                        std::vector<float>& kData, 
                                        std::vector<float>& vData) const {
    if (layerIdx >= kCaches_.size()) {
        CLLM_ERROR("[KVCacheManager] Invalid layer index: %zu", layerIdx);
        return false;
    }
    
    if (position >= currentLen_) {
        CLLM_ERROR("[KVCacheManager] Position %zu >= currentLen %zu", position, currentLen_);
        return false;
    }
    
    const ggml_tensor* kCache = kCaches_[layerIdx];
    const ggml_tensor* vCache = vCaches_[layerIdx];
    
    if (!kCache || !vCache || !kCache->data || !vCache->data) {
        return false;
    }
    
    const size_t headDim = kCache->ne[0];
    const size_t nKVHeads = kCache->ne[2];
    const size_t totalElements = headDim * nKVHeads;
    
    kData.resize(totalElements);
    vData.resize(totalElements);
    
    const float* kSrc = static_cast<const float*>(kCache->data);
    const float* vSrc = static_cast<const float*>(vCache->data);
    
    const size_t nb1 = kCache->nb[1] / sizeof(float);
    const size_t nb2 = kCache->nb[2] / sizeof(float);
    
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

} // namespace kylin
} // namespace cllm
