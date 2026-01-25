/**
 * @file kv_cache_pool.cpp
 * @brief Per-Request KV Cache 管理器实现
 */

#include "cllm/kylin/hf/kv_cache_pool.h"
#include "cllm/common/logger.h"

namespace cllm {
namespace kylin {

// ========== KVCachePool 实现 ==========

KVCachePool::KVCachePool(int maxSlots, int numLayers, int maxSeqLen, int numKVHeads, int headDim)
    : maxSlots_(maxSlots)
    , numLayers_(numLayers)
    , maxSeqLen_(maxSeqLen)
    , numKVHeads_(numKVHeads)
    , headDim_(headDim)
    , perLayerCacheSize_(static_cast<size_t>(maxSeqLen) * numKVHeads * headDim)
{
    CLLM_INFO("[KVCachePool] Initializing: maxSlots=%d, layers=%d, maxSeqLen=%d, kvHeads=%d, headDim=%d",
              maxSlots, numLayers, maxSeqLen, numKVHeads, headDim);
    
    // 预分配所有槽位
    slots_.reserve(maxSlots);
    freeSlots_.reserve(maxSlots);
    
    size_t totalCacheSize = static_cast<size_t>(numLayers) * perLayerCacheSize_;
    size_t perSlotMemory = totalCacheSize * 2 * sizeof(float);  // K + V
    
    for (int i = 0; i < maxSlots; ++i) {
        auto slot = std::make_unique<KVCacheSlot>();
        slot->kCache.resize(totalCacheSize, 0.0f);
        slot->vCache.resize(totalCacheSize, 0.0f);
        slot->currentLen = 0;
        slot->requestId = 0;
        slot->inUse = false;
        slots_.push_back(std::move(slot));
        freeSlots_.push_back(i);
    }
    
    CLLM_INFO("[KVCachePool] Allocated %d slots, %.2f MB per slot, %.2f MB total",
              maxSlots,
              perSlotMemory / (1024.0 * 1024.0),
              maxSlots * perSlotMemory / (1024.0 * 1024.0));
}

KVCacheSlot* KVCachePool::allocate(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查是否已分配
    auto it = requestToSlot_.find(requestId);
    if (it != requestToSlot_.end()) {
        return slots_[it->second].get();
    }
    
    // 分配新槽位
    if (freeSlots_.empty()) {
        CLLM_WARN("[KVCachePool] No free slots available for request %zu", requestId);
        return nullptr;
    }
    
    int slotIdx = freeSlots_.back();
    freeSlots_.pop_back();
    
    KVCacheSlot* slot = slots_[slotIdx].get();
    slot->currentLen = 0;
    slot->requestId = requestId;
    slot->inUse = true;
    
    requestToSlot_[requestId] = slotIdx;
    
    CLLM_DEBUG("[KVCachePool] Allocated slot %d for request %zu (free: %zu)",
               slotIdx, requestId, freeSlots_.size());
    
    return slot;
}

bool KVCachePool::release(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = requestToSlot_.find(requestId);
    if (it == requestToSlot_.end()) {
        return false;
    }
    
    int slotIdx = it->second;
    KVCacheSlot* slot = slots_[slotIdx].get();
    slot->reset();
    
    freeSlots_.push_back(slotIdx);
    requestToSlot_.erase(it);
    
    CLLM_DEBUG("[KVCachePool] Released slot %d for request %zu (free: %zu)",
               slotIdx, requestId, freeSlots_.size());
    
    return true;
}

KVCacheSlot* KVCachePool::get(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = requestToSlot_.find(requestId);
    if (it == requestToSlot_.end()) {
        return nullptr;
    }
    
    return slots_[it->second].get();
}

KVCacheSlot* KVCachePool::getOrAllocate(size_t requestId) {
    // 先尝试获取
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = requestToSlot_.find(requestId);
        if (it != requestToSlot_.end()) {
            return slots_[it->second].get();
        }
    }
    
    // 不存在则分配
    return allocate(requestId);
}

int KVCachePool::availableSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(freeSlots_.size());
}

int KVCachePool::usedSlots() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return maxSlots_ - static_cast<int>(freeSlots_.size());
}

// ========== WorkBufferPool 实现 ==========

WorkBufferPool::WorkBufferPool(int maxSlots, int hiddenSize, int intermediateSize,
                               int vocabSize, int numHeads, int numKVHeads, int headDim, int maxSeqLen) {
    CLLM_INFO("[WorkBufferPool] Initializing: maxSlots=%d", maxSlots);
    
    slots_.reserve(maxSlots);
    freeSlots_.reserve(maxSlots);
    
    int qSize = numHeads * headDim;
    int kvSize = numKVHeads * headDim;
    
    for (int i = 0; i < maxSlots; ++i) {
        auto slot = std::make_unique<WorkBufferSlot>();
        
        slot->hiddenStates.resize(hiddenSize);
        slot->residual.resize(hiddenSize);
        slot->normOutput.resize(hiddenSize);
        slot->attnOutput.resize(hiddenSize);
        slot->ffnOutput.resize(hiddenSize);
        slot->qkvBuffer.resize(qSize + 2 * kvSize);
        slot->qBuffer.resize(qSize);
        slot->kBuffer.resize(kvSize);
        slot->vBuffer.resize(kvSize);
        slot->attnScores.resize(numHeads * maxSeqLen);
        slot->attnOutBuffer.resize(qSize);
        slot->gateBuffer.resize(intermediateSize);
        slot->upBuffer.resize(intermediateSize);
        slot->gateUpBuffer.resize(intermediateSize * 2);
        slot->logits.resize(vocabSize);
        slot->inUse = false;
        
        slots_.push_back(std::move(slot));
        freeSlots_.push_back(i);
    }
    
    CLLM_INFO("[WorkBufferPool] Allocated %d work buffer slots", maxSlots);
}

WorkBufferSlot* WorkBufferPool::allocate() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (freeSlots_.empty()) {
        return nullptr;
    }
    
    int idx = freeSlots_.back();
    freeSlots_.pop_back();
    
    WorkBufferSlot* slot = slots_[idx].get();
    slot->inUse = true;
    
    return slot;
}

void WorkBufferPool::release(WorkBufferSlot* slot) {
    if (!slot) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    slot->inUse = false;
    
    // 找到槽位索引
    for (size_t i = 0; i < slots_.size(); ++i) {
        if (slots_[i].get() == slot) {
            freeSlots_.push_back(static_cast<int>(i));
            break;
        }
    }
}

} // namespace kylin
} // namespace cllm
