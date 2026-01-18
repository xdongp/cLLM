/**
 * @file kv_cache_manager.cpp
 * @brief KV缓存统计管理器实现（Phase 4）
 * 
 * @author cLLM Team
 * @date 2026-01-18
 */

#include "cllm/inference/kv_cache_manager.h"
#include "cllm/common/logger.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <vector>

namespace cllm {
namespace inference {

KVCacheManager::KVCacheManager(size_t maxItems, size_t maxMemoryMb)
    : totalItems_(0)
    , totalMemoryMb_(0)
    , maxItems_(maxItems)
    , maxMemoryMb_(maxMemoryMb) {
    CLLM_INFO("[KVCacheManager] Initialized with maxItems=%zu, maxMemoryMb=%zu", 
              maxItems_, maxMemoryMb_);
}

void KVCacheManager::updateKVCacheStats(size_t requestId, size_t sequenceLength) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 如果已存在统计信息，先减去旧的条目数和内存占用
    auto it = statsMap_.find(requestId);
    if (it != statsMap_.end()) {
        totalItems_ -= it->second.itemCount;
        totalMemoryMb_ -= it->second.memoryMb;
    }
    
    // 计算新的内存占用
    size_t memoryMb = calculateMemoryUsage(sequenceLength);
    
    // 更新或创建统计信息
    KVCacheStats stats(requestId, sequenceLength, memoryMb);
    stats.lastAccessTime = std::chrono::steady_clock::now();
    statsMap_[requestId] = stats;
    
    // 更新总条目数和内存占用
    totalItems_ += sequenceLength;
    totalMemoryMb_ += memoryMb;
    
    CLLM_DEBUG("[KVCacheManager] Updated stats for requestId=%zu: items=%zu, memory=%zuMB, totalItems=%zu, totalMemory=%zuMB",
               requestId, sequenceLength, memoryMb, totalItems_, totalMemoryMb_);
}

KVCacheStats KVCacheManager::getKVCacheStats(size_t requestId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = statsMap_.find(requestId);
    if (it != statsMap_.end()) {
        // 更新最后访问时间（用于LRU）
        // 注意：这里需要修改 const 对象，所以使用 mutable mutex_，并且修改统计信息
        // 但 KVCacheStats 不是 mutable 的，所以我们需要创建一个新的对象
        KVCacheStats stats = it->second;
        stats.lastAccessTime = std::chrono::steady_clock::now();
        
        // 注意：这里不能直接修改 it->second，因为 const 方法
        // 如果需要更新访问时间，应该在非 const 方法中处理
        return stats;
    }
    
    return KVCacheStats();  // 返回默认值
}

bool KVCacheManager::hasKVCacheStats(size_t requestId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return statsMap_.find(requestId) != statsMap_.end();
}

bool KVCacheManager::removeKVCache(llama_context* ctx, size_t requestId, int32_t seqId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = statsMap_.find(requestId);
    if (it == statsMap_.end()) {
        CLLM_WARN("[KVCacheManager] Request %zu not found in stats map", requestId);
        return false;
    }
    
    const KVCacheStats& stats = it->second;
    
    // 1. 调用 llama_memory_seq_rm 清理 llama.cpp 中的KV缓存
    if (ctx && seqId >= 0) {
        // llama_memory_seq_rm 的签名：bool llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq_id, llama_pos p0, llama_pos p1)
        // p0 和 p1 指定要清理的位置范围，-1 表示清理整个序列
        // 首先需要从 ctx 获取 memory handle
        llama_memory_t mem = llama_get_memory(ctx);
        if (mem) {
            llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seqId), -1, -1);
            CLLM_DEBUG("[KVCacheManager] Called llama_memory_seq_rm for requestId=%zu, seqId=%d", 
                       requestId, seqId);
        } else {
            CLLM_WARN("[KVCacheManager] Cannot clean KV cache: llama_get_memory returned nullptr");
        }
    } else {
        if (!ctx) {
            CLLM_WARN("[KVCacheManager] Cannot clean KV cache: ctx is nullptr");
        }
        if (seqId < 0) {
            CLLM_WARN("[KVCacheManager] Cannot clean KV cache: seqId is invalid (%d)", seqId);
        }
    }
    
    // 2. 从总条目数和内存占用中减去该请求的统计信息
    totalItems_ -= stats.itemCount;
    totalMemoryMb_ -= stats.memoryMb;
    
    // 3. 删除统计信息
    statsMap_.erase(it);
    
    // 4. 删除请求状态（如果存在）
    requestStatus_.erase(requestId);
    
    CLLM_DEBUG("[KVCacheManager] Removed KV cache for requestId=%zu: items=%zu, memory=%zuMB, remainingItems=%zu, remainingMemory=%zuMB",
               requestId, stats.itemCount, stats.memoryMb, totalItems_, totalMemoryMb_);
    
    return true;
}

size_t KVCacheManager::getTotalItems() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalItems_;
}

size_t KVCacheManager::getTotalMemoryMb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalMemoryMb_;
}

size_t KVCacheManager::getCacheCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return statsMap_.size();
}

void KVCacheManager::updateRequestStatus(size_t requestId, RequestStatus status) {
    std::lock_guard<std::mutex> lock(mutex_);
    requestStatus_[requestId] = status;
    CLLM_DEBUG("[KVCacheManager] Updated request status for requestId=%zu: status=%d", 
               requestId, static_cast<int>(status));
}

RequestStatus KVCacheManager::getRequestStatus(size_t requestId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = requestStatus_.find(requestId);
    if (it != requestStatus_.end()) {
        return it->second;
    }
    return RequestStatus::PENDING;  // 默认返回 PENDING
}

size_t KVCacheManager::estimateMemoryPerItem(size_t vocabSize, size_t hiddenSize) {
    // 粗略估算：假设每个条目（token）的KV缓存占用
    // 对于 transformer 模型，每个 token 的 KV cache 大小约为：
    // 2 × (hidden_size × num_layers × sizeof(float))
    // 这里使用一个简化的估算：假设每个条目占用约 2MB
    // 实际值取决于模型大小（hidden_size, num_layers, num_heads等）
    
    // 更精确的估算（可选）：
    // size_t kvCacheSizePerItem = 2 * hiddenSize * sizeof(float) / (1024 * 1024);  // 粗略估算
    // 这里先使用固定值 2MB，后续可以根据配置或模型参数调整
    
    return 2;  // 2MB per item (粗略估算)
}

size_t KVCacheManager::calculateMemoryUsage(size_t itemCount) const {
    // 使用估算值：每个条目占用约 2MB
    // 这只是一个粗略估算，实际值取决于模型大小
    return itemCount * estimateMemoryPerItem();
}

bool KVCacheManager::shouldEvict(double evictionThreshold) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查条目数是否超过阈值
    size_t itemsThreshold = static_cast<size_t>(maxItems_ * evictionThreshold);
    if (totalItems_ > itemsThreshold) {
        CLLM_DEBUG("[KVCacheManager] Should evict: totalItems=%zu > itemsThreshold=%zu (threshold=%.2f)",
                   totalItems_, itemsThreshold, evictionThreshold);
        return true;
    }
    
    // 检查内存是否超过阈值
    size_t memoryThreshold = static_cast<size_t>(maxMemoryMb_ * evictionThreshold);
    if (totalMemoryMb_ > memoryThreshold) {
        CLLM_DEBUG("[KVCacheManager] Should evict: totalMemory=%zuMB > memoryThreshold=%zuMB (threshold=%.2f)",
                   totalMemoryMb_, memoryThreshold, evictionThreshold);
        return true;
    }
    
    return false;
}

size_t KVCacheManager::evictLRUCache(llama_context* ctx, double evictionThreshold,
                                      std::function<int32_t(size_t)> getSeqIdCallback) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t evictedCount = 0;
    size_t itemsThreshold = static_cast<size_t>(maxItems_ * evictionThreshold);
    size_t memoryThreshold = static_cast<size_t>(maxMemoryMb_ * evictionThreshold);
    
    CLLM_DEBUG("[KVCacheManager] Starting LRU eviction: totalItems=%zu/%zu, totalMemory=%zuMB/%zuMB (threshold=%.2f)",
               totalItems_, itemsThreshold, totalMemoryMb_, memoryThreshold, evictionThreshold);
    
    // 持续淘汰直到条目数和内存都低于阈值
    while (totalItems_ > itemsThreshold || totalMemoryMb_ > memoryThreshold) {
        // 找到最久未使用的、可淘汰的请求
        // 1. 按照最后访问时间排序
        std::vector<std::pair<size_t, std::chrono::steady_clock::time_point>> sortedRequests;
        for (const auto& pair : statsMap_) {
            sortedRequests.emplace_back(pair.first, pair.second.lastAccessTime);
        }
        
        // 按最后访问时间升序排序（最久未使用的在前）
        std::sort(sortedRequests.begin(), sortedRequests.end(),
                  [](const auto& a, const auto& b) {
                      return a.second < b.second;
                  });
        
        // 2. 找到第一个可淘汰的请求（状态为 PENDING 或 COMPLETED）
        bool foundEvictable = false;
        for (const auto& pair : sortedRequests) {
            size_t requestId = pair.first;
            
            // 检查请求状态
            auto statusIt = requestStatus_.find(requestId);
            RequestStatus status = (statusIt != requestStatus_.end()) ? statusIt->second : RequestStatus::PENDING;
            
            // 只淘汰 PENDING 或 COMPLETED 状态的请求
            // 保护：不淘汰 PROCESSING 状态的请求
            if (status == RequestStatus::PENDING || status == RequestStatus::COMPLETED) {
                auto statsIt = statsMap_.find(requestId);
                if (statsIt != statsMap_.end()) {
                    const KVCacheStats& stats = statsIt->second;
                    
                    // 获取 seqId（通过回调函数）
                    int32_t seqId = -1;
                    if (getSeqIdCallback) {
                        seqId = getSeqIdCallback(requestId);
                    }
                    
                    // 调用 llama_memory_seq_rm 清理 llama.cpp 中的KV缓存
                    if (ctx && seqId >= 0) {
                        llama_memory_t mem = llama_get_memory(ctx);
                        if (mem) {
                            llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seqId), -1, -1);
                            CLLM_DEBUG("[KVCacheManager] Evicted KV cache for requestId=%zu, seqId=%d, items=%zu, memory=%zuMB",
                                       requestId, seqId, stats.itemCount, stats.memoryMb);
                        }
                    }
                    
                    // 从总条目数和内存占用中减去该请求的统计信息
                    totalItems_ -= stats.itemCount;
                    totalMemoryMb_ -= stats.memoryMb;
                    
                    // 删除统计信息和请求状态
                    statsMap_.erase(statsIt);
                    requestStatus_.erase(requestId);
                    
                    evictedCount++;
                    foundEvictable = true;
                    
                    // 跳出内层循环，继续外层循环检查是否还需要淘汰
                    break;
                }
            }
        }
        
        // 如果没有找到可淘汰的请求，说明所有请求都在 PROCESSING 状态，无法继续淘汰
        if (!foundEvictable) {
            CLLM_WARN("[KVCacheManager] No evictable requests found (all requests are in PROCESSING state)");
            break;
        }
    }
    
    CLLM_DEBUG("[KVCacheManager] LRU eviction completed: evictedCount=%zu, totalItems=%zu, totalMemory=%zuMB",
               evictedCount, totalItems_, totalMemoryMb_);
    
    return evictedCount;
}

} // namespace inference
} // namespace cllm