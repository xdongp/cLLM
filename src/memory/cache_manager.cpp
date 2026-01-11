/**
 * @file cache_manager.cpp
 * @brief KV缓存内存管理器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/memory/cache_manager.h"
#include <algorithm>

namespace cllm {

KVCacheMemoryManager::KVCacheMemoryManager(size_t maxMemoryMb)
    : maxMemoryBytes_(maxMemoryMb * 1024 * 1024), usedMemoryBytes_(0) {
}

bool KVCacheMemoryManager::insert(const std::string& requestId,
                                   const std::vector<float>& keyCache,
                                   const std::vector<float>& valueCache) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t memoryUsage = calculateMemoryUsage(keyCache, valueCache);
    
    if (memoryUsage > maxMemoryBytes_) {
        return false;
    }
    
    while (usedMemoryBytes_ + memoryUsage > maxMemoryBytes_ && !cacheList_.empty()) {
        evictOldest();
    }
    
    if (usedMemoryBytes_ + memoryUsage > maxMemoryBytes_) {
        return false;
    }
    
    auto it = cacheMap_.find(requestId);
    if (it != cacheMap_.end()) {
        usedMemoryBytes_ -= calculateMemoryUsage(it->second->kvCache.keyCache, it->second->kvCache.valueCache);
        cacheList_.erase(it->second);
    }
    
    MemoryCacheEntry entry;
    entry.requestId = requestId;
    entry.kvCache.keyCache.resize(keyCache.size());
    entry.kvCache.valueCache.resize(valueCache.size());
    
    // Copy data from vectors to FloatArray
    for (size_t i = 0; i < keyCache.size(); ++i) {
        entry.kvCache.keyCache[i] = keyCache[i];
    }
    
    for (size_t i = 0; i < valueCache.size(); ++i) {
        entry.kvCache.valueCache[i] = valueCache[i];
    }
    
    entry.sequenceLength = keyCache.size();
    entry.lastAccess = std::chrono::steady_clock::now();
    
    cacheList_.push_front(entry);
    cacheMap_[requestId] = cacheList_.begin();
    
    usedMemoryBytes_ += memoryUsage;
    
    return true;
}

bool KVCacheMemoryManager::get(const std::string& requestId,
                                std::vector<float>& keyCache,
                                std::vector<float>& valueCache) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cacheMap_.find(requestId);
    if (it == cacheMap_.end()) {
        return false;
    }
    
    MemoryCacheEntry& entry = *it->second;
    entry.lastAccess = std::chrono::steady_clock::now();
    entry.kvCache.accessCount++;
    
    cacheList_.splice(cacheList_.begin(), cacheList_, it->second);
    cacheMap_[requestId] = cacheList_.begin();
    
    // Convert FloatArray to std::vector<float>
    keyCache.resize(entry.kvCache.keyCache.size());
    valueCache.resize(entry.kvCache.valueCache.size());
    
    for (size_t i = 0; i < keyCache.size(); ++i) {
        keyCache[i] = entry.kvCache.keyCache[i];
    }
    
    for (size_t i = 0; i < valueCache.size(); ++i) {
        valueCache[i] = entry.kvCache.valueCache[i];
    }
    
    return true;
}

void KVCacheMemoryManager::evict(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = cacheMap_.find(requestId);
    if (it != cacheMap_.end()) {
        usedMemoryBytes_ -= calculateMemoryUsage(it->second->kvCache.keyCache, it->second->kvCache.valueCache);
        
        if (evictionCallback_) {
            evictionCallback_(requestId);
        }
        
        cacheList_.erase(it->second);
        cacheMap_.erase(it);
    }
}

size_t KVCacheMemoryManager::getUsedMemory() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return usedMemoryBytes_;
}

size_t KVCacheMemoryManager::getTotalMemory() const {
    return maxMemoryBytes_;
}

void KVCacheMemoryManager::setEvictionCallback(EvictionCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    evictionCallback_ = callback;
}

void KVCacheMemoryManager::evictOldest() {
    if (cacheList_.empty()) {
        return;
    }
    
    auto oldest = cacheList_.back();
    std::string requestId = oldest.requestId;
    
    // Calculate memory usage from the KVCacheEntry
    size_t memoryUsage = calculateMemoryUsage(oldest.kvCache.keyCache, oldest.kvCache.valueCache);
    usedMemoryBytes_ -= memoryUsage;
    
    if (evictionCallback_) {
        evictionCallback_(requestId);
    }
    
    cacheMap_.erase(requestId);
    cacheList_.pop_back();
}

size_t KVCacheMemoryManager::calculateMemoryUsage(const FloatArray& keyCache,
                                                   const FloatArray& valueCache) {
    return (keyCache.size() + valueCache.size()) * sizeof(float);
}

size_t KVCacheMemoryManager::calculateMemoryUsage(const std::vector<float>& keyCache,
                                                   const std::vector<float>& valueCache) {
    return (keyCache.size() + valueCache.size()) * sizeof(float);
}

}  // namespace cllm
