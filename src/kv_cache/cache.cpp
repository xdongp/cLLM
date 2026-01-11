#include "cllm/kv_cache/cache.h"
#include <algorithm>
#include <stdexcept>

namespace cllm {

KVCache::KVCache(size_t maxSize, size_t maxMemoryMB)
    : maxSize_((maxSize != 0) ? maxSize : Config::instance().cacheDefaultMaxSize())
    , maxMemoryMB_((maxMemoryMB != 0) ? maxMemoryMB : Config::instance().cacheDefaultMaxMemoryMb())
    , memoryUsage_(0.0f) {
}

KVCache::~KVCache() {
    clear();
}

bool KVCache::get(size_t sequenceId, KVCacheEntry& entry) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    auto it = cache_.find(sequenceId);
    if (it != cache_.end()) {
        entry = it->second;
        entry.updateAccess();
        
        accessList_.remove(sequenceId);
        accessList_.push_back(sequenceId);
        
        stats_.updateHit();
        return true;
    }
    
    stats_.updateMiss();
    return false;
}

void KVCache::put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    float newMemoryUsage = calculateMemoryUsage(keyCache, valueCache);
    
    auto it = cache_.find(sequenceId);
    if (it != cache_.end()) {
        memoryUsage_ -= it->second.memoryUsage;
        it->second = KVCacheEntry(keyCache, valueCache, sequenceId);
        it->second.memoryUsage = newMemoryUsage;
        it->second.updateAccess();
        
        accessList_.remove(sequenceId);
        accessList_.push_back(sequenceId);
        
        memoryUsage_ += newMemoryUsage;
    } else {
        ensureMemoryLimitForNewEntry(newMemoryUsage);
        
        if (cache_.size() >= maxSize_) {
            evictOldest();
        }
        
        cache_[sequenceId] = KVCacheEntry(keyCache, valueCache, sequenceId);
        cache_[sequenceId].memoryUsage = newMemoryUsage;
        accessList_.push_back(sequenceId);
        
        memoryUsage_ += newMemoryUsage;
    }
}

void KVCache::remove(size_t sequenceId) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    auto it = cache_.find(sequenceId);
    if (it != cache_.end()) {
        memoryUsage_ -= it->second.memoryUsage;
        cache_.erase(it);
        accessList_.remove(sequenceId);
    }
}

void KVCache::clear() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    cache_.clear();
    accessList_.clear();
    memoryUsage_ = 0.0f;
}

size_t KVCache::size() const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cache_.size();
}

bool KVCache::contains(size_t sequenceId) const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return cache_.find(sequenceId) != cache_.end();
}

CacheStats KVCache::getStats() const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return stats_;
}

void KVCache::resetStats() {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    stats_.reset();
}

float KVCache::getMemoryUsageMB() const {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    return static_cast<float>(memoryUsage_);
}

size_t KVCache::getMaxSize() const {
    return maxSize_;
}

size_t KVCache::getMaxMemoryMB() const {
    return maxMemoryMB_;
}

void KVCache::updateIncremental(
    size_t sequenceId,
    const FloatArray& newKeyPart,
    const FloatArray& newValuePart
) {
    std::lock_guard<std::mutex> lock(cacheMutex_);
    
    auto it = cache_.find(sequenceId);
    if (it == cache_.end()) {
        return;
    }
    
    KVCacheEntry& entry = it->second;
    
    size_t oldKeySize = entry.keyCache.size();
    size_t newValueSize = newKeyPart.size();
    
    FloatArray updatedKey(oldKeySize + newValueSize);
    FloatArray updatedValue(oldKeySize + newValueSize);
    
    std::copy(entry.keyCache.data(), entry.keyCache.data() + oldKeySize, updatedKey.data());
    std::copy(newKeyPart.data(), newKeyPart.data() + newValueSize, updatedKey.data() + oldKeySize);
    
    std::copy(entry.valueCache.data(), entry.valueCache.data() + oldKeySize, updatedValue.data());
    std::copy(newValuePart.data(), newValuePart.data() + newValueSize, updatedValue.data() + oldKeySize);
    
    memoryUsage_ -= entry.memoryUsage;
    
    entry.keyCache = updatedKey;
    entry.valueCache = updatedValue;
    entry.updateMemoryUsage();
    entry.updateAccess();
    
    memoryUsage_ += entry.memoryUsage;
}

void KVCache::evictOldest() {
    if (accessList_.empty()) {
        return;
    }
    
    size_t oldestId = accessList_.front();
    accessList_.pop_front();
    
    auto it = cache_.find(oldestId);
    if (it != cache_.end()) {
        memoryUsage_ -= it->second.memoryUsage;
        cache_.erase(it);
        stats_.updateEviction();
    }
}

void KVCache::ensureMemoryLimit() {
    if (maxMemoryMB_ == 0) {
        return;
    }
    
    while (memoryUsage_ > maxMemoryMB_ && !cache_.empty()) {
        evictOldest();
        stats_.updateMemoryReclaim();
    }
}

void KVCache::ensureMemoryLimitForNewEntry(double newEntrySize) {
    if (maxMemoryMB_ == 0) {
        return;
    }
    
    while (memoryUsage_ + newEntrySize > maxMemoryMB_ && !cache_.empty()) {
        evictOldest();
        stats_.updateMemoryReclaim();
    }
}

double KVCache::calculateMemoryUsage(const FloatArray& keyCache, const FloatArray& valueCache) {
    size_t totalSize = keyCache.size() + valueCache.size();
    return static_cast<double>(totalSize * sizeof(float)) / (1024.0 * 1024.0);
}

}
