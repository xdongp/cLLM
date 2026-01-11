#include "cllm/kv_cache/manager.h"
#include <stdexcept>

namespace cllm {

KVCacheManager::KVCacheManager(const CacheConfig& config)
    : config_(config) {
}

KVCacheManager::~KVCacheManager() {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    for (auto& pair : caches_) {
        delete pair.second;
    }
    
    caches_.clear();
}

KVCache* KVCacheManager::createCache(const std::string& cacheName) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    auto it = caches_.find(cacheName);
    if (it != caches_.end()) {
        return it->second;
    }
    
    KVCache* cache = new KVCache(
        config_.defaultMaxSize,
        config_.defaultMaxMemoryMB
    );
    
    caches_[cacheName] = cache;
    return cache;
}

KVCache* KVCacheManager::getCache(const std::string& cacheName) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    auto it = caches_.find(cacheName);
    if (it != caches_.end()) {
        return it->second;
    }
    
    return nullptr;
}

void KVCacheManager::removeCache(const std::string& cacheName) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    auto it = caches_.find(cacheName);
    if (it != caches_.end()) {
        delete it->second;
        caches_.erase(it);
    }
}

std::vector<std::string> KVCacheManager::getCacheNames() const {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    std::vector<std::string> names;
    for (const auto& pair : caches_) {
        names.push_back(pair.first);
    }
    
    return names;
}

size_t KVCacheManager::getTotalMemoryUsage() const {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    size_t totalMemory = 0;
    for (const auto& pair : caches_) {
        totalMemory += static_cast<size_t>(pair.second->getMemoryUsageMB());
    }
    
    return totalMemory;
}

ManagerStats KVCacheManager::getStats() const {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    ManagerStats stats;
    stats.totalCaches = caches_.size();
    stats.totalMemoryUsage = getTotalMemoryUsage();
    
    for (const auto& pair : caches_) {
        CacheStats cacheStats = pair.second->getStats();
        stats.totalHits += cacheStats.hits.load();
        stats.totalMisses += cacheStats.misses.load();
    }
    
    return stats;
}

}
