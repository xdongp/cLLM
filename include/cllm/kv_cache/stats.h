/**
 * @file stats.h
 * @brief KV缓存统计信息结构
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_CACHE_STATS_H
#define CLLM_CACHE_STATS_H

#include <string>
#include <atomic>
#include <cstddef>

namespace cllm {

/**
 * @brief KV缓存统计信息结构
 * 
 * 记录KV缓存的运行统计信息，包括命中率、淘汰次数等。
 * 使用原子变量保证线程安全。
 */
struct CacheStats {
    std::atomic<size_t> hits;            ///< 缓存命中次数
    std::atomic<size_t> misses;          ///< 缓存未命中次数
    std::atomic<size_t> evictions;       ///< 缓存淘汰次数
    std::atomic<size_t> memoryReclaims;  ///< 内存回收次数
    
    CacheStats() 
        : hits(0)
        , misses(0)
        , evictions(0)
        , memoryReclaims(0) {}
    
    CacheStats(const CacheStats& other)
        : hits(other.hits.load())
        , misses(other.misses.load())
        , evictions(other.evictions.load())
        , memoryReclaims(other.memoryReclaims.load()) {}
    
    CacheStats& operator=(const CacheStats& other) {
        if (this != &other) {
            hits.store(other.hits.load());
            misses.store(other.misses.load());
            evictions.store(other.evictions.load());
            memoryReclaims.store(other.memoryReclaims.load());
        }
        return *this;
    }
    
    /**
     * @brief 获取缓存命中率
     * @return 命中率（0.0-1.0）
     */
    float getHitRate() const {
        size_t totalHits = hits.load();
        size_t totalMisses = misses.load();
        size_t total = totalHits + totalMisses;
        
        if (total == 0) {
            return 0.0f;
        }
        
        return static_cast<float>(totalHits) / total;
    }
    
    /**
     * @brief 更新命中计数
     */
    void updateHit() {
        hits.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief 更新未命中计数
     */
    void updateMiss() {
        misses.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief 更新淘汰计数
     */
    void updateEviction() {
        evictions.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief 更新内存回收计数
     */
    void updateMemoryReclaim() {
        memoryReclaims.fetch_add(1, std::memory_order_relaxed);
    }
    
    /**
     * @brief 重置统计信息
     */
    void reset() {
        hits.store(0, std::memory_order_relaxed);
        misses.store(0, std::memory_order_relaxed);
        evictions.store(0, std::memory_order_relaxed);
        memoryReclaims.store(0, std::memory_order_relaxed);
    }
    
    /**
     * @brief 转换为字符串
     * @return 格式化的统计信息字符串
     */
    std::string toString() const {
        return "CacheStats{hits=" + std::to_string(hits.load()) +
               ", misses=" + std::to_string(misses.load()) +
               ", evictions=" + std::to_string(evictions.load()) +
               ", memoryReclaims=" + std::to_string(memoryReclaims.load()) +
               ", hitRate=" + std::to_string(getHitRate()) + "}";
    }
};

}

#endif
