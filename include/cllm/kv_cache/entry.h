/**
 * @file entry.h
 * @brief KV缓存条目结构
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_KV_CACHE_ENTRY_H
#define CLLM_KV_CACHE_ENTRY_H

#include "cllm/memory/float_array.h"
#include <cstddef>

namespace cllm {

/**
 * @brief KV缓存条目结构
 * 
 * 存储一个KV缓存条目的完整信息，包括缓存数据和元数据。
 */
struct KVCacheEntry {
    FloatArray keyCache;       ///< 键缓存数据
    FloatArray valueCache;     ///< 值缓存数据
    size_t sequenceId;         ///< 序列ID
    size_t lastAccessTime;     ///< 最后访问时间
    size_t hitCount;           ///< 命中次数
    size_t createdTime;        ///< 创建时间
    size_t lastUpdateTime;     ///< 最后更新时间
    float memoryUsage;         ///< 内存使用量(MB)
    size_t accessCount;        ///< 访问总次数
    
    /**
     * @brief 默认构造函数
     */
    KVCacheEntry()
        : sequenceId(0)
        , lastAccessTime(0)
        , hitCount(0)
        , createdTime(0)
        , lastUpdateTime(0)
        , memoryUsage(0.0f)
        , accessCount(0) {}
    
    /**
     * @brief 构造函数
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     * @param sequenceId 序列ID
     */
    KVCacheEntry(
        const FloatArray& keyCache,
        const FloatArray& valueCache,
        size_t sequenceId
    ) : keyCache(keyCache)
        , valueCache(valueCache)
        , sequenceId(sequenceId)
        , lastAccessTime(0)
        , hitCount(0)
        , createdTime(0)
        , lastUpdateTime(0)
        , memoryUsage(0.0f)
        , accessCount(0) {
        updateMemoryUsage();
    }
    
    /**
     * @brief 更新访问时间和计数器
     */
    void updateAccess() {
        lastAccessTime = getCurrentTime();
        hitCount++;
        accessCount++;
    }
    
    /**
     * @brief 更新内存使用量统计
     */
    void updateMemoryUsage() {
        memoryUsage = calculateMemoryUsage();
    }
    
private:
    size_t getCurrentTime() const {
        static size_t counter = 0;
        return ++counter;
    }
    
    float calculateMemoryUsage() const {
        size_t totalSize = keyCache.size() + valueCache.size();
        return static_cast<float>(totalSize * sizeof(float)) / (1024.0f * 1024.0f);
    }
};

}

#endif
