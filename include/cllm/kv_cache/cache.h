/**
 * @file cache.h
 * @brief KV缓存类，实现LRU缓存策略
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_KV_CACHE_H
#define CLLM_KV_CACHE_H

#include "cllm/kv_cache/entry.h"
#include "cllm/kv_cache/stats.h"
#include "cllm/memory/float_array.h"
#include "cllm/common/config.h"
#include <map>
#include <list>
#include <mutex>
#include <cstddef>

namespace cllm {

/**
 * @brief KV缓存类
 * 
 * 实现基于LRU策略的KV缓存，支持内存限制和统计功能。
 * 线程安全，支持并发访问。
 */
class KVCache {
public:
    /**
     * @brief 构造函数
     * @param maxSize 最大缓存数量，默认10
     * @param maxMemoryMB 最大内存限制(MB)，0表示不限制
     */
    explicit KVCache(size_t maxSize = 10, size_t maxMemoryMB = 0);
    
    /**
     * @brief 析构函数
     */
    ~KVCache();
    
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;
    
    /**
     * @brief 获取缓存条目
     * @param sequenceId 序列ID
     * @param entry 输出参数，缓存条目
     * @return true 如果找到，false 否则
     */
    bool get(size_t sequenceId, KVCacheEntry& entry);
    
    /**
     * @brief 存储缓存条目
     * @param sequenceId 序列ID
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     */
    void put(size_t sequenceId, const FloatArray& keyCache, const FloatArray& valueCache);
    
    /**
     * @brief 移除缓存条目
     * @param sequenceId 序列ID
     */
    void remove(size_t sequenceId);
    
    /**
     * @brief 清空缓存
     */
    void clear();
    
    /**
     * @brief 获取缓存大小
     * @return 缓存条目数
     */
    size_t size() const;
    
    /**
     * @brief 判断是否包含指定序列
     * @param sequenceId 序列ID
     * @return true 如果包含，false 否则
     */
    bool contains(size_t sequenceId) const;
    
    /**
     * @brief 获取统计信息
     * @return 缓存统计信息
     */
    CacheStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
    /**
     * @brief 获取内存使用量
     * @return 内存使用量(MB)
     */
    float getMemoryUsageMB() const;
    
    /**
     * @brief 获取最大缓存数量
     * @return 最大缓存数量
     */
    size_t getMaxSize() const;
    
    /**
     * @brief 获取最大内存限制
     * @return 最大内存限制(MB)
     */
    size_t getMaxMemoryMB() const;
    
    /**
     * @brief 增量更新缓存
     * @param sequenceId 序列ID
     * @param newKeyPart 新的键部分
     * @param newValuePart 新的值部分
     */
    void updateIncremental(
        size_t sequenceId,
        const FloatArray& newKeyPart,
        const FloatArray& newValuePart
    );
    
private:
    void evictOldest();  ///< 淘汰最旧条目
    void ensureMemoryLimit();  ///< 确保内存不超限
    void ensureMemoryLimitForNewEntry(double newEntrySize);  ///< 确保添加新条目后内存不超限
    double calculateMemoryUsage(const FloatArray& keyCache, const FloatArray& valueCache);  ///< 计算内存使用量
    
    std::map<size_t, KVCacheEntry> cache_;  ///< 缓存映射表
    std::list<size_t> accessList_;          ///< LRU访问列表
    
    size_t maxSize_;          ///< 最大缓存数量
    size_t maxMemoryMB_;      ///< 最大内存限制(MB)
    double memoryUsage_;      ///< 当前内存使用量(MB)
    
    mutable std::mutex cacheMutex_;  ///< 缓存互斥锁
    
    CacheStats stats_;  ///< 统计信息
};

}

#endif
