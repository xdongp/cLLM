/**
 * @file cache_manager.h
 * @brief KV缓存内存管理器，管理KV缓存的内存分配和淘汰
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <string>
#include <vector>
#include <list>
#include <unordered_map>
#include <mutex>
#include <functional>
#include <chrono>
#include "cllm/kv_cache/entry.h"

namespace cllm {

/**
 * @brief 内存缓存条目结构
 * 
 * 存储KV缓存条目及其元数据，用于LRU策略管理。
 */
struct MemoryCacheEntry {
    std::string requestId;                             ///< 请求ID
    KVCacheEntry kvCache;                              ///< KV缓存条目
    size_t sequenceLength;                             ///< 序列长度
    std::chrono::steady_clock::time_point lastAccess;  ///< 最后访问时间
};

/**
 * @brief KV缓存内存管理器类
 * 
 * 负责管理KV缓存的内存分配、释放和LRU淘汰策略。
 * 实现线程安全的缓存操作，支持内存限制和自动淘汰。
 * 使用LRU（最近最少使用）策略进行缓存淘汰。
 */
class KVCacheMemoryManager {
public:
    /// 淘汰回调函数类型
    typedef std::function<void(const std::string&)> EvictionCallback;
    
    /**
     * @brief 构造函数
     * @param maxMemoryMb 最大内存限制（MB）
     */
    explicit KVCacheMemoryManager(size_t maxMemoryMb);
    
    /**
     * @brief 插入KV缓存条目
     * @param requestId 请求ID
     * @param keyCache 键缓存数据
     * @param valueCache 值缓存数据
     * @return true 如果插入成功，false 否则
     */
    bool insert(const std::string& requestId,
                const std::vector<float>& keyCache,
                const std::vector<float>& valueCache);
    
    /**
     * @brief 获取KV缓存条目
     * @param requestId 请求ID
     * @param keyCache 输出参数，键缓存数据
     * @param valueCache 输出参数，值缓存数据
     * @return true 如果找到，false 否则
     */
    bool get(const std::string& requestId,
             std::vector<float>& keyCache,
             std::vector<float>& valueCache);
    
    /**
     * @brief 淘汰指定的缓存条目
     * @param requestId 要淘汰的请求ID
     */
    void evict(const std::string& requestId);
    
    /**
     * @brief 获取已使用的内存大小
     * @return 已使用的内存（字节）
     */
    size_t getUsedMemory() const;
    
    /**
     * @brief 获取总内存限制
     * @return 总内存限制（字节）
     */
    size_t getTotalMemory() const;
    
    /**
     * @brief 设置淘汰回调函数
     * @param callback 回调函数，当缓存被淘汰时调用
     */
    void setEvictionCallback(EvictionCallback callback);
    
private:
    /**
     * @brief 淘汰最旧的缓存条目（LRU策略）
     */
    void evictOldest();
    
    /**
     * @brief 计算缓存条目的内存使用量
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     * @return 内存使用量（字节）
     */
    size_t calculateMemoryUsage(const FloatArray& keyCache,
                                const FloatArray& valueCache);
    
    /**
     * @brief 计算缓存条目的内存使用量（vector版本）
     * @param keyCache 键缓存
     * @param valueCache 值缓存
     * @return 内存使用量（字节）
     */
    size_t calculateMemoryUsage(const std::vector<float>& keyCache,
                                const std::vector<float>& valueCache);
    
    std::unordered_map<std::string, std::list<MemoryCacheEntry>::iterator> cacheMap_;  ///< 缓存映射表（用于快速查找）
    std::list<MemoryCacheEntry> cacheList_;          ///< 缓存列表（LRU顺序）
    mutable std::mutex mutex_;                       ///< 互斥锁，保护并发访问
    size_t maxMemoryBytes_;                          ///< 最大内存限制（字节）
    size_t usedMemoryBytes_;                         ///< 已使用内存（字节）
    EvictionCallback evictionCallback_;              ///< 淘汰回调函数
};

}  // namespace cllm
