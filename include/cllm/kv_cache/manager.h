/**
 * @file manager.h
 * @brief KV缓存管理器，管理多个缓存实例
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_KV_CACHE_MANAGER_H
#define CLLM_KV_CACHE_MANAGER_H

#include "cllm/kv_cache/cache.h"
#include "cllm/kv_cache/config.h"
#include <map>
#include <string>
#include <vector>
#include <mutex>

namespace cllm {

/**
 * @brief 管理器统计信息结构
 * 
 * 记录KV缓存管理器的全局统计信息。
 */
struct ManagerStats {
    size_t totalCaches;       ///< 总缓存数
    size_t totalMemoryUsage;  ///< 总内存使用量
    size_t totalHits;         ///< 总命中次数
    size_t totalMisses;       ///< 总未命中次数
    
    ManagerStats()
        : totalCaches(0)
        , totalMemoryUsage(0)
        , totalHits(0)
        , totalMisses(0) {}
    
    /**
     * @brief 获取全局命中率
     * @return 命中率(0.0-1.0)
     */
    float getGlobalHitRate() const {
        size_t total = totalHits + totalMisses;
        if (total == 0) {
            return 0.0f;
        }
        return static_cast<float>(totalHits) / total;
    }
    
    /**
     * @brief 转换为字符串
     * @return 格式化的统计信息
     */
    std::string toString() const {
        return "ManagerStats{totalCaches=" + std::to_string(totalCaches) +
               ", totalMemoryUsage=" + std::to_string(totalMemoryUsage) +
               ", totalHits=" + std::to_string(totalHits) +
               ", totalMisses=" + std::to_string(totalMisses) +
               ", globalHitRate=" + std::to_string(getGlobalHitRate()) + "}";
    }
};

/**
 * @brief KV缓存管理器类
 * 
 * 管理多个KV缓存实例，提供创建、获取和移除缓存的功能。
 */
class KVCacheManager {
public:
    /**
     * @brief 构造函数
     * @param config 缓存配置
     */
    explicit KVCacheManager(const CacheConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~KVCacheManager();
    
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    
    /**
     * @brief 创建新的缓存实例
     * @param cacheName 缓存名称
     * @return 缓存实例指针
     */
    KVCache* createCache(const std::string& cacheName);
    
    /**
     * @brief 获取缓存实例
     * @param cacheName 缓存名称
     * @return 缓存实例指针，如果不存在则返回nullptr
     */
    KVCache* getCache(const std::string& cacheName);
    
    /**
     * @brief 移除缓存实例
     * @param cacheName 缓存名称
     */
    void removeCache(const std::string& cacheName);
    
    /**
     * @brief 获取所有缓存名称
     * @return 缓存名称列表
     */
    std::vector<std::string> getCacheNames() const;
    
    /**
     * @brief 获取总内存使用量
     * @return 总内存使用量(MB)
     */
    size_t getTotalMemoryUsage() const;
    
    /**
     * @brief 获取统计信息
     * @return 管理器统计信息
     */
    ManagerStats getStats() const;
    
private:
    std::map<std::string, KVCache*> caches_;  ///< 缓存实例映射表
    CacheConfig config_;                      ///< 缓存配置
    
    mutable std::mutex managerMutex_;  ///< 管理器互斥锁
};

}

#endif
