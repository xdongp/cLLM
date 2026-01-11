/**
 * @file config.h
 * @brief KV缓存配置结构
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_CACHE_CONFIG_H
#define CLLM_CACHE_CONFIG_H

#include <cstddef>

namespace cllm {

/**
 * @brief KV缓存配置结构
 * 
 * 存储KV缓存的配置参数，包括大小限制、淘汰策略等。
 */
struct CacheConfig {
    size_t defaultMaxSize;        ///< 默认最大缓存数量
    size_t defaultMaxMemoryMB;    ///< 默认最大内存限制(MB)
    bool enableLRU;               ///< 是否启用LRU淘汰
    bool enableMemoryLimit;       ///< 是否启用内存限制
    bool enableStats;             ///< 是否启用统计功能
    
    float evictionThreshold;      ///< 淘汰阈值（0.0-1.0）
    size_t cleanupInterval;       ///< 清理间隔（毫秒）
    
    CacheConfig()
        : defaultMaxSize(10)
        , defaultMaxMemoryMB(0)
        , enableLRU(true)
        , enableMemoryLimit(false)
        , enableStats(true)
        , evictionThreshold(0.9f)
        , cleanupInterval(1000) {}
};

}

#endif
