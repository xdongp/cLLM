#pragma once

#include <cstddef>
#include <string>

namespace cllm {

/**
 * @brief 分词器性能配置
 * 
 * 控制分词器的性能相关参数，包括：
 * - 缓存设置
 * - 批处理设置
 * - 线程配置
 * - 监控选项
 * - 资源限制
 */
struct TokenizerPerformanceConfig {
    // ===== 缓存配置 =====
    
    /// 是否启用 Token 缓存
    bool cacheEnabled = true;
    
    /// 缓存最大条目数（encode + decode 总和）
    size_t cacheMaxSize = 10000;
    
    /// 缓存淘汰策略："lru" (Least Recently Used) 或 "fifo" (First In First Out)
    std::string cacheEvictionPolicy = "lru";
    
    // ===== 批处理配置 =====
    
    /// 批处理是否启用
    bool batchEnabled = true;
    
    /// 批处理大小（单次批处理的最大文本数）
    size_t batchSize = 32;
    
    /// 批处理超时（毫秒）- 即使未满 batch_size 也会触发处理
    size_t batchTimeoutMs = 10;
    
    // ===== 线程配置 =====
    
    /// 并行处理线程数（0 = 自动检测 CPU 核心数）
    size_t numThreads = 0;
    
    /// 单个任务触发多线程的最小规模（例如文本长度或 batch 大小）
    size_t parallelThreshold = 100;
    
    // ===== 性能监控 =====
    
    /// 是否启用性能指标收集
    bool metricsEnabled = true;
    
    /// 延迟采样数（用于 P50/P95/P99 计算，蓄水池采样）
    size_t metricsReservoirSize = 1000;
    
    // ===== 资源限制 =====
    
    /// 内存限制（字节，0 = 无限制）
    size_t memoryLimit = 0;
    
    /// 单次 encode/decode 的最大输入长度（字符数/token 数）
    size_t maxInputLength = 1000000;
    
    // ===== 辅助方法 =====
    
    /**
     * @brief 从 JSON 对象加载配置
     * @param json JSON 配置对象（nlohmann::json）
     */
    void loadFromJson(const void* json);
    
    /**
     * @brief 验证配置合法性
     * @return 如果配置有效返回 true
     */
    bool validate() const;
    
    /**
     * @brief 获取默认配置
     * @return 默认性能配置
     */
    static TokenizerPerformanceConfig getDefault();
    
    /**
     * @brief 获取高性能配置（适用于服务器环境）
     * @return 高性能配置
     */
    static TokenizerPerformanceConfig getHighPerformance();
    
    /**
     * @brief 获取低内存配置（适用于资源受限环境）
     * @return 低内存配置
     */
    static TokenizerPerformanceConfig getLowMemory();
};

} // namespace cllm
