#include "cllm/CTokenizer/performance_config.h"
#include <nlohmann/json.hpp>
#include <thread>

namespace cllm {

void TokenizerPerformanceConfig::loadFromJson(const void* jsonPtr) {
    if (!jsonPtr) return;
    
    try {
        const auto& json = *static_cast<const nlohmann::json*>(jsonPtr);
        
        // 缓存配置
        if (json.contains("cache_enabled")) {
            cacheEnabled = json["cache_enabled"].get<bool>();
        }
        if (json.contains("cache_size")) {
            cacheMaxSize = json["cache_size"].get<size_t>();
        }
        if (json.contains("cache_eviction_policy")) {
            cacheEvictionPolicy = json["cache_eviction_policy"].get<std::string>();
        }
        
        // 批处理配置
        if (json.contains("batch_enabled")) {
            batchEnabled = json["batch_enabled"].get<bool>();
        }
        if (json.contains("batch_size")) {
            batchSize = json["batch_size"].get<size_t>();
        }
        if (json.contains("batch_timeout_ms")) {
            batchTimeoutMs = json["batch_timeout_ms"].get<size_t>();
        }
        
        // 线程配置
        if (json.contains("num_threads")) {
            numThreads = json["num_threads"].get<size_t>();
        }
        if (json.contains("parallel_threshold")) {
            parallelThreshold = json["parallel_threshold"].get<size_t>();
        }
        
        // 性能监控
        if (json.contains("enable_metrics")) {
            metricsEnabled = json["enable_metrics"].get<bool>();
        }
        if (json.contains("metrics_reservoir_size")) {
            metricsReservoirSize = json["metrics_reservoir_size"].get<size_t>();
        }
        
        // 资源限制
        if (json.contains("memory_limit")) {
            memoryLimit = json["memory_limit"].get<size_t>();
        }
        if (json.contains("max_input_length")) {
            maxInputLength = json["max_input_length"].get<size_t>();
        }
        
    } catch (const std::exception&) {
        // JSON 解析失败，保持默认值
    }
}

bool TokenizerPerformanceConfig::validate() const {
    // 检查配置合法性
    if (cacheMaxSize == 0 && cacheEnabled) {
        return false; // 缓存大小不能为 0
    }
    
    if (batchSize == 0 && batchEnabled) {
        return false; // 批大小不能为 0
    }
    
    if (cacheEvictionPolicy != "lru" && cacheEvictionPolicy != "fifo") {
        return false; // 非法的淘汰策略
    }
    
    if (metricsReservoirSize == 0 && metricsEnabled) {
        return false; // 监控采样数不能为 0
    }
    
    return true;
}

TokenizerPerformanceConfig TokenizerPerformanceConfig::getDefault() {
    TokenizerPerformanceConfig config;
    config.cacheEnabled = true;
    config.cacheMaxSize = 10000;
    config.cacheEvictionPolicy = "lru";
    
    config.batchEnabled = true;
    config.batchSize = 32;
    config.batchTimeoutMs = 10;
    
    config.numThreads = 0; // 自动检测
    config.parallelThreshold = 100;
    
    config.metricsEnabled = true;
    config.metricsReservoirSize = 1000;
    
    config.memoryLimit = 0; // 无限制
    config.maxInputLength = 1000000;
    
    return config;
}

TokenizerPerformanceConfig TokenizerPerformanceConfig::getHighPerformance() {
    TokenizerPerformanceConfig config;
    
    // 大缓存
    config.cacheEnabled = true;
    config.cacheMaxSize = 100000;
    config.cacheEvictionPolicy = "lru";
    
    // 大批处理
    config.batchEnabled = true;
    config.batchSize = 128;
    config.batchTimeoutMs = 5;
    
    // 最大并行度
    config.numThreads = std::thread::hardware_concurrency();
    config.parallelThreshold = 50;
    
    // 启用监控
    config.metricsEnabled = true;
    config.metricsReservoirSize = 5000;
    
    // 宽松限制
    config.memoryLimit = 0;
    config.maxInputLength = 10000000;
    
    return config;
}

TokenizerPerformanceConfig TokenizerPerformanceConfig::getLowMemory() {
    TokenizerPerformanceConfig config;
    
    // 小缓存
    config.cacheEnabled = true;
    config.cacheMaxSize = 1000;
    config.cacheEvictionPolicy = "fifo";
    
    // 小批处理
    config.batchEnabled = true;
    config.batchSize = 8;
    config.batchTimeoutMs = 20;
    
    // 少线程
    config.numThreads = 2;
    config.parallelThreshold = 200;
    
    // 轻量监控
    config.metricsEnabled = true;
    config.metricsReservoirSize = 100;
    
    // 严格限制
    config.memoryLimit = 100 * 1024 * 1024; // 100 MB
    config.maxInputLength = 100000;
    
    return config;
}

} // namespace cllm
