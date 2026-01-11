#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <mutex>

namespace cllm {

class Config {
public:
    static Config& instance() {
        static Config instance;
        return instance;
    }

    void load(const std::string& configPath);
    
    // HTTP配置
    int httpMaxInputTokens() const;
    int httpPort() const;
    int httpTimeoutMs() const;
    
    // 调度器配置
    int schedulerMaxIterations() const;
    int schedulerBatchTimeoutMs() const;
    int schedulerMaxBatchSize() const;
    
    // 模型配置
    int modelMaxContextLength() const;
    int modelDefaultMaxTokens() const;
    float modelTemperature() const;
    int modelTopK() const;
    float modelTopP() const;
    
    // 队列配置
    int queueMaxSize() const;
    float queuePriorityWeight() const;
    
    // 内存配置
    int memoryCacheLimitMb() const;
    int memoryMonitorIntervalMs() const;
    
    // 服务器资源配置
    int serverMaxBatchSize() const;
    int serverMaxContextLength() const;
    int serverKvCacheMaxSize() const;
    int serverKvCacheMaxMemoryMb() const;
    int serverMemoryLimitMb() const;
    int serverNumThreads() const;
    std::string serverModelPath() const;
    std::string serverQuantization() const;
    
    // 采样器配置
    float samplerTemperature() const;
    int samplerTopK() const;
    float samplerTopP() const;
    float samplerGreedyThreshold() const;
    
    // 调度器其他配置
    float schedulerContextUsageThreshold() const;
    float schedulerDefaultTemperature() const;
    float schedulerDefaultTopP() const;
    int schedulerDefaultTopK() const;
    int schedulerDefaultMaxTokens() const;
    float schedulerRequestTimeout() const;
    int schedulerLoopInterval() const;
    int schedulerIdleLoopInterval() const;
    
    // 缓存配置
    int cacheDefaultMaxSize() const;
    int cacheDefaultMaxMemoryMb() const;
    bool cacheEnableLru() const;
    bool cacheEnableMemoryLimit() const;
    bool cacheEnableStats() const;
    float cacheEvictionThreshold() const;
    int cacheCleanupInterval() const;

private:
    Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    YAML::Node config_;
    mutable std::mutex mutex_;
};

} // namespace cllm