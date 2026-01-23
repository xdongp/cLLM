#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
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
    int serverPort() const;
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
    int serverMinThreads() const;
    std::string serverModelPath() const;
    std::string serverQuantization() const;
    bool serverUseLibTorch() const;
    
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
    int schedulerWaitPollIntervalMs() const;
    
    // 缓存配置
    int cacheDefaultMaxSize() const;
    int cacheDefaultMaxMemoryMb() const;
    bool cacheEnableLru() const;
    bool cacheEnableMemoryLimit() const;
    bool cacheEnableStats() const;
    float cacheEvictionThreshold() const;
    int cacheCleanupInterval() const;
    
    // 服务器配置
    std::string serverHost() const;
    
    // 后端配置
    std::string backendType() const;

    // LibTorch后端配置
    std::vector<int> backendLibTorchSeqLenCandidates() const;
    int backendLibTorchFallbackSeqLen() const;

    // llama.cpp 后端配置
    int backendLlamaCppBatchSize() const;
    int backendLlamaCppNumThreads() const;
    int backendLlamaCppGpuLayers() const;
    int backendLlamaCppNSeqMax() const;  // n_seq_max (1-256, default: 1)
    bool backendLlamaCppUseMmap() const;
    bool backendLlamaCppUseMlock() const;
    
    // Kylin 后端配置
    std::string backendKylinDeviceBackend() const;
    std::string backendKylinOperatorBackend() const;
    
    // API端点配置
    std::string apiEndpointHealthName() const;
    std::string apiEndpointHealthPath() const;
    std::string apiEndpointHealthMethod() const;
    std::string apiEndpointGenerateName() const;
    std::string apiEndpointGeneratePath() const;
    std::string apiEndpointGenerateMethod() const;
    std::string apiEndpointGenerateStreamName() const;
    std::string apiEndpointGenerateStreamPath() const;
    std::string apiEndpointGenerateStreamMethod() const;
    std::string apiEndpointEncodeName() const;
    std::string apiEndpointEncodePath() const;
    std::string apiEndpointEncodeMethod() const;
    
    // API默认参数
    std::string apiDefaultPrompt() const;
    int apiDefaultMaxTokens() const;
    float apiDefaultTemperature() const;
    float apiDefaultTopP() const;
    
    // API响应配置
    std::string apiResponseContentTypeJson() const;
    std::string apiResponseContentTypeStream() const;
    std::string apiResponseHeaderCacheControl() const;
    std::string apiResponseHeaderConnection() const;
    
    // 超时和限制
    float apiTimeoutMin() const;
    float apiTimeoutMax() const;
    float apiTimeoutTokenFactor() const;
    
    // 日志配置
    std::string loggingLevel() const;
    std::string loggingFile() const;

private:
    Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;

    YAML::Node config_;
    mutable std::mutex mutex_;
};

} // namespace cllm