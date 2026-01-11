#include "cllm/common/config.h"
#include "cllm/common/logger.h"
#include <fstream>

namespace cllm {

void Config::load(const std::string& configPath) {
    std::lock_guard<std::mutex> lock(mutex_);
    try {
        config_ = YAML::LoadFile(configPath);
        CLLM_INFO("Configuration loaded from: %s", configPath.c_str());
    } catch (const YAML::Exception& e) {
        CLLM_ERROR("Failed to load config file: %s. Error: %s", 
                  configPath.c_str(), e.what());
        throw std::runtime_error("Config load failed");
    }
}

// HTTP配置
int Config::httpMaxInputTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["http"]["max_input_tokens"].as<int>(7);
}

int Config::httpPort() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["http"]["port"].as<int>(8080);
}

int Config::httpTimeoutMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["http"]["timeout_ms"].as<int>(30000);
}

// 调度器配置
int Config::schedulerMaxIterations() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["max_iterations"].as<int>(1000);
}

int Config::schedulerBatchTimeoutMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["batch_timeout_ms"].as<int>(500);
}

int Config::schedulerMaxBatchSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["max_batch_size"].as<int>(32);
}

// 模型配置
int Config::modelMaxContextLength() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["max_context_length"].as<int>(2048);
}

int Config::modelDefaultMaxTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["default_max_tokens"].as<int>(100);
}

float Config::modelTemperature() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["temperature"].as<float>(0.7f);
}

int Config::modelTopK() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["top_k"].as<int>(50);
}

float Config::modelTopP() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["top_p"].as<float>(0.9f);
}

// 队列配置
int Config::queueMaxSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["queue"]["max_size"].as<int>(1000);
}

float Config::queuePriorityWeight() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["queue"]["priority_weight"].as<float>(1.0f);
}

// 内存配置
int Config::memoryCacheLimitMb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["memory"]["cache_limit_mb"].as<int>(2048);
}

int Config::memoryMonitorIntervalMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["memory"]["monitor_interval_ms"].as<int>(1000);
}

// 服务器资源配置
int Config::serverMaxBatchSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["max_batch_size"].as<int>(8);
}

int Config::serverMaxContextLength() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["max_context_length"].as<int>(2048);
}

int Config::serverKvCacheMaxSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["kv_cache_max_size"].as<int>(100);
}

int Config::serverKvCacheMaxMemoryMb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["kv_cache_max_memory_mb"].as<int>(4096);
}

int Config::serverMemoryLimitMb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["memory_limit_mb"].as<int>(16384);
}

int Config::serverNumThreads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["resources"]["num_threads"].as<int>(8);
}

std::string Config::serverModelPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["path"].as<std::string>("/path/to/model");
}

std::string Config::serverQuantization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["model"]["quantization"].as<std::string>("fp16");
}

// 采样器配置
float Config::samplerTemperature() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_ && config_["sampler"]) {
        return config_["sampler"]["temperature"].as<float>(0.7f);
    }
    return 0.7f;
}

int Config::samplerTopK() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_ && config_["sampler"]) {
        return config_["sampler"]["top_k"].as<int>(50);
    }
    return 50;
}

float Config::samplerTopP() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_ && config_["sampler"]) {
        return config_["sampler"]["top_p"].as<float>(0.9f);
    }
    return 0.9f;
}

float Config::samplerGreedyThreshold() const {
    std::lock_guard<std::mutex> lock(mutex_);
    if (config_ && config_["sampler"]) {
        return config_["sampler"]["greedy_threshold"].as<float>(0.1f);
    }
    return 0.1f;
}

// 调度器其他配置
float Config::schedulerContextUsageThreshold() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["context_usage_threshold"].as<float>(0.75f);
}

float Config::schedulerDefaultTemperature() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["default_temperature"].as<float>(0.7f);
}

float Config::schedulerDefaultTopP() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["default_top_p"].as<float>(0.9f);
}

int Config::schedulerDefaultTopK() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["default_top_k"].as<int>(50);
}

int Config::schedulerDefaultMaxTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["default_max_tokens"].as<int>(100);
}

float Config::schedulerRequestTimeout() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["request_timeout"].as<float>(300.0f);
}

int Config::schedulerLoopInterval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["loop_interval"].as<int>(100);
}

int Config::schedulerIdleLoopInterval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["scheduler"]["idle_loop_interval"].as<int>(10000);
}

// 缓存配置
int Config::cacheDefaultMaxSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["default_max_size"].as<int>(10);
}

int Config::cacheDefaultMaxMemoryMb() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["default_max_memory_mb"].as<int>(0);
}

bool Config::cacheEnableLru() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["enable_lru"].as<bool>(true);
}

bool Config::cacheEnableMemoryLimit() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["enable_memory_limit"].as<bool>(false);
}

bool Config::cacheEnableStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["enable_stats"].as<bool>(true);
}

float Config::cacheEvictionThreshold() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["eviction_threshold"].as<float>(0.9f);
}

int Config::cacheCleanupInterval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_["cache"]["cleanup_interval"].as<int>(1000);
}

} // namespace cllm