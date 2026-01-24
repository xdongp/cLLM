#include "cllm/common/config.h"
#include "cllm/common/logger.h"

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
    
    return config_["http"]["max_input_tokens"].as<int>(120);
}

int Config::serverPort() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["server"]["port"].as<int>(8080);
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
    
    return config_["server"]["num_threads"].as<int>(4);
}

int Config::serverMinThreads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["server"]["min_threads"].as<int>(2);
}

std::string Config::serverModelPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["model"]["path"].as<std::string>("/path/to/model");
}

std::string Config::serverQuantization() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["model"]["quantization"].as<std::string>("fp16");
}

bool Config::serverUseLibTorch() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["server"]["use_libtorch"].as<bool>(false);
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
    
    return config_["scheduler"]["loop_interval"].as<int>(10);
}

int Config::schedulerIdleLoopInterval() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["scheduler"]["idle_loop_interval"].as<int>(100);
}

int Config::schedulerWaitPollIntervalMs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["scheduler"]["wait_poll_interval_ms"].as<int>(10);
}

Config::DynamicBatchTunerConfig Config::dynamicBatchTunerConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);

    DynamicBatchTunerConfig cfg{};
    const auto node = config_["dynamic_batch_tuner"];

    cfg.enabled = node["enabled"].as<bool>(false);
    cfg.strategy = node["strategy"].as<std::string>("static");
    cfg.fixedBatchSize = node["fixed_batch_size"].as<int>(0);

    cfg.minBatchSize = node["min_batch_size"].as<int>(1);
    cfg.maxBatchSize = node["max_batch_size"].as<int>(64);
    cfg.initialBatchSize = node["initial_batch_size"].as<int>(8);

    cfg.probingGrowthFactor = node["probing_growth_factor"].as<double>(2.0);
    cfg.maxProbingAttempts = node["max_probing_attempts"].as<int>(10);
    cfg.timeIncreaseThreshold = node["time_increase_threshold"].as<double>(0.30);
    cfg.adjustmentFactor = node["adjustment_factor"].as<double>(0.30);

    cfg.validationInterval = node["validation_interval"].as<int>(50);
    cfg.explorationInterval = node["exploration_interval"].as<int>(200);
    cfg.probeBatchCount = node["probe_batch_count"].as<int>(10);
    cfg.validationBatchCount = node["validation_batch_count"].as<int>(10);
    cfg.maxConsecutiveTimeIncreases = node["max_consecutive_time_increases"].as<int>(5);

    return cfg;
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

// 服务器配置
std::string Config::serverHost() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["server"]["host"].as<std::string>("0.0.0.0");
}

// 后端配置
std::string Config::backendType() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["type"].as<std::string>("");
}

// LibTorch后端配置
std::vector<int> Config::backendLibTorchSeqLenCandidates() const {
    std::lock_guard<std::mutex> lock(mutex_);
    

    std::vector<int> out;
    auto node = config_["backend"]["libtorch"]["seq_len_candidates"];
    if (node && node.IsSequence()) {
        out.reserve(node.size());
        for (const auto& v : node) {
            out.push_back(v.as<int>());
        }
    }

    if (out.empty()) {
        out = {8, 16, 32, 64, 128, 256};
    }
    return out;
}

int Config::backendLibTorchFallbackSeqLen() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["libtorch"]["fallback_seq_len"].as<int>(8);
}

// llama.cpp 后端配置
int Config::backendLlamaCppBatchSize() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["llama_cpp"]["n_batch"].as<int>(512);
}

int Config::backendLlamaCppNumThreads() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["llama_cpp"]["n_threads"].as<int>(0);
}

int Config::backendLlamaCppGpuLayers() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["llama_cpp"]["n_gpu_layers"].as<int>(0);
}

int Config::backendLlamaCppNSeqMax() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 默认值为1（llama.cpp的默认值），范围：1-256（LLAMA_MAX_SEQ）
    int nSeqMax = config_["backend"]["llama_cpp"]["n_seq_max"].as<int>(1);
    // 验证范围：1-256
    if (nSeqMax < 1) {
        CLLM_WARN("backend.llama_cpp.n_seq_max (%d) is less than 1, using default value 1", nSeqMax);
        return 1;
    }
    if (nSeqMax > 256) {
        CLLM_WARN("backend.llama_cpp.n_seq_max (%d) exceeds LLAMA_MAX_SEQ (256), using 256", nSeqMax);
        return 256;
    }
    return nSeqMax;
}

bool Config::backendLlamaCppUseMmap() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["llama_cpp"]["use_mmap"].as<bool>(true);
}

bool Config::backendLlamaCppUseMlock() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["llama_cpp"]["use_mlock"].as<bool>(false);
}

// Kylin 后端配置
std::string Config::backendKylinDeviceBackend() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["kylin"]["device_backend"].as<std::string>("cpu");
}

std::string Config::backendKylinOperatorBackend() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["backend"]["kylin"]["operator_backend"].as<std::string>("ggml");
}

// API端点配置
std::string Config::apiEndpointHealthName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["health"]["name"].as<std::string>("health");
}

std::string Config::apiEndpointHealthPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["health"]["path"].as<std::string>("/health");
}

std::string Config::apiEndpointHealthMethod() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["health"]["method"].as<std::string>("GET");
}

std::string Config::apiEndpointGenerateName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate"]["name"].as<std::string>("generate");
}

std::string Config::apiEndpointGeneratePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate"]["path"].as<std::string>("/generate");
}

std::string Config::apiEndpointGenerateMethod() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate"]["method"].as<std::string>("POST");
}

std::string Config::apiEndpointGenerateStreamName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate_stream"]["name"].as<std::string>("generate_stream");
}

std::string Config::apiEndpointGenerateStreamPath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate_stream"]["path"].as<std::string>("/generate_stream");
}

std::string Config::apiEndpointGenerateStreamMethod() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["generate_stream"]["method"].as<std::string>("POST");
}

std::string Config::apiEndpointEncodeName() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["encode"]["name"].as<std::string>("encode");
}

std::string Config::apiEndpointEncodePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["encode"]["path"].as<std::string>("/encode");
}

std::string Config::apiEndpointEncodeMethod() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["endpoints"]["encode"]["method"].as<std::string>("POST");
}

// API默认参数
std::string Config::apiDefaultPrompt() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["defaults"]["prompt"].as<std::string>("");
}

int Config::apiDefaultMaxTokens() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // API 默认 max_tokens：优先 api.defaults.max_tokens；否则回退到 model.default_max_tokens
    const int modelDefault = config_["model"]["default_max_tokens"].as<int>(100);
    return config_["api"]["defaults"]["max_tokens"].as<int>(modelDefault);
}

float Config::apiDefaultTemperature() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["defaults"]["temperature"].as<float>(0.7f);
}

float Config::apiDefaultTopP() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["defaults"]["top_p"].as<float>(0.9f);
}

// API响应配置
std::string Config::apiResponseContentTypeJson() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["response"]["content_type"]["json"].as<std::string>("application/json");
}

std::string Config::apiResponseContentTypeStream() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["response"]["content_type"]["stream"].as<std::string>("text/event-stream");
}

std::string Config::apiResponseHeaderCacheControl() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["response"]["headers"]["cache_control"].as<std::string>("no-cache");
}

std::string Config::apiResponseHeaderConnection() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["response"]["headers"]["connection"].as<std::string>("keep-alive");
}

// 超时和限制
float Config::apiTimeoutMin() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["timeouts"]["min"].as<float>(60.0f);
}

float Config::apiTimeoutMax() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["timeouts"]["max"].as<float>(600.0f);
}

float Config::apiTimeoutTokenFactor() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["api"]["timeouts"]["token_factor"].as<float>(10.0f);
}

// 日志配置
std::string Config::loggingLevel() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["logging"]["level"].as<std::string>("info");
}

std::string Config::loggingFile() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    return config_["logging"]["file"].as<std::string>("");
}

} // namespace cllm
