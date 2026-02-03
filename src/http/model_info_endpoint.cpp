#include "cllm/http/model_info_endpoint.h"
#include "cllm/http/response_builder.h"
#include "cllm/common/json.h"
#include <algorithm>
#include <cctype>

namespace cllm {

ModelInfoEndpoint::ModelInfoEndpoint(ModelExecutor* executor, const std::string& modelPath)
    : ApiEndpoint("ModelInfo", "/model/info", "GET"),
      executor_(executor),
      modelPath_(modelPath) {
}

ModelInfoEndpoint::~ModelInfoEndpoint() {
}

HttpResponse ModelInfoEndpoint::handle(const HttpRequest& request) {
    nlohmann::json responseJson;
    
    // 基本信息
    responseJson["model_path"] = modelPath_;
    responseJson["model_name"] = extractModelName();
    responseJson["model_family"] = detectModelFamily();
    
    // 从 ModelExecutor 获取详细配置
    if (executor_) {
        const ModelConfig& config = executor_->getConfig();
        
        responseJson["model_type"] = config.modelType;
        responseJson["vocab_size"] = config.vocabSize;
        responseJson["tokenizer_vocab_size"] = config.tokenizerVocabSize;
        responseJson["hidden_size"] = config.hiddenSize;
        responseJson["num_layers"] = config.numLayers;
        responseJson["num_attention_heads"] = config.numAttentionHeads;
        responseJson["num_key_value_heads"] = config.numKeyValueHeads;
        responseJson["max_sequence_length"] = config.maxSequenceLength;
        responseJson["intermediate_size"] = config.intermediateSize;
        
        // RoPE 参数
        responseJson["rope_theta"] = config.ropeTheta;
        responseJson["rms_norm_eps"] = config.rmsNormEps;
        
        // 量化信息
        responseJson["quantization"] = {
            {"enabled", config.useQuantization},
            {"type", config.quantizationType}
        };
        
        // llama.cpp 后端参数
        responseJson["llama_cpp"] = {
            {"batch_size", config.llamaBatchSize},
            {"num_threads", config.llamaNumThreads},
            {"gpu_layers", config.llamaGpuLayers},
            {"use_mmap", config.llamaUseMmap},
            {"use_mlock", config.llamaUseMlock}
        };
        
        // KV Cache
        responseJson["use_kv_cache"] = config.useKVCache;
    }
    
    return ResponseBuilder::success(responseJson);
}

std::string ModelInfoEndpoint::extractModelName() const {
    if (modelPath_.empty()) {
        return "unknown";
    }
    
    // 从路径中提取文件名
    size_t lastSlash = modelPath_.find_last_of("/\\");
    std::string filename = (lastSlash != std::string::npos) 
        ? modelPath_.substr(lastSlash + 1) 
        : modelPath_;
    
    // 移除扩展名
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        filename = filename.substr(0, lastDot);
    }
    
    return filename;
}

std::string ModelInfoEndpoint::detectModelFamily() const {
    // 转换路径为小写进行匹配
    std::string pathLower = modelPath_;
    std::transform(pathLower.begin(), pathLower.end(), pathLower.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    
    // 检测模型系列
    if (pathLower.find("qwen") != std::string::npos) {
        return "qwen";
    } else if (pathLower.find("deepseek") != std::string::npos) {
        return "deepseek";
    } else if (pathLower.find("llama") != std::string::npos) {
        return "llama";
    } else if (pathLower.find("mistral") != std::string::npos) {
        return "mistral";
    } else if (pathLower.find("phi") != std::string::npos) {
        return "phi";
    } else if (pathLower.find("gemma") != std::string::npos) {
        return "gemma";
    } else if (pathLower.find("yi") != std::string::npos) {
        return "yi";
    } else if (pathLower.find("baichuan") != std::string::npos) {
        return "baichuan";
    } else if (pathLower.find("chatglm") != std::string::npos) {
        return "chatglm";
    } else if (pathLower.find("internlm") != std::string::npos) {
        return "internlm";
    }
    
    // 如果有 executor，从 modelType 检测
    if (executor_) {
        const std::string& modelType = executor_->getConfig().modelType;
        if (!modelType.empty()) {
            return modelType;
        }
    }
    
    return "unknown";
}

}
