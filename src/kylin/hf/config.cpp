/**
 * @file hf_config.cpp
 * @brief HuggingFace 配置加载器实现
 */

#include "cllm/kylin/hf/config.h"
#include "cllm/common/logger.h"

#include <fstream>
#include <sstream>
#include <algorithm>

namespace cllm {
namespace kylin {

// 简单的 JSON 值提取辅助函数
static std::string extractJsonString(const std::string& json, const std::string& key) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    
    size_t colonPos = json.find(':', pos);
    if (colonPos == std::string::npos) return "";
    
    size_t valueStart = json.find('"', colonPos);
    if (valueStart == std::string::npos) return "";
    
    size_t valueEnd = json.find('"', valueStart + 1);
    if (valueEnd == std::string::npos) return "";
    
    return json.substr(valueStart + 1, valueEnd - valueStart - 1);
}

static int extractJsonInt(const std::string& json, const std::string& key, int defaultValue = 0) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return defaultValue;
    
    size_t colonPos = json.find(':', pos);
    if (colonPos == std::string::npos) return defaultValue;
    
    // 跳过空格
    size_t valueStart = colonPos + 1;
    while (valueStart < json.size() && (json[valueStart] == ' ' || json[valueStart] == '\n' || json[valueStart] == '\t')) {
        valueStart++;
    }
    
    // 提取数字
    size_t valueEnd = valueStart;
    while (valueEnd < json.size() && (isdigit(json[valueEnd]) || json[valueEnd] == '-')) {
        valueEnd++;
    }
    
    if (valueEnd == valueStart) return defaultValue;
    
    try {
        return std::stoi(json.substr(valueStart, valueEnd - valueStart));
    } catch (...) {
        return defaultValue;
    }
}

static float extractJsonFloat(const std::string& json, const std::string& key, float defaultValue = 0.0f) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return defaultValue;
    
    size_t colonPos = json.find(':', pos);
    if (colonPos == std::string::npos) return defaultValue;
    
    // 跳过空格
    size_t valueStart = colonPos + 1;
    while (valueStart < json.size() && (json[valueStart] == ' ' || json[valueStart] == '\n' || json[valueStart] == '\t')) {
        valueStart++;
    }
    
    // 提取数字
    size_t valueEnd = valueStart;
    while (valueEnd < json.size() && (isdigit(json[valueEnd]) || json[valueEnd] == '-' || json[valueEnd] == '.' || json[valueEnd] == 'e' || json[valueEnd] == 'E' || json[valueEnd] == '+')) {
        valueEnd++;
    }
    
    if (valueEnd == valueStart) return defaultValue;
    
    try {
        return std::stof(json.substr(valueStart, valueEnd - valueStart));
    } catch (...) {
        return defaultValue;
    }
}

static bool extractJsonBool(const std::string& json, const std::string& key, bool defaultValue = false) {
    std::string pattern = "\"" + key + "\"";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return defaultValue;
    
    size_t colonPos = json.find(':', pos);
    if (colonPos == std::string::npos) return defaultValue;
    
    // 查找 true 或 false
    size_t truePos = json.find("true", colonPos);
    size_t falsePos = json.find("false", colonPos);
    
    // 确保找到的值在合理范围内
    if (truePos != std::string::npos && truePos < colonPos + 20) {
        return true;
    }
    if (falsePos != std::string::npos && falsePos < colonPos + 20) {
        return false;
    }
    
    return defaultValue;
}

HFModelConfig loadHFConfig(const std::string& configPath) {
    CLLM_INFO("[HFConfig] Loading config from: %s", configPath.c_str());
    
    std::ifstream file(configPath);
    if (!file.is_open()) {
        CLLM_ERROR("[HFConfig] Failed to open config file: %s", configPath.c_str());
        return HFModelConfig{};
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string json = buffer.str();
    
    HFModelConfig config;
    
    // 基本信息
    config.modelType = extractJsonString(json, "model_type");
    config.torchDtype = extractJsonString(json, "torch_dtype");
    config.hiddenAct = extractJsonString(json, "hidden_act");
    
    // architectures 数组中的第一个
    size_t archPos = json.find("\"architectures\"");
    if (archPos != std::string::npos) {
        size_t arrayStart = json.find('[', archPos);
        size_t firstQuote = json.find('"', arrayStart);
        size_t secondQuote = json.find('"', firstQuote + 1);
        if (firstQuote != std::string::npos && secondQuote != std::string::npos) {
            config.architecture = json.substr(firstQuote + 1, secondQuote - firstQuote - 1);
        }
    }
    
    // 模型结构
    config.hiddenSize = extractJsonInt(json, "hidden_size");
    config.numHiddenLayers = extractJsonInt(json, "num_hidden_layers");
    config.numAttentionHeads = extractJsonInt(json, "num_attention_heads");
    config.numKeyValueHeads = extractJsonInt(json, "num_key_value_heads", config.numAttentionHeads);
    config.intermediateSize = extractJsonInt(json, "intermediate_size");
    config.vocabSize = extractJsonInt(json, "vocab_size");
    config.headDim = extractJsonInt(json, "head_dim", 0);
    config.maxPositionEmbeddings = extractJsonInt(json, "max_position_embeddings");
    
    // 归一化参数
    config.rmsNormEps = extractJsonFloat(json, "rms_norm_eps", 1e-6f);
    
    // RoPE 参数
    config.ropeTheta = extractJsonFloat(json, "rope_theta", 10000.0f);
    
    // 特殊配置
    config.tieWordEmbeddings = extractJsonBool(json, "tie_word_embeddings");
    config.attentionBias = extractJsonBool(json, "attention_bias");
    
    // Token IDs
    config.bosTokenId = extractJsonInt(json, "bos_token_id");
    config.eosTokenId = extractJsonInt(json, "eos_token_id");
    config.padTokenId = extractJsonInt(json, "pad_token_id", -1);
    
    if (config.isValid()) {
        CLLM_INFO("[HFConfig] Loaded: %s, hidden=%d, layers=%d, heads=%d, kv_heads=%d, vocab=%d",
                  config.modelType.c_str(), config.hiddenSize, config.numHiddenLayers,
                  config.numAttentionHeads, config.numKeyValueHeads, config.vocabSize);
    } else {
        CLLM_ERROR("[HFConfig] Invalid config");
    }
    
    return config;
}

HFModelConfig loadHFConfigFromDir(const std::string& modelDir) {
    std::string configPath = modelDir;
    if (configPath.back() != '/') {
        configPath += '/';
    }
    configPath += "config.json";
    return loadHFConfig(configPath);
}

void HFModelConfig::print() const {
    CLLM_INFO("=== HuggingFace Model Config ===");
    CLLM_INFO("  Architecture: %s", architecture.c_str());
    CLLM_INFO("  Model Type: %s", modelType.c_str());
    CLLM_INFO("  Hidden Size: %d", hiddenSize);
    CLLM_INFO("  Num Layers: %d", numHiddenLayers);
    CLLM_INFO("  Num Attention Heads: %d", numAttentionHeads);
    CLLM_INFO("  Num KV Heads: %d", getNumKVHeads());
    CLLM_INFO("  Head Dim: %d", getHeadDim());
    CLLM_INFO("  Intermediate Size: %d", intermediateSize);
    CLLM_INFO("  Vocab Size: %d", vocabSize);
    CLLM_INFO("  RMS Norm Eps: %.2e", rmsNormEps);
    CLLM_INFO("  RoPE Theta: %.0f", ropeTheta);
    CLLM_INFO("  Tie Embeddings: %s", tieWordEmbeddings ? "true" : "false");
    CLLM_INFO("  Torch Dtype: %s", torchDtype.c_str());
}

} // namespace kylin
} // namespace cllm
