#include "cllm/model/config.h"
#include <fstream>
#include <sstream>
#include <algorithm>

namespace cllm {

void ModelConfig::loadFromConfigFile(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return;
    }
    
    std::string line;
    while (std::getline(file, line)) {
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            
            key.erase(0, key.find_first_not_of(" \t"));
            key.erase(key.find_last_not_of(" \t") + 1);
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (key == "model_type") {
                modelType = value;
            } else if (key == "vocab_size") {
                vocabSize = std::stoul(value);
            } else if (key == "hidden_size") {
                hiddenSize = std::stoul(value);
            } else if (key == "num_layers") {
                numLayers = std::stoul(value);
            } else if (key == "num_attention_heads") {
                numAttentionHeads = std::stoul(value);
            } else if (key == "max_sequence_length") {
                maxSequenceLength = std::stoul(value);
            } else if (key == "intermediate_size") {
                intermediateSize = std::stoul(value);
            } else if (key == "use_kv_cache") {
                useKVCache = (value == "true" || value == "1");
            } else if (key == "use_quantization") {
                useQuantization = (value == "true" || value == "1");
            } else if (key == "quantization_type") {
                quantizationType = value;
            }
        }
    }
}

std::string ModelConfig::toString() const {
    std::ostringstream oss;
    oss << "ModelConfig{"
        << "modelType=" << modelType
        << ", vocabSize=" << vocabSize
        << ", hiddenSize=" << hiddenSize
        << ", numLayers=" << numLayers
        << ", numAttentionHeads=" << numAttentionHeads
        << ", maxSequenceLength=" << maxSequenceLength
        << ", intermediateSize=" << intermediateSize
        << ", useKVCache=" << (useKVCache ? "true" : "false")
        << ", useQuantization=" << (useQuantization ? "true" : "false")
        << ", quantizationType=" << quantizationType
        << "}";
    return oss.str();
}

}
