#include "cllm/CTokenizer/manager.h"
#include "cllm/CTokenizer/model_detector.h"
#include <fstream>

    namespace cllm {

ModelType TokenizerManager::detectModelType(const std::string& configPath) {
    return ModelDetector::detectModelType(configPath);
}

std::unique_ptr<CTokenizer> TokenizerManager::createTokenizer(const std::string& modelType) {
    ModelType type = stringToModelType(modelType);
    
    switch (type) {
        case ModelType::QWEN:
        case ModelType::QWEN2:
            return std::make_unique<QwenTokenizer>();
        case ModelType::DEEPSEEK_LLM:
        case ModelType::DEEPSEEK_CODER:
        case ModelType::DEEPSEEK3_LLM:
            return std::make_unique<DeepSeekTokenizer>(type);
        default:
            return std::make_unique<SentencePieceTokenizer>(type);
    }
}

ModelType TokenizerManager::stringToModelType(const std::string& modelTypeStr) {
    if (modelTypeStr == "qwen" || modelTypeStr == "Qwen") {
        return ModelType::QWEN;
    } else if (modelTypeStr == "qwen2" || modelTypeStr == "Qwen2") {
        return ModelType::QWEN2;
    } else if (modelTypeStr == "deepseek-llm" || modelTypeStr == "DeepSeek-LLM") {
        return ModelType::DEEPSEEK_LLM;
    } else if (modelTypeStr == "deepseek-coder" || modelTypeStr == "DeepSeek-Coder") {
        return ModelType::DEEPSEEK_CODER;
    } else if (modelTypeStr == "deepseek3-llm" || modelTypeStr == "DeepSeek3-LLM") {
        return ModelType::DEEPSEEK3_LLM;
    } else if (modelTypeStr == "llama" || modelTypeStr == "Llama") {
        return ModelType::LLAMA;
    } else if (modelTypeStr == "bert" || modelTypeStr == "Bert") {
        return ModelType::BERT;
    } else if (modelTypeStr == "gpt2" || modelTypeStr == "GPT2") {
        return ModelType::GPT2;
    } else if (modelTypeStr == "spm" || modelTypeStr == "SPM") {
        return ModelType::SPM;
    } else if (modelTypeStr == "bpe" || modelTypeStr == "BPE") {
        return ModelType::BPE;
    } else if (modelTypeStr == "wpm" || modelTypeStr == "WPM") {
        return ModelType::WPM;
    } else {
        return ModelType::AUTO;
    }
}

} // namespace cllm