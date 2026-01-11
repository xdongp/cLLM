#pragma once

#include "sentencepiece_tokenizer.h"
#include <string>
#include <vector>

namespace cllm {

class DeepSeekTokenizer : public SentencePieceTokenizer {
public:
    explicit DeepSeekTokenizer(ModelType modelType) : SentencePieceTokenizer(modelType) {}
    
    std::vector<llama_token> encode(const std::string& text, bool addSpecialTokens = true) override {
        // 应用DeepSeek特定的预处理
        std::string processedText = applyDeepSeekPreprocessing(text);
        return SentencePieceTokenizer::encode(processedText, addSpecialTokens);
    }
    
private:
    std::string applyDeepSeekPreprocessing(const std::string& text);
    
    std::string applyDeepSeekLLMPreprocessing(const std::string& text);
    
    std::string applyDeepSeekCoderPreprocessing(const std::string& text);
    
    std::string applyDeepSeek3Preprocessing(const std::string& text);
};

} // namespace cllm