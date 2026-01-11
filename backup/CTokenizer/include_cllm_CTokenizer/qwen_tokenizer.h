#pragma once

#include "sentencepiece_tokenizer.h"
#include <string>
#include <vector>

namespace cllm {

class QwenTokenizer : public SentencePieceTokenizer {
public:
    explicit QwenTokenizer() : SentencePieceTokenizer(ModelType::QWEN) {}
    
    std::vector<llama_token> encode(const std::string& text, bool addSpecialTokens = true) override {
        // Qwen特定的FIM（Fill-in-the-Middle）处理
        if (needsFimProcessing(text)) {
            return encodeWithFim(text, addSpecialTokens);
        }
        return SentencePieceTokenizer::encode(text, addSpecialTokens);
    }
    
private:
    bool needsFimProcessing(const std::string& text);
    
    std::vector<llama_token> encodeWithFim(const std::string& text, bool addSpecialTokens);
    
    std::string applyQwenPreprocessing(const std::string& text);
};

} // namespace cllm