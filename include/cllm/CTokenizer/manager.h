#pragma once

#include "tokenizer.h"
#include "sentencepiece_tokenizer.h"
#include "qwen_tokenizer.h"
#include "deepseek_tokenizer.h"
#include <unordered_map>
#include <memory>
#include <string>

namespace cllm {

class TokenizerManager {
private:
    std::unordered_map<std::string, std::unique_ptr<CTokenizer>> tokenizers_;
    
public:
    CTokenizer* getTokenizer(const std::string& modelType) {
        // 根据模型类型创建对应的分词器
        auto it = tokenizers_.find(modelType);
        if (it != tokenizers_.end()) {
            return it->second.get();
        }
        
        // 创建新的分词器实例
        std::unique_ptr<CTokenizer> tokenizer = createTokenizer(modelType);
        if (!tokenizer) {
            return nullptr;
        }
        
        CTokenizer* ptr = tokenizer.get();
        tokenizers_[modelType] = std::move(tokenizer);
        return ptr;
    }
    
    ModelType detectModelType(const std::string& configPath);
    
private:
    std::unique_ptr<CTokenizer> createTokenizer(const std::string& modelType);
    
    ModelType stringToModelType(const std::string& modelTypeStr);
};

} // namespace cllm