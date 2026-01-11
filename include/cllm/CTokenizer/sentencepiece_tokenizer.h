#pragma once

#include "tokenizer.h"
#include "token_cache.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>

// Forward declaration for SentencePieceProcessor
namespace sentencepiece {
    class SentencePieceProcessor;
}

namespace cllm {

class SentencePieceTokenizer : public CTokenizer {
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
    ModelType modelType_;
    std::unordered_map<std::string, int> specialTokens_;
    std::unordered_map<int, std::string> idToTokenMap_;
    
    // 特殊token ID
    llama_token bosId_{-1};
    llama_token eosId_{-1};
    llama_token padId_{-1};
    llama_token unkId_{-1};

    // Token 缓存（文本 <-> token 序列）
    TokenCache tokenCache_;
    bool cacheEnabled_{true};
    
public:
    explicit SentencePieceTokenizer(ModelType modelType);
    ~SentencePieceTokenizer() override;
    
    bool load(const std::string& modelPath) override;
    std::vector<llama_token> encode(const std::string& text, bool addSpecialTokens = true) override;
    std::string decode(const std::vector<llama_token>& ids, bool skipSpecialTokens = true) override;
    
    int getVocabSize() const override;
    std::string idToToken(llama_token id) const override;
    llama_token tokenToId(const std::string& token) const override;
    
    llama_token getBosId() const override { return bosId_; }
    llama_token getEosId() const override { return eosId_; }
    llama_token getPadId() const override { return padId_; }
    llama_token getUnkId() const override { return unkId_; }
    
    ModelType getModelType() const override { return modelType_; }

    // 性能配置应用
    void applyPerformanceConfig() override {
        // 应用缓存配置
        cacheEnabled_ = perfConfig_.cacheEnabled;
        if (cacheEnabled_ && tokenCache_.maxSize() != perfConfig_.cacheMaxSize) {
            tokenCache_.setMaxSize(perfConfig_.cacheMaxSize);
        }
        
        // 应用性能监控配置
        if (perfConfig_.metricsEnabled && !perfMonitor_) {
            enablePerformanceMonitor(true);
        } else if (!perfConfig_.metricsEnabled && perfMonitor_) {
            enablePerformanceMonitor(false);
        }
    }

    // 简单的缓存控制接口（后续可与性能配置选项打通）
    void enableCache(bool enable) { cacheEnabled_ = enable; }
    bool isCacheEnabled() const { return cacheEnabled_; }
    void clearCache() { tokenCache_.clear(); }
    
private:
    void loadModelConfig(const std::string& configPath);
    void loadSpecialTokens(const std::string& configPath);
    void initializeRegexPatterns();
};

} // namespace cllm