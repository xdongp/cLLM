#pragma once

#include "tokenizer.h"
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
    
private:
    void loadModelConfig(const std::string& configPath);
    void loadSpecialTokens(const std::string& configPath);
    void initializeRegexPatterns();
};

} // namespace cllm