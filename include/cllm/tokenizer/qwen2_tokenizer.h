#pragma once

#include "cllm/tokenizer/tokenizer_base.h"
#include <sentencepiece_processor.h>
#include <unordered_map>
#include <memory>

namespace cllm {

class Qwen2Tokenizer : public TokenizerBase {
public:
    explicit Qwen2Tokenizer(const std::string& modelPath);
    ~Qwen2Tokenizer() override;

    std::vector<int> encode(const std::string& text, bool addSpecialTokens = false) override;
    std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) override;
    int getVocabSize() const override;
    std::string getTokenText(int tokenId) const override;
    bool isSpecialToken(int tokenId) const override;

private:
    void loadSpecialTokens(const std::string& configPath);
    
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
    std::unordered_map<std::string, int> tokenToId_;
    std::unordered_map<int, std::string> idToToken_;
    std::unordered_map<std::string, int> specialTokenToId_;
    std::unordered_map<int, std::string> idToSpecialToken_;
    int padTokenId_ = -1;
    int eosTokenId_ = -1;
    int bosTokenId_ = -1;
    int unkTokenId_ = -1;
};

} // namespace cllm