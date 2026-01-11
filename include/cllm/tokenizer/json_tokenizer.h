#pragma once

#include "cllm/tokenizer/tokenizer_base.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cllm {

class JsonTokenizer : public TokenizerBase {
public:
    explicit JsonTokenizer(const std::string& jsonPath);
    ~JsonTokenizer() override;

    std::vector<int> encode(const std::string& text, bool addSpecialTokens = false) override;
    std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) override;
    int getVocabSize() const override;
    std::string getTokenText(int tokenId) const override;
    bool isSpecialToken(int tokenId) const override;

private:
    void loadFromJson(const std::string& jsonPath);
    
    std::unordered_map<std::string, int> tokenToId_;
    std::unordered_map<int, std::string> idToToken_;
    int padTokenId_ = -1;
    int eosTokenId_ = -1;
    int bosTokenId_ = -1;
    int unkTokenId_ = -1;
};

} // namespace cllm