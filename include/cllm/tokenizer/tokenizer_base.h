#pragma once

#include <string>
#include <vector>

namespace cllm {

class TokenizerBase {
public:
    virtual ~TokenizerBase() = default;
    
    virtual std::vector<int> encode(const std::string& text, bool addSpecialTokens = false) = 0;
    virtual std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) = 0;
    virtual int getVocabSize() const = 0;
    virtual std::string getTokenText(int tokenId) const = 0;
    virtual bool isSpecialToken(int tokenId) const = 0;
};

} // namespace cllm