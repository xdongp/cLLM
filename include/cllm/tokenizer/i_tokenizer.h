#pragma once

#include <string>
#include <vector>

namespace cllm {

enum class ModelType {
    AUTO,
    QWEN,
    QWEN2,
    DEEPSEEK_LLM,
    DEEPSEEK_CODER,
    DEEPSEEK3_LLM,
    LLAMA,
    BERT,
    GPT2,
    SPM,
    BPE,
    WPM
};

class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    
    virtual bool load(const std::string& modelPath) = 0;
    virtual std::vector<int> encode(const std::string& text, bool addSpecialTokens) = 0;
    virtual std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) = 0;
    
    virtual int getVocabSize() const = 0;
    virtual std::string idToToken(int id) const = 0;
    virtual int tokenToId(const std::string& token) const = 0;
    
    virtual int getBosId() const = 0;
    virtual int getEosId() const = 0;
    virtual int getPadId() const = 0;
    virtual int getUnkId() const = 0;
    
    virtual ModelType getModelType() const = 0;
};

} // namespace cllm
