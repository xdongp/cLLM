#pragma once

#include "tokenizer.h"
#include <memory>
#include <string>
#include <vector>

// Forward declarations to avoid direct llama.h dependency in header
struct llama_vocab;
struct llama_context;
using llama_token = int32_t;

namespace cllm {

class LlamaTokenizer : public CTokenizer {
private:
    void* vocab_;      // Opaque pointer to llama_vocab
    void* context_;    // Opaque pointer to llama_context
    ModelType modelType_;

    // 特殊token ID
    llama_token bosId_{-1};
    llama_token eosId_{-1};
    llama_token padId_{-1};
    llama_token unkId_{-1};

public:
    explicit LlamaTokenizer(ModelType modelType);
    ~LlamaTokenizer() override;

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
};

} // namespace cllm