#pragma once

#include "i_tokenizer.h"
#include <memory>
#include <unordered_set>

#ifdef USE_TOKENIZERS_CPP
#include <tokenizers_cpp.h>
#endif

namespace cllm {

/**
 * @brief HFTokenizer - HuggingFace Tokenizer实现
 * 
 * 使用tokenizers-cpp库实现HuggingFace格式分词器支持
 * 支持tokenizer.json格式的模型
 */
class HFTokenizer : public ITokenizer {
public:
    explicit HFTokenizer(ModelType modelType = ModelType::AUTO);
    ~HFTokenizer() override;

    // ITokenizer接口实现
    bool load(const std::string& modelPath) override;
    std::vector<int> encode(const std::string& text, bool addSpecialTokens = true) override;
    std::string decode(const std::vector<int>& ids, bool skipSpecialTokens = true) override;
    
    int getVocabSize() const override;
    std::string idToToken(int id) const override;
    int tokenToId(const std::string& token) const override;
    
    int getBosId() const override;
    int getEosId() const override;
    int getPadId() const override;
    int getUnkId() const override;
    
    ModelType getModelType() const override;
    
    // HF特有功能
    std::vector<std::string> tokenize(const std::string& text);
    bool isSpecialToken(int tokenId) const;

private:
    void loadConfig(const std::string& modelPath);
    void loadSpecialTokens(const std::string& configPath);

#ifdef USE_TOKENIZERS_CPP
    std::unique_ptr<tokenizers::Tokenizer> tokenizer_;  // tokenizers-cpp实例
#else
    std::unique_ptr<ITokenizer> tokenizer_;  // 回退实现
#endif
    
    ModelType modelType_;
    
    // 特殊Token IDs
    int bosId_ = -1;
    int eosId_ = -1;
    int padId_ = -1;
    int unkId_ = -1;
    int maxSpecialTokenId_ = -1;
    
    std::unordered_set<int> specialTokenIds_;
};

} // namespace cllm