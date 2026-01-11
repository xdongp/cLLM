#pragma once

#include "cllm/tokenizer/tokenizer_base.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace cllm {

/**
 * @brief 通用分词器，支持多种模型类型（DeepSeek、Qwen等）
 */
class UnifiedTokenizer : public TokenizerBase {
public:

    explicit UnifiedTokenizer(const std::string& modelPath, ModelType modelType = ModelType::AUTO);
    ~UnifiedTokenizer() override;

    std::vector<int> encode(const std::string& text, bool addSpecialTokens = false) override;
    std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) override;
    int getVocabSize() const override;
    std::string getTokenText(int tokenId) const override;
    bool isSpecialToken(int tokenId) const override;

    /**
     * @brief 从配置文件自动检测模型类型
     * @param configPath 配置文件路径
     * @return 检测到的模型类型
     */
    static ModelType detectModelType(const std::string& configPath);

    /**
     * @brief 设置特殊token ID
     */
    void setBosToken(int tokenId) { bosTokenId_ = tokenId; }
    void setEosToken(int tokenId) { eosTokenId_ = tokenId; }
    void setPadToken(int tokenId) { padTokenId_ = tokenId; }
    void setUnkToken(int tokenId) { unkTokenId_ = tokenId; }

    /**
     * @brief 获取特殊token ID
     */
    int getBosToken() const { return bosTokenId_; }
    int getEosToken() const { return eosTokenId_; }
    int getPadToken() const { return padTokenId_; }
    int getUnkToken() const { return unkTokenId_; }

private:
    /**
     * @brief 初始化分词器内部实现
     */
    void initializeTokenizer();

    /**
     * @brief 加载模型配置
     */
    void loadModelConfig(const std::string& configPath);

    /**
     * @brief 加载特殊tokens映射
     */
    void loadSpecialTokens(const std::string& configPath);

    std::string modelPath_;
    ModelType modelType_;
    
    // 内部分词器实现指针（具体实现在cpp文件中）
    std::unique_ptr<struct UnifiedTokenizerImpl> tokenizerImpl_;

    // 特殊token ID
    int bosTokenId_ = -1;
    int eosTokenId_ = -1;
    int padTokenId_ = -1;
    int unkTokenId_ = -1;

    // 词汇表相关信息
    int vocabSize_ = 0;
    
    // 特殊token映射
    std::unordered_map<std::string, int> specialTokenToId_;
    std::unordered_map<int, std::string> idToSpecialToken_;
};

} // namespace cllm