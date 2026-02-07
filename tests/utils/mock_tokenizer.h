#pragma once

#include <cllm/tokenizer/i_tokenizer.h>
#include <string>
#include <vector>

namespace cllm {
namespace test {

/**
 * @brief Mock Tokenizer用于测试
 * 提供简单的token编解码实现，不依赖真实模型
 */
class MockTokenizer : public ITokenizer {
public:
    MockTokenizer()
        : eosId_(2)
        , padId_(0)
        , bosId_(1)
        , unkId_(3)
        , vocabSize_(1000) {}
    
    ~MockTokenizer() override = default;
    
    bool load(const std::string& modelPath) override {
        return true;
    }
    
    /**
     * @brief 编码文本为token序列
     * 简单实现：将字符转换为token ID
     */
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        std::vector<int> tokens;
        
        if (addSpecialTokens) {
            tokens.push_back(bosId_);
        }
        
        for (char c : text) {
            tokens.push_back(static_cast<int>(c) % vocabSize_);
        }
        
        return tokens;
    }
    
    /**
     * @brief 解码token序列为文本
     * 简单实现：将token ID转换回字符
     */
    std::string decode(const std::vector<int>& tokens, bool skipSpecialTokens) override {
        std::string text;
        for (int token : tokens) {
            if (skipSpecialTokens && isSpecialToken(token)) {
                continue;
            }
            text += static_cast<char>(token % 128);
        }
        return text;
    }
    
    int getVocabSize() const override {
        return vocabSize_;
    }
    
    std::string idToToken(int id) const override {
        if (id == bosId_) return "<BOS>";
        if (id == eosId_) return "<EOS>";
        if (id == padId_) return "<PAD>";
        if (id == unkId_) return "<UNK>";
        return std::string(1, static_cast<char>(id % 128));
    }
    
    int tokenToId(const std::string& token) const override {
        if (token == "<BOS>") return bosId_;
        if (token == "<EOS>") return eosId_;
        if (token == "<PAD>") return padId_;
        if (token == "<UNK>") return unkId_;
        if (token.empty()) return unkId_;
        return static_cast<int>(token[0]) % vocabSize_;
    }
    
    int getBosId() const override { return bosId_; }
    int getEosId() const override { return eosId_; }
    int getPadId() const override { return padId_; }
    int getUnkId() const override { return unkId_; }
    
    ModelType getModelType() const override { return ModelType::LLAMA; }
    
    bool isSpecialToken(int tokenId) const override {
        return tokenId == eosId_ || tokenId == padId_ || 
               tokenId == bosId_ || tokenId == unkId_;
    }
    
    /**
     * @brief 设置词汇表大小（用于测试配置）
     */
    void setVocabSize(int size) {
        vocabSize_ = size;
    }
    
    /**
     * @brief 设置特殊token ID（用于测试配置）
     */
    void setSpecialTokenIds(int bos, int eos, int pad, int unk) {
        bosId_ = bos;
        eosId_ = eos;
        padId_ = pad;
        unkId_ = unk;
    }

private:
    int eosId_;
    int padId_;
    int bosId_;
    int unkId_;
    int vocabSize_;
};

/**
 * @brief 简化版Mock Tokenizer
 * 用于不需要复杂编解码逻辑的测试场景
 */
class SimpleMockTokenizer : public ITokenizer {
public:
    SimpleMockTokenizer() = default;
    ~SimpleMockTokenizer() override = default;
    
    bool load(const std::string& modelPath) override { return true; }
    
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override {
        // 返回固定长度的token序列
        return {1, 2, 3, 4, 5};
    }
    
    std::string decode(const std::vector<int>& tokens, bool skipSpecialTokens) override {
        return "mock_decoded_text";
    }
    
    int getVocabSize() const override { return 1000; }
    std::string idToToken(int id) const override { return "token"; }
    int tokenToId(const std::string& token) const override { return 1; }
    
    int getPadId() const override { return 0; }
    int getEosId() const override { return 2; }
    int getBosId() const override { return 1; }
    int getUnkId() const override { return 3; }
    
    ModelType getModelType() const override { return ModelType::AUTO; }
    
    bool isSpecialToken(int tokenId) const override {
        return tokenId >= 0 && tokenId <= 3;
    }
};

} // namespace test
} // namespace cllm
