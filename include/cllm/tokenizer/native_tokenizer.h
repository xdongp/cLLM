#pragma once

#include "cllm/tokenizer/tokenizer.h"
#include "i_tokenizer.h"
#include "unicode_utils.h"
#include <sentencepiece_processor.h>
#include <shared_mutex>
#include <unordered_map>

namespace cllm {

/**
 * @brief NativeTokenizer - 自研分词器实现
 * 
 * 基于SentencePiece的自研分词器，支持：
 * - BPE、Unigram、WordPiece算法
 * - 特殊Token处理
 * - 轻量级无外部依赖
 */
class NativeTokenizer : public cllm::ITokenizer {
public:
    explicit NativeTokenizer(cllm::ModelType modelType);
    ~NativeTokenizer() override;

    // ITokenizer接口实现
    bool load(const std::string& modelPath) override;
    std::vector<int> encode(const std::string& text, bool addSpecialTokens) override;
    std::string decode(const std::vector<int>& ids, bool skipSpecialTokens) override;
    
    int getVocabSize() const override;
    std::string idToToken(int id) const override;
    int tokenToId(const std::string& token) const override;
    
    int getBosId() const override;
    int getEosId() const override;
    int getPadId() const override;
    int getUnkId() const override;
    bool isSpecialToken(int tokenId) const override;
    
    cllm::ModelType getModelType() const override;

private:
    void loadSpecialTokens(const std::string& configPath);
    std::string preprocessText(const std::string& text);

    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_;
    cllm::ModelType modelType_;
    
    // 特殊Token IDs
    int bosId_ = -1;
    int eosId_ = -1;
    int padId_ = -1;
    int unkId_ = -1;
    
    // 模型路径
    std::string modelPath_;
    
    // 测试模式词汇表
    std::unordered_map<std::string, int> vocab_;
    std::unordered_map<int, std::string> idToToken_;
    
    // 缓存
    mutable std::unordered_map<std::string, std::vector<int>> encodeCache_;
    mutable std::shared_mutex cacheMutex_;
};

} // namespace cllm