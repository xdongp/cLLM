#pragma once

#include "i_tokenizer.h"
#include "cllm/model/gguf_loader_new.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>

namespace cllm {

/**
 * @brief GGUFTokenizer - GGUF格式分词器实现
 * 
 * 从GGUF文件中加载Tokenizer配置和词汇表
 * 支持从GGUF元数据和张量中提取完整的Tokenizer信息
 */
class GGUFTokenizer : public ITokenizer {
public:
    explicit GGUFTokenizer();
    ~GGUFTokenizer() override;

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
    
    // GGUF特有功能
    bool loadFromGGUFLoader(const GGUFLoader& loader);
    
private:
    // 加载词汇表
    void loadVocabulary(const GGUFLoader& loader);
    
    // 加载特殊tokens
    void loadSpecialTokens(const GGUFLoader& loader);
    
    // 加载合并规则
    void loadMergeRules(const GGUFLoader& loader);
    
    // 初始化编码/解码逻辑
    void initializeEncoding();

    // 预分词/编码
    std::vector<std::string> preTokenize(const std::string& text) const;
    std::vector<std::string> bpe(const std::string& token) const;
    void buildByteEncoder();

    // 核心数据结构
    std::unordered_map<int, std::string> idToTokenMap_;
    std::unordered_map<std::string, int> tokenToIdMap_;

    // BPE合并规则
    std::vector<std::pair<std::string, std::string>> mergeRules_;
    std::unordered_map<std::string, int> bpeRanks_;

    // Byte-level encoder/decoder
    std::unordered_map<unsigned char, std::string> byteEncoder_;
    std::unordered_map<std::string, unsigned char> byteDecoder_;

    // 特殊token
    std::unordered_set<int> specialTokenIds_;
    std::vector<std::string> specialTokenStrings_;

    // 特殊token ID
    int bosTokenId_ = -1;
    int eosTokenId_ = -1;
    int padTokenId_ = -1;
    int unkTokenId_ = -1;

    // 词汇表大小
    int vocabSize_ = 0;

    // 模型类型
    ModelType modelType_ = ModelType::LLAMA;

    // tokenizer pre type
    std::string preTokenizer_;

    // 是否已加载
    bool loaded_ = false;

    // 原始模型路径
    std::string modelPath_;
};

} // namespace cllm