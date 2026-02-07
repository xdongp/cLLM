/**
 * @file tokenizer.h
 * @brief Tokenizer类，负责文本编解码
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_TOKENIZER_H
#define CLLM_TOKENIZER_H

#include <sentencepiece_processor.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include "i_tokenizer.h"

namespace cllm {

/**
 * @brief Tokenizer类
 * 
 * 负责将文本编码为token IDs，以及token IDs解码为文本。
 */
class Tokenizer : public ITokenizer {
public:
    /**
     * @brief 构造函数
     * @param modelPath 模型路径
     */
    explicit Tokenizer(const std::string& modelPath);
    
    /**
     * @brief 析构函数
     */
    ~Tokenizer();
    
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
    
    cllm::ModelType getModelType() const override;
    
    /**
     * @brief 获取token文本
     * @param tokenId Token ID
     * @return Token文本
     */
    std::string getTokenText(int tokenId) const;
    
    /**
     * @brief 检查是否为特殊token
     * @param tokenId Token ID
     * @return true 如果是特殊token，false 否则
     */
    bool isSpecialToken(int tokenId) const override;
    
    void setPadToken(int tokenId);  ///< 设置填充token
    void setEosToken(int tokenId);  ///< 设置结束token
    void setBosToken(int tokenId);  ///< 设置开始token
    
    int getPadToken() const;  ///< 获取填充token
    int getEosToken() const;  ///< 获取结束token
    int getBosToken() const;  ///< 获取开始token
    
    /**
     * @brief 加载模型
     * @param modelPath 模型路径
     */
    void loadModel(const std::string& modelPath);
    
    /**
     * @brief 卸载模型
     */
    void unloadModel();
    
    /**
     * @brief 检查模型是否已加载
     * @return true 如果已加载，false 否则
     */
    bool isLoaded() const;
    
private:
    std::unique_ptr<sentencepiece::SentencePieceProcessor> processor_; ///< SentencePiece处理器
    std::string modelPath_;      ///< 模型路径
    
    int padTokenId_;             ///< 填充token ID
    int eosTokenId_;             ///< 结束token ID
    int bosTokenId_;             ///< 开始token ID
    int unkTokenId_;             ///< 未知token ID
    
    cllm::ModelType modelType_{cllm::ModelType::SPM};  ///< 模型类型
    
    bool loaded_;                ///< 加载状态
};

}

#endif