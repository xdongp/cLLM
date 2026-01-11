#pragma once

#include "cllm/tokenizer/i_tokenizer.h"
#include <string>
#include <vector>

namespace cllm {

/**
 * @brief ITokenizer接口 - 分词器基接口
 * 
 * 定义了分词器的基本功能接口，包括文本编码、解码等操作。
 */
class ITokenizer {
public:
    virtual ~ITokenizer() = default;
    
    /**
     * @brief 编码文本为token IDs
     * @param text 输入文本
     * @param addSpecialTokens 是否添加特殊tokens
     * @return Token IDs列表
     */
    virtual std::vector<int> encode(const std::string& text, bool addSpecialTokens = false) = 0;
    
    /**
     * @brief 解码token IDs为文本
     * @param tokenIds Token IDs列表
     * @param skipSpecialTokens 是否跳过特殊tokens
     * @return 解码后的文本
     */
    virtual std::string decode(const std::vector<int>& tokenIds, bool skipSpecialTokens = true) = 0;
    
    /**
     * @brief 获取词表大小
     * @return 词表大小
     */
    virtual int getVocabSize() const = 0;
    
    /**
     * @brief 获取token文本
     * @param tokenId Token ID
     * @return Token文本
     */
    virtual std::string getTokenText(int tokenId) const = 0;
    
    /**
     * @brief 加载模型
     * @param modelPath 模型路径
     */
    virtual void loadModel(const std::string& modelPath) = 0;
    
    /**
     * @brief 卸载模型
     */
    virtual void unloadModel() = 0;
    
    /**
     * @brief 检查模型是否已加载
     * @return true 如果已加载，false 否则
     */
    virtual bool isLoaded() const = 0;
};

} // namespace cllm