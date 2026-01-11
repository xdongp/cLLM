/**
 * @file request.h
 * @brief 生成请求类
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_GENERATION_REQUEST_H
#define CLLM_GENERATION_REQUEST_H

#include <string>
#include <vector>
#include "cllm/tokenizer/config.h"

namespace cllm {

/**
 * @brief 生成请求类
 * 
 * 封装文本生成的请求信息。
 */
class GenerationRequest {
public:
    /**
     * @brief 默认构造函数
     */
    GenerationRequest();
    
    /**
     * @brief 析构函数
     */
    ~GenerationRequest();
    
    void setRequestId(const std::string& requestId);  ///< 设置请求ID
    void setPrompt(const std::string& prompt);  ///< 设置提示词
    void setConfig(const TokenizerConfig& config);  ///< 设置配置
    void setStream(bool stream);  ///< 设置流式模式
    
    std::string getRequestId() const;  ///< 获取请求ID
    std::string getPrompt() const;  ///< 获取提示词
    TokenizerConfig getConfig() const;  ///< 获取配置
    bool isStream() const;  ///< 检查是否流式模式
    
    std::vector<int> getEncodedPrompt() const;  ///< 获取编码后的提示词
    void setEncodedPrompt(const std::vector<int>& encodedPrompt);  ///< 设置编码后的提示词
    
private:
    std::string requestId_;          ///< 请求ID
    std::string prompt_;             ///< 提示词
    TokenizerConfig config_;         ///< 配置
    bool stream_;                    ///< 流式模式
    std::vector<int> encodedPrompt_; ///< 编码后的提示词
};

}

#endif
