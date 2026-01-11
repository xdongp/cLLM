/**
 * @file response.h
 * @brief 生成响应类
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_GENERATION_RESPONSE_H
#define CLLM_GENERATION_RESPONSE_H

#include <string>
#include <vector>

namespace cllm {

/**
 * @brief 生成响应类
 * 
 * 封装文本生成的响应结果。
 */
class GenerationResponse {
public:
    /**
     * @brief 默认构造函数
     */
    GenerationResponse();
    
    /**
     * @brief 析构函数
     */
    ~GenerationResponse();
    
    void setRequestId(const std::string& requestId);  ///< 设置请求ID
    void setText(const std::string& text);  ///< 设置文本
    void setTokens(const std::vector<int>& tokens);  ///< 设置tokens
    void setFinished(bool finished);  ///< 设置完成状态
    void setError(const std::string& error);  ///< 设置错误信息
    void setResponseTime(float responseTime);  ///< 设置响应时间
    
    std::string getRequestId() const;  ///< 获取请求ID
    std::string getText() const;  ///< 获取文本
    std::vector<int> getTokens() const;  ///< 获取tokens
    bool isFinished() const;  ///< 检查是否完成
    std::string getError() const;  ///< 获取错误信息
    float getResponseTime() const;  ///< 获取响应时间
    
private:
    std::string requestId_;      ///< 请求ID
    std::string text_;           ///< 生成的文本
    std::vector<int> tokens_;    ///< 生成的tokens
    bool finished_;              ///< 是否完成
    std::string error_;          ///< 错误信息
    float responseTime_;         ///< 响应时间
};

}

#endif
