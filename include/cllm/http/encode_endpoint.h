/**
 * @file encode_endpoint.h
 * @brief 文本编码API端点，用于将文本转换为令牌序列
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_ENCODE_ENDPOINT_H
#define CLLM_ENCODE_ENDPOINT_H

#include "cllm/http/api_endpoint.h"
#include "cllm/http/request.h"
#include "cllm/http/response.h"

namespace cllm {

class Tokenizer;

/**
 * @brief 文本编码API端点，用于将文本转换为令牌序列
 * 
 * 该类实现了一个API端点，用于接收文本并使用指定的分词器将其转换为令牌序列。
 */
class EncodeEndpoint : public ApiEndpoint {
public:
    /**
     * @brief 构造函数，创建一个编码端点
     * @param tokenizer 分词器指针，用于将文本转换为令牌
     */
    EncodeEndpoint(Tokenizer* tokenizer);
    
    /**
     * @brief 析构函数
     */
    ~EncodeEndpoint();
    
    /**
     * @brief 处理文本编码请求
     * @param request HTTP请求对象，包含要编码的文本
     * @return HTTP响应对象，包含编码后的令牌序列
     */
    HttpResponse handle(const HttpRequest& request) override;
    
    /**
     * @brief 设置分词器
     * @param tokenizer 分词器指针
     */
    void setTokenizer(Tokenizer* tokenizer);
    
private:
    /**
     * @brief 编码请求结构体，包含要编码的文本
     */
    struct EncodeRequest {
        std::string text;  ///< 要编码的文本
    };
    
    /**
     * @brief 解析HTTP请求为编码请求结构体
     * @param request HTTP请求对象
     * @return 解析后的编码请求结构体
     */
    EncodeRequest parseRequest(const HttpRequest& request);
    
    Tokenizer* tokenizer_;  ///< 分词器指针，用于文本编码
};

}

#endif
