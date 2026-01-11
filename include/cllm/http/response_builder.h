/**
 * @file response_builder.h
 * @brief HTTP响应构建器，用于链式构建HTTP响应
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_RESPONSE_BUILDER_H
#define CLLM_RESPONSE_BUILDER_H

#include <string>
#include <map>
#include "cllm/http/response.h"

namespace cllm {

/**
 * @brief HTTP响应构建器，用于链式构建HTTP响应
 * 
 * 该类提供了链式API，用于方便地构建HTTP响应对象。
 * 支持设置状态码、头部和响应体等。
 */
class ResponseBuilder {
public:
    /**
     * @brief 构造函数，创建一个空的响应构建器
     */
    ResponseBuilder();
    
    /**
     * @brief 析构函数
     */
    ~ResponseBuilder();
    
    /**
     * @brief 设置HTTP响应状态码
     * @param code HTTP状态码
     * @return 响应构建器引用，用于链式调用
     */
    ResponseBuilder& setStatus(int code);
    
    /**
     * @brief 设置HTTP响应头部
     * @param name 头部名称
     * @param value 头部值
     * @return 响应构建器引用，用于链式调用
     */
    ResponseBuilder& setHeader(const std::string& name, const std::string& value);
    
    /**
     * @brief 设置HTTP响应体
     * @param body 响应体内容
     * @return 响应构建器引用，用于链式调用
     */
    ResponseBuilder& setBody(const std::string& body);
    
    /**
     * @brief 构建HTTP响应对象
     * @return 构建完成的HTTP响应对象
     */
    HttpResponse build();
    
    /**
     * @brief 创建一个成功响应的构建器
     * @return 响应构建器引用，用于链式调用
     */
    static ResponseBuilder ok();
    
    /**
     * @brief 创建一个错误响应的构建器
     * @param code 错误状态码
     * @param message 错误消息
     * @return 响应构建器引用，用于链式调用
     */
    static ResponseBuilder error(int code, const std::string& message);
    
private:
    int statusCode_;                 ///< HTTP状态码
    std::map<std::string, std::string> headers_;  ///< HTTP响应头部
    std::string body_;               ///< HTTP响应体
};

}

#endif
