/**
 * @file api_endpoint.h
 * @brief API端点基类，定义了API端点的基本接口
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_API_ENDPOINT_H
#define CLLM_API_ENDPOINT_H

#include <string>
#include <functional>
#include "cllm/http/request.h"
#include "cllm/http/response.h"

namespace cllm {

/**
 * @brief API端点基类，定义了API端点的基本接口
 * 
 * 该类是所有API端点的基类，提供了API端点的基本功能和接口定义。
 * 子类需要实现handle方法来处理具体的请求。
 */
class ApiEndpoint {
public:
    /**
     * @brief API端点处理函数类型定义
     */
    typedef std::function<HttpResponse(const HttpRequest&)> HandlerFunc;
    
    /**
     * @brief 构造函数
     * @param name 端点名称
     * @param path 请求路径
     * @param method 请求方法
     */
    ApiEndpoint(
        const std::string& name,
        const std::string& path,
        const std::string& method
    );
    
    /**
     * @brief 析构函数
     */
    virtual ~ApiEndpoint();
    
    /**
     * @brief 获取端点名称
     * @return 端点名称
     */
    std::string getName() const;
    
    /**
     * @brief 获取请求路径
     * @return 请求路径
     */
    std::string getPath() const;
    
    /**
     * @brief 获取请求方法
     * @return 请求方法
     */
    std::string getMethod() const;
    
    /**
     * @brief 设置请求处理函数
     * @param handler 处理函数
     */
    void setHandler(HandlerFunc handler);
    
    /**
     * @brief 获取请求处理函数
     * @return 处理函数
     */
    HandlerFunc getHandler() const;
    
    /**
     * @brief 验证请求的有效性
     * @param request HTTP请求对象
     */
    void validateRequest(const HttpRequest& request);
    
    /**
     * @brief 处理HTTP请求，纯虚函数，子类必须实现
     * @param request HTTP请求对象
     * @return HTTP响应对象
     */
    virtual HttpResponse handle(const HttpRequest& request) = 0;
    
protected:
    std::string name_;     ///< 端点名称
    std::string path_;     ///< 请求路径
    std::string method_;   ///< 请求方法
    HandlerFunc handler_;  ///< 请求处理函数
};

}

#endif
