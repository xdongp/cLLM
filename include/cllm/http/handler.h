/**
 * @file handler.h
 * @brief HTTP请求处理器，用于注册和处理HTTP请求
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_HTTP_HANDLER_H
#define CLLM_HTTP_HANDLER_H

#include <string>
#include <map>
#include <functional>
#include "cllm/http/request.h"
#include "cllm/http/response.h"

namespace cllm {

/**
 * @brief HTTP请求处理器，用于注册和处理HTTP请求
 * 
 * 该类提供了注册不同HTTP方法（GET、POST、PUT、DELETE）处理器的方法，
 * 并负责根据请求的方法和路径匹配对应的处理器函数。
 */
class HttpHandler {
public:
    /**
     * @brief HTTP请求处理函数类型定义
     */
    typedef std::function<HttpResponse(const HttpRequest&)> HandlerFunc;
    
    /**
     * @brief 构造函数
     */
    HttpHandler();
    
    /**
     * @brief 析构函数
     */
    ~HttpHandler();
    
    /**
     * @brief 注册GET请求处理器
     * @param path 请求路径
     * @param handler 处理函数
     */
    void get(const std::string& path, HandlerFunc handler);
    
    /**
     * @brief 注册POST请求处理器
     * @param path 请求路径
     * @param handler 处理函数
     */
    void post(const std::string& path, HandlerFunc handler);
    
    /**
     * @brief 注册PUT请求处理器
     * @param path 请求路径
     * @param handler 处理函数
     */
    void put(const std::string& path, HandlerFunc handler);
    
    /**
     * @brief 注册DELETE请求处理器
     * @param path 请求路径
     * @param handler 处理函数
     */
    void del(const std::string& path, HandlerFunc handler);
    
    /**
     * @brief 处理HTTP请求
     * @param request HTTP请求对象
     * @return HTTP响应对象
     */
    virtual HttpResponse handleRequest(const HttpRequest& request);
    
    /**
     * @brief 检查是否存在指定方法和路径的处理器
     * @param method HTTP方法
     * @param path 请求路径
     * @return 如果存在则返回true，否则返回false
     */
    bool hasHandler(const std::string& method, const std::string& path) const;
    
private:
    /**
     * @brief 规范化请求路径
     * @param path 原始请求路径
     * @return 规范化后的路径
     */
    std::string normalizePath(const std::string& path) const;
    
    /**
     * @brief 匹配路径模式
     * @param pattern 路径模式
     * @param path 请求路径
     * @return 如果匹配则返回true，否则返回false
     */
    bool matchPath(const std::string& pattern, const std::string& path) const;
    
    std::map<std::string, HandlerFunc> getHandlers_;     ///< GET请求处理器映射
    std::map<std::string, HandlerFunc> postHandlers_;    ///< POST请求处理器映射
    std::map<std::string, HandlerFunc> putHandlers_;     ///< PUT请求处理器映射
    std::map<std::string, HandlerFunc> deleteHandlers_;  ///< DELETE请求处理器映射
};

}

#endif
