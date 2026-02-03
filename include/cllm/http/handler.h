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
#include <vector>
#include "cllm/http/request.h"
#include "cllm/http/response.h"

namespace cllm {

/**
 * @brief HTTP中间件函数类型定义
 */
typedef std::function<HttpResponse(const HttpRequest&, const std::function<HttpResponse(const HttpRequest&)>&)> MiddlewareFunc;

/**
 * @brief 流式写入回调类型
 * 
 * 用于流式响应，每次调用发送一个数据块到客户端。
 * @param chunk 要发送的数据块
 * @return true 继续发送，false 停止（客户端断开或出错）
 */
using StreamingWriteCallback = std::function<bool(const std::string& chunk)>;

/**
 * @brief 流式请求处理函数类型
 * 
 * 用于处理需要流式响应的请求（如 SSE）。
 * @param request HTTP请求对象
 * @param writeCallback 流式写入回调
 */
using StreamingHandlerFunc = std::function<void(const HttpRequest& request, StreamingWriteCallback writeCallback)>;

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
     * @brief 注册流式POST请求处理器
     * @param path 请求路径
     * @param handler 流式处理函数
     */
    void postStreaming(const std::string& path, StreamingHandlerFunc handler);
    
    /**
     * @brief 添加中间件
     * @param middleware 中间件函数
     */
    void addMiddleware(MiddlewareFunc middleware);
    
    /**
     * @brief 处理HTTP请求
     * @param request HTTP请求对象
     * @return HTTP响应对象
     */
    virtual HttpResponse handleRequest(const HttpRequest& request);
    
    /**
     * @brief 检查是否为流式请求路径
     * @param method HTTP方法
     * @param path 请求路径
     * @return true 如果是流式请求路径
     */
    bool isStreamingRequest(const std::string& method, const std::string& path) const;
    
    /**
     * @brief 处理流式请求
     * @param request HTTP请求对象
     * @param writeCallback 流式写入回调
     */
    void handleStreamingRequest(const HttpRequest& request, StreamingWriteCallback writeCallback);
    
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
    
    /**
     * @brief 应用中间件链
     * @param request HTTP请求对象
     * @param handler 最终处理函数
     * @param middlewareIndex 当前中间件索引
     * @return HTTP响应对象
     */
    HttpResponse applyMiddleware(const HttpRequest& request, const HandlerFunc& handler, size_t middlewareIndex = 0);
    
    std::map<std::string, HandlerFunc> getHandlers_;     ///< GET请求处理器映射
    std::map<std::string, HandlerFunc> postHandlers_;    ///< POST请求处理器映射
    std::map<std::string, HandlerFunc> putHandlers_;     ///< PUT请求处理器映射
    std::map<std::string, HandlerFunc> deleteHandlers_;  ///< DELETE请求处理器映射
    std::map<std::string, StreamingHandlerFunc> streamingPostHandlers_;  ///< 流式POST请求处理器映射
    std::vector<MiddlewareFunc> middlewares_;            ///< 中间件列表
};

} // namespace cllm

#endif
