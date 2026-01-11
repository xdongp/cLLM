/**
 * @file response.h
 * @brief HTTP响应类，用于构建和管理HTTP响应
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_HTTP_RESPONSE_H
#define CLLM_HTTP_RESPONSE_H

#include <string>
#include <map>
#include <vector>

namespace cllm {

/**
 * @brief HTTP响应类，用于构建和管理HTTP响应
 * 
 * 该类提供了构建HTTP响应的所有必要方法，包括设置状态码、头部、响应体等。
 * 支持流式响应，允许分块发送数据。
 */
class HttpResponse {
public:
    /**
     * @brief 构造函数，创建一个空的HTTP响应
     */
    HttpResponse();
    
    /**
     * @brief 析构函数
     */
    ~HttpResponse();
    
    /**
     * @brief 设置HTTP响应状态码
     * @param code HTTP状态码（如200、404等）
     */
    void setStatusCode(int code);
    
    /**
     * @brief 设置HTTP响应头部
     * @param name 头部名称
     * @param value 头部值
     */
    void setHeader(const std::string& name, const std::string& value);
    
    /**
     * @brief 设置HTTP响应体
     * @param body 响应体内容
     */
    void setBody(const std::string& body);
    
    /**
     * @brief 设置HTTP响应的Content-Type头部
     * @param contentType MIME类型（如application/json、text/plain等）
     */
    void setContentType(const std::string& contentType);
    
    /**
     * @brief 获取HTTP响应状态码
     * @return HTTP状态码
     */
    int getStatusCode() const;
    
    /**
     * @brief 获取指定名称的HTTP响应头部
     * @param name 头部名称
     * @return 头部值，如果不存在则返回空字符串
     */
    std::string getHeader(const std::string& name) const;
    
    /**
     * @brief 获取HTTP响应体
     * @return 响应体内容
     */
    std::string getBody() const;
    
    /**
     * @brief 获取HTTP响应的Content-Type头部
     * @return Content-Type值
     */
    std::string getContentType() const;
    
    /**
     * @brief 获取所有HTTP响应头部
     * @return 包含所有头部的键值对映射
     */
    std::map<std::string, std::string> getAllHeaders() const;
    
    /**
     * @brief 设置错误响应
     * @param code 错误状态码
     * @param message 错误消息
     */
    void setError(int code, const std::string& message);
    
    /**
     * @brief 启用流式响应
     */
    void enableStreaming();
    
    /**
     * @brief 检查是否为流式响应
     * @return 如果是流式响应则返回true，否则返回false
     */
    bool isStreaming() const;
    
    /**
     * @brief 向流式响应添加数据块
     * @param chunk 要添加的数据块
     */
    void addChunk(const std::string& chunk);
    
    /**
     * @brief 获取所有响应数据块
     * @return 数据块向量
     */
    std::vector<std::string> getChunks() const;
    
    /**
     * @brief 创建一个成功的HTTP响应
     * @param body 响应体内容，默认为空
     * @return 成功的HTTP响应对象
     */
    static HttpResponse ok(const std::string& body = "");
    
    /**
     * @brief 创建一个404 Not Found响应
     * @return 404响应对象
     */
    static HttpResponse notFound();
    
    /**
     * @brief 创建一个400 Bad Request响应
     * @param message 错误消息，默认为空
     * @return 400响应对象
     */
    static HttpResponse badRequest(const std::string& message = "");
    
    /**
     * @brief 创建一个500 Internal Server Error响应
     * @param message 错误消息，默认为空
     * @return 500响应对象
     */
    static HttpResponse internalError(const std::string& message = "");
    
private:
    int statusCode_;                 ///< HTTP状态码
    std::map<std::string, std::string> headers_;  ///< HTTP响应头部
    std::string body_;               ///< HTTP响应体
    bool streaming_;                 ///< 是否为流式响应
    std::vector<std::string> chunks_;  ///< 流式响应的数据块
};

}

#endif
