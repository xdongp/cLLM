/**
 * @file request.h
 * @brief HTTP请求类，用于解析和管理HTTP请求
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_HTTP_REQUEST_H
#define CLLM_HTTP_REQUEST_H

#include <string>
#include <map>

namespace cllm {

/**
 * @brief HTTP请求类，用于解析和管理HTTP请求
 * 
 * 该类提供了访问和修改HTTP请求各个部分的方法，包括请求方法、路径、头部、
 * 请求体和查询参数等。
 */
class HttpRequest {
public:
    /**
     * @brief 构造函数，创建一个空的HTTP请求
     */
    HttpRequest();
    
    /**
     * @brief 析构函数
     */
    ~HttpRequest();
    
    /**
     * @brief 获取HTTP请求方法
     * @return 请求方法（如GET、POST等）
     */
    std::string getMethod() const;
    
    /**
     * @brief 获取HTTP请求路径
     * @return 请求路径
     */
    std::string getPath() const;
    
    /**
     * @brief 获取指定名称的HTTP请求头部
     * @param name 头部名称
     * @return 头部值，如果不存在则返回空字符串
     */
    std::string getHeader(const std::string& name) const;
    
    /**
     * @brief 获取HTTP请求体
     * @return 请求体内容
     */
    std::string getBody() const;
    
    /**
     * @brief 获取指定键的查询参数
     * @param key 参数键
     * @return 参数值，如果不存在则返回空字符串
     */
    std::string getQuery(const std::string& key) const;
    
    /**
     * @brief 设置HTTP请求方法
     * @param method 请求方法
     */
    void setMethod(const std::string& method);
    
    /**
     * @brief 设置HTTP请求路径
     * @param path 请求路径
     */
    void setPath(const std::string& path);
    
    /**
     * @brief 设置HTTP请求头部
     * @param name 头部名称
     * @param value 头部值
     */
    void setHeader(const std::string& name, const std::string& value);
    
    /**
     * @brief 设置HTTP请求体
     * @param body 请求体内容
     */
    void setBody(const std::string& body);
    
    /**
     * @brief 设置查询参数
     * @param key 参数键
     * @param value 参数值
     */
    void setQuery(const std::string& key, const std::string& value);
    
    /**
     * @brief 检查是否包含指定名称的请求头部
     * @param name 头部名称
     * @return 如果包含则返回true，否则返回false
     */
    bool hasHeader(const std::string& name) const;
    
    /**
     * @brief 检查是否包含指定键的查询参数
     * @param key 参数键
     * @return 如果包含则返回true，否则返回false
     */
    bool hasQuery(const std::string& key) const;
    
    /**
     * @brief 获取所有HTTP请求头部
     * @return 包含所有头部的键值对映射
     */
    std::map<std::string, std::string> getAllHeaders() const;
    
    /**
     * @brief 获取所有查询参数
     * @return 包含所有查询参数的键值对映射
     */
    std::map<std::string, std::string> getAllQueries() const;
    
private:
    std::string method_;             ///< HTTP请求方法
    std::string path_;               ///< HTTP请求路径
    std::map<std::string, std::string> headers_;  ///< HTTP请求头部
    std::string body_;               ///< HTTP请求体
    std::map<std::string, std::string> queries_;  ///< 查询参数
};

}

#endif
