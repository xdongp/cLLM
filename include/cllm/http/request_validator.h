/**
 * @file request_validator.h
 * @brief HTTP请求验证器，用于验证HTTP请求的参数
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_REQUEST_VALIDATOR_H
#define CLLM_REQUEST_VALIDATOR_H

#include <string>
#include "cllm/http/request.h"

namespace cllm {

/**
 * @brief HTTP请求验证器，用于验证HTTP请求的参数
 * 
 * 该类提供了验证HTTP请求参数的各种方法，包括必填字段验证、类型验证、范围验证等。
 */
class RequestValidator {
public:
    /**
     * @brief 构造函数
     */
    RequestValidator();
    
    /**
     * @brief 析构函数
     */
    ~RequestValidator();
    
    /**
     * @brief 验证请求中是否包含必填字段
     * @param request HTTP请求对象
     * @param field 要验证的字段名
     * @return 如果包含必填字段则返回true，否则返回false
     */
    bool validateRequired(const HttpRequest& request, const std::string& field);
    
    /**
     * @brief 验证值的类型是否符合预期
     * @param value 要验证的值
     * @param expectedType 预期的类型
     * @return 如果类型符合预期则返回true，否则返回false
     */
    bool validateType(const std::string& value, const std::string& expectedType);
    
    /**
     * @brief 验证整数值是否在指定范围内
     * @param value 要验证的整数值
     * @param min 最小值
     * @param max 最大值
     * @return 如果在范围内则返回true，否则返回false
     */
    bool validateRange(int value, int min, int max);
    
    /**
     * @brief 验证浮点数值是否在指定范围内
     * @param value 要验证的浮点数值
     * @param min 最小值
     * @param max 最大值
     * @return 如果在范围内则返回true，否则返回false
     */
    bool validateRange(float value, float min, float max);
    
    /**
     * @brief 获取最后一次验证错误的信息
     * @return 错误信息字符串
     */
    std::string getLastError() const;
    
private:
    std::string lastError_;  ///< 最后一次验证错误的信息
};

}

#endif
