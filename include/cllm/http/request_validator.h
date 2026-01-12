/**
 * @file request_validator.h
 * @brief HTTP请求验证器，用于验证HTTP请求的参数
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_REQUEST_VALIDATOR_H
#define CLLM_REQUEST_VALIDATOR_H

#include <string>
#include <vector>
#include <functional>
#include <regex>
#include "cllm/http/request.h"
#include "cllm/common/json.h"

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
     * @brief 验证JSON请求体中是否包含必填字段
     * @param jsonBody JSON对象
     * @param field 要验证的字段名
     * @return 如果包含必填字段则返回true，否则返回false
     */
    bool validateRequired(const nlohmann::json& jsonBody, const std::string& field);
    
    /**
     * @brief 验证值的类型是否符合预期
     * @param value 要验证的值
     * @param expectedType 预期的类型
     * @return 如果类型符合预期则返回true，否则返回false
     */
    bool validateType(const std::string& value, const std::string& expectedType);
    
    /**
     * @brief 验证JSON字段类型
     * @param jsonBody JSON对象
     * @param field 字段名
     * @param expectedType 预期的类型 ("string", "number", "integer", "boolean", "array", "object")
     * @return 如果类型符合预期则返回true，否则返回false
     */
    bool validateType(const nlohmann::json& jsonBody, const std::string& field, const std::string& expectedType);
    
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
     * @brief 验证字符串长度是否在指定范围内
     * @param value 要验证的字符串
     * @param minLength 最小长度
     * @param maxLength 最大长度
     * @return 如果在范围内则返回true，否则返回false
     */
    bool validateLength(const std::string& value, size_t minLength, size_t maxLength);
    
    /**
     * @brief 验证数组大小是否在指定范围内
     * @param array 要验证的数组
     * @param minSize 最小大小
     * @param maxSize 最大大小
     * @return 如果在范围内则返回true，否则返回false
     */
    bool validateSize(const nlohmann::json& array, size_t minSize, size_t maxSize);
    
    /**
     * @brief 验证字符串是否匹配正则表达式
     * @param value 要验证的字符串
     * @param pattern 正则表达式模式
     * @return 如果匹配则返回true，否则返回false
     */
    bool validatePattern(const std::string& value, const std::string& pattern);
    
    /**
     * @brief 验证值是否在允许的枚举列表中
     * @param value 要验证的值
     * @param allowedValues 允许的值列表
     * @return 如果在列表中则返回true，否则返回false
     */
    bool validateEnum(const std::string& value, const std::vector<std::string>& allowedValues);
    
    /**
     * @brief 使用自定义验证器验证
     * @param value 要验证的值
     * @param validator 自定义验证函数
     * @return 如果验证通过则返回true，否则返回false
     */
    bool validateCustom(const std::string& value, std::function<bool(const std::string&)> validator);
    
    /**
     * @brief 验证JSON请求体
     * @param requestBody 请求体字符串
     * @param jsonBody 输出的JSON对象
     * @return 如果是有效的JSON则返回true，否则返回false
     */
    bool validateJson(const std::string& requestBody, nlohmann::json& jsonBody);
    
    /**
     * @brief 批量验证多个必填字段
     * @param jsonBody JSON对象
     * @param fields 要验证的字段列表
     * @return 如果所有字段都存在则返回true，否则返回false
     */
    bool validateRequiredFields(const nlohmann::json& jsonBody, const std::vector<std::string>& fields);
    
    /**
     * @brief 获取最后一次验证错误的信息
     * @return 错误信息字符串
     */
    std::string getLastError() const;
    
    /**
     * @brief 清除最后一次验证错误的信息
     */
    void clearLastError();
    
private:
    std::string lastError_;
    
    void setError(const std::string& error);
};

}

#endif
