/**
 * @file exceptions.h
 * @brief 通用异常处理类，定义项目中通用的异常类型
 * @author cLLM Team
 * @date 2026-01-10
 */

#pragma once

#include <stdexcept>
#include <string>

namespace cllm {

/**
 * @brief 通用错误类型枚举
 */
enum class CommonError {
    INVALID_ARGUMENT,      // 无效参数
    OUT_OF_MEMORY,         // 内存不足
    RESOURCE_NOT_FOUND,    // 资源未找到
    OPERATION_FAILED,      // 操作失败
    TIMEOUT,               // 超时
    PERMISSION_DENIED      // 权限不足
};

/**
 * @brief 通用异常基类
 */
class BaseException : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param message 错误消息
     */
    explicit BaseException(const std::string& message)
        : std::runtime_error(message) {}

    /**
     * @brief 构造函数
     * @param error 错误类型
     * @param message 错误消息
     */
    BaseException(CommonError error, const std::string& message)
        : std::runtime_error(message), error_(error) {}

    /**
     * @brief 获取错误类型
     * @return 错误类型
     */
    CommonError getError() const { return error_; }

private:
    CommonError error_{CommonError::OPERATION_FAILED};  // 默认错误类型
};

/**
 * @brief 模型错误类型枚举
 */
enum class ModelError {
    MODEL_LOAD_FAILED,    // 模型加载失败
    MODEL_NOT_LOADED,     // 模型未加载
    INVALID_INPUT,        // 无效输入
    INFERENCE_FAILED,     // 推理失败
    QUANTIZATION_FAILED,  // 量化失败
    OUT_OF_MEMORY         // 内存不足
};

/**
 * @brief 模型异常类
 */
class ModelException : public BaseException {
public:
    /**
     * @brief 构造函数
     * @param error 错误类型
     * @param message 错误消息
     */
    ModelException(ModelError error, const std::string& message)
        : BaseException(message), error_(error) {}

    /**
     * @brief 获取错误类型
     * @return 错误类型
     */
    ModelError getError() const { return error_; }

private:
    ModelError error_;  // 错误类型
};

/**
 * @brief 线程池异常类
 */
class ThreadPoolException : public BaseException {
public:
    explicit ThreadPoolException(const std::string& message)
        : BaseException(message) {}
};

}  // namespace cllm