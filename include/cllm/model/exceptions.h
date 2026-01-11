/**
 * @file exceptions.h
 * @brief 模型执行器模块的错误处理类
 * @author cLLM Team
 * @date 2026-01-08
 * 
 * Note: ModelError 和 ModelException 已经在 common/exceptions.h 中定义
 *       此文件保留用于未来的模型特定异常扩展
 */
#ifndef CLLM_MODEL_EXCEPTIONS_H
#define CLLM_MODEL_EXCEPTIONS_H

#include "cllm/common/exceptions.h"

// ModelError 和 ModelException 已在 common/exceptions.h 中定义
// 此处不再重复定义

namespace cllm {

// 未来可以在这里添加模型特定的异常类型

}  // namespace cllm

#endif  // CLLM_MODEL_EXCEPTIONS_H