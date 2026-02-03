/**
 * @file model_info_endpoint.h
 * @brief 模型信息 API 端点
 * @author cLLM Team
 * @date 2024-02-03
 */
#ifndef CLLM_MODEL_INFO_ENDPOINT_H
#define CLLM_MODEL_INFO_ENDPOINT_H

#include "cllm/http/api_endpoint.h"
#include "cllm/model/executor.h"
#include <string>

namespace cllm {

/**
 * @brief 模型信息 API 端点
 * 
 * 返回当前加载模型的详细信息，包括：
 * - 模型类型（qwen/llama/deepseek等）
 * - 模型路径
 * - 词表大小
 * - 层数、隐藏层大小等架构信息
 * - 量化类型
 */
class ModelInfoEndpoint : public ApiEndpoint {
public:
    /**
     * @brief 构造函数
     * @param executor 模型执行器指针
     * @param modelPath 模型文件路径
     */
    ModelInfoEndpoint(ModelExecutor* executor, const std::string& modelPath);
    
    /**
     * @brief 析构函数
     */
    ~ModelInfoEndpoint();
    
    /**
     * @brief 处理模型信息请求
     * @param request HTTP 请求对象
     * @return HTTP 响应对象
     */
    HttpResponse handle(const HttpRequest& request) override;
    
private:
    ModelExecutor* executor_;
    std::string modelPath_;
    
    /**
     * @brief 从模型路径中提取模型名称
     * @return 模型名称（如 qwen3-1.7b-q4_k_m）
     */
    std::string extractModelName() const;
    
    /**
     * @brief 检测模型系列（qwen/llama/deepseek等）
     * @return 模型系列名称
     */
    std::string detectModelFamily() const;
};

}

#endif
