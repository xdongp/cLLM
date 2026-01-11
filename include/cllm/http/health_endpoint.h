/**
 * @file health_endpoint.h
 * @brief 健康检查API端点，用于监控服务状态
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_HEALTH_ENDPOINT_H
#define CLLM_HEALTH_ENDPOINT_H

#include "cllm/http/api_endpoint.h"

namespace cllm {

/**
 * @brief 健康检查API端点，用于监控服务状态
 * 
 * 该类实现了一个健康检查端点，用于检查服务是否正常运行。
 * 通常用于监控系统或负载均衡器来检测服务状态。
 */
class HealthEndpoint : public ApiEndpoint {
public:
    /**
     * @brief 构造函数，创建一个健康检查端点
     */
    HealthEndpoint();
    
    /**
     * @brief 析构函数
     */
    ~HealthEndpoint();
    
    /**
     * @brief 处理健康检查请求
     * @param request HTTP请求对象
     * @return HTTP响应对象，通常返回200 OK表示服务正常
     */
    HttpResponse handle(const HttpRequest& request) override;
};

}

#endif
