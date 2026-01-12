/**
 * @file health_endpoint.h
 * @brief 健康检查API端点，用于监控服务状态
 * @author cLLM Team
 * @date 2024-01-09
 */
#ifndef CLLM_HEALTH_ENDPOINT_H
#define CLLM_HEALTH_ENDPOINT_H

#include "cllm/http/api_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/http/request_queue.h"
#include "cllm/http/metrics_collector.h"
#include <map>

namespace cllm {

class HttpRequestQueue;

/**
 * @brief 健康检查API端点，用于监控服务状态
 * 
 * 该类实现了一个健康检查端点，用于检查服务是否正常运行。
 * 通常用于监控系统或负载均衡器来检测服务状态。
 * 支持不同的详细级别：Basic、Detailed、Full
 */
class HealthEndpoint : public ApiEndpoint {
public:
    enum class DetailLevel {
        Basic,      
        Detailed,   
        Full        
    };
    
    /**
     * @brief 构造函数，创建一个基础健康检查端点
     */
    HealthEndpoint();
    
    /**
     * @brief 构造函数，创建一个增强健康检查端点
     * @param scheduler 调度器指针
     * @param executor 模型执行器指针
     * @param tokenizer 分词器管理器指针
     * @param queue 请求队列指针
     * @param metrics 指标收集器指针
     * @param defaultLevel 默认详细级别
     */
    HealthEndpoint(
        Scheduler* scheduler,
        ModelExecutor* executor,
        TokenizerManager* tokenizer,
        HttpRequestQueue* queue,
        MetricsCollector* metrics,
        DetailLevel defaultLevel = DetailLevel::Basic
    );
    
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
    
    /**
     * @brief 设置默认详细级别
     * @param level 详细级别
     */
    void setDefaultDetailLevel(DetailLevel level);
    
private:
    struct HealthStatus {
        bool healthy;
        std::string status;
        std::map<std::string, bool> componentStatus;
        std::map<std::string, std::string> metrics;
        std::map<std::string, std::string> dependencies;
    };
    
    DetailLevel parseDetailLevel(const HttpRequest& request);
    HttpResponse handleBasicHealth();
    HttpResponse handleDetailedHealth();
    HttpResponse handleFullHealth();
    
    HealthStatus checkHealth();
    bool checkScheduler();
    bool checkModelExecutor();
    bool checkTokenizer();
    bool checkRequestQueue();
    
    std::map<std::string, std::string> collectMetrics();
    
    Scheduler* scheduler_;
    ModelExecutor* executor_;
    TokenizerManager* tokenizer_;
    HttpRequestQueue* queue_;
    MetricsCollector* metrics_;
    DetailLevel defaultLevel_;
};

}

#endif
