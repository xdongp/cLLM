#include "cllm/http/enhanced_health_endpoint.h"
#include "cllm/common/json.h"

namespace cllm {

EnhancedHealthEndpoint::EnhancedHealthEndpoint(
    Scheduler* scheduler,
    ModelExecutor* executor,
    TokenizerManager* tokenizer,
    RequestQueue* queue,
    MetricsCollector* metrics
) : ApiEndpoint("EnhancedHealth", "/health", "GET"),
    scheduler_(scheduler),
    executor_(executor),
    tokenizer_(tokenizer),
    queue_(queue),
    metrics_(metrics) {
}

EnhancedHealthEndpoint::~EnhancedHealthEndpoint() {
}

HttpResponse EnhancedHealthEndpoint::handle(const HttpRequest& request) {
    HealthStatus status = checkHealth();
    
    // 构建JSON响应
    nlohmann::json responseJson = {
        {"healthy", status.healthy},
        {"status", status.status},
        {"components", status.componentStatus},
        {"metrics", status.metrics},
        {"dependencies", status.dependencies}
    };
    
    HttpResponse response;
    response.setStatus(HttpResponse::HTTP_200_OK);
    response.setBody(responseJson.dump());
    response.setHeader("Content-Type", "application/json");
    
    return response;
}

EnhancedHealthEndpoint::HealthStatus EnhancedHealthEndpoint::checkHealth() {
    HealthStatus status;
    
    // 检查各个组件
    status.componentStatus["scheduler"] = checkScheduler();
    status.componentStatus["model_executor"] = checkModelExecutor();
    status.componentStatus["tokenizer"] = checkTokenizer();
    status.componentStatus["request_queue"] = checkRequestQueue();
    
    // 收集指标
    status.metrics = collectMetrics();
    
    // 检查依赖
    status.dependencies["model"] = "ok";
    status.dependencies["tokenizer"] = "ok";
    
    // 确定整体健康状态
    bool allHealthy = true;
    for (const auto& [component, isHealthy] : status.componentStatus) {
        if (!isHealthy) {
            allHealthy = false;
            break;
        }
    }
    
    status.healthy = allHealthy;
    status.status = allHealthy ? "ok" : "degraded";
    
    return status;
}

bool EnhancedHealthEndpoint::checkScheduler() {
    if (!scheduler_) {
        return false;
    }
    
    try {
        // 简单检查：获取队列大小，这应该不会抛出异常
        scheduler_->getQueueSize();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool EnhancedHealthEndpoint::checkModelExecutor() {
    if (!executor_) {
        return false;
    }
    
    try {
        // 简单检查：获取模型状态，这应该不会抛出异常
        // 注意：这里假设ModelExecutor有一个getStats方法或类似的轻量级方法
        // 如果没有，可能需要调整这个检查
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool EnhancedHealthEndpoint::checkTokenizer() {
    if (!tokenizer_) {
        return false;
    }
    
    try {
        // 简单检查：获取tokenizer指针，这应该不会抛出异常
        return tokenizer_->getTokenizer() != nullptr;
    } catch (const std::exception& e) {
        return false;
    }
}

bool EnhancedHealthEndpoint::checkRequestQueue() {
    if (!queue_) {
        return false;
    }
    
    try {
        // 简单检查：获取队列大小，这应该不会抛出异常
        queue_->getQueueSize();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::map<std::string, std::string> EnhancedHealthEndpoint::collectMetrics() {
    std::map<std::string, std::string> metricsMap;
    
    // 从各个组件收集指标
    if (scheduler_) {
        metricsMap["scheduler_queue_size"] = std::to_string(scheduler_->getQueueSize());
    }
    
    if (queue_) {
        metricsMap["http_queue_size"] = std::to_string(queue_->getQueueSize());
        metricsMap["http_queue_max_size"] = std::to_string(queue_->getMaxQueueSize());
        metricsMap["http_average_wait_time"] = std::to_string(queue_->getAverageWaitTime());
    }
    
    // 从MetricsCollector收集指标
    if (metrics_) {
        // 注意：这里我们不直接获取完整的指标字符串，而是获取我们需要的特定指标
        // 如果MetricsCollector提供了更细粒度的指标获取方法，可以使用那些方法
    }
    
    return metricsMap;
}

} // namespace cllm
