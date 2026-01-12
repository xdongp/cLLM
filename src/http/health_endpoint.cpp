#include "cllm/http/health_endpoint.h"
#include "cllm/http/response_builder.h"
#include "cllm/common/config.h"
#include "cllm/common/json.h"
#include <sstream>

namespace cllm {

HealthEndpoint::HealthEndpoint()
    : ApiEndpoint(cllm::Config::instance().apiEndpointHealthName(), cllm::Config::instance().apiEndpointHealthPath(), cllm::Config::instance().apiEndpointHealthMethod()),
      scheduler_(nullptr),
      executor_(nullptr),
      tokenizer_(nullptr),
      queue_(nullptr),
      metrics_(nullptr),
      defaultLevel_(DetailLevel::Basic) {
}

HealthEndpoint::HealthEndpoint(
    Scheduler* scheduler,
    ModelExecutor* executor,
    TokenizerManager* tokenizer,
    HttpRequestQueue* queue,
    MetricsCollector* metrics,
    DetailLevel defaultLevel
) : ApiEndpoint("EnhancedHealth", "/health", "GET"),
    scheduler_(scheduler),
    executor_(executor),
    tokenizer_(tokenizer),
    queue_(queue),
    metrics_(metrics),
    defaultLevel_(defaultLevel) {
}

HealthEndpoint::~HealthEndpoint() {
}

void HealthEndpoint::setDefaultDetailLevel(DetailLevel level) {
    defaultLevel_ = level;
}

HealthEndpoint::DetailLevel HealthEndpoint::parseDetailLevel(const HttpRequest& request) {
    std::string levelParam = request.getQuery("level");
    
    if (levelParam == "basic") {
        return DetailLevel::Basic;
    } else if (levelParam == "detailed") {
        return DetailLevel::Detailed;
    } else if (levelParam == "full") {
        return DetailLevel::Full;
    }
    
    return defaultLevel_;
}

HttpResponse HealthEndpoint::handle(const HttpRequest& request) {
    DetailLevel level = parseDetailLevel(request);
    
    switch (level) {
        case DetailLevel::Basic:
            return handleBasicHealth();
        case DetailLevel::Detailed:
            return handleDetailedHealth();
        case DetailLevel::Full:
            return handleFullHealth();
        default:
            return handleBasicHealth();
    }
}

HttpResponse HealthEndpoint::handleBasicHealth() {
    nlohmann::json responseJson = {
        {"status", "healthy"},
        {"model_loaded", true}
    };
    
    return ResponseBuilder::success(responseJson);
}

HttpResponse HealthEndpoint::handleDetailedHealth() {
    HealthStatus status = checkHealth();
    
    nlohmann::json responseJson = {
        {"healthy", status.healthy},
        {"status", status.status},
        {"components", status.componentStatus}
    };
    
    return ResponseBuilder::success(responseJson);
}

HttpResponse HealthEndpoint::handleFullHealth() {
    HealthStatus status = checkHealth();
    
    nlohmann::json responseJson = {
        {"healthy", status.healthy},
        {"status", status.status},
        {"components", status.componentStatus},
        {"metrics", status.metrics},
        {"dependencies", status.dependencies}
    };
    
    return ResponseBuilder::success(responseJson);
}

HealthEndpoint::HealthStatus HealthEndpoint::checkHealth() {
    HealthStatus status;
    
    status.componentStatus["scheduler"] = checkScheduler();
    status.componentStatus["model_executor"] = checkModelExecutor();
    status.componentStatus["tokenizer"] = checkTokenizer();
    status.componentStatus["request_queue"] = checkRequestQueue();
    
    status.metrics = collectMetrics();
    
    status.dependencies["model"] = "ok";
    status.dependencies["tokenizer"] = "ok";
    
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

bool HealthEndpoint::checkScheduler() {
    if (!scheduler_) {
        return false;
    }
    
    try {
        scheduler_->getQueueSize();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool HealthEndpoint::checkModelExecutor() {
    if (!executor_) {
        return false;
    }
    
    try {
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

bool HealthEndpoint::checkTokenizer() {
    if (!tokenizer_) {
        return false;
    }
    
    try {
        return tokenizer_->getTokenizer() != nullptr;
    } catch (const std::exception& e) {
        return false;
    }
}

bool HealthEndpoint::checkRequestQueue() {
    if (!queue_) {
        return false;
    }
    
    try {
        queue_->getQueueSize();
        return true;
    } catch (const std::exception& e) {
        return false;
    }
}

std::map<std::string, std::string> HealthEndpoint::collectMetrics() {
    std::map<std::string, std::string> metricsMap;
    
    if (scheduler_) {
        metricsMap["scheduler_queue_size"] = std::to_string(scheduler_->getQueueSize());
    }
    
    if (queue_) {
        metricsMap["http_queue_size"] = std::to_string(queue_->getQueueSize());
        metricsMap["http_queue_max_size"] = std::to_string(queue_->getMaxQueueSize());
        metricsMap["http_average_wait_time"] = std::to_string(queue_->getAverageWaitTime());
    }
    
    return metricsMap;
}

}
