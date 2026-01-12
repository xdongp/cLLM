#pragma once
#include "cllm/http/api_endpoint.h"
#include "cllm/scheduler/scheduler.h"
#include "cllm/model/executor.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/http/request_queue.h"
#include "cllm/http/metrics_collector.h"

namespace cllm {

class EnhancedHealthEndpoint : public ApiEndpoint {
public:
    EnhancedHealthEndpoint(
        Scheduler* scheduler,
        ModelExecutor* executor,
        TokenizerManager* tokenizer,
        RequestQueue* queue,
        MetricsCollector* metrics
    );
    ~EnhancedHealthEndpoint();
    
    HttpResponse handle(const HttpRequest& request) override;
    
private:
    struct HealthStatus {
        bool healthy;
        std::string status;
        std::map<std::string, bool> componentStatus;
        std::map<std::string, std::string> metrics;
        std::map<std::string, std::string> dependencies;
    };
    
    HealthStatus checkHealth();
    bool checkScheduler();
    bool checkModelExecutor();
    bool checkTokenizer();
    bool checkRequestQueue();
    
    std::map<std::string, std::string> collectMetrics();
    
    Scheduler* scheduler_;
    ModelExecutor* executor_;
    TokenizerManager* tokenizer_;
    RequestQueue* queue_;
    MetricsCollector* metrics_;
};

} // namespace cllm
