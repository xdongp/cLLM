#pragma once
#include <string>
#include <map>
#include <mutex>
#include <atomic>

namespace cllm {

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();
    
    void recordRequest(const std::string& endpoint, size_t durationMs);
    void recordError(const std::string& endpoint, const std::string& errorType);
    void recordQueueSize(size_t queueSize);
    void recordActiveRequests(size_t activeRequests);
    
    std::string getMetrics() const;
    std::string getPrometheusMetrics() const;
    
private:
    struct EndpointMetrics {
        size_t totalRequests = 0;
        size_t totalErrors = 0;
        size_t totalDuration = 0;
        size_t maxDuration = 0;
        size_t minDuration = std::numeric_limits<size_t>::max();
        std::map<std::string, size_t> errorCounts;
    };
    
    std::map<std::string, EndpointMetrics> endpointMetrics_;
    std::mutex metricsMutex_;
    
    std::atomic<size_t> currentQueueSize_;
    std::atomic<size_t> currentActiveRequests_;
};

} // namespace cllm
