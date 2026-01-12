#include "cllm/http/metrics_collector.h"
#include <sstream>
#include <limits>
#include <chrono>

namespace cllm {

MetricsCollector::MetricsCollector()
    : currentQueueSize_(0),
      currentActiveRequests_(0) {
}

MetricsCollector::~MetricsCollector() {
}

void MetricsCollector::recordRequest(const std::string& endpoint, size_t durationMs) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    EndpointMetrics& metrics = endpointMetrics_[endpoint];
    metrics.totalRequests++;
    metrics.totalDuration += durationMs;
    
    if (durationMs > metrics.maxDuration) {
        metrics.maxDuration = durationMs;
    }
    
    if (durationMs < metrics.minDuration) {
        metrics.minDuration = durationMs;
    }
}

void MetricsCollector::recordError(const std::string& endpoint, const std::string& errorType) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    EndpointMetrics& metrics = endpointMetrics_[endpoint];
    metrics.totalErrors++;
    metrics.errorCounts[errorType]++;
}

void MetricsCollector::recordQueueSize(size_t queueSize) {
    currentQueueSize_ = queueSize;
}

void MetricsCollector::recordActiveRequests(size_t activeRequests) {
    currentActiveRequests_ = activeRequests;
}

std::string MetricsCollector::getMetrics() const {
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    oss << "=== HTTP Server Metrics ===\n";
    oss << "Current queue size: " << currentQueueSize_ << "\n";
    oss << "Current active requests: " << currentActiveRequests_ << "\n\n";
    
    for (const auto& [endpoint, metrics] : endpointMetrics_) {
        oss << "Endpoint: " << endpoint << "\n";
        oss << "  Total requests: " << metrics.totalRequests << "\n";
        oss << "  Total errors: " << metrics.totalErrors << "\n";
        
        if (metrics.totalRequests > 0) {
            double avgDuration = static_cast<double>(metrics.totalDuration) / metrics.totalRequests;
            oss << "  Average duration: " << avgDuration << " ms\n";
        }
        
        oss << "  Max duration: " << metrics.maxDuration << " ms\n";
        oss << "  Min duration: " << (metrics.minDuration == std::numeric_limits<size_t>::max() ? 0 : metrics.minDuration) << " ms\n";
        
        if (!metrics.errorCounts.empty()) {
            oss << "  Error breakdown: " << "\n";
            for (const auto& [errorType, count] : metrics.errorCounts) {
                oss << "    " << errorType << ": " << count << "\n";
            }
        }
        
        oss << "\n";
    }
    
    return oss.str();
}

std::string MetricsCollector::getPrometheusMetrics() const {
    std::ostringstream oss;
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    // Queue size metric
    oss << "# HELP cllm_queue_size Current queue size\n";
    oss << "# TYPE cllm_queue_size gauge\n";
    oss << "cllm_queue_size " << currentQueueSize_ << "\n\n";
    
    // Active requests metric
    oss << "# HELP cllm_active_requests Current active requests\n";
    oss << "# TYPE cllm_active_requests gauge\n";
    oss << "cllm_active_requests " << currentActiveRequests_ << "\n\n";
    
    for (const auto& [endpoint, metrics] : endpointMetrics_) {
        // Total requests counter
        oss << "# HELP cllm_requests_total Total number of requests\n";
        oss << "# TYPE cllm_requests_total counter\n";
        oss << "cllm_requests_total{endpoint=\"" << endpoint << "\"} " << metrics.totalRequests << "\n\n";
        
        // Total errors counter
        oss << "# HELP cllm_errors_total Total number of errors\n";
        oss << "# TYPE cllm_errors_total counter\n";
        oss << "cllm_errors_total{endpoint=\"" << endpoint << "\"} " << metrics.totalErrors << "\n\n";
        
        // Total duration summary
        oss << "# HELP cllm_request_duration_seconds Total request duration in seconds\n";
        oss << "# TYPE cllm_request_duration_seconds summary\n";
        oss << "cllm_request_duration_seconds_sum{endpoint=\"" << endpoint << "\"} " << metrics.totalDuration / 1000.0 << "\n";
        oss << "cllm_request_duration_seconds_count{endpoint=\"" << endpoint << "\"} " << metrics.totalRequests << "\n";
        
        if (metrics.totalRequests > 0) {
            oss << "cllm_request_duration_seconds_max{endpoint=\"" << endpoint << "\"} " << metrics.maxDuration / 1000.0 << "\n";
            oss << "cllm_request_duration_seconds_min{endpoint=\"" << endpoint << "\"} " << (metrics.minDuration == std::numeric_limits<size_t>::max() ? 0 : metrics.minDuration / 1000.0) << "\n\n";
        }
        
        // Error type breakdown
        for (const auto& [errorType, count] : metrics.errorCounts) {
            oss << "# HELP cllm_error_type_total Total number of errors by type\n";
            oss << "# TYPE cllm_error_type_total counter\n";
            oss << "cllm_error_type_total{endpoint=\"" << endpoint << "\",error_type=\"" << errorType << "\"} " << count << "\n\n";
        }
    }
    
    return oss.str();
}

} // namespace cllm
