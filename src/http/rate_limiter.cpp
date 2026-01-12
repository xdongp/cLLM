#include "cllm/http/rate_limiter.h"
#include "cllm/common/time_utils.h"
#include <chrono>
#include <thread>

namespace cllm {

RateLimiter::RateLimiter(size_t maxRequests, size_t timeWindowMs)
    : maxRequests_(maxRequests),
      timeWindowMs_(timeWindowMs) {
}

RateLimiter::~RateLimiter() {
}

bool RateLimiter::allowRequest(const std::string& clientId) {
    return allowRequest(clientId, 1);
}

bool RateLimiter::allowRequest(const std::string& clientId, size_t requestCount) {
    std::lock_guard<std::mutex> lock(clientInfoMutex_);
    
    size_t currentTime = time_utils::getCurrentTimeMs();
    
    // 获取或创建客户端信息
    ClientInfo& client = clientInfo_[clientId];
    
    // 清理过期的请求记录
    cleanupOldRequests(client.requestTimestamps);
    
    // 检查是否允许请求
    if (client.requestTimestamps.size() + requestCount <= maxRequests_) {
        // 添加新的请求记录
        for (size_t i = 0; i < requestCount; ++i) {
            client.requestTimestamps.push_back(currentTime);
        }
        client.totalRequests += requestCount;
        return true;
    }
    
    return false;
}

void RateLimiter::setMaxRequests(size_t maxRequests) {
    std::lock_guard<std::mutex> lock(clientInfoMutex_);
    maxRequests_ = maxRequests;
}

void RateLimiter::setTimeWindow(size_t timeWindowMs) {
    std::lock_guard<std::mutex> lock(clientInfoMutex_);
    timeWindowMs_ = timeWindowMs;
    
    // 清理所有客户端的过期请求记录
    for (auto& [clientId, client] : clientInfo_) {
        cleanupOldRequests(client.requestTimestamps);
    }
}

size_t RateLimiter::getMaxRequests() const {
    return maxRequests_;
}

size_t RateLimiter::getTimeWindow() const {
    return timeWindowMs_;
}

void RateLimiter::reset() {
    std::lock_guard<std::mutex> lock(clientInfoMutex_);
    clientInfo_.clear();
}

void RateLimiter::cleanupOldRequests(std::deque<size_t>& timestamps) {
    size_t currentTime = time_utils::getCurrentTimeMs();
    size_t cutoffTime = currentTime - timeWindowMs_;
    
    // 移除所有过期的请求记录
    while (!timestamps.empty() && timestamps.front() < cutoffTime) {
        timestamps.pop_front();
    }
}

} // namespace cllm
