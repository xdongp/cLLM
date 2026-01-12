#pragma once
#include <map>
#include <mutex>
#include <deque>

namespace cllm {

class RateLimiter {
public:
    RateLimiter(size_t maxRequests, size_t timeWindowMs);
    ~RateLimiter();
    
    bool allowRequest(const std::string& clientId);
    bool allowRequest(const std::string& clientId, size_t requestCount);
    
    void setMaxRequests(size_t maxRequests);
    void setTimeWindow(size_t timeWindowMs);
    
    size_t getMaxRequests() const;
    size_t getTimeWindow() const;
    
    void reset();
    
private:
    struct ClientInfo {
        std::deque<size_t> requestTimestamps;
        size_t totalRequests;
    };
    
    std::map<std::string, ClientInfo> clientInfo_;
    std::mutex clientInfoMutex_;
    
    size_t maxRequests_;
    size_t timeWindowMs_;
    
    void cleanupOldRequests(std::deque<size_t>& timestamps);
};

} // namespace cllm
