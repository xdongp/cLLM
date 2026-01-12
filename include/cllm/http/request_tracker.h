#pragma once
#include <string>
#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include <utility>

namespace cllm {

struct RequestInfo {
    std::string requestId;
    std::string endpoint;
    size_t startTime;
    size_t endTime;
    std::vector<std::pair<size_t, std::string>> events;
    bool completed;
};

class RequestTracker {
public:
    RequestTracker();
    ~RequestTracker();
    
    std::string generateRequestId();
    void startRequest(const std::string& requestId, const std::string& endpoint);
    void endRequest(const std::string& requestId);
    void recordEvent(const std::string& requestId, const std::string& event);
    
    RequestInfo getRequestInfo(const std::string& requestId) const;
    size_t getActiveRequestCount() const;
    
private:
    std::map<std::string, RequestInfo> requestInfos_;
    std::mutex requestInfosMutex_;
    
    std::atomic<size_t> requestCounter_;
    
    size_t getCurrentTimeMs() const;
};

} // namespace cllm
