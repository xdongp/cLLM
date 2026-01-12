#include "cllm/http/request_tracker.h"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace cllm {

RequestTracker::RequestTracker()
    : requestCounter_(0) {
}

RequestTracker::~RequestTracker() {
}

std::string RequestTracker::generateRequestId() {
    size_t counter = requestCounter_++;
    size_t currentTime = getCurrentTimeMs();
    
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(16) << std::hex << currentTime;
    oss << "-" << std::setw(8) << std::hex << counter;
    
    return oss.str();
}

void RequestTracker::startRequest(const std::string& requestId, const std::string& endpoint) {
    std::lock_guard<std::mutex> lock(requestInfosMutex_);
    
    size_t currentTime = getCurrentTimeMs();
    
    RequestInfo info = {
        requestId,
        endpoint,
        currentTime,
        0,
        {{currentTime, "Request started"}},
        false
    };
    
    requestInfos_[requestId] = info;
}

void RequestTracker::endRequest(const std::string& requestId) {
    std::lock_guard<std::mutex> lock(requestInfosMutex_);
    
    auto it = requestInfos_.find(requestId);
    if (it != requestInfos_.end()) {
        size_t currentTime = getCurrentTimeMs();
        it->second.endTime = currentTime;
        it->second.completed = true;
        it->second.events.emplace_back(currentTime, "Request ended");
    }
}

void RequestTracker::recordEvent(const std::string& requestId, const std::string& event) {
    std::lock_guard<std::mutex> lock(requestInfosMutex_);
    
    auto it = requestInfos_.find(requestId);
    if (it != requestInfos_.end()) {
        size_t currentTime = getCurrentTimeMs();
        it->second.events.emplace_back(currentTime, event);
    }
}

RequestInfo RequestTracker::getRequestInfo(const std::string& requestId) const {
    std::lock_guard<std::mutex> lock(requestInfosMutex_);
    
    auto it = requestInfos_.find(requestId);
    if (it != requestInfos_.end()) {
        return it->second;
    }
    
    // 返回空的RequestInfo
    return RequestInfo();
}

size_t RequestTracker::getActiveRequestCount() const {
    std::lock_guard<std::mutex> lock(requestInfosMutex_);
    
    size_t count = 0;
    for (const auto& [requestId, info] : requestInfos_) {
        if (!info.completed) {
            count++;
        }
    }
    
    return count;
}

size_t RequestTracker::getCurrentTimeMs() const {
    auto now = std::chrono::system_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
}

} // namespace cllm
