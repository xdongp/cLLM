#include "cllm/scheduler/tracker.h"
#include "cllm/common/request_state.h"
#include <stdexcept>

namespace cllm {

RequestTracker::RequestTracker() {
}

RequestTracker::~RequestTracker() {
}

size_t RequestTracker::addRequest(const RequestState& request) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t requestId = nextRequestId_++;
    
    RequestState newRequest = request;
    newRequest.requestId = requestId;
    
    runningRequests_[requestId] = newRequest;
    
    return requestId;
}

bool RequestTracker::updateRequest(size_t requestId, const RequestState& updated) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = runningRequests_.find(requestId);
    if (it != runningRequests_.end()) {
        it->second = updated;
        return true;
    }
    
    it = completedRequests_.find(requestId);
    if (it != completedRequests_.end()) {
        it->second = updated;
        return true;
    }
    
    return false;
}

bool RequestTracker::removeRequest(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (runningRequests_.erase(requestId) > 0) {
        return true;
    }
    
    return completedRequests_.erase(requestId) > 0;
}

RequestState RequestTracker::getRequest(size_t requestId) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = runningRequests_.find(requestId);
    if (it != runningRequests_.end()) {
        return it->second;
    }
    
    it = completedRequests_.find(requestId);
    if (it != completedRequests_.end()) {
        return it->second;
    }
    
    throw std::runtime_error("Request not found: " + std::to_string(requestId));
}

std::vector<RequestState> RequestTracker::getRunningRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<RequestState> requests;
    requests.reserve(runningRequests_.size());
    
    for (const auto& pair : runningRequests_) {
        requests.push_back(pair.second);
    }
    
    return requests;
}

std::vector<RequestState> RequestTracker::getCompletedRequests() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<RequestState> requests;
    requests.reserve(completedRequests_.size());
    
    for (const auto& pair : completedRequests_) {
        requests.push_back(pair.second);
    }
    
    return requests;
}

bool RequestTracker::markAsRunning(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = runningRequests_.find(requestId);
    if (it != runningRequests_.end()) {
        it->second.isRunning = true;
        condition_.notify_all();
        return true;
    }
    
    return false;
}

bool RequestTracker::markAsCompleted(size_t requestId) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = runningRequests_.find(requestId);
    if (it != runningRequests_.end()) {
        it->second.isRunning = false;
        it->second.isCompleted = true;
        completedRequests_[requestId] = it->second;
        runningRequests_.erase(it);
        condition_.notify_all();
        return true;
    }
    
    return false;
}

bool RequestTracker::markAsFailed(size_t requestId, const std::string& error) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = runningRequests_.find(requestId);
    if (it != runningRequests_.end()) {
        it->second.isRunning = false;
        it->second.isFailed = true;
        it->second.errorMessage = error;
        completedRequests_[requestId] = it->second;
        runningRequests_.erase(it);
        condition_.notify_all();
        return true;
    }
    
    return false;
}

size_t RequestTracker::getRunningCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return runningRequests_.size();
}

size_t RequestTracker::getCompletedCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return completedRequests_.size();
}

}
