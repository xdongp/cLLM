#include "cllm/common/queue.h"
#include <chrono>
#include <algorithm>

namespace cllm {

RequestQueue::RequestQueue(size_t maxQueueSize, size_t maxContextLength)
    : maxQueueSize_(maxQueueSize)
    , maxContextLength_(maxContextLength)
    , totalRequests_(0)
    , completedRequests_(0)
    , totalWaitTime_(0)
    , stopFlag_(false) {
}

RequestQueue::~RequestQueue() {
    stopFlag_ = true;
    condition_.notify_all();
}

bool RequestQueue::addRequest(const RequestState& request) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    if (queue_.size() >= maxQueueSize_) {
        return false;
    }
    
    RequestState req = request;
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    req.arrivalTime = currentTime;
    req.priority = req.calculatePriority(currentTime);
    
    queue_.push(req);
    totalRequests_++;
    
    condition_.notify_one();
    return true;
}

bool RequestQueue::removeRequest(size_t requestId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    std::vector<RequestState> tempQueue;
    bool found = false;
    
    while (!queue_.empty()) {
        auto req = queue_.top();
        queue_.pop();
        
        if (req.requestId == requestId) {
            found = true;
        } else {
            tempQueue.push_back(req);
        }
    }
    
    for (const auto& req : tempQueue) {
        queue_.push(req);
    }
    
    return found;
}

RequestState RequestQueue::getNextRequest() {
    std::unique_lock<std::mutex> lock(queueMutex_);
    
    condition_.wait(lock, [this] {
        return !queue_.empty() || stopFlag_;
    });
    
    if (stopFlag_ && queue_.empty()) {
        return RequestState{};
    }
    
    RequestState request = queue_.top();
    queue_.pop();
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    size_t waitTime = currentTime - request.arrivalTime;
    totalWaitTime_ += waitTime;
    
    return request;
}

bool RequestQueue::tryGetNextRequest(RequestState& request) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    request = queue_.top();
    queue_.pop();
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    size_t waitTime = currentTime - request.arrivalTime;
    totalWaitTime_ += waitTime;
    
    return true;
}

std::vector<RequestState> RequestQueue::formBatch(size_t maxContextLength) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    std::vector<RequestState> batch;
    size_t currentBatchLength = 0;
    
    size_t runningLength = 0;
    for (const auto& req : runningRequests_) {
        runningLength += req.getTotalLength();
    }
    
    if (runningLength > maxContextLength * 0.75f) {
        return batch;
    }
    
    std::vector<RequestState> tempQueue;
    while (!queue_.empty()) {
        tempQueue.push_back(queue_.top());
        queue_.pop();
    }
    
    size_t optimalBatchSize = calculateOptimalBatchSize(tempQueue);
    size_t availableContext = maxContextLength - runningLength;
    
    for (auto& req : tempQueue) {
        if (batch.size() >= optimalBatchSize) {
            queue_.push(req);
            continue;
        }
        
        if (currentBatchLength + req.getPromptLength() <= availableContext) {
            batch.push_back(req);
            currentBatchLength += req.getPromptLength();
        } else {
            queue_.push(req);
        }
    }
    
    return batch;
}

std::vector<RequestState> RequestQueue::getPendingRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    std::vector<RequestState> pending;
    pending.reserve(queue_.size());
    
    std::vector<RequestState> tempQueue;
    tempQueue.reserve(queue_.size());
    
    // 一次性提取所有请求
    while (!queue_.empty()) {
        tempQueue.push_back(queue_.top());
        queue_.pop();
    }
    
    // 复制到结果并重建队列
    for (auto& req : tempQueue) {
        pending.push_back(std::move(req));
        queue_.push(pending.back());
    }
    
    return pending;
}

size_t RequestQueue::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return queue_.size();
}

size_t RequestQueue::getRunningRequestsLength() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    size_t total = 0;
    for (const auto& req : runningRequests_) {
        total += req.getTotalLength();
    }
    return total;
}

float RequestQueue::getAverageWaitTime() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    size_t completedCount = completedRequests_.load();
    if (completedCount == 0 && queue_.empty()) {
        return 0.0f;
    }
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    // 使用向量避免多次锁定
    std::vector<RequestState> tempQueue;
    tempQueue.reserve(queue_.size());
    
    size_t queueWaitTime = 0;
    
    // 一次性提取并计算
    while (!queue_.empty()) {
        auto req = queue_.top();
        queue_.pop();
        queueWaitTime += (currentTime - req.arrivalTime);
        tempQueue.push_back(req);
    }
    
    // 重建队列
    for (auto& req : tempQueue) {
        queue_.push(req);
    }
    
    size_t totalWait = totalWaitTime_.load() + queueWaitTime;
    size_t totalCount = completedCount + queue_.size();
    
    return totalCount > 0 ? static_cast<float>(totalWait) / totalCount : 0.0f;
}

size_t RequestQueue::getAverageRequestLength() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    if (queue_.empty()) {
        return 0;
    }
    
    std::vector<RequestState> tempQueue;
    tempQueue.reserve(queue_.size());
    
    size_t totalLength = 0;
    
    // 一次性提取
    while (!queue_.empty()) {
        auto req = queue_.top();
        queue_.pop();
        totalLength += req.getPromptLength();
        tempQueue.push_back(req);
    }
    
    // 重建队列
    for (auto& req : tempQueue) {
        queue_.push(req);
    }
    
    return totalLength / queue_.size();
}

void RequestQueue::updateRunningRequests(const std::vector<RequestState>& running) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    for (const auto& oldReq : runningRequests_) {
        bool found = false;
        for (const auto& newReq : running) {
            if (oldReq.requestId == newReq.requestId) {
                found = true;
                break;
            }
        }
        if (!found && oldReq.isCompleted) {
            completedRequests_++;
        }
    }
    
    runningRequests_ = running;
}

void RequestQueue::clear() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    while (!queue_.empty()) {
        queue_.pop();
    }
    
    runningRequests_.clear();
}

void RequestQueue::updatePriorities() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    std::vector<RequestState> tempQueue;
    while (!queue_.empty()) {
        auto req = queue_.top();
        queue_.pop();
        req.priority = req.calculatePriority(currentTime);
        tempQueue.push_back(req);
    }
    
    for (const auto& req : tempQueue) {
        queue_.push(req);
    }
}

size_t RequestQueue::calculateOptimalBatchSize(const std::vector<RequestState>& requests) {
    if (requests.empty()) {
        return 0;
    }
    
    size_t avgLength = 0;
    for (const auto& req : requests) {
        avgLength += req.getPromptLength();
    }
    avgLength /= requests.size();
    
    size_t availableContext = maxContextLength_;
    size_t optimalBatchSize = 0;
    
    if (avgLength < 100) {
        optimalBatchSize = std::min(requests.size(), availableContext / avgLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(16));
    } else if (avgLength < 500) {
        optimalBatchSize = std::min(requests.size(), availableContext / avgLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(8));
    } else {
        optimalBatchSize = std::min(requests.size(), availableContext / avgLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(4));
    }
    
    return optimalBatchSize;
}

}  // namespace cllm