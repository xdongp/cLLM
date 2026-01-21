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
    , stopFlag_(false)
    , queueDirty_(false)
    , cachedQueueSize_(0) {
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
    
    queue_.push_back(req);
    totalRequests_++;
    cachedQueueSize_.store(queue_.size(), std::memory_order_release);
    queueDirty_.store(true, std::memory_order_release);
    
    condition_.notify_one();
    return true;
}

bool RequestQueue::removeRequest(size_t requestId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    auto it = std::remove_if(queue_.begin(), queue_.end(),
        [requestId](const RequestState& req) {
            return req.requestId == requestId;
        });
    
    bool found = it != queue_.end();
    queue_.erase(it, queue_.end());
    cachedQueueSize_.store(queue_.size(), std::memory_order_release);
    
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
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    RequestState request;
    if (queueDirty_.load(std::memory_order_acquire)) {
        std::vector<RequestState> sortedQueue = queue_;
        std::sort(sortedQueue.begin(), sortedQueue.end(), RequestComparator());
        queueDirty_.store(false, std::memory_order_release);
        
        if (!sortedQueue.empty()) {
            auto it = std::find(queue_.begin(), queue_.end(), sortedQueue.front());
            if (it != queue_.end()) {
                request = *it;
                queue_.erase(it);
            }
        }
    } else {
        if (!queue_.empty()) {
            request = queue_.front();
            queue_.erase(queue_.begin());
        }
    }
    
    if (request.requestId == 0) {
        return RequestState{};
    }
    
    size_t waitTime = currentTime - request.arrivalTime;
    totalWaitTime_ += waitTime;
    cachedQueueSize_.store(queue_.size(), std::memory_order_release);
    
    return request;
}

bool RequestQueue::tryGetNextRequest(RequestState& request) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    if (queueDirty_.load(std::memory_order_acquire)) {
        std::vector<RequestState> sortedQueue = queue_;
        std::sort(sortedQueue.begin(), sortedQueue.end(), RequestComparator());
        queueDirty_.store(false, std::memory_order_release);
        
        auto it = std::find(queue_.begin(), queue_.end(), sortedQueue.front());
        if (it != queue_.end()) {
            request = *it;
            queue_.erase(it);
        }
    } else {
        request = queue_.front();
        queue_.erase(queue_.begin());
    }
    
    size_t waitTime = currentTime - request.arrivalTime;
    totalWaitTime_ += waitTime;
    cachedQueueSize_.store(queue_.size(), std::memory_order_release);
    
    return true;
}

std::vector<RequestState> RequestQueue::formBatch(size_t maxContextLength) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    std::vector<RequestState> batch;
    size_t currentBatchLength = 0;
    
    std::lock_guard<std::mutex> runningLock(runningMutex_);
    
    size_t runningLength = 0;
    for (const auto& req : runningRequests_) {
        runningLength += req.getTotalLength();
    }
    
    if (runningLength > maxContextLength * 0.75f) {
        return batch;
    }
    
    if (queueDirty_.load(std::memory_order_acquire)) {
        std::vector<RequestState> sortedQueue = queue_;
        std::sort(sortedQueue.begin(), sortedQueue.end(), RequestComparator());
        queueDirty_.store(false, std::memory_order_release);
        
        size_t optimalBatchSize = calculateOptimalBatchSize(sortedQueue);
        size_t availableContext = maxContextLength - runningLength;
        
        auto it = queue_.begin();
        auto sortedIt = sortedQueue.begin();
        while (sortedIt != sortedQueue.end() && batch.size() < optimalBatchSize) {
            if (currentBatchLength + sortedIt->getPromptLength() <= availableContext) {
                it = std::find(queue_.begin(), queue_.end(), *sortedIt);
                if (it != queue_.end()) {
                    batch.push_back(*it);
                    currentBatchLength += it->getPromptLength();
                    it = queue_.erase(it);
                }
                ++sortedIt;
            } else {
                ++sortedIt;
            }
        }
    } else {
        size_t optimalBatchSize = calculateOptimalBatchSize(queue_);
        size_t availableContext = maxContextLength - runningLength;
        
        auto it = queue_.begin();
        while (it != queue_.end() && batch.size() < optimalBatchSize) {
            if (currentBatchLength + it->getPromptLength() <= availableContext) {
                batch.push_back(*it);
                currentBatchLength += it->getPromptLength();
                it = queue_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    cachedQueueSize_.store(queue_.size(), std::memory_order_release);
    
    return batch;
}

std::vector<RequestState> RequestQueue::getPendingRequests() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    std::vector<RequestState> pending = queue_;
    
    if (queueDirty_.load(std::memory_order_acquire)) {
        std::sort(pending.begin(), pending.end(), RequestComparator());
    }
    
    return pending;
}

size_t RequestQueue::getQueueSize() const {
    return cachedQueueSize_.load(std::memory_order_acquire);
}

size_t RequestQueue::getRunningRequestsLength() const {
    std::lock_guard<std::mutex> lock(runningMutex_);
    
    size_t total = 0;
    for (const auto& req : runningRequests_) {
        total += req.getTotalLength();
    }
    return total;
}

float RequestQueue::getAverageWaitTime() const {
    size_t completedCount = completedRequests_.load();
    size_t queueSize = cachedQueueSize_.load(std::memory_order_acquire);
    
    if (completedCount == 0 && queueSize == 0) {
        return 0.0f;
    }
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    size_t queueWaitTime = 0;
    for (const auto& req : queue_) {
        queueWaitTime += (currentTime - req.arrivalTime);
    }
    
    size_t totalWait = totalWaitTime_.load() + queueWaitTime;
    size_t totalCount = completedCount + queue_.size();
    
    return totalCount > 0 ? static_cast<float>(totalWait) / totalCount : 0.0f;
}

size_t RequestQueue::getAverageRequestLength() const {
    size_t queueSize = cachedQueueSize_.load(std::memory_order_acquire);
    
    if (queueSize == 0) {
        return 0;
    }
    
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    size_t totalLength = 0;
    for (const auto& req : queue_) {
        totalLength += req.getPromptLength();
    }
    
    return totalLength / queue_.size();
}

void RequestQueue::updateRunningRequests(const std::vector<RequestState>& running) {
    std::lock_guard<std::mutex> lock(runningMutex_);
    
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
    std::lock_guard<std::mutex> runningLock(runningMutex_);
    
    queue_.clear();
    runningRequests_.clear();
    cachedQueueSize_.store(0, std::memory_order_release);
}

void RequestQueue::updatePriorities() {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    auto currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    for (auto& req : queue_) {
        req.priority = req.calculatePriority(currentTime);
    }
    
    queueDirty_.store(true, std::memory_order_release);
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
