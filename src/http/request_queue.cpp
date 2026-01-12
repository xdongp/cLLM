#include "cllm/http/request_queue.h"
#include "cllm/common/time_utils.h"
#include <chrono>
#include <thread>

namespace cllm {

HttpRequestQueue::HttpRequestQueue(size_t maxQueueSize, size_t maxWaitTime)
    : maxQueueSize_(maxQueueSize),
      maxWaitTime_(maxWaitTime),
      totalWaitTime_(0),
      completedRequests_(0),
      running_(true) {
}

HttpRequestQueue::~HttpRequestQueue() {
    running_ = false;
    queueCV_.notify_all();
}

bool HttpRequestQueue::enqueue(const HttpRequest& request, size_t requestId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    if (queue_.size() >= maxQueueSize_) {
        return false; // 队列已满
    }
    
    size_t currentTime = time_utils::getCurrentTimeMs();
    int priority = 0;
    
    if (priorityCallback_) {
        priority = priorityCallback_(request);
    }
    
    QueuedRequest queuedRequest = {
        request,
        requestId,
        currentTime,
        priority,
        false
    };
    
    queue_.push(queuedRequest);
    requestArrivalTimes_[requestId] = currentTime;
    
    queueCV_.notify_one();
    return true;
}

bool HttpRequestQueue::dequeue(HttpRequest& request, size_t& requestId) {
    std::unique_lock<std::mutex> lock(queueMutex_);
    
    // 等待队列中有请求或者停止信号
    queueCV_.wait(lock, [this] { return !running_ || !queue_.empty(); });
    
    if (!running_) {
        return false;
    }
    
    if (queue_.empty()) {
        return false;
    }
    
    // 清理标记为删除的请求
    cleanupMarkedRequests();
    
    if (queue_.empty()) {
        return false;
    }
    
    const QueuedRequest& queuedRequest = queue_.top();
    size_t currentTime = time_utils::getCurrentTimeMs();
    size_t waitTime = currentTime - queuedRequest.arrivalTime;
    
    // 更新统计信息
    totalWaitTime_ += waitTime;
    completedRequests_++;
    requestArrivalTimes_.erase(queuedRequest.requestId);
    
    // 返回请求
    request = queuedRequest.request;
    requestId = queuedRequest.requestId;
    
    queue_.pop();
    return true;
}

bool HttpRequestQueue::removeRequest(size_t requestId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    
    auto it = requestArrivalTimes_.find(requestId);
    if (it == requestArrivalTimes_.end()) {
        return false;
    }
    
    requestArrivalTimes_.erase(it);
    deletedRequestIds_.insert(requestId);
    
    // 标记队首元素（如果匹配）
    if (!queue_.empty() && queue_.top().requestId == requestId) {
        cleanupMarkedRequests();
    }
    
    return true;
}

size_t HttpRequestQueue::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return queue_.size();
}

size_t HttpRequestQueue::getMaxQueueSize() const {
    return maxQueueSize_;
}

double HttpRequestQueue::getAverageWaitTime() const {
    size_t completed = completedRequests_.load();
    if (completed == 0) {
        return 0.0;
    }
    
    size_t totalWait = totalWaitTime_.load();
    return static_cast<double>(totalWait) / completed;
}

void HttpRequestQueue::setPriorityCallback(std::function<int(const HttpRequest&)> callback) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    priorityCallback_ = callback;
}

void HttpRequestQueue::cleanupMarkedRequests() {
    while (!queue_.empty()) {
        const QueuedRequest& topRequest = queue_.top();
        
        auto it = deletedRequestIds_.find(topRequest.requestId);
        if (it != deletedRequestIds_.end()) {
            queue_.pop();
            deletedRequestIds_.erase(it);
        } else {
            break;
        }
    }
}

} // namespace cllm
