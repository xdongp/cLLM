#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <map>
#include <set>
#include <atomic>
#include "cllm/http/request.h"

namespace cllm {

class HttpRequestQueue {
public:
    HttpRequestQueue(size_t maxQueueSize = 1000, size_t maxWaitTime = 300);
    ~HttpRequestQueue();
    
    bool enqueue(const HttpRequest& request, size_t requestId);
    bool dequeue(HttpRequest& request, size_t& requestId);
    bool removeRequest(size_t requestId);
    
    size_t getQueueSize() const;
    size_t getMaxQueueSize() const;
    double getAverageWaitTime() const;
    
    void setPriorityCallback(std::function<int(const HttpRequest&)> callback);
    
private:
    struct QueuedRequest {
        HttpRequest request;
        size_t requestId;
        size_t arrivalTime;
        int priority;
        bool markedForDeletion;
        
        // 重载比较运算符，用于优先级队列
        bool operator<(const QueuedRequest& other) const {
            return priority < other.priority; // 优先级高的请求排在前面
        }
    };
    
    std::priority_queue<QueuedRequest> queue_;
    mutable std::mutex queueMutex_;
    std::condition_variable queueCV_;
    
    size_t maxQueueSize_;
    size_t maxWaitTime_;
    std::function<int(const HttpRequest&)> priorityCallback_;
    
    std::map<size_t, size_t> requestArrivalTimes_;
    std::set<size_t> deletedRequestIds_;
    
    std::atomic<size_t> totalWaitTime_;
    std::atomic<size_t> completedRequests_;
    std::atomic<bool> running_;
    
    void cleanupMarkedRequests();
};

} // namespace cllm
