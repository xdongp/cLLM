/**
 * @file queue.h
 * @brief 请求队列，管理请求的优先级调度和批处理
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include "cllm/common/request_state.h"
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace cllm {

/**
 * @brief 请求比较器
 * 
 * 用于优先级队列的比较操作，优先级高的和到达时间早的请求优先处理。
 */
struct RequestComparator {
    /**
     * @brief 比较两个请求的优先级
     * @param a 请求A
     * @param b 请求B
     * @return true 如果A的优先级低于B
     */
    bool operator()(const RequestState& a, const RequestState& b) const {
        if (a.priority != b.priority) {
            return a.priority > b.priority;
        }
        return a.arrivalTime > b.arrivalTime;
    }
};

/**
 * @brief 请求队列类
 * 
 * 管理请求的优先级调度、批处理组装和队列统计。
 * 实现线程安全的队列操作，支持动态优先级调整。
 */
class RequestQueue {
public:
    /**
     * @brief 构造函数
     * @param maxQueueSize 最大队列长度，默认1000
     * @param maxContextLength 最大上下文长度，默认2048
     */
    explicit RequestQueue(size_t maxQueueSize = 1000, size_t maxContextLength = 2048);
    
    /**
     * @brief 析构函数
     */
    ~RequestQueue();
    
    RequestQueue(const RequestQueue&) = delete;
    RequestQueue& operator=(const RequestQueue&) = delete;
    
    /**
     * @brief 添加请求到队列
     * @param request 请求状态
     * @return true 如果添加成功，false 否则
     */
    bool addRequest(const RequestState& request);
    
    /**
     * @brief 移除指定的请求
     * @param requestId 请求ID
     * @return true 如果移除成功，false 否则
     */
    bool removeRequest(size_t requestId);
    
    /**
     * @brief 获取下一个请求（最高优先级）
     * @return 下一个要处理的请求
     */
    RequestState getNextRequest();
    
    /**
     * @brief 获取下一个请求（非阻塞模式）
     * @param request 输出参数，存储获取的请求
     * @return 是否成功获取请求
     */
    bool tryGetNextRequest(RequestState& request);
    
    /**
     * @brief 组装批处理
     * @param maxContextLength 批处理的最大上下文长度
     * @return 批处理请求列表
     */
    std::vector<RequestState> formBatch(size_t maxContextLength);
    
    /**
     * @brief 获取所有待处理请求
     * @return 待处理请求列表
     */
    std::vector<RequestState> getPendingRequests() const;
    
    /**
     * @brief 获取队列大小
     * @return 队列中的请求数
     */
    size_t getQueueSize() const;
    
    /**
     * @brief 获取运行中请求的总长度
     * @return 运行中请求的token总数
     */
    size_t getRunningRequestsLength() const;
    
    /**
     * @brief 获取平均等待时间
     * @return 平均等待时间（毫秒）
     */
    float getAverageWaitTime() const;
    
    /**
     * @brief 获取平均请求长度
     * @return 平均请求长度（token数）
     */
    size_t getAverageRequestLength() const;
    
    /**
     * @brief 更新运行中的请求列表
     * @param running 运行中的请求列表
     */
    void updateRunningRequests(const std::vector<RequestState>& running);
    
    /**
     * @brief 清空队列
     */
    void clear();
    
private:
    /**
     * @brief 更新请求优先级
     */
    void updatePriorities();
    
    /**
     * @brief 计算最优批处理大小
     * @param requests 请求列表
     * @return 最优批大小
     */
    size_t calculateOptimalBatchSize(const std::vector<RequestState>& requests);
    
    mutable std::priority_queue<RequestState, std::vector<RequestState>, RequestComparator> queue_;  ///< 优先级队列
    std::vector<RequestState> runningRequests_;     ///< 运行中的请求列表
    mutable std::mutex queueMutex_;                 ///< 队列互斥锁
    std::condition_variable condition_;             ///< 条件变量
    
    size_t maxQueueSize_;                           ///< 最大队列大小
    size_t maxContextLength_;                       ///< 最大上下文长度
    std::atomic<size_t> totalRequests_;             ///< 总请求数
    std::atomic<size_t> completedRequests_;         ///< 已完成请求数
    std::atomic<size_t> totalWaitTime_;             ///< 总等待时间（毫秒）
    bool stopFlag_;                                 ///< 停止标志
};

}  // namespace cllm