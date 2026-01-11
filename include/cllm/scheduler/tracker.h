/**
 * @file tracker.h
 * @brief 请求跟踪器，管理请求的生命周期
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_REQUEST_TRACKER_H
#define CLLM_REQUEST_TRACKER_H

#include <map>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <vector>

namespace cllm {

struct RequestState;

/**
 * @brief 请求跟踪器类
 * 
 * 负责跟踪和管理请求的生命周期，维护运行中和已完成的请求。
 * 所有操作都是线程安全的。
 */
class RequestTracker {
public:
    /**
     * @brief 构造函数
     */
    explicit RequestTracker();
    
    /**
     * @brief 析构函数
     */
    ~RequestTracker();
    
    /**
     * @brief 添加请求
     * @param request 请求状态对象
     * @return 分配的请求ID
     */
    size_t addRequest(const RequestState& request);
    
    /**
     * @brief 更新请求
     * @param requestId 请求ID
     * @param updated 更新后的请求状态
     * @return true 如果成功更新，false 否则
     */
    bool updateRequest(size_t requestId, const RequestState& updated);
    
    /**
     * @brief 移除请求
     * @param requestId 请求ID
     * @return true 如果成功移除，false 否则
     */
    bool removeRequest(size_t requestId);
    
    /**
     * @brief 获取请求
     * @param requestId 请求ID
     * @return 请求状态对象
     */
    RequestState getRequest(size_t requestId) const;
    
    /**
     * @brief 获取所有运行中的请求
     * @return 运行中请求的向量
     */
    std::vector<RequestState> getRunningRequests() const;
    
    /**
     * @brief 获取所有已完成的请求
     * @return 已完成请求的向量
     */
    std::vector<RequestState> getCompletedRequests() const;
    
    /**
     * @brief 标记请求为运行中
     * @param requestId 请求ID
     * @return true 如果成功标记，false 否则
     */
    bool markAsRunning(size_t requestId);
    
    /**
     * @brief 标记请求为已完成
     * @param requestId 请求ID
     * @return true 如果成功标记，false 否则
     */
    bool markAsCompleted(size_t requestId);
    
    /**
     * @brief 标记请求为失败
     * @param requestId 请求ID
     * @param error 错误消息
     * @return true 如果成功标记，false 否则
     */
    bool markAsFailed(size_t requestId, const std::string& error);
    
    /**
     * @brief 获取运行中请求的数量
     * @return 运行中请求数
     */
    size_t getRunningCount() const;
    
    /**
     * @brief 获取已完成请求的数量
     * @return 已完成请求数
     */
    size_t getCompletedCount() const;
    
private:
    std::map<size_t, RequestState> runningRequests_;    ///< 运行中的请求
    std::map<size_t, RequestState> completedRequests_;  ///< 已完成的请求
    
    mutable std::mutex mutex_;              ///< 互斥锁
    std::condition_variable condition_;     ///< 条件变量
    
    std::atomic<size_t> nextRequestId_{1};  ///< 下一个请求ID
};

}

#endif
