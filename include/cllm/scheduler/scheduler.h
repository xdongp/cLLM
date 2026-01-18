/**
 * @file scheduler.h
 * @brief 调度器核心类，负责请求调度和批处理
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_SCHEDULER_H
#define CLLM_SCHEDULER_H

#include <string>
#include <vector>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>

#include "cllm/scheduler/config.h"
#include "cllm/scheduler/stats.h"
#include "cllm/scheduler/tracker.h"
#include "cllm/scheduler/batch_processor.h"
#include "cllm/common/queue.h"
#include "cllm/common/request_state.h"
#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/common/config.h"

namespace cllm {

/**
 * @brief 调度器错误类型枚举
 */
enum class SchedulerError {
    SCHEDULER_NOT_RUNNING,       ///< 调度器未运行
    REQUEST_NOT_FOUND,           ///< 请求未找到
    REQUEST_TIMEOUT,             ///< 请求超时
    REQUEST_QUEUE_FULL,          ///< 请求队列已满
    BATCH_PROCESSING_FAILED,     ///< 批处理失败
    INVALID_REQUEST              ///< 无效请求
};

/**
 * @brief 调度器异常类
 */
class SchedulerException : public std::runtime_error {
public:
    /**
     * @brief 构造函数
     * @param error 错误类型
     * @param message 错误消息
     */
    SchedulerException(SchedulerError error, const std::string& message)
        : std::runtime_error(message), error_(error) {}
    
    /**
     * @brief 获取错误类型
     * @return 错误类型
     */
    SchedulerError getError() const { return error_; }
    
private:
    SchedulerError error_;  ///< 错误类型
};

/**
 * @brief Phase 7: 响应回调函数类型
 * @param requestId 请求ID
 * @param state 请求状态
 */
using ResponseCallback = std::function<void(size_t requestId, const RequestState& state)>;

/**
 * @brief 调度器类
 * 
 * 负责请求的调度、批处理和执行管理。
 * 维护请求队列，协调模型执行器和KV缓存，处理多个并发请求。
 */
class Scheduler {
public:
    /**
     * @brief 构造函数
     * @param modelExecutor 模型执行器实例
     * @param maxBatchSize 最大批处理大小
     * @param maxContextLength 最大上下文长度
     */
    Scheduler(
        ModelExecutor* modelExecutor,
        size_t maxBatchSize = 8,
        size_t maxContextLength = 2048
    );
    
    /**
     * @brief 构造函数（兼容旧接口，仅用于测试）
     * @param modelPath 模型路径
     * @param quantization 量化类型
     * @param maxBatchSize 最大批处理大小
     * @param maxContextLength 最大上下文长度
     */
    Scheduler(
        const std::string& modelPath,
        const std::string& quantization = "",
        size_t maxBatchSize = 8,
        size_t maxContextLength = 2048
    );

    
    /**
     * @brief 析构函数
     */
    ~Scheduler();
    
    /**
     * @brief 启动调度器
     */
    void start();
    
    /**
     * @brief 停止调度器
     */
    void stop();
    
    /**
     * @brief 添加请求到队列
     * @param request 请求状态对象
     * @return 请求ID
     */
    size_t addRequest(const RequestState& request);
    
    /**
     * @brief 移除请求
     * @param requestId 请求ID
     * @return true 如果成功移除，false 否则
     */
    bool removeRequest(size_t requestId);
    
    /**
     * @brief 获取请求结果
     * @param requestId 请求ID
     * @return 请求状态对象
     */
    RequestState getRequestResult(size_t requestId);
    
    /**
     * @brief 等待请求完成
     * @param requestId 请求ID
     * @param timeout 超时时间（秒）
     * @return true 如果请求完成，false 如果超时
     */
    bool waitForRequest(size_t requestId, float timeout = 300.0f);
    
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
     * @brief 获取队列大小
     * @return 队列中等待的请求数量
     */
    size_t getQueueSize() const;
    
    /**
     * @brief 获取统计信息
     * @return 调度器统计信息
     */
    SchedulerStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
    /**
     * @brief Phase 6: 获取运行中请求数量
     * @return 运行中请求数量
     */
    size_t getRunningCount() const;
    
    /**
     * @brief Phase 6: 获取最大并发请求数
     * @return 最大并发请求数
     */
    size_t getMaxConcurrentRequests() const;
    
    /**
     * @brief Phase 7: 设置响应回调函数
     * @param callback 回调函数
     */
    void setResponseCallback(ResponseCallback callback);
    
    // Phase 7: 触发响应回调（供内部使用）
    void triggerResponseCallback(size_t requestId, const RequestState& state);
    
private:
    void schedulerLoop();  ///< 调度器主循环
    void processRequests();  ///< 处理请求
    void processBatch(std::vector<RequestState>& batch);  ///< 处理批次
    void checkRequestTimeout();  ///< Phase 3: 检查请求超时
    void checkKVCachEviction();  ///< Phase 5: 检查KV缓存淘汰
    size_t getCurrentTime();  ///< 获取当前时间（毫秒）
    
    RequestQueue requestQueue_;        ///< 请求队列
    BatchManager batchManager_;        ///< 批处理管理器
    ModelExecutor* modelExecutor_;  ///< 模型执行器
    KVCache* kvCache_;              ///< KV缓存
    bool ownsModelExecutor_;        ///< 是否拥有模型执行器所有权
    RequestTracker requestTracker_;    ///< 请求跟踪器
    
    std::map<size_t, RequestState> runningRequests_;    ///< 运行中的请求
    std::map<size_t, RequestState> completedRequests_;  ///< 已完成的请求
    
    std::thread schedulerThread_;      ///< 调度器线程
    std::atomic<bool> running_{false}; ///< 运行状态
    
    size_t maxBatchSize_;              ///< 最大批处理大小
    size_t maxContextLength_;          ///< 最大上下文长度
    SchedulerConfig config_;           ///< 调度器配置
    
    mutable std::mutex queueMutex_;     ///< 队列互斥锁
    mutable std::mutex requestsMutex_;  ///< 请求互斥锁
    mutable std::mutex statsMutex_;     ///< 统计互斥锁
    std::condition_variable resultCondition_;  ///< 结果条件变量
    std::condition_variable queueCondition_;   ///< 队列条件变量
    
    SchedulerStats stats_;             ///< 统计信息
    
    // Phase 7: 响应回调
    ResponseCallback responseCallback_;  ///< 响应回调函数
    mutable std::mutex callbackMutex_;   ///< 回调互斥锁
};

}

#endif
