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
#include <shared_mutex>
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

class DynamicBatchTuner;
class HybridBatchStrategy;

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
 * @brief 批处理池（优化：减少内存分配）
 * 
 * 预分配批处理对象，避免频繁的内存分配和释放。
 * 提高CPU缓存友好性，减少内存碎片。
 */
class BatchPool {
private:
    static constexpr size_t POOL_SIZE = 16;
    static constexpr size_t BATCH_CAPACITY = 32;
    
    std::array<std::vector<RequestState>, POOL_SIZE> pool_;
    std::atomic<size_t> nextIndex_{0};
    
public:
    BatchPool() {
        for (auto& batch : pool_) {
            batch.reserve(BATCH_CAPACITY);
        }
    }
    
    /**
     * @brief 从池中获取一个批处理对象
     * @return 批处理对象的引用
     */
    std::vector<RequestState>& acquire() {
        size_t index = nextIndex_.fetch_add(1, std::memory_order_relaxed) % POOL_SIZE;
        auto& batch = pool_[index];
        batch.clear();
        return batch;
    }
    
    /**
     * @brief 释放批处理对象（实际上不需要做任何事）
     * @param batch 批处理对象
     */
    void release(std::vector<RequestState>& batch) {
        // 不需要做任何事，对象在池中复用
        (void)batch;
    }
};

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
    
    /**
     * @brief 判断后端是否需要外部 KVCache
     * @param backendType 后端类型
     * @return true 如果需要外部 KVCache，false 否则（如 llama.cpp 后端内部管理）
     */
    bool needsExternalKVCache(const std::string& backendType) const;
    
    RequestQueue requestQueue_;        ///< 请求队列
    BatchManager batchManager_;        ///< 批处理管理器
    ModelExecutor* modelExecutor_;  ///< 模型执行器
    KVCache* kvCache_ = nullptr;    ///< KV缓存（llama.cpp 后端为 nullptr）
    bool ownsModelExecutor_;        ///< 是否拥有模型执行器所有权
    RequestTracker requestTracker_;    ///< 请求跟踪器
    
    BatchPool batchPool_;  ///< 批处理池（优化：减少内存分配）
    
    std::map<size_t, RequestState> runningRequests_;    ///< 运行中的请求
    std::map<size_t, RequestState> completedRequests_;  ///< 已完成的请求
    
    std::thread schedulerThread_;      ///< 调度器线程
    std::thread cleanupThread_;        ///< 异步清理线程
    std::atomic<bool> running_{false}; ///< 运行状态
    
    size_t maxBatchSize_;              ///< 最大批处理大小
    size_t maxContextLength_;          ///< 最大上下文长度
    SchedulerConfig config_;           ///< 调度器配置
    
    mutable std::mutex queueMutex_;     ///< 队列互斥锁
    mutable std::shared_mutex requestsMutex_;  ///< 请求读写锁（优化：读多写少场景）
    mutable std::mutex statsMutex_;     ///< 统计互斥锁
    std::condition_variable_any resultCondition_;  ///< 结果条件变量（优化：支持shared_mutex）
    
    // 🔥 优化步骤1: 原子操作只读缓存（减少锁竞争）
    std::atomic<size_t> cachedQueueSize_{0};      ///< 队列大小缓存（原子操作，快速读取）
    std::atomic<size_t> cachedRunningCount_{0};   ///< 运行中请求数缓存（原子操作，快速读取）
    std::condition_variable queueCondition_;   ///< 队列条件变量
    
    SchedulerStats stats_;             ///< 统计信息
    
    // Phase 7: 响应回调
    ResponseCallback responseCallback_;  ///< 响应回调函数
    mutable std::mutex callbackMutex_;   ///< 回调互斥锁
    
    // 异步资源清理
    std::queue<size_t> cleanupQueue_;    ///< 清理任务队列
    mutable std::mutex cleanupMutex_;    ///< 清理队列互斥锁
    std::condition_variable cleanupCondition_;  ///< 清理条件变量
    void cleanupLoop();                 ///< 清理线程循环
    void cleanupRequestAsync(size_t requestId);  ///< 异步清理请求资源
    
    std::unique_ptr<HybridBatchStrategy> hybridStrategy_;  ///< 混合批处理策略
    size_t staticBatchSize_;  ///< 静态批处理策略的固定 batch size
};

}

#endif
