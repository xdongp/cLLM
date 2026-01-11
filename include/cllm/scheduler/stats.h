/**
 * @file stats.h
 * @brief 调度器统计信息结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_SCHEDULER_STATS_H
#define CLLM_SCHEDULER_STATS_H

#include <atomic>
#include <string>
#include <vector>

namespace cllm {

struct RequestState;

/**
 * @brief 调度器统计信息结构
 * 
 * 记录调度器的运行统计信息，所有字段都是线程安全的原子类型。
 */
struct SchedulerStats {
    std::atomic<size_t> totalRequests{0};      ///< 总请求数
    std::atomic<size_t> completedRequests{0};  ///< 已完成请求数
    std::atomic<size_t> failedRequests{0};     ///< 失败请求数
    std::atomic<size_t> totalBatches{0};       ///< 总批次数
    std::atomic<float> averageBatchSize{0.0f}; ///< 平均批次大小
    std::atomic<float> averageRequestTime{0.0f};  ///< 平均请求时间
    std::atomic<float> averageWaitTime{0.0f};  ///< 平均等待时间
    std::atomic<size_t> peakQueueSize{0};      ///< 峰值队列大小
    
    /**
     * @brief 默认构造函数
     */
    SchedulerStats() = default;
    
    /**
     * @brief 拷贝构造函数
     * @param other 源对象
     */
    SchedulerStats(const SchedulerStats& other)
        : totalRequests(other.totalRequests.load())
        , completedRequests(other.completedRequests.load())
        , failedRequests(other.failedRequests.load())
        , totalBatches(other.totalBatches.load())
        , averageBatchSize(other.averageBatchSize.load())
        , averageRequestTime(other.averageRequestTime.load())
        , averageWaitTime(other.averageWaitTime.load())
        , peakQueueSize(other.peakQueueSize.load()) {}
    
    /**
     * @brief 拷贝赋值运算符
     * @param other 源对象
     * @return 当前对象引用
     */
    SchedulerStats& operator=(const SchedulerStats& other) {
        if (this != &other) {
            totalRequests.store(other.totalRequests.load());
            completedRequests.store(other.completedRequests.load());
            failedRequests.store(other.failedRequests.load());
            totalBatches.store(other.totalBatches.load());
            averageBatchSize.store(other.averageBatchSize.load());
            averageRequestTime.store(other.averageRequestTime.load());
            averageWaitTime.store(other.averageWaitTime.load());
            peakQueueSize.store(other.peakQueueSize.load());
        }
        return *this;
    }
    
    /**
     * @brief 更新统计信息（单个请求）
     * @param request 请求状态对象
     */
    void update(const RequestState& request);
    
    /**
     * @brief 更新统计信息（批次）
     * @param batch 请求批次
     */
    void updateBatch(const std::vector<RequestState>& batch);
    
    /**
     * @brief 重置所有统计信息
     */
    void reset();
    
    /**
     * @brief 转换为字符串表示
     * @return 统计信息的字符串表示
     */
    std::string toString() const;
};

}

#endif
