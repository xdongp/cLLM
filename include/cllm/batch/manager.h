/**
 * @file manager.h
 * @brief 批处理管理器，负责批次组装和处理
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_BATCH_MANAGER_H
#define CLLM_BATCH_MANAGER_H

#include "cllm/common/request_state.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/batch/stats.h"
#include "cllm/sampler.h"
#include "cllm/common/config.h"
#include <vector>
#include <mutex>
#include <cstddef>

namespace cllm {

struct RequestState;
class ModelExecutor;

/**
 * @brief 批处理管理器类
 * 
 * 负责将请求组装成批次，准备批处理输入，处理批处理输出。
 * 支持动态批处理大小和上下文长度管理。
 */
class BatchManager {
public:
    /**
     * @brief 构造函数
     * @param maxContextLength 最大上下文长度
     * @param maxBatchSize 最大批处理大小
     */
    explicit BatchManager(size_t maxContextLength, size_t maxBatchSize = 32);
    
    /**
     * @brief 构造函数（带模型执行器）
     * @param maxContextLength 最大上下文长度
     * @param maxBatchSize 最大批处理大小
     * @param executor 模型执行器指针
     */
    explicit BatchManager(size_t maxContextLength, size_t maxBatchSize, ModelExecutor* executor);
    
    /**
     * @brief 析构函数
     */
    ~BatchManager();
    
    /**
     * @brief 禁止拷贝构造
     */
    BatchManager(const BatchManager&) = delete;
    
    /**
     * @brief 禁止拷贝赋值
     */
    BatchManager& operator=(const BatchManager&) = delete;
    
    /**
     * @brief 组装一个批次
     * @param pendingRequests 待处理请求
     * @param runningRequests 运行中请求
     * @param availableSeqIds 可用的序列ID数量（用于限制批处理大小，0表示不限制）
     * @return 组装好的批次请求
     */
    std::vector<RequestState> formBatch(
        const std::vector<RequestState>& pendingRequests,
        const std::vector<RequestState>& runningRequests,
        size_t availableSeqIds = 0
    );
    
    /**
     * @brief 组装多个批次
     * @param pendingRequests 待处理请求
     * @param runningRequests 运行中请求
     * @return 组装好的多个批次
     */
    std::vector<RequestState> formMultipleBatches(
        const std::vector<RequestState>& pendingRequests,
        const std::vector<RequestState>& runningRequests
    );
    
    /**
     * @brief 准备批处理输入
     * @param batch 请求批次
     * @return 批处理输入数据
     */
    BatchInput prepareBatchInput(const std::vector<RequestState>& batch);
    
    /**
     * @brief 🔥 优化: 增量准备批处理输入（只更新新增的tokens，减少数据复制）
     * @param batch 请求批次（包含更新后的generatedTokens）
     * @param previousInput 上次的批处理输入（用于增量更新）
     * @param previousTokenCounts 上次每个请求的token数量（用于检测变化）
     * @return 更新后的批处理输入数据
     */
    BatchInput prepareBatchInputIncremental(
        const std::vector<RequestState>& batch,
        const BatchInput& previousInput,
        const std::vector<size_t>& previousTokenCounts
    );
    
    /**
     * @brief 处理批处理输出
     * @param batch 请求批次
     * @param output 批处理输出数据
     */
    void processBatchOutput(
        std::vector<RequestState>& batch,
        const BatchOutput& output
    );
    
    /**
     * @brief 计算最优批处理大小
     * @param requests 请求列表
     * @param avgRequestLength 平均请求长度
     * @return 最优批处理大小
     */
    size_t calculateOptimalBatchSize(
        const std::vector<RequestState>& requests,
        size_t avgRequestLength
    );

    /**
     * @brief 更新最大批处理大小
     * @param maxBatchSize 最大批处理大小
     */
    void setMaxBatchSize(size_t maxBatchSize);

    /**
     * @brief 获取当前最大批处理大小
     * @return 最大批处理大小
     */
    size_t getMaxBatchSize() const;
    
    /**
     * @brief 根据系统负载动态调整批处理大小
     * @param queueSize 队列大小
     * @param runningCount 运行中请求数
     * @return 动态批处理大小
     */
    size_t adaptiveBatchSize(size_t queueSize, size_t runningCount);
    
    /**
     * @brief 更新批处理时间
     * @param processingTimeMs 批处理时间（毫秒）
     */
    void updateBatchProcessingTime(size_t processingTimeMs);
    
    /**
     * @brief 检查是否可以将请求添加到批次
     * @param request 请求对象
     * @param currentBatch 当前批次
     * @param currentBatchLength 当前批次长度
     * @param dynamicBatchSize 动态批处理大小
     * @return true 如果可以添加，false 否则
     */
    bool canAddToBatch(
        const RequestState& request,
        const std::vector<RequestState>& currentBatch,
        size_t currentBatchLength,
        size_t dynamicBatchSize
    );
    
    /**
     * @brief 获取统计信息
     * @return 批处理统计信息
     */
    BatchStats getStats() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStats();
    
private:
    size_t calculateRunningRequestsLength(  ///< 计算运行中请求的总长度
        const std::vector<RequestState>& runningRequests
    );
    
    size_t calculateAverageRequestLength(  ///< 计算平均请求长度
        const std::vector<RequestState>& requests
    );
    
    void updateStats(const std::vector<RequestState>& batch);  ///< 更新统计信息
    
    void checkStoppingConditions(RequestState& request, int nextToken);  ///< 检查停止条件
    
    size_t maxContextLength_;       ///< 最大上下文长度
    size_t maxBatchSize_;           ///< 最大批处理大小
    float contextUsageThreshold_;   ///< 上下文使用阈值
    
    Sampler sampler_;               ///< 采样器
    ModelExecutor* executor_;       ///< 模型执行器指针
    
    mutable std::mutex statsMutex_; ///< 统计信息互斥锁
    BatchStats stats_;              ///< 统计信息
    
    size_t lastBatchProcessingTimeMs_;  ///< 上次批处理时间（毫秒）
    size_t adaptiveBatchSize_;          ///< 自适应批处理大小
    size_t minAdaptiveBatchSize_;      ///< 最小自适应批处理大小
    size_t maxAdaptiveBatchSize_;      ///< 最大自适应批处理大小
};

}

#endif
