/**
 * @file batch_processor.h
 * @brief 调度器批处理处理器
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_SCHEDULER_BATCH_PROCESSOR_H
#define CLLM_SCHEDULER_BATCH_PROCESSOR_H

#include <vector>
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"

namespace cllm {

struct RequestState;
class Scheduler;
class ModelExecutor;
class KVCache;
class BatchManager;

/**
 * @brief 调度器批处理处理器类
 * 
 * 负责处理批次请求，协调模型执行器、KV缓存和批处理管理器。
 */
class SchedulerBatchProcessor {
public:
    /**
     * @brief 构造函数
     * @param scheduler 调度器指针
     * @param executor 模型执行器指针
     * @param cache KV缓存指针
     * @param batchManager 批处理管理器指针
     */
    explicit SchedulerBatchProcessor(
        Scheduler* scheduler,
        ModelExecutor* executor,
        KVCache* cache,
        BatchManager* batchManager
    );
    
    /**
     * @brief 析构函数
     */
    ~SchedulerBatchProcessor();
    
    /**
     * @brief 处理批次
     * @param batch 请求批次
     */
    void processBatch(std::vector<RequestState>& batch);
    
    /**
     * @brief 检查批次是否完成
     * @param batch 请求批次
     * @return true 如果所有请求都完成，false 否则
     */
    bool isBatchComplete(const std::vector<RequestState>& batch) const;
    
    /**
     * @brief 获取活跃的请求
     * @param batch 请求批次
     * @return 活跃请求的向量
     */
    std::vector<RequestState> getActiveRequests(
        const std::vector<RequestState>& batch
    ) const;
    
private:
    void processIteration(std::vector<RequestState>& batch);  ///< 处理一次迭代
    void updateRequestStates(  ///< 更新请求状态
        std::vector<RequestState>& batch,
        const BatchOutput& output
    );
    
    Scheduler* scheduler_;         ///< 调度器指针
    ModelExecutor* executor_;      ///< 模型执行器指针
    KVCache* cache_;               ///< KV缓存指针
    BatchManager* batchManager_;   ///< 批处理管理器指针
};

}

#endif
