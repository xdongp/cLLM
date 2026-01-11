/**
 * @file processor.h
 * @brief 批处理处理器
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_BATCH_PROCESSOR_H
#define CLLM_BATCH_PROCESSOR_H

#include "cllm/batch/manager.h"
#include <vector>

namespace cllm {

/**
 * @brief 批处理处理器类
 * 
 * 负责处理批次请求，检查完成状态，管理活跃和已完成的请求。
 */
class BatchProcessor {
public:
    /**
     * @brief 构造函数
     * @param manager 批处理管理器指针
     */
    explicit BatchProcessor(BatchManager* manager);
    
    /**
     * @brief 析构函数
     */
    ~BatchProcessor();
    
    /**
     * @brief 禁止拷贝构造
     */
    BatchProcessor(const BatchProcessor&) = delete;
    
    /**
     * @brief 禁止拷贝赋值
     */
    BatchProcessor& operator=(const BatchProcessor&) = delete;
    
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
    
    /**
     * @brief 获取已完成的请求
     * @param batch 请求批次
     * @return 已完成请求的向量
     */
    std::vector<RequestState> getCompletedRequests(
        const std::vector<RequestState>& batch
    ) const;
    
private:
    void checkStoppingConditions(RequestState& request, int nextToken);  ///< 检查停止条件
    
    BatchManager* manager_;  ///< 批处理管理器指针
};

}

#endif
