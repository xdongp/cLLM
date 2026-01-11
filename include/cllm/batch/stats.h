/**
 * @file stats.h
 * @brief 批处理统计信息结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_BATCH_STATS_H
#define CLLM_BATCH_STATS_H

#include <string>
#include <cstddef>

namespace cllm {

/**
 * @brief 批处理统计信息结构
 * 
 * 记录批处理的运行统计信息。
 */
struct BatchStats {
    size_t totalBatches;        ///< 总批次数
    size_t totalRequests;       ///< 总请求数
    size_t totalTokens;         ///< 总tokens数
    float averageBatchSize;     ///< 平均批次大小
    float averageBatchLength;   ///< 平均批次长度
    size_t maxBatchSize;        ///< 最大批次大小
    size_t minBatchSize;        ///< 最小批次大小
    
    /**
     * @brief 默认构造函数
     */
    BatchStats() 
        : totalBatches(0)
        , totalRequests(0)
        , totalTokens(0)
        , averageBatchSize(0.0f)
        , averageBatchLength(0.0f)
        , maxBatchSize(0)
        , minBatchSize(0) {}
    
    /**
     * @brief 更新统计信息
     * @param batchSize 批次大小
     * @param batchLength 批次长度
     */
    void update(size_t batchSize, size_t batchLength) {
        totalBatches++;
        totalRequests += batchSize;
        totalTokens += batchLength;
        
        averageBatchSize = static_cast<float>(totalRequests) / totalBatches;
        averageBatchLength = static_cast<float>(totalTokens) / totalBatches;
        
        if (batchSize > maxBatchSize) {
            maxBatchSize = batchSize;
        }
        
        if (minBatchSize == 0 || batchSize < minBatchSize) {
            minBatchSize = batchSize;
        }
    }
    
    /**
     * @brief 重置所有统计信息
     */
    void reset() {
        totalBatches = 0;
        totalRequests = 0;
        totalTokens = 0;
        averageBatchSize = 0.0f;
        averageBatchLength = 0.0f;
        maxBatchSize = 0;
        minBatchSize = 0;
    }
    
    /**
     * @brief 转换为字符串表示
     * @return 统计信息的字符串表示
     */
    std::string toString() const {
        return "BatchStats{totalBatches=" + std::to_string(totalBatches) +
               ", totalRequests=" + std::to_string(totalRequests) +
               ", totalTokens=" + std::to_string(totalTokens) +
               ", averageBatchSize=" + std::to_string(averageBatchSize) +
               ", averageBatchLength=" + std::to_string(averageBatchLength) +
               ", maxBatchSize=" + std::to_string(maxBatchSize) +
               ", minBatchSize=" + std::to_string(minBatchSize) + "}";
    }
};

}

#endif
