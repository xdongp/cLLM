/**
 * @file input.h
 * @brief 批处理输入数据结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_BATCH_INPUT_H
#define CLLM_BATCH_INPUT_H

#include <vector>
#include <utility>
#include <cstddef>

namespace cllm {

/**
 * @brief 批处理输入结构
 * 
 * 包含批处理所需的输入数据，包括输入token ID、请求位置和序列ID。
 */
struct BatchInput {
    std::vector<int> inputIds;                              ///< 输入token ID列表
    std::vector<std::pair<size_t, size_t>> requestPositions;  ///< 每个请求的起始和结束位置
    size_t batchSize;                                       ///< 批次大小
    std::vector<size_t> sequenceIds;                        ///< 序列ID列表
    
    /**
     * @brief 默认构造函数
     */
    BatchInput() : batchSize(0) {}
    
    /**
     * @brief 获取总 token 数量
     * @return 输入中的总 token 数
     */
    size_t getTotalTokens() const {
        return inputIds.size();
    }
    
    /**
     * @brief 清空所有数据
     */
    void clear() {
        inputIds.clear();
        requestPositions.clear();
        sequenceIds.clear();
        batchSize = 0;
    }
    
    /**
     * @brief 检查是否为空
     * @return true 如果批次为空，false 否则
     */
    bool empty() const {
        return batchSize == 0;
    }
};

}

#endif
