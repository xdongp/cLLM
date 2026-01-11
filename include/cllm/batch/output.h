/**
 * @file output.h
 * @brief 批处理输出数据结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_BATCH_OUTPUT_H
#define CLLM_BATCH_OUTPUT_H

#include "cllm/memory/float_array.h"
#include "cllm/common/logger.h"
#include <vector>
#include <utility>
#include <cstddef>

namespace cllm {

/**
 * @brief 批处理输出结构
 * 
 * 包含批处理的输出logits和请求位置信息。
 */
struct BatchOutput {
    FloatArray logits;                                      ///< 输出logits
    std::vector<std::pair<size_t, size_t>> requestPositions;  ///< 每个请求的起始和结束位置
    std::vector<size_t> sequenceIds;                        ///< 序列ID列表
    
    /**
     * @brief 获取指定请求的logits
     * @param requestIndex 请求索引
     * @param vocabSize 词表大小
     * @return 请求对应的logits（vocab_size维度）
     */
    FloatArray getLogitsForRequest(size_t requestIndex, size_t vocabSize = 32000) const {
        CLLM_DEBUG("getLogitsForRequest({}, vocabSize={})", requestIndex, vocabSize);
        CLLM_DEBUG("  requestPositions.size(): {}", requestPositions.size());
        CLLM_DEBUG("  logits.size(): {}", logits.size());
        
        if (requestIndex >= requestPositions.size()) {
            CLLM_DEBUG("  ERROR: requestIndex {} >= requestPositions.size() {}", requestIndex, requestPositions.size());
            return FloatArray();
        }
        
        auto [start, end] = requestPositions[requestIndex];
        CLLM_DEBUG("  Request positions: [{}, {}]", start, end);
        
        size_t lastTokenPos = end - 1;
        CLLM_DEBUG("  Last token position: {}", lastTokenPos);
        
        FloatArray result(vocabSize);
        size_t logitsOffset = lastTokenPos * vocabSize;
        CLLM_DEBUG("  Logits offset: {} ({} * {})", logitsOffset, lastTokenPos, vocabSize);
        
        if (logitsOffset + vocabSize > logits.size()) {
            CLLM_DEBUG("  WARNING: logitsOffset + vocabSize ({}) > logits.size() ({})", logitsOffset + vocabSize, logits.size());
            size_t availableSize = (logits.size() > start) ? std::min(vocabSize, logits.size() - start) : 0;
            CLLM_DEBUG("  Using fallback: availableSize = {}", availableSize);
            for (size_t i = 0; i < availableSize; ++i) {
                result[i] = logits[start + i];
            }
            for (size_t i = availableSize; i < vocabSize; ++i) {
                result[i] = 0.0f;
            }
        } else {
            CLLM_DEBUG("  Extracting logits from offset {}", logitsOffset);
            for (size_t i = 0; i < vocabSize; ++i) {
                result[i] = logits[logitsOffset + i];
            }
        }
        
        CLLM_DEBUG("  Returning logits with size: {}", result.size());
        CLLM_DEBUG("  First 5 logits: [");
        for (size_t i = 0; i < std::min((size_t)5, result.size()); ++i) {
            CLLM_DEBUG(" {}", result[i]);
        }
        CLLM_DEBUG(" ...]");
        
        return result;
    }
    
    /**
     * @brief 清空所有数据
     */
    void clear() {
        logits.clear();
        requestPositions.clear();
        sequenceIds.clear();
    }
    
    /**
     * @brief 检查是否为空
     * @return true 如果输出为空，false 否则
     */
    bool empty() const {
        return logits.empty();
    }
};

}

#endif
