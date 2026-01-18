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
        CLLM_DEBUG("getLogitsForRequest(requestIndex=%zu, vocabSize=%zu)", requestIndex, vocabSize);
        CLLM_DEBUG("  requestPositions.size(): %zu", requestPositions.size());
        CLLM_DEBUG("  logits.size(): %zu", logits.size());
        
        if (requestIndex >= requestPositions.size()) {
            CLLM_DEBUG("  ERROR: requestIndex %zu >= requestPositions.size() %zu", requestIndex, requestPositions.size());
            return FloatArray();
        }
        
        auto [start, end] = requestPositions[requestIndex];
        CLLM_DEBUG("  Request positions: [start=%zu, end=%zu]", start, end);
        
        size_t lastTokenPos = end - 1;
        CLLM_DEBUG("  Last token position: %zu (end - 1)", lastTokenPos);
        
        FloatArray result(vocabSize);
        size_t logitsOffset = lastTokenPos * vocabSize;
        CLLM_DEBUG("  Logits offset calculation: %zu = %zu * %zu", logitsOffset, lastTokenPos, vocabSize);
        CLLM_DEBUG("  Boundary check: logitsOffset + vocabSize = %zu, logits.size() = %zu", logitsOffset + vocabSize, logits.size());
        
        if (logitsOffset + vocabSize > logits.size()) {
            CLLM_DEBUG("  WARNING: logitsOffset + vocabSize (%zu) > logits.size() (%zu)", logitsOffset + vocabSize, logits.size());
            size_t availableSize = (logits.size() > start) ? std::min(vocabSize, logits.size() - start) : 0;
            CLLM_DEBUG("  Using fallback: availableSize = %zu (logits.size()=%zu, start=%zu)", availableSize, logits.size(), start);
            for (size_t i = 0; i < availableSize; ++i) {
                result[i] = logits[start + i];
            }
            for (size_t i = availableSize; i < vocabSize; ++i) {
                result[i] = 0.0f;
            }
        } else {
            CLLM_DEBUG("  Extracting logits from offset %zu to %zu", logitsOffset, logitsOffset + vocabSize);
            for (size_t i = 0; i < vocabSize; ++i) {
                result[i] = logits[logitsOffset + i];
            }
        }
        
        // 检查提取的 logits 值
        size_t nonZeroCount = 0;
        float maxLogit = result.empty() ? 0.0f : result[0];
        float minLogit = result.empty() ? 0.0f : result[0];
        for (size_t i = 0; i < result.size(); ++i) {
            if (result[i] != 0.0f) nonZeroCount++;
            if (result[i] > maxLogit) maxLogit = result[i];
            if (result[i] < minLogit) minLogit = result[i];
        }
        
        CLLM_DEBUG("  Returning logits with size: %zu", result.size());
        CLLM_DEBUG("  Extracted logits stats: non_zero=%zu, max=%.6f, min=%.6f", nonZeroCount, maxLogit, minLogit);
        CLLM_DEBUG("  First 10 logits: [%.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f %.6f ...]", 
                   result.size() > 0 ? result[0] : 0.0f,
                   result.size() > 1 ? result[1] : 0.0f,
                   result.size() > 2 ? result[2] : 0.0f,
                   result.size() > 3 ? result[3] : 0.0f,
                   result.size() > 4 ? result[4] : 0.0f,
                   result.size() > 5 ? result[5] : 0.0f,
                   result.size() > 6 ? result[6] : 0.0f,
                   result.size() > 7 ? result[7] : 0.0f,
                   result.size() > 8 ? result[8] : 0.0f,
                   result.size() > 9 ? result[9] : 0.0f);
        
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
