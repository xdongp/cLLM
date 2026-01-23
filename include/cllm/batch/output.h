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
#include "cllm/kylin/core/tensor.h"
#include <vector>
#include <utility>
#include <cstddef>
#include <cstring>
#include <memory>
#include <cstring>  // for memset

namespace cllm {

/**
 * @brief 批处理输出结构
 * 
 * 包含批处理的输出logits和请求位置信息。
 * 
 * 🔥 优化：支持直接使用Tensor，避免数据拷贝
 */
struct BatchOutput {
    FloatArray logits;                                      ///< 输出logits（兼容旧代码）
    std::unique_ptr<kylin::Tensor> logitsTensor;            ///< 🔥 优化：直接使用Tensor，避免拷贝（如果存在则优先使用）
    std::vector<std::pair<size_t, size_t>> requestPositions;  ///< 每个请求的起始和结束位置
    std::vector<size_t> sequenceIds;                        ///< 序列ID列表
    
    /**
     * @brief 获取指定请求的logits
     * @param requestIndex 请求索引
     * @param vocabSize 词表大小
     * @return 请求对应的logits（vocab_size维度）
     */
    FloatArray getLogitsForRequest(size_t requestIndex, size_t vocabSize = 32000) const {
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("getLogitsForRequest(requestIndex=%zu, vocabSize=%zu)", requestIndex, vocabSize);
        CLLM_DEBUG("  requestPositions.size(): %zu", requestPositions.size());
        #endif
        
        if (requestIndex >= requestPositions.size()) {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  ERROR: requestIndex %zu >= requestPositions.size() %zu", requestIndex, requestPositions.size());
            #endif
            return FloatArray();
        }
        
        auto [start, end] = requestPositions[requestIndex];
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("  Request positions: [start=%zu, end=%zu]", start, end);
        #endif
        
        size_t lastTokenPos = end - 1;
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("  Last token position: %zu (end - 1)", lastTokenPos);
        #endif
        
        FloatArray result(vocabSize);
        size_t logitsOffset = lastTokenPos * vocabSize;
        
        // 🔥 优化：优先使用Tensor，避免数据拷贝
        const float* srcData = nullptr;
        size_t totalSize = 0;
        
        if (logitsTensor && logitsTensor->size() > 0) {
            // 使用Tensor（零拷贝）
            srcData = logitsTensor->data();
            totalSize = logitsTensor->size();
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  Using Tensor (zero-copy), size: %zu", totalSize);
            #endif
        } else if (logits.size() > 0) {
            // 回退到FloatArray（兼容旧代码）
            srcData = logits.data();
            totalSize = logits.size();
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  Using FloatArray (fallback), size: %zu", totalSize);
            #endif
        } else {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  ERROR: No logits data available");
            #endif
            return FloatArray();
        }
        
        #ifdef CLLM_DEBUG_MODE
        CLLM_DEBUG("  Logits offset calculation: %zu = %zu * %zu", logitsOffset, lastTokenPos, vocabSize);
        CLLM_DEBUG("  Boundary check: logitsOffset + vocabSize = %zu, totalSize = %zu", logitsOffset + vocabSize, totalSize);
        #endif
        
        if (logitsOffset + vocabSize > totalSize) {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  WARNING: logitsOffset + vocabSize (%zu) > totalSize (%zu)", logitsOffset + vocabSize, totalSize);
            #endif
            size_t availableSize = (totalSize > start) ? std::min(vocabSize, totalSize - start) : 0;
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  Using fallback: availableSize = %zu (totalSize=%zu, start=%zu)", availableSize, totalSize, start);
            #endif
            // 🔥 优化：使用memcpy替代循环拷贝
            if (availableSize > 0) {
                std::memcpy(result.data(), srcData + start, availableSize * sizeof(float));
            }
            if (availableSize < vocabSize) {
                std::memset(result.data() + availableSize, 0, (vocabSize - availableSize) * sizeof(float));
            }
        } else {
            #ifdef CLLM_DEBUG_MODE
            CLLM_DEBUG("  Extracting logits from offset %zu to %zu", logitsOffset, logitsOffset + vocabSize);
            #endif
            // 🔥 优化：使用memcpy替代循环拷贝，提升性能
            std::memcpy(result.data(), srcData + logitsOffset, vocabSize * sizeof(float));
        }
        
        // 🔥 优化：减少不必要的调试日志和统计计算（在生产环境中关闭）
        // 这些操作在性能测试中会产生额外开销
        #ifdef CLLM_DEBUG_MODE
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
        #endif
        
        return result;
    }
    
    /**
     * @brief 清空所有数据
     */
    void clear() {
        logits.clear();
        logitsTensor.reset();
        requestPositions.clear();
        sequenceIds.clear();
    }
    
    /**
     * @brief 检查是否为空
     * @return true 如果输出为空，false 否则
     */
    bool empty() const {
        return (logitsTensor && logitsTensor->size() > 0) ? false : logits.empty();
    }
};

}

#endif
