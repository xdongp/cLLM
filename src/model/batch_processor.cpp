/**
 * @file batch_processor.cpp
 * @brief 批处理处理器的实现
 * @author cLLM Team
 * @date 2026-01-08
 */
#include "cllm/model/batch_processor.h"
#include "cllm/model/executor.h"
#include <algorithm>
#include <stdexcept>

namespace cllm {

BatchProcessor::BatchProcessor(ModelExecutor* executor)
    : executor_(executor) {
    if (!executor) {
        throw std::invalid_argument("ModelExecutor pointer cannot be null");
    }
}

BatchProcessor::~BatchProcessor() {
    // 清理资源
}

BatchOutput BatchProcessor::processBatch(const BatchInput& input) {
    if (input.requestPositions.empty()) {
        throw std::invalid_argument("Batch size cannot be zero");
    }
    
    // 🔥 优化：对于单请求场景，跳过prepareBatchInput()和processBatchOutput()
    // 因为它们对于单请求都是直接返回，但函数调用仍有开销
    if (input.requestPositions.size() == 1) {
        // 单请求场景：直接调用executor，跳过中间处理
        return executor_->forward(input);
    }
    
    // 多请求场景：需要填充/去填充处理
    prepareBatchInput(input);
    
    BatchOutput output = executor_->forward(input);
    
    processBatchOutput(output);
    
    return output;
}

void BatchProcessor::prepareBatchInput(const BatchInput& input) {
    // 准备批处理输入
    // 简化实现，仅用于演示
    if (input.requestPositions.size() == 1) {
        // 单个请求，无需特殊处理
        return;
    }
    
    // 计算最大序列长度
    size_t maxSeqLength = 0;
    for (const auto& pos : input.requestPositions) {
        maxSeqLength = std::max(maxSeqLength, pos.second);
    }
    
    // 对输入进行填充
    std::vector<int> paddedInputIds = input.inputIds;
    _padBatch(paddedInputIds, maxSeqLength * input.requestPositions.size());
}

void BatchProcessor::processBatchOutput(BatchOutput& output) {
    // 处理批处理输出
    // 简化实现，仅用于演示
    if (output.requestPositions.size() == 1) {
        // 单个请求，无需特殊处理
        return;
    }
    
    // 计算每个请求的输出长度
    std::vector<size_t> originalLengths;
    for (const auto& pos : output.requestPositions) {
        originalLengths.push_back(pos.second - pos.first);
    }
    
    // 对输出进行去填充
    _unpadBatch(output, originalLengths);
}

void BatchProcessor::_padBatch(std::vector<int>& inputIds, size_t targetLength) {
    // 对批处理进行填充
    size_t currentSize = inputIds.size();
    if (currentSize >= targetLength) {
        return;
    }
    
    // 使用0作为填充token
    inputIds.resize(targetLength, 0);
}

void BatchProcessor::_unpadBatch(BatchOutput& output, const std::vector<size_t>& originalLengths) {
    // 对批处理输出进行去填充
    // 简化实现，仅用于演示
    if (output.requestPositions.size() != originalLengths.size()) {
        throw std::invalid_argument("Batch size mismatch in unpadBatch");
    }
    
    // 这里可以根据需要对输出进行去填充处理
    // 例如，调整logits的大小以匹配原始长度
}

}  // namespace cllm