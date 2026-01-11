/**
 * @file batch_processor.h
 * @brief 批处理处理器，负责批处理输入的准备和输出的处理
 * @author cLLM Team
 * @date 2026-01-08
 */
#ifndef CLLM_MODEL_BATCH_PROCESSOR_H
#define CLLM_MODEL_BATCH_PROCESSOR_H

#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include <vector>

namespace cllm {
    // 前向声明，解决循环引用问题
    class ModelExecutor;

/**
 * @brief 批处理处理器类，负责批处理输入的准备和输出的处理
 */
class BatchProcessor {
public:
    /**
     * @brief 构造函数
     * @param executor 模型执行器指针
     */
    explicit BatchProcessor(ModelExecutor* executor);
    
    /**
     * @brief 析构函数
     */
    ~BatchProcessor();
    
    /**
     * @brief 处理批处理输入
     * @param input 批处理输入
     * @return 批处理输出
     */
    BatchOutput processBatch(const BatchInput& input);
    
    /**
     * @brief 准备批处理输入
     * @param input 批处理输入
     */
    void prepareBatchInput(const BatchInput& input);
    
    /**
     * @brief 处理批处理输出
     * @param output 批处理输出
     */
    void processBatchOutput(BatchOutput& output);
    
private:
    /**
     * @brief 对批处理进行填充
     * @param inputIds 输入Token ID列表
     * @param targetLength 目标长度
     */
    void _padBatch(std::vector<int>& inputIds, size_t targetLength);
    
    /**
     * @brief 对批处理输出进行去填充
     * @param output 批处理输出
     * @param originalLengths 原始长度列表
     */
    void _unpadBatch(BatchOutput& output, const std::vector<size_t>& originalLengths);
    
    ModelExecutor* executor_;  // 模型执行器指针
};

}  // namespace cllm

#endif  // CLLM_MODEL_BATCH_PROCESSOR_H