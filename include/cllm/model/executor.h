/**
 * @file executor.h
 * @brief 模型执行器模块，负责模型加载、前向传播和生成
 * @author cLLM Team
 * @date 2026-01-08
 */
#ifndef CLLM_MODEL_EXECUTOR_H
#define CLLM_MODEL_EXECUTOR_H

#include "cllm/model/config.h"
#include "cllm/model/stats.h"
#include "cllm/model/exceptions.h"
#include "cllm/model/quantization_manager.h"
#include "cllm/model/inference_optimizer.h"
#include "cllm/model/batch_processor.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/memory/float_array.h"
#include "cllm/inference/inference_engine.h"
#include "cllm/inference/kv_cache_manager.h"
#include <string>
#include <vector>
#include <mutex>
#include <memory>

namespace cllm {

class Sampler;
class KVCache;

/**
 * @brief 模型执行器类，负责模型加载、前向传播和生成
 */
class ModelExecutor {
public:
    /**
     * @brief 构造函数
     * @param modelPath 模型文件路径
     * @param quantization 量化类型（支持int8, int4）
     * @param enableSIMD 是否启用SIMD优化
     * @param useLibTorch 是否使用 LibTorch 后端（默认 false，使用自研引擎）
     */
    ModelExecutor(
        const std::string& modelPath,
        const std::string& quantization = "",
        bool enableSIMD = true,
        bool useLibTorch = false,
        const std::string& backendType = "",
        const ModelConfig* initialConfig = nullptr
    );
    
    /**
     * @brief 析构函数
     */
    ~ModelExecutor();
    
    ModelExecutor(const ModelExecutor&) = delete;
    ModelExecutor& operator=(const ModelExecutor&) = delete;
    
    /**
     * @brief 前向传播
     * @param input 批处理输入
     * @return 批处理输出
     */
    BatchOutput forward(const BatchInput& input);
    
    /**
     * @brief 生成文本
     * @param inputIds 输入Token ID列表
     * @param maxNewTokens 最大生成Token数
     * @param temperature 温度参数
     * @return 生成的Token ID列表
     */
    std::vector<int> generate(
        const std::vector<int>& inputIds,
        size_t maxNewTokens = 100,
        float temperature = 0.7f
    );
    
    /**
     * @brief 采样单个Token
     * @param inputIds 输入Token ID列表
     * @param temperature 温度参数
     * @return 采样的Token ID
     */
    int sampleToken(
        const std::vector<int>& inputIds,
        float temperature = 0.7f
    );
    
    /**
     * @brief 加载模型
     */
    void loadModel();
    
    /**
     * @brief 卸载模型
     */
    void unloadModel();
    
    /**
     * @brief 获取模型统计信息
     * @return 模型统计信息
     */
    ModelStats getStats() const;
    
    /**
     * @brief 重置模型统计信息
     */
    void resetStats();
    
    /**
     * @brief 获取采样器
     * @return 采样器指针
     */
    Sampler* getSampler() const;
    
    /**
     * @brief 获取KV缓存
     * @return KV缓存指针
     */
    KVCache* getKVCache() const;
    
    /**
     * @brief 获取模型配置
     * @return 模型配置
     */
    const ModelConfig& getConfig() const;
    
    /**
     * @brief 设置模型配置（用于 LibTorch 后端动态更新）
     * @param config 新的模型配置
     */
    void setConfig(const ModelConfig& config);
    
    /**
     * @brief 设置 tokenizer 的 vocab_size（不影响 InferenceEngine）
     * @param tokenizerVocabSize tokenizer 的词表大小
     */
    void setTokenizerVocabSize(size_t tokenizerVocabSize);
    
    /**
     * @brief 检查模型是否已加载
     * @return true if model is loaded, false otherwise
     */
    bool isLoaded() const {
        return isModelLoaded_;
    }
    
    /**
     * @brief 释放序列ID（Phase 2: 序列ID管理）
     * 
     * 当请求完成时，释放对应的序列ID，使其可以被新请求重用
     * 只对支持序列ID管理的后端（如 llama.cpp）有效
     * 
     * @param requestId 请求ID
     * @return true 如果成功释放，false 如果后端不支持或请求ID不存在
     */
    bool releaseSequenceId(size_t requestId) const;

    /**
     * @brief 清理KV缓存（Phase 4: KV缓存统计管理）
     * 
     * 当请求完成时，清理对应的KV缓存
     * 只对支持KV缓存管理的后端（如 llama.cpp）有效
     * 
     * @param requestId 请求ID
     * @return true 如果成功清理，false 如果后端不支持或请求ID不存在
     */
    bool cleanupKVCache(size_t requestId) const;

    /**
     * @brief 更新KV缓存请求状态（Phase 5）
     * @param requestId 请求ID
     * @param status 请求状态
     * @return true 如果成功更新，false 否则
     */
    bool updateKVCacheRequestStatus(size_t requestId, inference::RequestStatus status) const;

    /**
     * @brief 执行KV缓存LRU淘汰（Phase 5）
     * @param evictionThreshold 淘汰阈值
     * @return 淘汰的请求数量
     */
    size_t evictKVCachesIfNeeded(double evictionThreshold) const;
    
private:
    void _loadFullPrecisionModel();
    void _loadInt8QuantizedModel();
    void _loadInt4QuantizedModel();
    
    void _applyInferenceOptimizations();
    void _warmupModel();
    
    FloatArray _prepareInput(const std::vector<int>& inputIds);
    void _processOutput(FloatArray& logits, size_t batchSize, size_t vocabSize);
    
    FloatArray _executeModelInference(const BatchInput& input);
    
    void _optimizeMemoryUsage();
    void _enableMemoryCompression();
    
    std::string modelPath_;
    std::string quantization_;
    bool enableSIMD_;
    bool useLibTorch_;  // 是否使用 LibTorch 后端
    std::string backendType_;  // 指定后端类型（空表示自动选择）
    
    void* modelHandle_;
    void* modelWeights_;
    size_t modelSize_;
    
    std::unique_ptr<Sampler> sampler_;
    std::unique_ptr<KVCache> kvCache_;
    std::unique_ptr<QuantizationManager> quantizationManager_;
    std::unique_ptr<InferenceOptimizer> inferenceOptimizer_;
    std::unique_ptr<BatchProcessor> batchProcessor_;
    std::unique_ptr<inference::InferenceEngine> inferenceEngine_;
    
    ModelConfig config_;
    ModelStats stats_;
    
    mutable std::mutex modelMutex_;
    
    bool isModelLoaded_;
    
    FloatArray inferenceBuffer_;
    FloatArray inputBuffer_;
};

}

#endif
