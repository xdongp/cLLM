/**
 * @file llama_cpp_backend.h
 * @brief llama.cpp 推理后端实现（使用 llama.cpp C API）
 * 
 * 参考文档：llama.cpp后端集成设计.md
 */
#pragma once

#include "cllm/inference/backend_interface.h"
#include "cllm/inference/kv_cache_manager.h"
#include "cllm/kylin/tensor.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/gguf_tokenizer.h"

#include <memory>
#include <string>
#include <vector>

// Forward declarations for llama.cpp types
// 注意：实际类型定义在 llama.h 中
// 在实现文件中包含 llama.h
struct llama_model;
struct llama_context;
struct llama_model_params;
struct llama_context_params;

namespace cllm {
namespace inference {

/**
 * @brief llama.cpp 推理后端
 *
 * 使用 llama.cpp C API 进行推理：
 * - 加载 GGUF 格式模型
 * - 支持多种量化格式（Q4_K_M, Q8_0, F16, F32 等）
 * - 利用 llama.cpp 的优化实现
 * - 支持 GPU 加速（Metal/CUDA，如果编译时启用）
 * 
 * 实现 IBackend 接口，可与 Kylin、LibTorch 后端无缝切换
 */
class LlamaCppBackend : public IBackend {
public:
    /**
     * @brief 构造函数
     * @param config 模型配置
     * @param modelPath GGUF 模型路径（.gguf 文件）
     */
    explicit LlamaCppBackend(const ModelConfig &config, const std::string &modelPath);

    /**
     * @brief 析构函数
     */
    ~LlamaCppBackend() override;

    // ========== IBackend 接口实现 ==========

    /**
     * @brief 初始化加载模型
     * @return true 成功，false 失败
     */
    bool initialize() override;

    /**
     * @brief 单序列前向推理
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     */
    Tensor forward(const std::vector<int> &inputIds) override;

    /**
     * @brief 批处理前向推理
     * @param flatInputIds 展平后的所有 token id
     * @param requestPositions 每个请求的起止位置
     * @param batchSize 批大小
     * @param sequenceIds 每个请求的序列ID（requestId），用于序列ID管理
     * @return [total_tokens, vocab_size] logits 张量
     */
    Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize,
        const std::vector<size_t> &sequenceIds = {}
    ) override;

    /**
     * @brief 获取后端名称
     */
    std::string getName() const override { return "llama.cpp"; }

    /**
     * @brief 获取模型是否已加载
     */
    bool isInitialized() const override { return initialized_; }

    /**
     * @brief 获取模型配置
     */
    const ModelConfig &getConfig() const override { return config_; }

    /**
     * @brief 设置线程数（CPU 推理）
     * 
     * @param numThreads 线程数，0 表示使用默认值
     */
    void setNumThreads(int numThreads);

    /**
     * @brief 设置 GPU 层数（GPU 加速）
     * 
     * @param nGpuLayers GPU 层数，0 表示仅使用 CPU
     */
    void setNGpuLayers(int nGpuLayers);

    /**
     * @brief 为请求分配序列ID
     * @param requestId 请求ID
     * @return 分配的序列ID，如果失败返回 -1
     */
    int32_t allocateSequenceId(size_t requestId);

    /**
     * @brief 释放请求的序列ID
     * @param requestId 请求ID
     * @return true 如果成功释放，false 否则
     */
    bool releaseSequenceId(size_t requestId);

    /**
     * @brief 获取请求对应的序列ID
     * @param requestId 请求ID
     * @return 序列ID，如果未分配返回 -1
     */
    int32_t getSequenceId(size_t requestId) const;

    /**
     * @brief 清理请求的KV缓存（Phase 4）
     * @param requestId 请求ID
     * @return true 如果成功清理，false 否则
     * 
     * 协调 KV 缓存管理器清理 llama.cpp 中的 KV 缓存。
     */
    bool cleanupKVCache(size_t requestId);

    /**
     * @brief 获取KV缓存管理器（Phase 4）
     * @return KV缓存管理器的指针
     */
    KVCacheManager* getKVCacheManager() { return kvCacheManager_.get(); }

    /**
     * @brief 获取 llama.cpp 上下文句柄（Phase 5）
     * @return llama.cpp 上下文句柄
     */
    ::llama_context* getContext() { return ctx_; }

    /**
     * @brief 监控序列ID池使用情况
     * @return 序列ID池使用率（0.0-1.0）
     * 
     * 用于监控序列ID池的使用情况，当使用率超过阈值时记录警告。
     */
    double getSequenceIdPoolUsage() const;

    /**
     * @brief 获取可用序列ID数量
     * @return 可用序列ID数量
     */
    size_t getAvailableSequenceIdCount() const;

private:
    /**
     * @brief 从 ModelConfig 创建 llama_model_params
     */
    void createModelParams();

    /**
     * @brief 从 ModelConfig 创建 llama_context_params
     */
    void createContextParams();

    /**
     * @brief 校验 vocab_size 一致性
     * @return true 一致，false 不一致
     */
    bool validateVocabSize();

    /**
     * @brief 将 std::vector<int> 转换为 llama_token 数组
     */
    std::vector<int32_t> convertToLlamaTokens(const std::vector<int> &inputIds);

    /**
     * @brief 初始化序列ID池
     */
    void initializeSequenceIdPool();

    std::string modelPath_;              ///< GGUF 模型路径
    ModelConfig config_;                 ///< 模型配置
    ::llama_model* model_;         ///< llama.cpp 模型句柄
    ::llama_context* ctx_;          ///< llama.cpp 上下文句柄
    std::unique_ptr<GGUFTokenizer> tokenizer_; ///< GGUF tokenizer（用于编码/解码）
    bool initialized_;                   ///< 是否已初始化
    
    // llama.cpp 参数结构（在 initialize 时创建）
    // 使用指针避免在头文件中需要完整类型定义
    ::llama_model_params* modelParams_;
    ::llama_context_params* contextParams_;
    
    int numThreads_;                     ///< CPU 线程数
    int nGpuLayers_;                     ///< GPU 层数
    size_t currentPosition_;              ///< 当前推理位置（用于增量推理）

    // Phase 2: 序列ID管理
    std::map<size_t, int32_t> requestIdToSeqId_;  ///< requestId 到 seqId 的映射
    std::vector<int32_t> availableSeqIds_;         ///< 可用序列ID池
    mutable std::mutex sequenceIdMutex_;             ///< 序列ID管理互斥锁（mutable，允许 const 方法使用）
    int32_t nSeqMax_;                              ///< 最大序列数（从配置读取）

    // Phase 4: KV缓存统计管理
    std::unique_ptr<KVCacheManager> kvCacheManager_;  ///< KV缓存统计管理器
};

} // namespace inference
} // namespace cllm
