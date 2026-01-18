/**
 * @file llama_cpp_backend.h
 * @brief llama.cpp 推理后端实现（使用 llama.cpp C API）
 * 
 * 参考文档：llama.cpp后端集成设计.md
 */
#pragma once

#include "cllm/inference/backend_interface.h"
#include "cllm/kylin/tensor.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/gguf_tokenizer.h"

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

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
     * @return [total_tokens, vocab_size] logits 张量
     */
    Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize
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

    std::string modelPath_;              ///< GGUF 模型路径
    ModelConfig config_;                 ///< 模型配置
    struct llama_model* model_;         ///< llama.cpp 模型句柄
    struct llama_context* ctx_;          ///< llama.cpp 上下文句柄
    std::unique_ptr<GGUFTokenizer> tokenizer_; ///< GGUF tokenizer（用于编码/解码）
    bool initialized_;                   ///< 是否已初始化
    
    // llama.cpp 参数结构（在 initialize 时创建）
    // 使用指针避免在头文件中需要完整类型定义
    struct llama_model_params* modelParams_;
    struct llama_context_params* contextParams_;
    
    int numThreads_;                     ///< CPU 线程数
    int nGpuLayers_;                     ///< GPU 层数
    size_t currentPosition_;              ///< 当前位置（用于单序列 forward，已弃用，保留用于兼容）
    // 使用 int32_t 而不是 llama_seq_id，因为 llama_seq_id 在头文件中不可见（只在 .cpp 中包含 llama.h）
    mutable std::mutex seqPositionsMutex_;  ///< 保护 seqPositions_ 的互斥锁
    std::unordered_map<int32_t, size_t> seqPositions_;  ///< 每个序列的位置映射（seq_id -> position）
    std::unordered_map<int32_t, size_t> seqLengths_;   ///< 每个序列的上次长度（seq_id -> length），用于检测新请求
    
    // ========== 位置管理方法（统一 forward() 和 forwardBatch() 的逻辑）==========
    
    /**
     * @brief 获取序列的当前位置（线程安全）
     * @param seqId 序列 ID
     * @return 当前位置，如果不存在则返回 0
     */
    size_t getSeqPosition(int32_t seqId) const;
    
    /**
     * @brief 更新序列的位置（线程安全）
     * @param seqId 序列 ID
     * @param position 新位置
     */
    void updateSeqPosition(int32_t seqId, size_t position);
    
    /**
     * @brief 重置序列的位置（线程安全）
     * @param seqId 序列 ID
     */
    void resetSeqPosition(int32_t seqId);
    
    /**
     * @brief 检查序列是否已有位置记录（线程安全）
     * @param seqId 序列 ID
     * @return true 如果已有记录，false 否则
     */
    bool hasSeqPosition(int32_t seqId) const;
    
    /**
     * @brief 清空 llama.cpp KV cache 中指定序列的数据
     * @param seqId 序列 ID
     */
    void clearKVCacheForSequence(int32_t seqId);
};

} // namespace inference
} // namespace cllm
