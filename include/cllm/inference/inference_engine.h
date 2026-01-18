/**
 * @file inference_engine.h
 * @brief 推理引擎统一接口层
 * 
 * 参考文档：推理引擎接口设计.md
 * 
 * InferenceEngine 是统一的推理接口，支持多后端切换：
 * - LibTorch Backend：基于 PyTorch C++ API，快速原型和 GPU 推理
 * - Kylin Backend (麒麟)：自研高性能引擎，极致 CPU 性能
 */
#pragma once

#include "cllm/inference/backend_interface.h"
#include "cllm/inference/kv_cache_manager.h"
#include "cllm/kylin/tensor.h"
#include "cllm/model/config.h"

#include <memory>
#include <string>
#include <vector>

namespace cllm {
namespace inference {

/**
 * @brief 推理引擎统一接口
 * 
 * 提供：
 * - 统一的前向推理接口
 * - 多后端支持（LibTorch、Kylin）
 * - 批处理和流式生成
 * - 配置和生命周期管理
 * 
 * 后端选择：
 * - useLibTorch=true：使用 LibTorch 后端（快速原型、GPU 推理）
 * - useLibTorch=false：使用 Kylin 后端（极致 CPU 性能）
 * - backendType="llama_cpp"：使用 llama.cpp 后端（GGUF 格式，原生优化）
 */
class InferenceEngine {
public:
    /**
     * @brief 构造函数（使用布尔标志选择后端，向后兼容）
     * 
     * @param config 模型配置
     * @param modelPath 模型路径
     *                  - .pt 文件：用于 LibTorch 后端
     *                  - .bin 文件：用于 Kylin 后端
     *                  - .gguf 文件：自动使用 llama.cpp 后端（如果可用）
     *                  - 空字符串：使用占位权重（仅 Kylin，测试模式）
     * @param useLibTorch 是否使用 LibTorch 后端（默认 false，使用 Kylin）
     */
    explicit InferenceEngine(
        const ModelConfig &config,
        const std::string &modelPath = std::string(),
        bool useLibTorch = false
    );

    /**
     * @brief 构造函数（使用字符串指定后端类型）
     * 
     * @param config 模型配置
     * @param modelPath 模型路径
     * @param backendType 后端类型（"kylin", "libtorch", "llama_cpp"）
     */
    explicit InferenceEngine(
        const ModelConfig &config,
        const std::string &modelPath,
        const std::string &backendType
    );

    /**
     * @brief 初始化推理引擎
     * 
     * 根据 useLibTorch 参数选择后端并初始化：
     * - LibTorch: 加载 TorchScript 模型
     * - Kylin: 加载自研引擎权重或使用占位权重
     * 
     * @return true 成功，false 失败
     */
    bool initialize();

    /**
     * @brief 单序列前向推理
     * 
     * 输入 token id 序列，输出 logits 张量
     * 
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     */
    kylin::Tensor forward(const std::vector<int> &inputIds) const;

    /**
     * @brief 批处理前向推理
     * 
     * 与 BatchInput 语义对齐，支持多请求并行处理
     * 
     * @param flatInputIds 展平后的所有 token id（等同于 BatchInput::inputIds）
     * @param requestPositions 每个请求在 flatInputIds 中的起止位置 [start, end)
     * @param batchSize 批大小（请求数）
     * @param sequenceIds 每个请求的序列ID（requestId），用于序列ID管理（可选，默认空向量）
     * @return 形状为 [total_tokens, vocab_size] 的 logits Tensor
     */
    kylin::Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize,
        const std::vector<size_t> &sequenceIds = {}
    ) const;

    /**
     * @brief 获取模型配置
     */
    const ModelConfig &getConfig() const;

    /**
     * @brief 获取后端类型
     * 
     * @return 后端名称字符串（"LibTorch", "Kylin", 或 "llama.cpp"）
     */
    std::string getBackendType() const;

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const;
    
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
    bool updateKVCacheRequestStatus(size_t requestId, RequestStatus status) const;

    /**
     * @brief 执行KV缓存LRU淘汰（Phase 5）
     * @param evictionThreshold 淘汰阈值
     * @return 淘汰的请求数量
     */
    size_t evictKVCachesIfNeeded(double evictionThreshold) const;

private:
    /// 模型配置
    ModelConfig config_;

    /// 后端选择标志
    bool useLibTorch_;

    /// 后端实例（LibTorch 或 Kylin）
    std::unique_ptr<IBackend> backend_;
};

} // namespace inference
} // namespace cllm
