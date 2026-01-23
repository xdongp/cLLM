/**
 * @file ggml_transformer.h
 * @brief 基于 GGML 的 Transformer 模型
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 使用 GGML 原生张量进行量化推理，支持：
 * - 直接加载 GGUF 量化模型
 * - GGML 原生算子计算（保持量化格式）
 * - KV Cache 管理
 */
#pragma once

#include "cllm/kylin/gguf/context.h"
#include "cllm/kylin/gguf/loader.h"
#include "cllm/kylin/core/tensor_stats.h"
#include "cllm/kylin/ops/kv_cache_ops.h"
#include "cllm/kylin/ops/attention_graph.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief 基于 GGML 的 Transformer 模型
 * 
 * 特点：
 * - 使用 GGML 原生张量（支持量化格式）
 * - 直接在量化权重上计算（无需反量化）
 * - 支持 GQA（Grouped Query Attention）
 * - 内置 KV Cache 管理
 * - 支持 CPU/Metal/CUDA 后端
 */
class GGMLTransformerModel {
public:
    /**
     * @brief 构造函数
     * @param backend 计算后端类型（CPU/Metal/CUDA）
     */
    explicit GGMLTransformerModel(BackendType backend = BackendType::CPU);
    
    /**
     * @brief 析构函数
     */
    ~GGMLTransformerModel();
    
    // 禁止拷贝
    GGMLTransformerModel(const GGMLTransformerModel&) = delete;
    GGMLTransformerModel& operator=(const GGMLTransformerModel&) = delete;
    
    // ========== 模型加载 ==========
    
    /**
     * @brief 从 GGUF 文件加载模型
     * @param path GGUF 文件路径
     * @return true 成功，false 失败
     */
    bool loadFromGGUF(const std::string& path);
    
    /**
     * @brief 检查模型是否已加载
     */
    bool isLoaded() const { return loaded_; }
    
    /**
     * @brief 获取模型配置
     */
    const GGUFModelConfig& getConfig() const { return config_; }
    
    // ========== 推理接口 ==========
    
    /**
     * @brief 前向传播（全序列）
     * @param inputIds 输入 token ID 序列
     * @return logits [seq_len, vocab_size]（FP32 格式）
     */
    std::vector<float> forward(const std::vector<int32_t>& inputIds);
    
    /**
     * @brief 单 token 前向传播（增量推理）
     * @param tokenId 当前 token ID
     * @param position 当前位置
     * @return logits [vocab_size]（FP32 格式）
     */
    std::vector<float> forwardOneToken(int32_t tokenId, size_t position);
    
    // ========== KV Cache 管理 ==========
    
    /**
     * @brief 清空 KV Cache
     */
    void clearKVCache();
    
    /**
     * @brief 获取当前 KV Cache 长度
     */
    size_t getKVCacheLength() const { return kvCacheLen_; }
    
    /**
     * @brief 获取最大序列长度
     */
    size_t getMaxSeqLen() const { return config_.contextLength; }
    
    // ========== 调试接口 ==========
    
    /**
     * @brief 获取 Layer 0 调试节点统计信息
     */
    std::map<std::string, TensorStats> getLayer0DebugStats() const;
    
    /**
     * @brief 获取所有层的输出统计信息
     */
    std::vector<TensorStats> getAllLayerStats() const;
    
    /**
     * @brief 获取最终归一化输出统计信息
     */
    TensorStats getFinalNormStats() const;
    
    /**
     * @brief 获取 logits 输出统计信息
     */
    TensorStats getLogitsStats() const;
    
    // ========== KV Cache 验证接口 (Stage 8) ==========
    
    using KVCacheStats = cllm::kylin::KVCacheStats;
    
    /**
     * @brief 获取指定层的 KV Cache 统计信息
     */
    KVCacheStats getKVCacheStats(size_t layerIdx) const;
    
    /**
     * @brief 获取所有层的 KV Cache 统计信息
     */
    std::vector<KVCacheStats> getAllKVCacheStats() const;
    
    /**
     * @brief 验证 KV Cache 数据完整性
     */
    bool validateKVCacheIntegrity(size_t expectedLen) const;
    
    /**
     * @brief 获取指定层、指定位置的 KV 数据
     */
    bool getKVAtPosition(size_t layerIdx, size_t position, 
                         std::vector<float>& kData, std::vector<float>& vData) const;

private:
    // ========== 模型状态 ==========
    bool loaded_;
    GGUFModelConfig config_;
    std::unique_ptr<GGUFLoader> loader_;
    BackendType backendType_;
    
    // ========== GGML 上下文 ==========
    std::unique_ptr<GGMLContext> weightCtx_;
    std::unique_ptr<GGMLContext> computeCtx_;
    std::unique_ptr<GGMLContext> kvCacheCtx_;
    
    // ========== 权重张量 ==========
    ggml_tensor* tokEmbed_;
    ggml_tensor* outputNorm_;
    ggml_tensor* output_;
    
    struct LayerWeights {
        ggml_tensor* attnNorm;
        ggml_tensor* ffnNorm;
        ggml_tensor* wq;
        ggml_tensor* wk;
        ggml_tensor* wv;
        ggml_tensor* wo;
        ggml_tensor* attnQNorm;
        ggml_tensor* attnKNorm;
        ggml_tensor* wGate;
        ggml_tensor* wUp;
        ggml_tensor* wDown;
    };
    std::vector<LayerWeights> layers_;
    
    // ========== KV Cache ==========
    std::vector<ggml_tensor*> kCaches_;
    std::vector<ggml_tensor*> vCaches_;
    size_t kvCacheLen_;
    size_t maxKVCacheLen_;
    
    std::vector<ggml_tensor*> pendingK_;
    std::vector<ggml_tensor*> pendingV_;
    size_t pendingStartPos_;
    size_t pendingSeqLen_;
    
    // ========== 调试节点 ==========
    ggml_tensor* debugEmbedding_;
    ggml_tensor* debugLayer0Output_;
    std::vector<ggml_tensor*> debugLayerOutputs_;
    ggml_tensor* debugFinalNorm_;
    ggml_tensor* debugLogits_;
    
    struct Layer0DebugNodes {
        ggml_tensor* attnNormOutput = nullptr;
        ggml_tensor* qkvOutput = nullptr;
        ggml_tensor* qNormOutput = nullptr;
        ggml_tensor* kBeforeNorm = nullptr;      // RMS Norm 前的 K
        ggml_tensor* kAfterRmsNorm = nullptr;    // RMS Norm 后、乘法前的 K
        ggml_tensor* kNormOutput = nullptr;
        ggml_tensor* ropeQOutput = nullptr;
        ggml_tensor* ropeKOutput = nullptr;
        ggml_tensor* attentionOutput = nullptr;
        ggml_tensor* ffnNormOutput = nullptr;
        ggml_tensor* ffnGateOutput = nullptr;
        ggml_tensor* ffnUpOutput = nullptr;
        ggml_tensor* ffnHiddenOutput = nullptr;
        ggml_tensor* ffnOutput = nullptr;
    };
    Layer0DebugNodes debugLayer0Nodes_;
    
    // Layer 27 调试节点（用于定位最后一层的问题）
    ggml_tensor* debugLayer27AttnInput_ = nullptr;
    ggml_tensor* debugLayer27AttnOutput_ = nullptr;
    ggml_tensor* debugLayer27FfnInput_ = nullptr;
    ggml_tensor* debugLayer27FfnOutput_ = nullptr;
    
    // ========== 内部方法 ==========
    void mapWeights();
    void allocateKVCache();
    void flushKVCache();
    bool verifyKVCacheIntegrity(size_t layerIdx, size_t expectedLen) const;
    ggml_tensor* findWeight(const std::string& name);
    
    ggml_tensor* buildForwardGraph(const std::vector<int32_t>& inputIds, size_t startPos);
    ggml_tensor* buildLayerGraph(ggml_context* ctx, ggml_tensor* input, 
                                  const LayerWeights& layer, size_t layerIdx,
                                  size_t startPos, size_t seqLen);
    ggml_tensor* buildFFNGraph(ggml_context* ctx, ggml_tensor* input, const LayerWeights& layer);
};

} // namespace kylin
} // namespace cllm
