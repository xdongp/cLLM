/**
 * @file attention_graph.h
 * @brief GGML Attention 计算图构建器
 * 
 * 负责构建 Self-Attention 的 GGML 计算图，包括：
 * - QKV 投影
 * - Q/K 归一化（可选）
 * - RoPE 位置编码
 * - GQA（Grouped Query Attention）
 * - 注意力计算和输出投影
 */
#pragma once

#include <ggml.h>
#include <cstddef>
#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief Attention 层权重
 */
struct AttentionWeights {
    ggml_tensor* wq = nullptr;         ///< Q 投影 [hidden, n_heads * head_dim]
    ggml_tensor* wk = nullptr;         ///< K 投影 [hidden, n_kv_heads * head_dim]
    ggml_tensor* wv = nullptr;         ///< V 投影 [hidden, n_kv_heads * head_dim]
    ggml_tensor* wo = nullptr;         ///< O 投影 [n_heads * head_dim, hidden]
    ggml_tensor* attnQNorm = nullptr;  ///< Q 归一化 [head_dim] (可选)
    ggml_tensor* attnKNorm = nullptr;  ///< K 归一化 [head_dim] (可选)
};

/**
 * @brief Attention 配置参数
 */
struct AttentionConfig {
    size_t nHeads = 0;          ///< Query heads 数量
    size_t nKVHeads = 0;        ///< KV heads 数量
    size_t headDim = 0;         ///< Head 维度
    size_t contextLength = 0;   ///< 最大上下文长度
    float rmsNormEps = 1e-6f;   ///< RMS Norm epsilon
    float ropeFreqBase = 10000.0f;  ///< RoPE 频率基数
    int ropeType = 0;           ///< RoPE 类型 (0=NORMAL, 2=NEOX)
};

/**
 * @brief KV Cache 引用（用于图构建）
 */
struct KVCacheRef {
    ggml_tensor* kCache = nullptr;  ///< K cache 张量
    ggml_tensor* vCache = nullptr;  ///< V cache 张量
    size_t cacheLen = 0;            ///< 当前 cache 长度
    size_t maxLen = 0;              ///< 最大 cache 长度
};

/**
 * @brief Attention 计算结果
 */
struct AttentionResult {
    ggml_tensor* output = nullptr;   ///< 输出张量 [hidden, seq_len]
    ggml_tensor* kNew = nullptr;     ///< 新的 K 张量（用于写入 cache）
    ggml_tensor* vNew = nullptr;     ///< 新的 V 张量（用于写入 cache）
};

/**
 * @brief Attention 调试节点
 */
struct AttentionDebugNodes {
    ggml_tensor* qkvOutput = nullptr;      ///< QKV 投影输出（Q）
    ggml_tensor* qNormOutput = nullptr;    ///< Q 归一化后
    ggml_tensor* kNormOutput = nullptr;    ///< K 归一化后
    ggml_tensor* ropeQOutput = nullptr;    ///< RoPE 后的 Q
    ggml_tensor* ropeKOutput = nullptr;    ///< RoPE 后的 K
    ggml_tensor* kBeforeNorm = nullptr;    ///< RMS Norm 前的 K（调试用）
    ggml_tensor* kAfterRmsNorm = nullptr;  ///< RMS Norm 后、乘法前的 K（调试用）
};

/**
 * @brief Attention 图构建器
 * 
 * 负责构建 Self-Attention 的 GGML 计算图
 */
class AttentionGraphBuilder {
public:
    /**
     * @brief 构造函数
     * @param config Attention 配置
     */
    explicit AttentionGraphBuilder(const AttentionConfig& config);
    
    /**
     * @brief 构建 Attention 计算图
     * @param ctx GGML 计算上下文
     * @param input 输入张量 [hidden, seq_len]
     * @param weights 层权重
     * @param kvCache KV Cache 引用
     * @param startPos 起始位置（用于增量推理）
     * @param seqLen 当前序列长度
     * @param layerIdx 层索引（用于调试）
     * @return Attention 计算结果
     */
    AttentionResult build(
        ggml_context* ctx,
        ggml_tensor* input,
        const AttentionWeights& weights,
        const KVCacheRef& kvCache,
        size_t startPos,
        size_t seqLen,
        size_t layerIdx = 0
    );
    
    /**
     * @brief 获取调试节点
     */
    const AttentionDebugNodes& getDebugNodes() const { return debugNodes_; }
    
private:
    AttentionConfig config_;
    AttentionDebugNodes debugNodes_;
    
    /**
     * @brief 构建 QKV 投影
     */
    void buildQKVProjection(
        ggml_context* ctx,
        ggml_tensor* input,
        const AttentionWeights& weights,
        ggml_tensor*& q,
        ggml_tensor*& kNew,
        ggml_tensor*& vNew,
        size_t seqLen
    );
    
    /**
     * @brief 应用 Q/K 归一化
     */
    void applyQKNorm(
        ggml_context* ctx,
        ggml_tensor*& q,
        ggml_tensor*& kNew,
        const AttentionWeights& weights,
        size_t layerIdx
    );
    
    /**
     * @brief 应用 RoPE 位置编码
     */
    void applyRoPE(
        ggml_context* ctx,
        ggml_tensor*& q,
        ggml_tensor*& kNew,
        size_t startPos,
        size_t seqLen,
        size_t layerIdx
    );
    
    /**
     * @brief 构建完整的 KV 序列（从 cache + 新数据）
     */
    void buildFullKV(
        ggml_context* ctx,
        ggml_tensor* kNew,
        ggml_tensor* vNew,
        const KVCacheRef& kvCache,
        size_t startPos,
        size_t seqLen,
        ggml_tensor*& kFull,
        ggml_tensor*& vFull,
        size_t layerIdx
    );
    
    /**
     * @brief 计算注意力输出
     */
    ggml_tensor* computeAttention(
        ggml_context* ctx,
        ggml_tensor* q,
        ggml_tensor* kFull,
        ggml_tensor* vFull,
        const AttentionWeights& weights,
        size_t startPos,
        size_t seqLen,
        size_t totalLen,
        size_t layerIdx
    );
};

} // namespace kylin
} // namespace cllm
