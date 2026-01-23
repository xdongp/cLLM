/**
 * @file attention.h
 * @brief Multi-Head Attention 的简化实现（MVP，无 KV Cache）
 */
#pragma once

#include "cllm/kylin/core/tensor.h"
#include "cllm/kylin/core/kernels.h"
#include "cllm/kylin/ops/rope.h"
#include <memory>

namespace cllm {
namespace kylin {

/**
 * @brief 多头自注意力（不含KV缓存，MVP阶段）
 *
 * 假设输入形状为 [batch, seq_len, hidden_size]。
 * 支持 GQA (Grouped Query Attention)：Q heads 和 KV heads 可以不同。
 */
class MultiHeadAttention {
public:
    MultiHeadAttention(
        size_t hiddenSize,
        size_t numQHeads,
        size_t numKVHeads,
        float ropeTheta = 10000.0f,
        float rmsNormEps = 1e-5f,  // P4修复：从配置读取，而不是硬编码
        // P3修复：RoPE扩展参数
        size_t maxSequenceLength = 2048,
        size_t ropeNctxOrig = 0,
        float ropeFreqScale = 1.0f,
        int ropeType = 0,
        float ropeExtFactor = 1.0f
    );

    /// 设置权重（MVP阶段简单按引用保存，生命周期由上层保证）
    void setWeights(
        const Tensor& wq,
        const Tensor& wk,
        const Tensor& wv,
        const Tensor& wo
    );
    
    /// 设置Q/K的独立归一化权重（Qwen3等模型需要，可选）
    void setAttnQKNormWeights(
        const Tensor& attnQNormWeight,
        const Tensor& attnKNormWeight
    );

    /// 无 KV 的前向传播
    /// 输入: [batch, seq_len, hidden_size]
    /// 输出: [batch, seq_len, hidden_size]
    Tensor forwardNoKV(const Tensor& input) const;

private:
    size_t hiddenSize_;
    size_t numQHeads_;      // Query heads 数量
    size_t numKVHeads_;     // Key-Value heads 数量（GQA支持）
    size_t headDim_;        // 每个头的维度（Q和KV共享相同的head_dim）
    float ropeTheta_;       // RoPE theta 参数（从配置读取）
    float rmsNormEps_;      // RMSNorm epsilon（从配置读取，P4修复）
    // P3修复：RoPE扩展参数
    size_t maxSequenceLength_;  // 最大序列长度（从配置读取）
    size_t ropeNctxOrig_;       // 原始上下文长度
    float ropeFreqScale_;       // 频率缩放因子
    int ropeType_;              // RoPE类型
    float ropeExtFactor_;       // 扩展因子

    Tensor wq_;
    Tensor wk_;
    Tensor wv_;
    Tensor wo_;
    
    Tensor attnQNormWeight_;  // Q的独立归一化权重（可选）
    Tensor attnKNormWeight_;  // K的独立归一化权重（可选）
    bool hasAttnQKNorm_;      // 是否已设置Q/K归一化权重

    mutable std::unique_ptr<RoPE> rope_;  // 延迟初始化，根据实际的qHeadDim创建
};

}  // namespace kylin
}  // namespace cllm
