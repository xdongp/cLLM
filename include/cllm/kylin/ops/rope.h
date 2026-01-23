/**
 * @file rope.h
 * @brief Rotary Position Embedding (RoPE) 的简化实现
 */
#pragma once

#include <vector>
#include "cllm/kylin/core/tensor.h"

namespace cllm {
namespace kylin {

/**
 * @brief RoPE 实现（MVP阶段）
 *
 * 假设输入 Q/K 的形状为 [batch, num_heads, seq_len, head_dim]（row-major），
 * 仅支持偶数维 head_dim。
 */
class RoPE {
public:
    // P3修复：支持RoPE扩展参数（对齐llama.cpp的ggml_rope_ext）
    RoPE(size_t dimPerHead, size_t maxSeqLen, float theta = 10000.0f,
         size_t nCtxOrig = 0, float freqScale = 1.0f, 
         int ropeType = 0, float extFactor = 1.0f);
    // 向后兼容的构造函数
    RoPE(size_t dimPerHead, float theta = 10000.0f)
        : RoPE(dimPerHead, 2048, theta, 0, 1.0f, 0, 1.0f) {}

    /**
     * @brief 对 Q/K 应用旋转位置编码
     * @param q Query 张量，[batch, num_heads, seq_len, head_dim]
     * @param k Key 张量，[batch, num_heads, seq_len, head_dim]
     * @param seqLen 当前序列长度
     * @param posOffset 位置偏移（用于增量解码），MVP阶段可先传0
     */
    void apply(Tensor& q, Tensor& k, size_t seqLen, size_t posOffset = 0) const;

    /**
     * @brief 获取每个头的维度
     */
    size_t getDimPerHead() const { return dimPerHead_; }

private:
    size_t dimPerHead_;
    size_t maxSeqLen_;
    float theta_;
    // P3修复：RoPE扩展参数
    size_t nCtxOrig_;      // 原始上下文长度
    float freqScale_;      // 频率缩放因子
    int ropeType_;         // RoPE类型（0=标准RoPE）
    float extFactor_;      // 扩展因子
    std::vector<float> cosCache_;
    std::vector<float> sinCache_;
};

}  // namespace kylin
}  // namespace cllm
