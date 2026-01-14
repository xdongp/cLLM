/**
 * @file rope.h
 * @brief Rotary Position Embedding (RoPE) 的简化实现
 */
#pragma once

#include <vector>
#include "cllm/kylin/tensor.h"

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
    RoPE(size_t dimPerHead, size_t maxSeqLen, float theta = 10000.0f);
    RoPE(size_t dimPerHead, float theta = 10000.0f)
        : RoPE(dimPerHead, 2048, theta) {}

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
    std::vector<float> cosCache_;
    std::vector<float> sinCache_;
};

}  // namespace kylin
}  // namespace cllm
