/**
 * @file rope.cpp
 * @brief Rotary Position Embedding (RoPE) 的简化实现
 */

#include "cllm/kylin/rope.h"

#include <cmath>
#include <stdexcept>

namespace cllm {
namespace kylin {

RoPE::RoPE(size_t dimPerHead, size_t maxSeqLen, float theta,
           size_t nCtxOrig, float freqScale, int ropeType, float extFactor)
    : dimPerHead_(dimPerHead)
    , maxSeqLen_(maxSeqLen)
    , theta_(theta)
    , nCtxOrig_(nCtxOrig > 0 ? nCtxOrig : maxSeqLen)  // P3修复：如果未提供，使用maxSeqLen
    , freqScale_(freqScale)
    , ropeType_(ropeType)
    , extFactor_(extFactor)
    , cosCache_(maxSeqLen * (dimPerHead / 2))
    , sinCache_(maxSeqLen * (dimPerHead / 2)) {
    if (dimPerHead_ % 2 != 0) {
        throw std::invalid_argument("RoPE requires even dimPerHead");
    }
    
    // P3修复：计算RoPE缓存时考虑扩展参数
    // 参考llama.cpp的ggml_rope_ext实现
    for (size_t pos = 0; pos < maxSeqLen_; ++pos) {
        for (size_t i = 0; i < dimPerHead_ / 2; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(dimPerHead_);
            float thetaVal = std::pow(theta_, exponent);
            
            // 应用freq_scale和ext_factor
            // 对于扩展RoPE，位置需要根据n_ctx_orig和ext_factor调整
            float adjustedPos = static_cast<float>(pos);
            if (nCtxOrig_ > 0 && extFactor_ != 1.0f) {
                // 扩展RoPE的位置调整（简化实现，完整实现需要参考llama.cpp）
                adjustedPos = adjustedPos * extFactor_;
            }
            
            float freq = (adjustedPos * freqScale_) / thetaVal;
            size_t idx = pos * (dimPerHead_ / 2) + i;
            cosCache_[idx] = std::cos(freq);
            sinCache_[idx] = std::sin(freq);
        }
    }
}

void RoPE::apply(Tensor& q, Tensor& k, size_t seqLen, size_t posOffset) const {
    const auto& qShape = q.shape();
    const auto& kShape = k.shape();

    if (qShape.size() != 4) {
        throw std::invalid_argument("RoPE::apply expects q to be 4D [batch, heads, seq, dim]");
    }
    if (kShape.size() != 4) {
        throw std::invalid_argument("RoPE::apply expects k to be 4D [batch, heads, seq, dim]");
    }

    // 允许 Q/K 的 head 数量不同（GQA），但 batch/seq/dim 必须一致
    if (qShape[0] != kShape[0] || qShape[2] != kShape[2] || qShape[3] != kShape[3]) {
        throw std::invalid_argument("RoPE::apply requires q and k to have same [batch, seq, dim]");
    }

    const size_t batch = qShape[0];
    const size_t qHeads = qShape[1];
    const size_t kHeads = kShape[1];
    const size_t maxSeq = qShape[2];
    const size_t dim = qShape[3];

    if (seqLen > maxSeq) {
        throw std::invalid_argument("RoPE::apply seqLen exceeds tensor sequence dimension");
    }
    if (dim != dimPerHead_) {
        throw std::invalid_argument("RoPE::apply dimPerHead mismatch with tensor last dim");
    }
    if (dim % 2 != 0) {
        throw std::invalid_argument("RoPE::apply requires head_dim to be even");
    }

    auto apply_one = [&](Tensor& t, size_t heads) {
        float* data = t.data();
        if (!data) {
            throw std::runtime_error("RoPE::apply: null tensor data");
        }

        // shape: [batch, heads, seq, dim] row-major
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < heads; ++h) {
                for (size_t pos = 0; pos < seqLen; ++pos) {
                    const size_t position = posOffset + pos;
                    if (position >= maxSeqLen_) {
                        throw std::invalid_argument("RoPE::apply position exceeds maxSeqLen");
                    }

                    const size_t cacheBase = position * (dim / 2);
                    const size_t baseIndex = ((b * heads + h) * maxSeq + pos) * dim;

                    for (size_t i = 0; i < dim / 2; ++i) {
                        const float cosVal = cosCache_[cacheBase + i];
                        const float sinVal = sinCache_[cacheBase + i];

                        const size_t d0 = 2 * i;
                        const size_t d1 = 2 * i + 1;

                        const size_t idx0 = baseIndex + d0;
                        const size_t idx1 = baseIndex + d1;

                        const float x0 = data[idx0];
                        const float x1 = data[idx1];

                        data[idx0] = x0 * cosVal - x1 * sinVal;
                        data[idx1] = x0 * sinVal + x1 * cosVal;
                    }
                }
            }
        }
    };

    apply_one(q, qHeads);
    apply_one(k, kHeads);
}

}  // namespace kylin
}  // namespace cllm
