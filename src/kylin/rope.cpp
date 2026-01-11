/**
 * @file rope.cpp
 * @brief Rotary Position Embedding (RoPE) 的简化实现
 */

#include "cllm/kylin/rope.h"

#include <cmath>
#include <stdexcept>

namespace cllm {
namespace kylin {

RoPE::RoPE(size_t dimPerHead, size_t maxSeqLen, float theta)
    : dimPerHead_(dimPerHead)
    , maxSeqLen_(maxSeqLen)
    , theta_(theta)
    , cosCache_(maxSeqLen * (dimPerHead / 2))
    , sinCache_(maxSeqLen * (dimPerHead / 2)) {
    if (dimPerHead_ % 2 != 0) {
        throw std::invalid_argument("RoPE requires even dimPerHead");
    }
    for (size_t pos = 0; pos < maxSeqLen_; ++pos) {
        for (size_t i = 0; i < dimPerHead_ / 2; ++i) {
            float exponent = static_cast<float>(2 * i) / static_cast<float>(dimPerHead_);
            float thetaVal = std::pow(theta_, exponent);
            float freq = static_cast<float>(pos) / thetaVal;
            size_t idx = pos * (dimPerHead_ / 2) + i;
            cosCache_[idx] = std::cos(freq);
            sinCache_[idx] = std::sin(freq);
        }
    }
}

void RoPE::apply(Tensor& q, Tensor& k, size_t seqLen, size_t posOffset) const {
    const auto& shape = q.shape();
    if (shape.size() != 4) {
        throw std::invalid_argument("RoPE::apply expects q to be 4D [batch, heads, seq, dim]");
    }
    if (k.shape() != shape) {
        throw std::invalid_argument("RoPE::apply requires q and k to have same shape");
    }

    size_t batch = shape[0];
    size_t heads = shape[1];
    size_t maxSeq = shape[2];
    size_t dim = shape[3];

    if (seqLen > maxSeq) {
        throw std::invalid_argument("RoPE::apply seqLen exceeds tensor sequence dimension");
    }
    if (dim != dimPerHead_) {
        throw std::invalid_argument("RoPE::apply dimPerHead mismatch with tensor last dim");
    }
    if (dim % 2 != 0) {
        throw std::invalid_argument("RoPE::apply requires head_dim to be even");
    }

    float* qData = q.data();
    float* kData = k.data();

    // 形状为 [batch, heads, seq, dim]，row-major
    // index = ((b * heads + h) * maxSeq + pos) * dim + d
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < heads; ++h) {
            for (size_t pos = 0; pos < seqLen; ++pos) {
                size_t position = posOffset + pos;
                if (position >= maxSeqLen_) {
                    throw std::invalid_argument("RoPE::apply position exceeds maxSeqLen");
                }

                size_t cacheBase = position * (dim / 2);

                for (size_t i = 0; i < dim / 2; ++i) {
                    float cosVal = cosCache_[cacheBase + i];
                    float sinVal = sinCache_[cacheBase + i];

                    size_t d0 = 2 * i;
                    size_t d1 = 2 * i + 1;

                    size_t baseIndex = ((b * heads + h) * maxSeq + pos) * dim;
                    size_t idxQ0 = baseIndex + d0;
                    size_t idxQ1 = baseIndex + d1;
                    size_t idxK0 = baseIndex + d0;
                    size_t idxK1 = baseIndex + d1;

                    float q0 = qData[idxQ0];
                    float q1 = qData[idxQ1];
                    float k0 = kData[idxK0];
                    float k1 = kData[idxK1];

                    // 旋转变换
                    qData[idxQ0] = q0 * cosVal - q1 * sinVal;
                    qData[idxQ1] = q0 * sinVal + q1 * cosVal;

                    kData[idxK0] = k0 * cosVal - k1 * sinVal;
                    kData[idxK1] = k0 * sinVal + k1 * cosVal;
                }
            }
        }
    }
}

}  // namespace kylin
}  // namespace cllm
