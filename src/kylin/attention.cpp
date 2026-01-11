/**
 * @file attention.cpp
 * @brief Multi-Head Attention 的简化实现（MVP，无 KV Cache）
 */

#include "cllm/kylin/attention.h"

#include <stdexcept>
#include <cmath>
#include <limits>

namespace cllm {
namespace kylin {

MultiHeadAttention::MultiHeadAttention(
    size_t hiddenSize,
    size_t numHeads,
    float ropeTheta
)
    : hiddenSize_(hiddenSize)
    , numHeads_(numHeads)
    , headDim_(hiddenSize / numHeads)
    , wq_(nullptr)
    , wk_(nullptr)
    , wv_(nullptr)
    , wo_(nullptr)
    , rope_(headDim_, ropeTheta) {
    if (hiddenSize_ == 0 || numHeads_ == 0 || hiddenSize_ % numHeads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention: invalid hiddenSize/numHeads");
    }
}

void MultiHeadAttention::setWeights(
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo
) {
    wq_ = &wq;
    wk_ = &wk;
    wv_ = &wv;
    wo_ = &wo;
}

Tensor MultiHeadAttention::forwardNoKV(const Tensor& input) const {
    const auto& inShape = input.shape();
    if (inShape.size() != 3) {
        throw std::invalid_argument("MultiHeadAttention::forwardNoKV expects [batch, seq, hidden]");
    }
    if (!wq_ || !wk_ || !wv_ || !wo_) {
        throw std::runtime_error("MultiHeadAttention weights not set");
    }

    size_t batch = inShape[0];
    size_t seqLen = inShape[1];
    size_t hidden = inShape[2];
    if (hidden != hiddenSize_) {
        throw std::invalid_argument("MultiHeadAttention: input hidden size mismatch");
    }

    // 展平成二维： [B*S, H]
    size_t rows = batch * seqLen;
    size_t cols = hiddenSize_;

    // Q/K/V: [B*S, numHeads * headDim]
    Tensor q2d({rows, numHeads_ * headDim_});
    Tensor k2d({rows, numHeads_ * headDim_});
    Tensor v2d({rows, numHeads_ * headDim_});

    using namespace kernels;

    matmul(input.data(), wq_->data(), q2d.data(), rows, numHeads_ * headDim_, cols);
    matmul(input.data(), wk_->data(), k2d.data(), rows, numHeads_ * headDim_, cols);
    matmul(input.data(), wv_->data(), v2d.data(), rows, numHeads_ * headDim_, cols);

    // 重新组织为 [batch, heads, seq, headDim]
    Tensor q4d({batch, numHeads_, seqLen, headDim_});
    Tensor k4d({batch, numHeads_, seqLen, headDim_});
    Tensor v4d({batch, numHeads_, seqLen, headDim_});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t row = b * seqLen + s;
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t d = 0; d < headDim_; ++d) {
                    size_t srcIndex = row * (numHeads_ * headDim_) + h * headDim_ + d;
                    size_t dstIndex = ((b * numHeads_ + h) * seqLen + s) * headDim_ + d;
                    q4d[dstIndex] = q2d[srcIndex];
                    k4d[dstIndex] = k2d[srcIndex];
                    v4d[dstIndex] = v2d[srcIndex];
                }
            }
        }
    }

    // 应用 RoPE 到 Q/K
    rope_.apply(q4d, k4d, seqLen, 0);

    // 计算 attention
    // q4d/k4d/v4d: [B, H, S, D]
    // 对每个 (b, h) 计算 SxS 的注意力矩阵
    Tensor attnOut4d({batch, numHeads_, seqLen, headDim_});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numHeads_; ++h) {
            // Q, K, V 视为 [S, D]
            const float* qPtr = q4d.data() + ((b * numHeads_ + h) * seqLen * headDim_);
            const float* kPtr = k4d.data() + ((b * numHeads_ + h) * seqLen * headDim_);
            const float* vPtr = v4d.data() + ((b * numHeads_ + h) * seqLen * headDim_);
            float* outPtr = attnOut4d.data() + ((b * numHeads_ + h) * seqLen * headDim_);

            // scores: [S, S]
            Tensor scores({seqLen, seqLen});
            matmul(qPtr, kPtr, scores.data(), seqLen, seqLen, headDim_);

            // scale by sqrt(headDim)
            float scale = 1.0f / std::sqrt(static_cast<float>(headDim_));
            for (size_t i = 0; i < seqLen * seqLen; ++i) {
                scores[i] *= scale;
            }

            // causal mask: 只保留 i>=j 的位置，屏蔽未来信息
            float negInf = -std::numeric_limits<float>::infinity();
            for (size_t i = 0; i < seqLen; ++i) {
                for (size_t j = i + 1; j < seqLen; ++j) {
                    scores[i * seqLen + j] = negInf;
                }
            }

            // softmax over last dim (key dimension)
            Tensor probs({seqLen, seqLen});
            softmax_stable(scores.data(), probs.data(), seqLen, seqLen);

            // out = probs @ V  -> [S, D]
            matmul(probs.data(), vPtr, outPtr, seqLen, headDim_, seqLen);
        }
    }

    // 合并 heads: [B, S, H, D] -> [B, S, H*D]
    Tensor merged({batch, seqLen, numHeads_ * headDim_});
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t d = 0; d < headDim_; ++d) {
                    size_t srcIndex = ((b * numHeads_ + h) * seqLen + s) * headDim_ + d;
                    size_t dstIndex = (b * seqLen + s) * (numHeads_ * headDim_) + h * headDim_ + d;
                    merged[dstIndex] = attnOut4d[srcIndex];
                }
            }
        }
    }

    // 输出投影: [B*S, H*D] @ [H*D, hidden] -> [B*S, hidden]
    Tensor out2d({rows, hiddenSize_});
    matmul(merged.data(), wo_->data(), out2d.data(), rows, hiddenSize_, numHeads_ * headDim_);

    // 还原为 [B, S, H]
    Tensor output({batch, seqLen, hiddenSize_});
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t row = b * seqLen + s;
            for (size_t d = 0; d < hiddenSize_; ++d) {
                size_t srcIndex = row * hiddenSize_ + d;
                size_t dstIndex = (b * seqLen + s) * hiddenSize_ + d;
                output[dstIndex] = out2d[srcIndex];
            }
        }
    }

    return output;
}

}  // namespace kylin
}  // namespace cllm
