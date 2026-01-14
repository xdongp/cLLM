/**
 * @file attention.cpp
 * @brief Multi-Head Attention 的简化实现（MVP，无 KV Cache）
 */

#include "cllm/kylin/attention.h"

#include "cllm/common/logger.h"

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
    , rope_(nullptr) {  // 延迟初始化，在forwardNoKV中根据实际的qHeadDim创建
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

    // 验证权重形状
    const auto& wqShape = wq_->shape();
    const auto& wkShape = wk_->shape();
    const auto& wvShape = wv_->shape();
    const auto& woShape = wo_->shape();
    
    if (wqShape.size() != 2 || wkShape.size() != 2 || wvShape.size() != 2 || woShape.size() != 2) {
        throw std::runtime_error("MultiHeadAttention: weight shapes must be 2D");
    }
    
    // 从实际权重形状推断维度（支持 GQA，Q 和 KV 维度可能不同）
    size_t qDim = wqShape[1];  // wq: [hidden, qDim]
    size_t kvDim = wkShape[1]; // wk: [hidden, kvDim]
    
    // 验证权重维度
    if (wqShape[0] != hiddenSize_) {
        throw std::runtime_error("MultiHeadAttention: wq shape mismatch, expected [" + 
                                std::to_string(hiddenSize_) + ", ?], got [" + 
                                std::to_string(wqShape[0]) + ", " + std::to_string(wqShape[1]) + "]");
    }
    if (wkShape[0] != hiddenSize_) {
        throw std::runtime_error("MultiHeadAttention: wk shape mismatch, expected [" + 
                                std::to_string(hiddenSize_) + ", ?], got [" + 
                                std::to_string(wkShape[0]) + ", " + std::to_string(wkShape[1]) + "]");
    }
    if (wvShape[0] != hiddenSize_ || wvShape[1] != kvDim) {
        throw std::runtime_error("MultiHeadAttention: wv shape mismatch, expected [" + 
                                std::to_string(hiddenSize_) + ", " + std::to_string(kvDim) + 
                                "], got [" + std::to_string(wvShape[0]) + ", " + std::to_string(wvShape[1]) + "]");
    }
    if (woShape[0] != qDim || woShape[1] != hiddenSize_) {
        throw std::runtime_error("MultiHeadAttention: wo shape mismatch, expected [" + 
                                std::to_string(qDim) + ", " + std::to_string(hiddenSize_) + 
                                "], got [" + std::to_string(woShape[0]) + ", " + std::to_string(woShape[1]) + "]");
    }
    
    // 验证维度可整除
    if (qDim % numHeads_ != 0) {
        throw std::runtime_error("MultiHeadAttention: qDim (" + std::to_string(qDim) + 
                                ") must be divisible by numHeads (" + std::to_string(numHeads_) + ")");
    }
    if (kvDim % numHeads_ != 0) {
        throw std::runtime_error("MultiHeadAttention: kvDim (" + std::to_string(kvDim) + 
                                ") must be divisible by numHeads (" + std::to_string(numHeads_) + ")");
    }
    
    // 计算实际的 headDim（用于 Q 和 KV）
    // 注意：对于 GQA，Q 和 KV 可能有不同的 headDim
    size_t qHeadDim = qDim / numHeads_;  // Q 的 headDim
    size_t kvHeadDim = kvDim / numHeads_; // KV 的 headDim
    
    // 延迟初始化 RoPE：根据实际的 qHeadDim 创建（支持 GQA）
    // 如果 qHeadDim 改变，重新创建 RoPE
    if (!rope_ || rope_->getDimPerHead() != qHeadDim) {
        // 使用默认的 maxSeqLen=2048 和 theta=10000.0f
        // 如果需要更大的序列长度，可以在构造函数中传入
        rope_ = std::make_unique<RoPE>(qHeadDim, 2048, 10000.0f);
        CLLM_DEBUG("[MultiHeadAttention] Initialized RoPE with qHeadDim=%zu", qHeadDim);
    }

    // 展平成二维： [B*S, H]
    size_t rows = batch * seqLen;
    size_t cols = hiddenSize_;

    // Q/K/V: [B*S, qDim/kvDim]
    Tensor q2d({rows, qDim});
    Tensor k2d({rows, kvDim});
    Tensor v2d({rows, kvDim});

    using namespace kernels;

    // 验证输入数据有效性
    if (input.data() == nullptr || wq_->data() == nullptr || wk_->data() == nullptr || wv_->data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: null pointer detected");
    }

    CLLM_DEBUG("[MultiHeadAttention] matmul Q: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wqShape[0], wqShape[1], rows, qDim);
    matmul(input.data(), wq_->data(), q2d.data(), rows, qDim, cols);
    
    CLLM_DEBUG("[MultiHeadAttention] matmul K: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wkShape[0], wkShape[1], rows, kvDim);
    matmul(input.data(), wk_->data(), k2d.data(), rows, kvDim, cols);
    
    CLLM_DEBUG("[MultiHeadAttention] matmul V: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wvShape[0], wvShape[1], rows, kvDim);
    matmul(input.data(), wv_->data(), v2d.data(), rows, kvDim, cols);

    // 重新组织为 [batch, heads, seq, headDim]
    // 注意：qHeadDim 和 kvHeadDim 已经在上面计算过了
    Tensor q4d({batch, numHeads_, seqLen, qHeadDim});
    Tensor k4d({batch, numHeads_, seqLen, kvHeadDim});
    Tensor v4d({batch, numHeads_, seqLen, kvHeadDim});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t row = b * seqLen + s;
            for (size_t h = 0; h < numHeads_; ++h) {
                // Q: 使用 qHeadDim
                for (size_t d = 0; d < qHeadDim; ++d) {
                    size_t srcIndex = row * qDim + h * qHeadDim + d;
                    size_t dstIndex = ((b * numHeads_ + h) * seqLen + s) * qHeadDim + d;
                    q4d[dstIndex] = q2d[srcIndex];
                }
                // K/V: 使用 kvHeadDim
                for (size_t d = 0; d < kvHeadDim; ++d) {
                    size_t kSrcIndex = row * kvDim + h * kvHeadDim + d;
                    size_t vSrcIndex = row * kvDim + h * kvHeadDim + d;
                    size_t kDstIndex = ((b * numHeads_ + h) * seqLen + s) * kvHeadDim + d;
                    size_t vDstIndex = ((b * numHeads_ + h) * seqLen + s) * kvHeadDim + d;
                    k4d[kDstIndex] = k2d[kSrcIndex];
                    v4d[vDstIndex] = v2d[vSrcIndex];
                }
            }
        }
    }

    // 应用 RoPE 到 Q/K
    // 注意：RoPE 需要相同的 headDim，但 Q 和 K 可能有不同的 headDim（GQA）
    // 对于 GQA，如果 qHeadDim != kvHeadDim，需要分别处理
    // 简化：暂时只对 Q 和 K 应用 RoPE（如果 headDim 相同）
    // 如果不同，创建一个临时的 K 张量，使用 qHeadDim 的维度（简化处理）
    if (qHeadDim == kvHeadDim) {
        rope_->apply(q4d, k4d, seqLen, 0);
    } else {
        // GQA 情况：Q 和 K 有不同的 headDim
        // 创建一个临时的 k4d_q，使用 qHeadDim 的维度
        // 将 k4d 的数据复制到 k4d_q，只复制前 min(qHeadDim, kvHeadDim) 个维度
        Tensor k4d_q({batch, numHeads_, seqLen, qHeadDim});
        k4d_q.fill(0.0f);
        
        size_t copyDim = std::min(qHeadDim, kvHeadDim);
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t s = 0; s < seqLen; ++s) {
                    size_t kSrcOffset = ((b * numHeads_ + h) * seqLen + s) * kvHeadDim;
                    size_t kDstOffset = ((b * numHeads_ + h) * seqLen + s) * qHeadDim;
                    for (size_t d = 0; d < copyDim; ++d) {
                        k4d_q[kDstOffset + d] = k4d[kSrcOffset + d];
                    }
                }
            }
        }
        
        // 对 Q 和临时 K 应用 RoPE
        rope_->apply(q4d, k4d_q, seqLen, 0);
        
        // 将 RoPE 后的 K 数据复制回 k4d（只复制前 kvHeadDim 个维度）
        for (size_t b = 0; b < batch; ++b) {
            for (size_t h = 0; h < numHeads_; ++h) {
                for (size_t s = 0; s < seqLen; ++s) {
                    size_t kSrcOffset = ((b * numHeads_ + h) * seqLen + s) * qHeadDim;
                    size_t kDstOffset = ((b * numHeads_ + h) * seqLen + s) * kvHeadDim;
                    for (size_t d = 0; d < copyDim; ++d) {
                        k4d[kDstOffset + d] = k4d_q[kSrcOffset + d];
                    }
                }
            }
        }
        
        CLLM_DEBUG("[MultiHeadAttention] Applied RoPE with different headDim: qHeadDim=%zu, kvHeadDim=%zu", 
                  qHeadDim, kvHeadDim);
    }

    // 计算 attention
    // q4d/k4d/v4d: [B, H, S, D]
    // 对每个 (b, h) 计算 SxS 的注意力矩阵
    // 注意：对于 GQA，Q 和 KV 的 headDim 可能不同，但这里简化处理，使用 qHeadDim 作为 scale
    Tensor attnOut4d({batch, numHeads_, seqLen, kvHeadDim});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numHeads_; ++h) {
            // Q, K, V 视为 [S, D]
            const float* qPtr = q4d.data() + ((b * numHeads_ + h) * seqLen * qHeadDim);
            const float* kPtr = k4d.data() + ((b * numHeads_ + h) * seqLen * kvHeadDim);
            const float* vPtr = v4d.data() + ((b * numHeads_ + h) * seqLen * kvHeadDim);
            float* outPtr = attnOut4d.data() + ((b * numHeads_ + h) * seqLen * kvHeadDim);

            // scores: [S, S]
            Tensor scores({seqLen, seqLen});
            // 直接计算 Q*K^T，而不使用 matmul 函数
            // 注意：Q 和 K 的 headDim 可能不同，使用 min 来确保安全
            size_t minHeadDim = std::min(qHeadDim, kvHeadDim);
            for (size_t i = 0; i < seqLen; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < minHeadDim; ++k) {
                        // Q: [S, qHeadDim], K: [S, kvHeadDim]
                        // Q[i][k] * K[j][k]
                        sum += qPtr[i * qHeadDim + k] * kPtr[j * kvHeadDim + k];
                    }
                    scores[i * seqLen + j] = sum;
                }
            }

            // scale by sqrt(qHeadDim) - 使用 Q 的 headDim
            float scale = 1.0f / std::sqrt(static_cast<float>(qHeadDim));
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
            if (scores.data() == nullptr || probs.data() == nullptr) {
                throw std::runtime_error("MultiHeadAttention: scores or probs data is null");
            }
            softmax_stable(scores.data(), probs.data(), seqLen, seqLen);

            // out = probs @ V  -> [S, D]
            if (vPtr == nullptr || outPtr == nullptr) {
                throw std::runtime_error("MultiHeadAttention: vPtr or outPtr is null");
            }
            CLLM_DEBUG("[MultiHeadAttention] matmul probs@V: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
                      seqLen, seqLen, seqLen, kvHeadDim, seqLen, kvHeadDim);
            matmul(probs.data(), vPtr, outPtr, seqLen, kvHeadDim, seqLen);
        }
    }

    // 合并 heads: [B, S, H, kvHeadDim] -> [B, S, qDim]
    // 注意：对于 GQA，attnOut4d 的每个 head 输出是 kvHeadDim，但 wo 的输入是 qDim
    // 简化处理：假设 qDim = numHeads * qHeadDim，kvDim = numHeads * kvHeadDim
    // 如果 qDim != kvDim，需要将 kvDim 的输出映射到 qDim
    // 这里简化：直接按 head 顺序拼接，如果 qDim > kvDim，剩余部分填充 0
    Tensor merged({batch, seqLen, qDim});
    merged.fill(0.0f);  // 初始化为 0
    
    // 验证：qDim 应该等于 numHeads * qHeadDim，kvDim 应该等于 numHeads * kvHeadDim
    if (qDim != numHeads_ * qHeadDim) {
        CLLM_WARN("[MultiHeadAttention] qDim (%zu) != numHeads * qHeadDim (%zu * %zu = %zu)", 
                 qDim, numHeads_, qHeadDim, numHeads_ * qHeadDim);
    }
    if (kvDim != numHeads_ * kvHeadDim) {
        CLLM_WARN("[MultiHeadAttention] kvDim (%zu) != numHeads * kvHeadDim (%zu * %zu = %zu)", 
                 kvDim, numHeads_, kvHeadDim, numHeads_ * kvHeadDim);
    }
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t dstOffset = (b * seqLen + s) * qDim;
            size_t headOffset = 0;
            
            for (size_t h = 0; h < numHeads_; ++h) {
                // 从 attnOut4d 复制当前 head 的输出
                size_t srcOffset = ((b * numHeads_ + h) * seqLen + s) * kvHeadDim;
                
                // 复制 kvHeadDim 个元素到 merged
                // 注意：如果 qDim > kvDim，可能需要填充或插值
                // 这里简化：直接复制，如果 headOffset + kvHeadDim > qDim，只复制部分
                size_t copyLen = std::min(kvHeadDim, qDim - headOffset);
                for (size_t d = 0; d < copyLen; ++d) {
                    merged[dstOffset + headOffset + d] = attnOut4d[srcOffset + d];
                }
                
                headOffset += kvHeadDim;
                
                // 如果已经填满 qDim，停止
                if (headOffset >= qDim) {
                    break;
                }
            }
        }
    }

    // 输出投影: [B*S, qDim] @ [qDim, hidden] -> [B*S, hidden]
    CLLM_DEBUG("[MultiHeadAttention] Projecting output: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, qDim, woShape[0], woShape[1], rows, hiddenSize_);
    if (wo_->data() == nullptr || merged.data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: wo_ or merged data is null");
    }
    Tensor out2d({rows, hiddenSize_});
    if (out2d.data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: out2d data is null");
    }
    matmul(merged.data(), wo_->data(), out2d.data(), rows, hiddenSize_, qDim);

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
