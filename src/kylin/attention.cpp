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
    size_t numQHeads,
    size_t numKVHeads,
    float ropeTheta,
    float rmsNormEps,  // P4修复：从配置读取，而不是硬编码
    // P3修复：RoPE扩展参数
    size_t maxSequenceLength,
    size_t ropeNctxOrig,
    float ropeFreqScale,
    int ropeType,
    float ropeExtFactor
)
    : hiddenSize_(hiddenSize)
    , numQHeads_(numQHeads)
    , numKVHeads_(numKVHeads)
    , headDim_(hiddenSize / numQHeads)  // head_dim 基于 Q heads 计算
    , ropeTheta_(ropeTheta)  // 保存 ropeTheta，用于延迟初始化 RoPE
    , rmsNormEps_(rmsNormEps)  // P4修复：保存 rmsNormEps，用于 Q/K Norm
    , maxSequenceLength_(maxSequenceLength)  // P3修复：保存 maxSequenceLength
    , ropeNctxOrig_(ropeNctxOrig > 0 ? ropeNctxOrig : maxSequenceLength)  // P3修复：如果未提供，使用maxSequenceLength
    , ropeFreqScale_(ropeFreqScale)
    , ropeType_(ropeType)
    , ropeExtFactor_(ropeExtFactor)
    , hasAttnQKNorm_(false)
    , rope_(nullptr) {  // 延迟初始化，在forwardNoKV中根据实际的qHeadDim创建
    if (hiddenSize_ == 0 || numQHeads_ == 0 || numKVHeads_ == 0) {
        throw std::invalid_argument("MultiHeadAttention: invalid hiddenSize/numQHeads/numKVHeads");
    }
    if (hiddenSize_ % numQHeads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention: hiddenSize must be divisible by numQHeads");
    }
    if (numQHeads_ % numKVHeads_ != 0) {
        throw std::invalid_argument("MultiHeadAttention: numQHeads must be divisible by numKVHeads (GQA requirement)");
    }
    // 验证 Q 和 KV 的 head_dim 应该相同（GQA 标准）
    size_t kvHeadDim = (hiddenSize / numQHeads_) * (numQHeads_ / numKVHeads_);
    if (kvHeadDim != headDim_) {
        CLLM_WARN("MultiHeadAttention: Q head_dim (%zu) != KV head_dim (%zu), using Q head_dim", 
                 headDim_, kvHeadDim);
    }
}

void MultiHeadAttention::setWeights(
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    const Tensor& wo
) {
    wq_ = wq;
    wk_ = wk;
    wv_ = wv;
    wo_ = wo;
}

void MultiHeadAttention::setAttnQKNormWeights(
    const Tensor& attnQNormWeight,
    const Tensor& attnKNormWeight
) {
    attnQNormWeight_ = attnQNormWeight;
    attnKNormWeight_ = attnKNormWeight;
    hasAttnQKNorm_ = true;
    CLLM_DEBUG("[MultiHeadAttention] Set Q/K norm weights (Q shape: %zu, K shape: %zu)", 
              attnQNormWeight_.shape().empty() ? 0 : attnQNormWeight_.shape()[0],
              attnKNormWeight_.shape().empty() ? 0 : attnKNormWeight_.shape()[0]);
}

Tensor MultiHeadAttention::forwardNoKV(const Tensor& input) const {
    const auto& inShape = input.shape();
    if (inShape.size() != 3) {
        throw std::invalid_argument("MultiHeadAttention::forwardNoKV expects [batch, seq, hidden]");
    }
    if (wq_.shape().empty() || wk_.shape().empty() || wv_.shape().empty() || wo_.shape().empty()) {
        throw std::runtime_error("MultiHeadAttention weights not set");
    }

    size_t batch = inShape[0];
    size_t seqLen = inShape[1];
    size_t hidden = inShape[2];
    if (hidden != hiddenSize_) {
        throw std::invalid_argument("MultiHeadAttention: input hidden size mismatch");
    }

    // 验证权重形状
    const auto& wqShape = wq_.shape();
    const auto& wkShape = wk_.shape();
    const auto& wvShape = wv_.shape();
    const auto& woShape = wo_.shape();
    
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
    
    // 验证维度可整除（GQA 正确实现）
    if (qDim % numQHeads_ != 0) {
        throw std::runtime_error("MultiHeadAttention: qDim (" + std::to_string(qDim) + 
                                ") must be divisible by numQHeads (" + std::to_string(numQHeads_) + ")");
    }
    if (kvDim % numKVHeads_ != 0) {
        throw std::runtime_error("MultiHeadAttention: kvDim (" + std::to_string(kvDim) + 
                                ") must be divisible by numKVHeads (" + std::to_string(numKVHeads_) + ")");
    }
    
    // 计算实际的 headDim（从权重形状推断，而不是假设标准公式）
    // P0修复：使用实际的 qHeadDim 和 kvHeadDim，而不是标准公式计算的 headDim
    size_t qHeadDim = qDim / numQHeads_;   // Q 的 headDim（从权重形状推断，如 Qwen3: 2048/16=128）
    size_t kvHeadDim = kvDim / numKVHeads_; // KV 的 headDim（从权重形状推断，如 Qwen3: 1024/16=64）
    
    // 计算标准 head_dim（从 hidden_size 和配置的 head 数量，仅用于对比）
    size_t standardHeadDim = hiddenSize_ / numQHeads_;
    
    // P2修复：移除容错逻辑，要求 Q 和 KV 的 head_dim 必须一致（Qwen3/llama.cpp 要求）
    // llama.cpp 强制断言 Q/K/V head_dim 与 n_rot 完全一致，否则直接失败
    // 我们也要遵循这个原则，避免"看似能跑"的错误输出
    
    // 检查 Q 和 KV 的 head_dim 是否一致
    if (qHeadDim != kvHeadDim) {
        CLLM_ERROR("[MultiHeadAttention] ❌ Q head_dim (%zu) != KV head_dim (%zu)，不符合 Qwen3/llama.cpp 要求！",
                  qHeadDim, kvHeadDim);
        CLLM_ERROR("  - qHeadDim (from weights): %zu (qDim=%zu / numQHeads=%zu)", qHeadDim, qDim, numQHeads_);
        CLLM_ERROR("  - kvHeadDim (from weights): %zu (kvDim=%zu / numKVHeads=%zu)", kvHeadDim, kvDim, numKVHeads_);
        throw std::runtime_error("Q and KV head_dim must be consistent (Qwen3/llama.cpp requirement)");
    }
    
    // 使用 qHeadDim（Q 和 KV 的 head_dim 应该相同）
    size_t unifiedHeadDim = qHeadDim;
    
    // 如果 qHeadDim != standardHeadDim，说明使用了扩展 head_dim（如 Qwen3 的 2x 扩展）
    if (qHeadDim != standardHeadDim) {
        size_t expansionFactor = qHeadDim / standardHeadDim;
        CLLM_INFO("[MultiHeadAttention] ✓ 使用扩展 head_dim：qHeadDim=kvHeadDim=%zu (标准=%zu, 扩展因子=%zu)",
                 qHeadDim, standardHeadDim, expansionFactor);
    } else {
        CLLM_DEBUG("[MultiHeadAttention] ✓ 使用标准 head_dim：%zu", unifiedHeadDim);
    }
    
    // 计算 GQA 分组数
    size_t gqa = numQHeads_ / numKVHeads_;  // 每个 KV head 对应多少个 Q heads
    if (numQHeads_ % numKVHeads_ != 0) {
        throw std::runtime_error("MultiHeadAttention: numQHeads (" + std::to_string(numQHeads_) + 
                                ") must be divisible by numKVHeads (" + std::to_string(numKVHeads_) + ")");
    }
    
    // P3修复：RoPE 初始化 - 使用配置的 maxSequenceLength 和扩展参数
    // llama.cpp 要求 Q/K head_dim 与 n_rot 完全一致
    size_t ropeHeadDim = unifiedHeadDim;  // Q 和 KV 的 head_dim 应该相同
    if (!rope_ || rope_->getDimPerHead() != ropeHeadDim) {
        // P3修复：使用配置的 maxSequenceLength 和 RoPE 扩展参数
        rope_ = std::make_unique<RoPE>(
            ropeHeadDim, 
            maxSequenceLength_,  // 使用配置的 maxSequenceLength，而不是固定的 2048
            ropeTheta_,
            ropeNctxOrig_,
            ropeFreqScale_,
            ropeType_,
            ropeExtFactor_
        );
        CLLM_DEBUG("[MultiHeadAttention] Initialized RoPE with headDim=%zu, theta=%f, maxSeqLen=%zu, nCtxOrig=%zu, freqScale=%f, type=%d, extFactor=%f", 
                  ropeHeadDim, ropeTheta_, maxSequenceLength_, ropeNctxOrig_, ropeFreqScale_, ropeType_, ropeExtFactor_);
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
    if (input.data() == nullptr || wq_.data() == nullptr || wk_.data() == nullptr || wv_.data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: null pointer detected");
    }

    CLLM_DEBUG("[MultiHeadAttention] matmul Q: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wqShape[0], wqShape[1], rows, qDim);
    matmul(input.data(), wq_.data(), q2d.data(), rows, qDim, cols);
    
    CLLM_DEBUG("[MultiHeadAttention] matmul K: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wkShape[0], wkShape[1], rows, kvDim);
    matmul(input.data(), wk_.data(), k2d.data(), rows, kvDim, cols);
    
    CLLM_DEBUG("[MultiHeadAttention] matmul V: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, cols, wvShape[0], wvShape[1], rows, kvDim);
    matmul(input.data(), wv_.data(), v2d.data(), rows, kvDim, cols);

    // 重新组织为 [batch, heads, seq, headDim]
    // GQA 正确实现：Q 为 [B, n_head, S, head_dim]，K/V 为 [B, n_head_kv, S, head_dim]
    // P2修复：Q 和 KV 的 head_dim 必须一致（已在上面检查），所以 kvHeadDimForReshape = unifiedHeadDim
    size_t kvHeadDimForReshape = unifiedHeadDim;
    Tensor q4d({batch, numQHeads_, seqLen, unifiedHeadDim});
    Tensor k4d({batch, numKVHeads_, seqLen, kvHeadDimForReshape});
    Tensor v4d({batch, numKVHeads_, seqLen, kvHeadDimForReshape});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t row = b * seqLen + s;
            // Q: reshape 为 [B, n_head, S, head_dim]
            // 注意：配置应该在 bindWeightsToModel 中已经修正，所以 numQHeads_ 应该是正确的
            // 但我们仍然需要处理可能的不匹配情况（以防配置修正失败）
            
            // 首先尝试按照 unifiedHeadDim 推断实际的 head 数量
            size_t actualNumQHeads = (unifiedHeadDim > 0 && qDim % unifiedHeadDim == 0) ? 
                                      (qDim / unifiedHeadDim) : numQHeads_;
            
            // 如果 qDim 等于期望值，说明配置是正确的，直接使用标准 reshape
            size_t expectedQDim = numQHeads_ * unifiedHeadDim;
            if (qDim == expectedQDim && actualNumQHeads == numQHeads_) {
                // 配置正确，直接复制
                for (size_t h = 0; h < numQHeads_; ++h) {
                    for (size_t d = 0; d < unifiedHeadDim; ++d) {
                        size_t srcIndex = row * qDim + h * unifiedHeadDim + d;
                        size_t dstIndex = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim + d;
                        if (srcIndex < q2d.size() && dstIndex < q4d.size()) {
                            q4d[dstIndex] = q2d[srcIndex];
                        }
                    }
                }
            } else if (actualNumQHeads != numQHeads_ && actualNumQHeads > 0) {
                // 配置可能未被修正，尝试映射（这种情况应该很少见，因为我们已经在 bindWeightsToModel 中修正了）
                // 权重按照 actualNumQHeads * unifiedHeadDim 组织
                // 需要将 actualNumQHeads 个权重 heads 映射到 numQHeads_ 个配置 heads
                size_t headsPerConfigHead = actualNumQHeads / numQHeads_;
                if (headsPerConfigHead > 0 && actualNumQHeads % numQHeads_ == 0) {
                    // 将多个权重 heads 的数据平均分配到配置 heads
                    for (size_t h = 0; h < numQHeads_; ++h) {
                        for (size_t d = 0; d < unifiedHeadDim; ++d) {
                            float sum = 0.0f;
                            for (size_t subH = 0; subH < headsPerConfigHead; ++subH) {
                                size_t weightHeadIdx = h * headsPerConfigHead + subH;
                                // 权重按照 unifiedHeadDim 组织：每个 head 有 unifiedHeadDim 个元素
                                size_t srcIndex = row * qDim + weightHeadIdx * unifiedHeadDim + d;
                                if (srcIndex < q2d.size()) {
                                    sum += q2d[srcIndex];
                                }
                            }
                            size_t dstIndex = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim + d;
                            if (dstIndex < q4d.size()) {
                                q4d[dstIndex] = sum / static_cast<float>(headsPerConfigHead);
                            }
                        }
                    }
                } else {
                    // 无法平均分配，使用前 numQHeads_ 个 heads
                    CLLM_WARN("[MultiHeadAttention] Cannot evenly map %zu weight heads to %zu config heads, using first %zu heads",
                             actualNumQHeads, numQHeads_, numQHeads_);
                    for (size_t h = 0; h < numQHeads_ && h < actualNumQHeads; ++h) {
                        for (size_t d = 0; d < unifiedHeadDim; ++d) {
                            size_t srcIndex = row * qDim + h * unifiedHeadDim + d;
                            size_t dstIndex = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim + d;
                            if (srcIndex < q2d.size() && dstIndex < q4d.size()) {
                                q4d[dstIndex] = q2d[srcIndex];
                            }
                        }
                    }
                }
            } else {
                // 配置匹配，或者无法推断，使用配置的 numQHeads
                // 但需要处理 head_dim 不匹配的情况
                size_t weightHeadDim = qDim / numQHeads_;  // 从权重推断的 head_dim
                size_t copyDim = std::min(weightHeadDim, unifiedHeadDim);
                
                for (size_t h = 0; h < numQHeads_; ++h) {
                    for (size_t d = 0; d < copyDim; ++d) {
                        size_t srcIndex = row * qDim + h * weightHeadDim + d;
                        size_t dstIndex = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim + d;
                        if (srcIndex < q2d.size() && dstIndex < q4d.size()) {
                            q4d[dstIndex] = q2d[srcIndex];
                        }
                    }
                    // 如果 unifiedHeadDim > weightHeadDim，剩余部分填充 0
                    for (size_t d = copyDim; d < unifiedHeadDim; ++d) {
                        size_t dstIndex = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim + d;
                        if (dstIndex < q4d.size()) {
                            q4d[dstIndex] = 0.0f;
                        }
                    }
                }
            }
            // K/V: reshape 为 [B, n_head_kv, S, head_dim]
            // P0修复：K/V 应该使用实际的 kvHeadDim，而不是 unifiedHeadDim（可能是 qHeadDim）
            // 注意：kvHeadDimForReshape 已在上面定义
            size_t actualNumKVHeads = (kvHeadDimForReshape > 0 && kvDim % kvHeadDimForReshape == 0) ? 
                                       (kvDim / kvHeadDimForReshape) : numKVHeads_;
            
            // 如果 kvDim 等于期望值，说明配置是正确的，直接使用标准 reshape
            size_t expectedKVDim = numKVHeads_ * kvHeadDimForReshape;
            if (kvDim == expectedKVDim && actualNumKVHeads == numKVHeads_) {
                // 配置正确，直接复制（使用 kvHeadDimForReshape）
                for (size_t h = 0; h < numKVHeads_; ++h) {
                    for (size_t d = 0; d < kvHeadDimForReshape; ++d) {
                        size_t kSrcIndex = row * kvDim + h * kvHeadDimForReshape + d;
                        size_t vSrcIndex = row * kvDim + h * kvHeadDimForReshape + d;
                        size_t kDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        size_t vDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        if (kSrcIndex < k2d.size() && kDstIndex < k4d.size()) {
                            k4d[kDstIndex] = k2d[kSrcIndex];
                        }
                        if (vSrcIndex < v2d.size() && vDstIndex < v4d.size()) {
                            v4d[vDstIndex] = v2d[vSrcIndex];
                        }
                    }
                }
            } else if (actualNumKVHeads != numKVHeads_ && actualNumKVHeads > 0) {
                // 配置可能未被修正，尝试映射（这种情况应该很少见）
                size_t headsPerConfigHead = actualNumKVHeads / numKVHeads_;
                if (headsPerConfigHead > 0 && actualNumKVHeads % numKVHeads_ == 0) {
                    // 将多个权重 heads 的数据平均分配到配置 heads（使用 kvHeadDimForReshape）
                    for (size_t h = 0; h < numKVHeads_; ++h) {
                        for (size_t d = 0; d < kvHeadDimForReshape; ++d) {
                            float kSum = 0.0f, vSum = 0.0f;
                            for (size_t subH = 0; subH < headsPerConfigHead; ++subH) {
                                size_t weightHeadIdx = h * headsPerConfigHead + subH;
                                // 权重按照 kvHeadDimForReshape 组织：每个 head 有 kvHeadDimForReshape 个元素
                                size_t kSrcIndex = row * kvDim + weightHeadIdx * kvHeadDimForReshape + d;
                                size_t vSrcIndex = row * kvDim + weightHeadIdx * kvHeadDimForReshape + d;
                                if (kSrcIndex < k2d.size()) {
                                    kSum += k2d[kSrcIndex];
                                }
                                if (vSrcIndex < v2d.size()) {
                                    vSum += v2d[vSrcIndex];
                                }
                            }
                            size_t kDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                            size_t vDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                            if (kDstIndex < k4d.size()) {
                                k4d[kDstIndex] = kSum / static_cast<float>(headsPerConfigHead);
                            }
                            if (vDstIndex < v4d.size()) {
                                v4d[vDstIndex] = vSum / static_cast<float>(headsPerConfigHead);
                            }
                        }
                    }
                } else {
                    // 无法平均分配，使用前 numKVHeads_ 个 heads（使用 kvHeadDimForReshape）
                    CLLM_WARN("[MultiHeadAttention] Cannot evenly map %zu KV weight heads to %zu config heads, using first %zu heads",
                             actualNumKVHeads, numKVHeads_, numKVHeads_);
                    for (size_t h = 0; h < numKVHeads_ && h < actualNumKVHeads; ++h) {
                        for (size_t d = 0; d < kvHeadDimForReshape; ++d) {
                            size_t kSrcIndex = row * kvDim + h * kvHeadDimForReshape + d;
                            size_t vSrcIndex = row * kvDim + h * kvHeadDimForReshape + d;
                            size_t kDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                            size_t vDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                            if (kSrcIndex < k2d.size() && kDstIndex < k4d.size()) {
                                k4d[kDstIndex] = k2d[kSrcIndex];
                            }
                            if (vSrcIndex < v2d.size() && vDstIndex < v4d.size()) {
                                v4d[vDstIndex] = v2d[vSrcIndex];
                            }
                        }
                    }
                }
            } else {
                // 配置匹配，或者无法推断，使用配置的 numKVHeads
                // 但需要处理 head_dim 不匹配的情况（使用 kvHeadDimForReshape）
                size_t weightKVHeadDim = kvDim / numKVHeads_;  // 从权重推断的 KV head_dim
                size_t copyDim = std::min(weightKVHeadDim, kvHeadDimForReshape);
                
                for (size_t h = 0; h < numKVHeads_; ++h) {
                    for (size_t d = 0; d < copyDim; ++d) {
                        size_t kSrcIndex = row * kvDim + h * weightKVHeadDim + d;
                        size_t vSrcIndex = row * kvDim + h * weightKVHeadDim + d;
                        size_t kDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        size_t vDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        if (kSrcIndex < k2d.size() && kDstIndex < k4d.size()) {
                            k4d[kDstIndex] = k2d[kSrcIndex];
                        }
                        if (vSrcIndex < v2d.size() && vDstIndex < v4d.size()) {
                            v4d[vDstIndex] = v2d[vSrcIndex];
                        }
                    }
                    // 如果 kvHeadDimForReshape > weightKVHeadDim，剩余部分填充 0
                    for (size_t d = copyDim; d < kvHeadDimForReshape; ++d) {
                        size_t kDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        size_t vDstIndex = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape + d;
                        if (kDstIndex < k4d.size()) {
                            k4d[kDstIndex] = 0.0f;
                        }
                        if (vDstIndex < v4d.size()) {
                            v4d[vDstIndex] = 0.0f;
                        }
                    }
                }
            }
        }
    }

    // 应用 Q/K 的独立归一化（如果已设置，Qwen3等模型需要）
    // 顺序：reshape → q_norm/k_norm → RoPE → attention
    // 对每个 (batch, head, seq) 的 head_dim 向量做 RMSNorm
    if (hasAttnQKNorm_) {
        using namespace kernels;
        
        // 对 Q 应用归一化：q4d: [B, n_head, S, head_dim]
        size_t qNormSize = attnQNormWeight_.shape().empty() ? 0 : attnQNormWeight_.shape()[0];
        if (qNormSize == unifiedHeadDim && attnQNormWeight_.data() != nullptr) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < numQHeads_; ++h) {
                    for (size_t s = 0; s < seqLen; ++s) {
                        size_t offset = ((b * numQHeads_ + h) * seqLen + s) * unifiedHeadDim;
                        float* qPtr = q4d.data() + offset;
                        rmsnorm(qPtr, qPtr, attnQNormWeight_.data(), 1, unifiedHeadDim, rmsNormEps_);  // P4修复：使用配置的 rmsNormEps
                    }
                }
            }
            CLLM_DEBUG("[MultiHeadAttention] Applied Q norm (weight size: %zu, head_dim: %zu)", 
                      qNormSize, unifiedHeadDim);
        } else {
            CLLM_WARN("[MultiHeadAttention] Q norm weight size (%zu) != unifiedHeadDim (%zu) or weight is null, skipping Q norm", 
                     qNormSize, unifiedHeadDim);
        }
        
        // 对 K 应用归一化：k4d: [B, n_head_kv, S, head_dim]
        // P0修复：使用 kvHeadDimForReshape 而不是 unifiedHeadDim
        // 注意：kvHeadDimForReshape 已在函数开始时定义（第226行）
        size_t kNormSize = attnKNormWeight_.shape().empty() ? 0 : attnKNormWeight_.shape()[0];
        if (kNormSize == kvHeadDimForReshape && attnKNormWeight_.data() != nullptr) {
            for (size_t b = 0; b < batch; ++b) {
                for (size_t h = 0; h < numKVHeads_; ++h) {
                    for (size_t s = 0; s < seqLen; ++s) {
                        size_t offset = ((b * numKVHeads_ + h) * seqLen + s) * kvHeadDimForReshape;
                        float* kPtr = k4d.data() + offset;
                        rmsnorm(kPtr, kPtr, attnKNormWeight_.data(), 1, kvHeadDimForReshape, rmsNormEps_);  // P4修复：使用配置的 rmsNormEps
                    }
                }
            }
            CLLM_DEBUG("[MultiHeadAttention] Applied K norm (weight size: %zu, head_dim: %zu)", 
                      kNormSize, kvHeadDimForReshape);
        } else {
            CLLM_WARN("[MultiHeadAttention] K norm weight size (%zu) != kvHeadDimForReshape (%zu) or weight is null, skipping K norm", 
                     kNormSize, kvHeadDimForReshape);
        }
    }
    
    // 应用 RoPE 到 Q/K
    // GQA：Q 和 K 的 head_dim 相同，但 head 数量不同
    // RoPE 需要处理不同 head 数量的情况
    // 注意：RoPE::apply 已经支持 Q 和 K 有不同的 head 数量（只要 head_dim 相同）
    
    // 验证 q4d 和 k4d 的维度
    const auto& q4dShape = q4d.shape();
    const auto& k4dShape = k4d.shape();
    CLLM_DEBUG("[MultiHeadAttention] Before RoPE: q4d shape=[%zu, %zu, %zu, %zu], k4d shape=[%zu, %zu, %zu, %zu], unifiedHeadDim=%zu",
              q4dShape.size() > 0 ? q4dShape[0] : 0,
              q4dShape.size() > 1 ? q4dShape[1] : 0,
              q4dShape.size() > 2 ? q4dShape[2] : 0,
              q4dShape.size() > 3 ? q4dShape[3] : 0,
              k4dShape.size() > 0 ? k4dShape[0] : 0,
              k4dShape.size() > 1 ? k4dShape[1] : 0,
              k4dShape.size() > 2 ? k4dShape[2] : 0,
              k4dShape.size() > 3 ? k4dShape[3] : 0,
              unifiedHeadDim);
    
    // P0修复：K/V 可能使用不同的 head_dim
    // 注意：kvHeadDimForReshape 已在函数开始时定义（第226行）
    if (q4dShape.size() == 4 && q4dShape[3] != unifiedHeadDim) {
        CLLM_ERROR("[MultiHeadAttention] q4d last dim (%zu) != unifiedHeadDim (%zu)", q4dShape[3], unifiedHeadDim);
        throw std::runtime_error("MultiHeadAttention: q4d dimension mismatch");
    }
    if (k4dShape.size() == 4 && k4dShape[3] != kvHeadDimForReshape) {
        CLLM_ERROR("[MultiHeadAttention] k4d last dim (%zu) != kvHeadDimForReshape (%zu)", k4dShape[3], kvHeadDimForReshape);
        throw std::runtime_error("MultiHeadAttention: k4d dimension mismatch");
    }
    
    // P2修复：移除容错逻辑 - Q 和 K 的 head_dim 必须一致（已在上面检查）
    // llama.cpp 要求 Q/K head_dim 与 n_rot 完全一致，否则直接失败
    // 由于我们已经确保 qHeadDim == kvHeadDim，所以 unifiedHeadDim == kvHeadDimForReshape
    if (unifiedHeadDim != kvHeadDimForReshape) {
        CLLM_ERROR("[MultiHeadAttention] ❌ unifiedHeadDim (%zu) != kvHeadDimForReshape (%zu)，内部状态不一致！",
                  unifiedHeadDim, kvHeadDimForReshape);
        throw std::runtime_error("Internal state inconsistency: unifiedHeadDim != kvHeadDimForReshape");
    }
    
    // Q 和 K 的 head_dim 相同，直接应用 RoPE
    rope_->apply(q4d, k4d, seqLen, 0);

    // 计算 attention（GQA 正确实现）
    // q4d: [B, n_head, S, head_dim]
    // k4d/v4d: [B, n_head_kv, S, head_dim]
    // 对每个 Q head，使用对应的 KV head（通过 gqa 映射）
    // P0修复：attention 输出的 head_dim 应该与 Q 的 head_dim 相同（用于后续的 wo 投影）
    // 注意：虽然标准的 attention 输出维度 = V 的维度，但 Qwen3 的 wo 权重期望输入是 qDim=2048
    // 所以我们需要将 V 的输出扩展到与 Q 相同的 head_dim
    // 方案：使用 Q 的 head_dim 作为输出维度，将 V 的输出投影/填充到该维度
    size_t attnOutHeadDim = unifiedHeadDim;  // 使用 Q 的 head_dim（与 wo 的输入维度匹配）
    Tensor attnOut4d({batch, numQHeads_, seqLen, attnOutHeadDim});

    for (size_t b = 0; b < batch; ++b) {
        for (size_t qh = 0; qh < numQHeads_; ++qh) {
            // GQA 映射：每个 Q head 对应一个 KV head
            size_t kvh = qh / gqa;  // kv_head = q_head / gqa
            
            // P2修复：Q 和 K 的 head_dim 必须一致（已在上面检查）
            // 由于 unifiedHeadDim == kvHeadDimForReshape，可以直接使用 unifiedHeadDim
            const float* qPtr = q4d.data() + ((b * numQHeads_ + qh) * seqLen * unifiedHeadDim);
            const float* kPtr = k4d.data() + ((b * numKVHeads_ + kvh) * seqLen * unifiedHeadDim);
            const float* vPtr = v4d.data() + ((b * numKVHeads_ + kvh) * seqLen * unifiedHeadDim);
            float* outPtr = attnOut4d.data() + ((b * numQHeads_ + qh) * seqLen * attnOutHeadDim);

            // scores: [S, S] = Q @ K^T
            // P2修复：Q 和 K 的 head_dim 必须一致，使用 unifiedHeadDim
            Tensor scores({seqLen, seqLen});
            for (size_t i = 0; i < seqLen; ++i) {
                for (size_t j = 0; j < seqLen; ++j) {
                    float sum = 0.0f;
                    for (size_t k = 0; k < unifiedHeadDim; ++k) {
                        sum += qPtr[i * unifiedHeadDim + k] * kPtr[j * unifiedHeadDim + k];
                    }
                    scores[i * seqLen + j] = sum;
                }
            }

            // scale by sqrt(head_dim)
            // P2修复：使用 unifiedHeadDim（Q 和 K 的 head_dim 相同）
            float scale = 1.0f / std::sqrt(static_cast<float>(unifiedHeadDim));
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

            // out = probs @ V  -> [S, head_dim]
            // P2修复：Q 和 KV 的 head_dim 必须一致，所以 attnOutHeadDim == unifiedHeadDim == kvHeadDimForReshape
            // 不需要扩展逻辑，直接计算
            if (vPtr == nullptr || outPtr == nullptr) {
                throw std::runtime_error("MultiHeadAttention: vPtr or outPtr is null");
            }
            
            // 验证维度一致性（P2 修复要求）
            if (attnOutHeadDim != unifiedHeadDim) {
                CLLM_ERROR("[MultiHeadAttention] ❌ attnOutHeadDim (%zu) != unifiedHeadDim (%zu)，内部状态不一致！",
                          attnOutHeadDim, unifiedHeadDim);
                throw std::runtime_error("Internal state inconsistency: attnOutHeadDim != unifiedHeadDim");
            }
            
            // 直接计算，不需要扩展
            CLLM_DEBUG("[MultiHeadAttention] matmul probs@V: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
                      seqLen, seqLen, seqLen, attnOutHeadDim, seqLen, attnOutHeadDim);
            matmul(probs.data(), vPtr, outPtr, seqLen, attnOutHeadDim, seqLen);
        }
    }

    // 合并 heads: [B, n_head, S, head_dim] -> [B, S, qDim]
    // GQA：attnOut4d 是 [B, n_head, S, attnOutHeadDim]，需要合并为 [B, S, qDim]
    // 注意：qDim 是实际的权重输出维度，应该等于 numQHeads * attnOutHeadDim
    Tensor merged({batch, seqLen, qDim});
    
    // 计算期望的输出维度（基于 attnOutHeadDim）
    size_t expectedQDim = numQHeads_ * attnOutHeadDim;
    if (qDim != expectedQDim) {
        CLLM_WARN("[MultiHeadAttention] qDim (%zu) != numQHeads * attnOutHeadDim (%zu * %zu = %zu), "
                 "this may indicate a model configuration mismatch", 
                 qDim, numQHeads_, attnOutHeadDim, expectedQDim);
    }
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seqLen; ++s) {
            size_t dstOffset = (b * seqLen + s) * qDim;
            
            for (size_t h = 0; h < numQHeads_; ++h) {
                // 从 attnOut4d 复制当前 Q head 的输出
                size_t srcOffset = ((b * numQHeads_ + h) * seqLen + s) * attnOutHeadDim;
                size_t headOffset = h * attnOutHeadDim;
                
                // 复制 head_dim 个元素（如果 headOffset + attnOutHeadDim 超出 qDim，只复制部分）
                size_t copyLen = std::min(attnOutHeadDim, qDim - headOffset);
                for (size_t d = 0; d < copyLen; ++d) {
                    merged[dstOffset + headOffset + d] = attnOut4d[srcOffset + d];
                }
            }
        }
    }

    // 输出投影: [B*S, qDim] @ [qDim, hidden] -> [B*S, hidden]
    CLLM_DEBUG("[MultiHeadAttention] Projecting output: [%zu, %zu] @ [%zu, %zu] -> [%zu, %zu]",
              rows, qDim, woShape[0], woShape[1], rows, hiddenSize_);
    if (wo_.data() == nullptr || merged.data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: wo_ or merged data is null");
    }
    Tensor out2d({rows, hiddenSize_});
    if (out2d.data() == nullptr) {
        throw std::runtime_error("MultiHeadAttention: out2d data is null");
    }
    matmul(merged.data(), wo_.data(), out2d.data(), rows, hiddenSize_, qDim);

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
