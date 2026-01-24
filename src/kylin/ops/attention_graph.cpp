/**
 * @file attention_graph.cpp
 * @brief GGML Attention 计算图构建器实现
 */

#include "cllm/kylin/ops/attention_graph.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <cstring>

namespace cllm {
namespace kylin {

AttentionGraphBuilder::AttentionGraphBuilder(const AttentionConfig& config)
    : config_(config) {
}

AttentionResult AttentionGraphBuilder::build(
    ggml_context* ctx,
    ggml_tensor* input,
    const AttentionWeights& weights,
    const KVCacheRef& kvCache,
    size_t startPos,
    size_t seqLen,
    size_t layerIdx
) {
    AttentionResult result;
    const size_t totalLen = startPos + seqLen;
    
    // 清空调试节点
    debugNodes_ = AttentionDebugNodes{};
    
    // 1. QKV 投影
    ggml_tensor* q = nullptr;
    ggml_tensor* kNew = nullptr;
    ggml_tensor* vNew = nullptr;
    buildQKVProjection(ctx, input, weights, q, kNew, vNew, seqLen);
    
    // 2. Q/K 归一化（可选）
    applyQKNorm(ctx, q, kNew, weights, layerIdx);
    
    // 3. RoPE 位置编码
    applyRoPE(ctx, q, kNew, startPos, seqLen, layerIdx);
    
    // 4. 转置 K/V 为 cache 格式
    // 从 [head_dim, n_kv_heads, seq_len] 到 [head_dim, seq_len, n_kv_heads]
    kNew = ggml_cont(ctx, ggml_permute(ctx, kNew, 0, 2, 1, 3));
    vNew = ggml_cont(ctx, ggml_permute(ctx, vNew, 0, 2, 1, 3));
    
    // 保存用于写入 cache
    result.kNew = kNew;
    result.vNew = vNew;
    
    // 5. 构建完整的 KV 序列
    ggml_tensor* kFull = nullptr;
    ggml_tensor* vFull = nullptr;
    buildFullKV(ctx, kNew, vNew, kvCache, startPos, seqLen, kFull, vFull, layerIdx);
    
    // 6. 计算注意力输出
    result.output = computeAttention(ctx, q, kFull, vFull, weights, startPos, seqLen, totalLen, layerIdx);
    
    return result;
}

void AttentionGraphBuilder::buildQKVProjection(
    ggml_context* ctx,
    ggml_tensor* input,
    const AttentionWeights& weights,
    ggml_tensor*& q,
    ggml_tensor*& kNew,
    ggml_tensor*& vNew,
    size_t seqLen
) {
    // input: [hidden, seq_len]
    // wq: [qDim, hidden] => [qDim, seq_len]
    q = ggml_mul_mat(ctx, weights.wq, input);
    kNew = ggml_mul_mat(ctx, weights.wk, input);
    vNew = ggml_mul_mat(ctx, weights.wv, input);
    
    // 保存 QKV 投影输出用于调试
    debugNodes_.qkvOutput = q;
    
    // 获取实际维度
    size_t qDim = q->ne[0];
    size_t kvDim = kNew->ne[0];
    size_t actualHeadDim = qDim / config_.nHeads;
    size_t kvHeadDim = kvDim / config_.nKVHeads;
    
    // 重塑为多头格式: [headDim, nHeads, seqLen]
    q = ggml_reshape_3d(ctx, q, actualHeadDim, config_.nHeads, seqLen);
    kNew = ggml_reshape_3d(ctx, kNew, kvHeadDim, config_.nKVHeads, seqLen);
    vNew = ggml_reshape_3d(ctx, vNew, kvHeadDim, config_.nKVHeads, seqLen);
}

void AttentionGraphBuilder::applyQKNorm(
    ggml_context* ctx,
    ggml_tensor*& q,
    ggml_tensor*& kNew,
    const AttentionWeights& weights,
    size_t layerIdx
) {
    size_t actualHeadDim = q->ne[0];
    size_t kvHeadDim = kNew->ne[0];
    
    if (weights.attnQNorm) {
        size_t qNormDim = weights.attnQNorm->ne[0];
        if (qNormDim == actualHeadDim) {
            // 保存 RMS Norm 前的 Q 值用于调试
            ggml_tensor* qBeforeNorm = q;
            
            q = ggml_rms_norm(ctx, q, config_.rmsNormEps);
            
            // 调试：打印 RMS Norm 后、乘法前的 Q 形状
            if (layerIdx == 0) {
                CLLM_DEBUG("[Attention L%zu] Q before mul: shape=[%lld, %lld, %lld], norm_weight shape=[%lld]",
                           layerIdx, q->ne[0], q->ne[1], q->ne[2], weights.attnQNorm->ne[0]);
            }
            
            q = ggml_mul(ctx, q, weights.attnQNorm);
            debugNodes_.qNormOutput = q;
            CLLM_DEBUG("[Attention L%zu] Applied Q norm (head_dim=%zu)", layerIdx, actualHeadDim);
        } else {
            CLLM_WARN("[Attention L%zu] Q norm size (%zu) != head_dim (%zu), skipping",
                     layerIdx, qNormDim, actualHeadDim);
        }
    }
    
    if (weights.attnKNorm) {
        size_t kNormDim = weights.attnKNorm->ne[0];
        if (kNormDim == kvHeadDim) {
            // 调试：保存 RMS Norm 前的 K
            if (layerIdx == 0) {
                debugNodes_.kBeforeNorm = kNew;  // 保存用于调试
                CLLM_DEBUG("[Attention L%zu] K before rms_norm: shape=[%lld, %lld, %lld]",
                           layerIdx, kNew->ne[0], kNew->ne[1], kNew->ne[2]);
            }
            
            kNew = ggml_rms_norm(ctx, kNew, config_.rmsNormEps);
            
            // 保存 RMS Norm 后、乘法前的 K
            if (layerIdx == 0) {
                debugNodes_.kAfterRmsNorm = kNew;  // 保存用于调试
                CLLM_DEBUG("[Attention L%zu] K after rms_norm: shape=[%lld, %lld, %lld]",
                           layerIdx, kNew->ne[0], kNew->ne[1], kNew->ne[2]);
            }
            
            kNew = ggml_mul(ctx, kNew, weights.attnKNorm);
            debugNodes_.kNormOutput = kNew;
            CLLM_DEBUG("[Attention L%zu] Applied K norm (head_dim=%zu)", layerIdx, kvHeadDim);
        } else {
            CLLM_WARN("[Attention L%zu] K norm size (%zu) != kv_head_dim (%zu), skipping",
                     layerIdx, kNormDim, kvHeadDim);
        }
    }
}

void AttentionGraphBuilder::applyRoPE(
    ggml_context* ctx,
    ggml_tensor*& q,
    ggml_tensor*& kNew,
    size_t startPos,
    size_t seqLen,
    size_t layerIdx
) {
    size_t actualHeadDim = q->ne[0];
    
    // 创建位置张量
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
    int32_t* posData = static_cast<int32_t*>(positions->data);
    for (size_t i = 0; i < seqLen; ++i) {
        posData[i] = static_cast<int32_t>(startPos + i);
    }
    
    // RoPE 参数 - 使用配置中的 ropeType
    // GGML_ROPE_TYPE_NORMAL = 0, GGML_ROPE_TYPE_NEOX = 2
    const int ropeMode = config_.ropeType;
    const int nCtxOrig = static_cast<int>(config_.contextLength);
    const float freqBase = config_.ropeFreqBase;
    const float freqScale = 1.0f;
    const float extFactor = 0.0f;
    const float attnFactor = 1.0f;
    const float betaFast = 32.0f;
    const float betaSlow = 1.0f;
    const int nRot = static_cast<int>(actualHeadDim);
    
    if (layerIdx == 0) {
        CLLM_DEBUG("[Attention L%zu] RoPE: type=%d, freq_base=%.0f, n_rot=%d, head_dim=%zu",
                   layerIdx, ropeMode, freqBase, nRot, actualHeadDim);
    }
    
    // 应用 RoPE
    q = ggml_rope_ext(ctx, q, positions, nullptr,
        nRot, ropeMode, nCtxOrig, freqBase, freqScale,
        extFactor, attnFactor, betaFast, betaSlow);
    
    kNew = ggml_rope_ext(ctx, kNew, positions, nullptr,
        nRot, ropeMode, nCtxOrig, freqBase, freqScale,
        extFactor, attnFactor, betaFast, betaSlow);
    
    debugNodes_.ropeQOutput = q;
    debugNodes_.ropeKOutput = kNew;
}

void AttentionGraphBuilder::buildFullKV(
    ggml_context* ctx,
    ggml_tensor* kNew,
    ggml_tensor* vNew,
    const KVCacheRef& kvCache,
    size_t startPos,
    size_t seqLen,
    ggml_tensor*& kFull,
    ggml_tensor*& vFull,
    size_t layerIdx
) {
    const size_t totalLen = startPos + seqLen;
    const size_t kvHeadDim = kNew->ne[0];
    const size_t nKVHeads = config_.nKVHeads;
    
    if (startPos == 0) {
        // 首次推理：直接使用新的 K/V
        kFull = kNew;
        vFull = vNew;
        CLLM_DEBUG("[Attention L%zu] First inference: seqLen=%zu", layerIdx, seqLen);
    } else {
        // 增量推理：从 cache 读取历史并连接
        ggml_tensor* kCache = kvCache.kCache;
        ggml_tensor* vCache = kvCache.vCache;
        
        if (!kCache || !vCache) {
            CLLM_ERROR("[Attention L%zu] KV cache is null", layerIdx);
            kFull = kNew;
            vFull = vNew;
            return;
        }
        
        const size_t cacheNb1 = kCache->nb[1];
        const size_t cacheNb2 = kCache->nb[2];
        
        // 读取历史 K/V
        ggml_tensor* kHist = ggml_view_3d(ctx, kCache,
            kvHeadDim, startPos, nKVHeads, cacheNb1, cacheNb2, 0);
        ggml_tensor* vHist = ggml_view_3d(ctx, vCache,
            kvHeadDim, startPos, nKVHeads, cacheNb1, cacheNb2, 0);
        
        // 检查连续性
        const size_t expectedNb2 = kvHeadDim * startPos * sizeof(float);
        if (kHist->nb[2] != expectedNb2) {
            kHist = ggml_cont(ctx, kHist);
        }
        if (vHist->nb[2] != expectedNb2) {
            vHist = ggml_cont(ctx, vHist);
        }
        
        // 连接历史和新数据
        kFull = ggml_concat(ctx, kHist, kNew, 1);
        vFull = ggml_concat(ctx, vHist, vNew, 1);
        
        CLLM_DEBUG("[Attention L%zu] Incremental: history=%zu + new=%zu = total=%zu",
                   layerIdx, startPos, seqLen, totalLen);
    }
}

ggml_tensor* AttentionGraphBuilder::computeAttention(
    ggml_context* ctx,
    ggml_tensor* q,
    ggml_tensor* kFull,
    ggml_tensor* vFull,
    const AttentionWeights& weights,
    size_t startPos,
    size_t seqLen,
    size_t totalLen,
    size_t layerIdx
) {
    const size_t nHeads = config_.nHeads;
    const size_t nKVHeads = config_.nKVHeads;
    const size_t actualHeadDim = q->ne[0];
    const size_t kvHeadDim = kFull->ne[0];
    
    const float scale = 1.0f / std::sqrt(static_cast<float>(actualHeadDim));
    
    // Q 调整维度: [headDim, nHeads, seqLen] -> [headDim, seqLen, nHeads]
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    
    // GQA 扩展
    ggml_tensor* kExpanded = kFull;
    ggml_tensor* vExpanded = vFull;
    
    if (nKVHeads < nHeads) {
        size_t gqa = nHeads / nKVHeads;
        
        ggml_tensor* k4d = ggml_reshape_4d(ctx, kFull, kvHeadDim, totalLen, 1, nKVHeads);
        ggml_tensor* v4d = ggml_reshape_4d(ctx, vFull, kvHeadDim, totalLen, 1, nKVHeads);
        
        ggml_tensor* kTarget = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kvHeadDim, totalLen, gqa, nKVHeads);
        ggml_tensor* vTarget = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, kvHeadDim, totalLen, gqa, nKVHeads);
        
        k4d = ggml_repeat(ctx, k4d, kTarget);
        v4d = ggml_repeat(ctx, v4d, vTarget);
        
        k4d = ggml_cont(ctx, ggml_permute(ctx, k4d, 0, 1, 3, 2));
        v4d = ggml_cont(ctx, ggml_permute(ctx, v4d, 0, 1, 3, 2));
        
        kExpanded = ggml_reshape_3d(ctx, k4d, kvHeadDim, totalLen, nHeads);
        vExpanded = ggml_reshape_3d(ctx, v4d, kvHeadDim, totalLen, nHeads);
        
        CLLM_DEBUG("[Attention L%zu] GQA: %zu KV heads -> %zu Q heads (gqa=%zu)",
                   layerIdx, nKVHeads, nHeads, gqa);
    }
    
    // 计算注意力分数
    // 关键：Q@K 矩阵乘法需要 F32 精度以避免数值不稳定
    // 参考 llama.cpp: "this op tends to require high floating point range"
    ggml_tensor* scores = ggml_mul_mat(ctx, kExpanded, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, scale);
    
    // 因果 mask
    if (layerIdx == 0) {
        CLLM_DEBUG("[Stage8 Debug L%zu] Causal mask: n_past=%zu, scores shape: [%lld, %lld, %lld]",
                   layerIdx, startPos, scores->ne[0], scores->ne[1], scores->ne[2]);
    }
    scores = ggml_diag_mask_inf(ctx, scores, static_cast<int>(startPos));
    
    // Softmax
    ggml_tensor* attnWeights = ggml_soft_max(ctx, scores);
    
    // V 转置: [headDim, totalLen, nHeads] -> [totalLen, headDim, nHeads]
    ggml_tensor* vT = ggml_cont(ctx, ggml_permute(ctx, vExpanded, 1, 0, 2, 3));
    
    // Attention @ V
    ggml_tensor* attnOutput = ggml_mul_mat(ctx, vT, attnWeights);
    
    // 重塑输出: [headDim, seqLen, nHeads] -> [headDim*nHeads, seqLen]
    attnOutput = ggml_permute(ctx, attnOutput, 0, 2, 1, 3);
    attnOutput = ggml_cont(ctx, attnOutput);
    attnOutput = ggml_reshape_2d(ctx, attnOutput, actualHeadDim * nHeads, seqLen);
    
    // 输出投影
    ggml_tensor* output = ggml_mul_mat(ctx, weights.wo, attnOutput);
    
    return output;
}

} // namespace kylin
} // namespace cllm
