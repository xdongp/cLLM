/**
 * @file operator_interface.cpp
 * @brief 算子接口基类和工厂实现
 */

#include "cllm/kylin/gguf/operator_interface.h"
#include "cllm/kylin/gguf/native_operator.h"
#include "cllm/common/logger.h"

#include <cstring>

// 条件编译 GGML 算子
#ifdef CLLM_ENABLE_GGML
#include "cllm/kylin/gguf/ggml_operator.h"
#endif

namespace cllm {
namespace kylin {

// ========== IOperator 默认实现（复合算子）==========

void IOperator::qkvProject(
    const Tensor& input,
    const Tensor& wq,
    const Tensor& wk,
    const Tensor& wv,
    Tensor& q,
    Tensor& k,
    Tensor& v
) {
    // 默认实现：三次独立的 matmul
    matmul(input, wq, q);
    matmul(input, wk, k);
    matmul(input, wv, v);
}

void IOperator::attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output,
    float scale,
    bool causal
) {
    // 默认实现：标准注意力计算
    // scores = Q @ K^T / sqrt(d_k)
    // attn = softmax(scores)
    // output = attn @ V
    
    const auto& qShape = query.shape();  // [batch, num_heads, seq_q, head_dim]
    const auto& kShape = key.shape();    // [batch, num_kv_heads, seq_k, head_dim]
    const auto& vShape = value.shape();  // [batch, num_kv_heads, seq_k, head_dim]
    
    if (qShape.size() != 4 || kShape.size() != 4 || vShape.size() != 4) {
        throw std::runtime_error("attention: expected 4D tensors");
    }
    
    const size_t batch = qShape[0];
    const size_t numHeads = qShape[1];
    const size_t seqQ = qShape[2];
    const size_t headDim = qShape[3];
    const size_t seqK = kShape[2];
    
    // 计算 scores = Q @ K^T
    Tensor scores({batch, numHeads, seqQ, seqK});
    
    // 批量处理每个 head
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numHeads; ++h) {
            // 获取当前 head 的 Q, K
            // Q: [seqQ, headDim], K: [seqK, headDim] -> K^T: [headDim, seqK]
            // scores = Q @ K^T = [seqQ, seqK]
            
            size_t qOffset = (b * numHeads + h) * seqQ * headDim;
            size_t kOffset = (b * numHeads + h) * seqK * headDim;  // 简化：假设 GQA 已处理
            size_t sOffset = (b * numHeads + h) * seqQ * seqK;
            
            // 简单的 matmul：Q @ K^T
            for (size_t i = 0; i < seqQ; ++i) {
                for (size_t j = 0; j < seqK; ++j) {
                    float sum = 0.0f;
                    for (size_t d = 0; d < headDim; ++d) {
                        sum += query.data()[qOffset + i * headDim + d] *
                               key.data()[kOffset + j * headDim + d];
                    }
                    scores.data()[sOffset + i * seqK + j] = sum * scale;
                    
                    // 因果 mask
                    if (causal && j > i) {
                        scores.data()[sOffset + i * seqK + j] = -1e9f;
                    }
                }
            }
        }
    }
    
    // 应用 softmax
    Tensor attnWeights({batch, numHeads, seqQ, seqK});
    softmax(scores, attnWeights, -1);
    
    // 计算 output = attn @ V
    output.resize({batch, numHeads, seqQ, headDim});
    
    for (size_t b = 0; b < batch; ++b) {
        for (size_t h = 0; h < numHeads; ++h) {
            size_t aOffset = (b * numHeads + h) * seqQ * seqK;
            size_t vOffset = (b * numHeads + h) * seqK * headDim;
            size_t oOffset = (b * numHeads + h) * seqQ * headDim;
            
            // attn @ V
            for (size_t i = 0; i < seqQ; ++i) {
                for (size_t d = 0; d < headDim; ++d) {
                    float sum = 0.0f;
                    for (size_t j = 0; j < seqK; ++j) {
                        sum += attnWeights.data()[aOffset + i * seqK + j] *
                               value.data()[vOffset + j * headDim + d];
                    }
                    output.data()[oOffset + i * headDim + d] = sum;
                }
            }
        }
    }
}

void IOperator::swiGLU(
    const Tensor& input,
    const Tensor& wGate,
    const Tensor& wUp,
    const Tensor& wDown,
    Tensor& output
) {
    // 默认实现：分步计算
    // gate = input @ wGate
    // up = input @ wUp
    // hidden = silu(gate) * up
    // output = hidden @ wDown
    
    const auto& inputShape = input.shape();
    const auto& gateShape = wGate.shape();
    
    size_t batchSeq = 1;
    for (size_t i = 0; i < inputShape.size() - 1; ++i) {
        batchSeq *= inputShape[i];
    }
    size_t hiddenSize = inputShape.back();
    size_t intermediateSize = gateShape.back();
    
    // 展平输入（直接使用，Tensor 已经是扁平的）
    Tensor flatInput({batchSeq, hiddenSize});
    std::memcpy(flatInput.data(), input.data(), input.size() * sizeof(float));
    
    // gate projection
    Tensor gate({batchSeq, intermediateSize});
    matmul(flatInput, wGate, gate);
    
    // up projection
    Tensor up({batchSeq, intermediateSize});
    matmul(flatInput, wUp, up);
    
    // silu(gate) * up
    Tensor siluGate({batchSeq, intermediateSize});
    silu(gate, siluGate);
    
    Tensor hidden({batchSeq, intermediateSize});
    mul(siluGate, up, hidden);
    
    // down projection
    Tensor flatOutput({batchSeq, hiddenSize});
    matmul(hidden, wDown, flatOutput);
    
    // 恢复形状
    output.resize(inputShape);
    std::memcpy(output.data(), flatOutput.data(), flatOutput.size() * sizeof(float));
}

// ========== 算子工厂 ==========

OperatorBackend OperatorFactory::defaultBackend_ = OperatorBackend::Auto;

std::unique_ptr<IOperator> OperatorFactory::create(OperatorBackend backend, BackendType deviceBackend) {
    if (backend == OperatorBackend::Auto) {
        backend = getDefaultBackend();
    }
    
    switch (backend) {
        case OperatorBackend::Native:
            CLLM_INFO("[OperatorFactory] Creating Native operator");
            return std::make_unique<NativeOperator>();
            
#ifdef CLLM_ENABLE_GGML
        case OperatorBackend::GGML:
            CLLM_INFO("[OperatorFactory] Creating GGML operator with device backend: %s",
                     deviceBackend == BackendType::Metal ? "Metal" :
                     deviceBackend == BackendType::CUDA ? "CUDA" : "CPU");
            return std::make_unique<GGMLOperator>(deviceBackend);
#endif
            
        case OperatorBackend::Auto:
        default:
            // Auto 或未知类型，回退到 Native
            CLLM_INFO("[OperatorFactory] Fallback to Native operator");
            return std::make_unique<NativeOperator>();
    }
}

bool OperatorFactory::isGGMLAvailable() {
#ifdef CLLM_ENABLE_GGML
    return true;
#else
    return false;
#endif
}

OperatorBackend OperatorFactory::getDefaultBackend() {
    if (defaultBackend_ == OperatorBackend::Auto) {
        // 自动选择：优先 GGML
#ifdef CLLM_ENABLE_GGML
        return OperatorBackend::GGML;
#else
        return OperatorBackend::Native;
#endif
    }
    return defaultBackend_;
}

void OperatorFactory::setDefaultBackend(OperatorBackend backend) {
    defaultBackend_ = backend;
    CLLM_INFO("[OperatorFactory] Default backend set to: %s",
              backend == OperatorBackend::Native ? "Native" :
              backend == OperatorBackend::GGML ? "GGML" : "Auto");
}

} // namespace kylin
} // namespace cllm
