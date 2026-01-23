/**
 * @file ggml_operator.cpp
 * @brief GGML 高性能算子实现
 * 
 * 使用 GGML 库实现的高性能算子
 */

#ifdef CLLM_ENABLE_GGML

#include "cllm/kylin/gguf/ggml_operator.h"
#include "cllm/common/logger.h"

#include <cstring>
#include <stdexcept>

namespace cllm {
namespace kylin {

GGMLOperator::GGMLOperator(BackendType backend, size_t memSize) {
    ctx_ = std::make_unique<GGMLContext>(memSize, backend);
    
    // 设置默认线程数
    nThreads_ = static_cast<int>(std::thread::hardware_concurrency());
    if (nThreads_ <= 0) {
        nThreads_ = 4;
    }
    
    CLLM_INFO("[GGMLOperator] Initialized with backend=%d, memSize=%zu, nThreads=%d",
              static_cast<int>(backend), memSize, nThreads_);
}

ggml_tensor* GGMLOperator::toGGMLTensor(const Tensor& tensor) {
    const auto& shape = tensor.shape();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* t = nullptr;
    
    // GGML 使用 column-major 存储，需要反转维度
    switch (shape.size()) {
        case 1:
            t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, shape[0]);
            break;
        case 2:
            t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, shape[1], shape[0]);
            break;
        case 3:
            t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, shape[2], shape[1], shape[0]);
            break;
        case 4:
            t = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, shape[3], shape[2], shape[1], shape[0]);
            break;
        default:
            throw std::runtime_error("GGMLOperator: unsupported tensor dimension");
    }
    
    if (!t) {
        throw std::runtime_error("GGMLOperator: failed to create ggml_tensor");
    }
    
    // 复制数据
    std::memcpy(t->data, tensor.data(), tensor.size() * sizeof(float));
    
    return t;
}

void GGMLOperator::fromGGMLTensor(ggml_tensor* src, Tensor& dst) {
    // 计算元素数量
    size_t numElements = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (src->ne[i] > 0) {
            numElements *= src->ne[i];
        }
    }
    
    // 构建形状（反转 GGML 的维度顺序）
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        if (src->ne[i] > 1 || (i == 0 && shape.empty())) {
            shape.push_back(src->ne[i]);
        }
    }
    if (shape.empty()) {
        shape.push_back(1);
    }
    
    dst.resize(shape);
    std::memcpy(dst.data(), src->data, numElements * sizeof(float));
}

void GGMLOperator::computeGraph(ggml_tensor* output) {
    ggml_context* ctx = ctx_->raw();
    
    // 创建计算图
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    
    // 执行计算
    ggml_graph_compute_with_ctx(ctx, graph, nThreads_);
}

void GGMLOperator::matmul(
    const Tensor& A,
    const Tensor& B,
    Tensor& C,
    bool transposeA,
    bool transposeB
) {
    ctx_->reset();  // 重置上下文
    ggml_context* ctx = ctx_->raw();
    
    // 转换为 GGML 张量
    ggml_tensor* a = toGGMLTensor(A);
    ggml_tensor* b = toGGMLTensor(B);
    
    // 处理转置
    if (transposeA) {
        a = ggml_transpose(ctx, a);
    }
    if (transposeB) {
        b = ggml_transpose(ctx, b);
    }
    
    // 矩阵乘法
    ggml_tensor* c = ggml_mul_mat(ctx, a, b);
    
    // 计算
    computeGraph(c);
    
    // 复制结果
    fromGGMLTensor(c, C);
}

void GGMLOperator::add(const Tensor& A, const Tensor& B, Tensor& C) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* a = toGGMLTensor(A);
    ggml_tensor* b = toGGMLTensor(B);
    
    ggml_tensor* c = ggml_add(ctx, a, b);
    
    computeGraph(c);
    fromGGMLTensor(c, C);
}

void GGMLOperator::mul(const Tensor& A, const Tensor& B, Tensor& C) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* a = toGGMLTensor(A);
    ggml_tensor* b = toGGMLTensor(B);
    
    ggml_tensor* c = ggml_mul(ctx, a, b);
    
    computeGraph(c);
    fromGGMLTensor(c, C);
}

void GGMLOperator::silu(const Tensor& input, Tensor& output) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* x = toGGMLTensor(input);
    
    // GGML 的 silu 实现
    ggml_tensor* y = ggml_silu(ctx, x);
    
    computeGraph(y);
    fromGGMLTensor(y, output);
}

void GGMLOperator::softmax(const Tensor& input, Tensor& output, int axis) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* x = toGGMLTensor(input);
    
    // GGML softmax（沿最后一维）
    ggml_tensor* y = ggml_soft_max(ctx, x);
    
    computeGraph(y);
    fromGGMLTensor(y, output);
}

void GGMLOperator::rmsNorm(
    const Tensor& input,
    const Tensor& weight,
    Tensor& output,
    float eps
) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* x = toGGMLTensor(input);
    ggml_tensor* w = toGGMLTensor(weight);
    
    // GGML RMS norm
    ggml_tensor* y = ggml_rms_norm(ctx, x, eps);
    
    // 乘以权重
    y = ggml_mul(ctx, y, w);
    
    computeGraph(y);
    fromGGMLTensor(y, output);
}

void GGMLOperator::rope(
    Tensor& q,
    Tensor& k,
    size_t startPos,
    float freqBase
) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* gq = toGGMLTensor(q);
    ggml_tensor* gk = toGGMLTensor(k);
    
    const auto& qShape = q.shape();  // [batch, heads, seq, dim]
    
    size_t seqLen = qShape[2];
    size_t headDim = qShape[3];
    
    // 创建位置索引
    ggml_tensor* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seqLen);
    int32_t* posData = static_cast<int32_t*>(positions->data);
    for (size_t i = 0; i < seqLen; ++i) {
        posData[i] = static_cast<int32_t>(startPos + i);
    }
    
    // 应用 RoPE
    // 注意：GGML 的 rope 接口可能与这里的签名不同
    // 这里使用简化的实现
    gq = ggml_rope(ctx, gq, positions, static_cast<int>(headDim), 0);
    gk = ggml_rope(ctx, gk, positions, static_cast<int>(headDim), 0);
    
    // 构建计算图并执行
    struct ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, gq);
    ggml_build_forward_expand(graph, gk);
    ggml_graph_compute_with_ctx(ctx, graph, nThreads_);
    
    // 复制结果回原张量
    fromGGMLTensor(gq, q);
    fromGGMLTensor(gk, k);
}

void GGMLOperator::attention(
    const Tensor& query,
    const Tensor& key,
    const Tensor& value,
    Tensor& output,
    float scale,
    bool causal
) {
    ctx_->reset();
    ggml_context* ctx = ctx_->raw();
    
    ggml_tensor* q = toGGMLTensor(query);
    ggml_tensor* k = toGGMLTensor(key);
    ggml_tensor* v = toGGMLTensor(value);
    
    // 使用 GGML 的 flash attention 或标准注意力
    // 注意：具体 API 取决于 GGML 版本
    
    // 标准实现：scores = Q @ K^T, attn = softmax(scores * scale), output = attn @ V
    
    // Q @ K^T
    ggml_tensor* kT = ggml_transpose(ctx, k);
    ggml_tensor* scores = ggml_mul_mat(ctx, q, kT);
    
    // 缩放
    scores = ggml_scale(ctx, scores, scale);
    
    // 因果 mask（如果需要）
    if (causal) {
        // GGML 的 diag_mask_inf
        scores = ggml_diag_mask_inf(ctx, scores, 0);
    }
    
    // Softmax
    ggml_tensor* attn = ggml_soft_max(ctx, scores);
    
    // attn @ V
    ggml_tensor* out = ggml_mul_mat(ctx, attn, v);
    
    computeGraph(out);
    fromGGMLTensor(out, output);
}

} // namespace kylin
} // namespace cllm

#endif // CLLM_ENABLE_GGML
