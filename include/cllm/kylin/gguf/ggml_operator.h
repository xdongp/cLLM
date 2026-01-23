/**
 * @file ggml_operator.h
 * @brief GGML 高性能算子实现
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 使用 GGML 库实现的高性能算子
 */
#pragma once

#include "cllm/kylin/gguf/operator_interface.h"
#include "cllm/kylin/gguf/context.h"

#include <memory>

namespace cllm {
namespace kylin {

/**
 * @brief GGML 高性能算子实现
 * 
 * 特点：
 * - 使用 GGML 库的优化实现
 * - 支持 SIMD 加速（AVX2/AVX-512/NEON）
 * - 支持量化计算
 * - 可选 GPU 加速（CUDA/Metal）
 * - 生产级性能
 */
class GGMLOperator : public IOperator {
public:
    /**
     * @brief 构造函数
     * @param backend 后端类型（CPU/CUDA/Metal）
     * @param memSize 上下文内存大小
     */
    explicit GGMLOperator(
        BackendType backend = BackendType::CPU,
        size_t memSize = 256 * 1024 * 1024  // 256MB 默认
    );
    
    ~GGMLOperator() override = default;
    
    std::string getName() const override { return "GGML"; }
    OperatorBackend getBackend() const override { return OperatorBackend::GGML; }
    
    // ========== 基础算子 ==========
    
    void matmul(
        const Tensor& A,
        const Tensor& B,
        Tensor& C,
        bool transposeA = false,
        bool transposeB = false
    ) override;
    
    void add(const Tensor& A, const Tensor& B, Tensor& C) override;
    
    void mul(const Tensor& A, const Tensor& B, Tensor& C) override;
    
    // ========== 激活函数 ==========
    
    void silu(const Tensor& input, Tensor& output) override;
    
    void softmax(const Tensor& input, Tensor& output, int axis = -1) override;
    
    // ========== 归一化 ==========
    
    void rmsNorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps = 1e-6f
    ) override;
    
    // ========== 位置编码 ==========
    
    void rope(
        Tensor& q,
        Tensor& k,
        size_t startPos,
        float freqBase = 10000.0f
    ) override;
    
    // ========== 复合算子优化 ==========
    
    void attention(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& output,
        float scale,
        bool causal = true
    ) override;
    
    // ========== GGML 特有方法 ==========
    
    /**
     * @brief 获取 GGML 上下文
     */
    GGMLContext* getContext() { return ctx_.get(); }
    
    /**
     * @brief 设置线程数
     */
    void setNumThreads(int nThreads) { nThreads_ = nThreads; }
    
    /**
     * @brief 获取线程数
     */
    int getNumThreads() const { return nThreads_; }

private:
    std::unique_ptr<GGMLContext> ctx_;    ///< GGML 上下文
    int nThreads_ = 4;                     ///< 计算线程数
    
    /**
     * @brief 将 Kylin Tensor 转换为 GGML 张量
     */
    ggml_tensor* toGGMLTensor(const Tensor& tensor);
    
    /**
     * @brief 将 GGML 张量数据复制到 Kylin Tensor
     */
    void fromGGMLTensor(ggml_tensor* src, Tensor& dst);
    
    /**
     * @brief 执行计算图
     */
    void computeGraph(ggml_tensor* output);
};

} // namespace kylin
} // namespace cllm
