/**
 * @file native_operator.h
 * @brief 自研 Native 算子实现
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 使用纯 C++ 实现的算子，适合学习和定制
 */
#pragma once

#include "cllm/kylin/gguf/operator_interface.h"

namespace cllm {
namespace kylin {

/**
 * @brief 自研 Native 算子实现
 * 
 * 特点：
 * - 纯 C++ 实现，无外部依赖
 * - 代码清晰，易于理解和修改
 * - 可作为学习 Transformer 的参考实现
 * - 性能中等，适合原型验证
 */
class NativeOperator : public IOperator {
public:
    NativeOperator() = default;
    ~NativeOperator() override = default;
    
    std::string getName() const override { return "Native"; }
    OperatorBackend getBackend() const override { return OperatorBackend::Native; }
    
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
};

} // namespace kylin
} // namespace cllm
