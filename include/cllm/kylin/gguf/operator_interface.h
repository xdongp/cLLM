/**
 * @file operator_interface.h
 * @brief Kylin 算子抽象接口
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 定义统一的算子接口，支持多种后端实现：
 * - Native: 自研 C++ 算子（学习/定制）
 * - GGML: 高性能 GGML 算子（生产）
 */
#pragma once

#include "cllm/kylin/core/tensor.h"
#include "cllm/kylin/gguf/context.h"  // 包含 BackendType 定义

#include <memory>
#include <string>

namespace cllm {
namespace kylin {

/**
 * @brief 算子后端类型
 */
enum class OperatorBackend {
    Native,   ///< 自研 C++ 算子
    GGML,     ///< GGML 高性能算子
    Auto      ///< 自动选择（优先 GGML，如果不可用则回退到 Native）
};

/**
 * @brief 算子接口基类
 * 
 * 定义 Transformer 所需的核心算子接口
 */
class IOperator {
public:
    virtual ~IOperator() = default;
    
    /**
     * @brief 获取算子后端名称
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief 获取算子后端类型
     */
    virtual OperatorBackend getBackend() const = 0;
    
    // ========== 基础算子 ==========
    
    /**
     * @brief 矩阵乘法: C = A @ B
     * @param A 输入矩阵 [M, K]
     * @param B 输入矩阵 [K, N]
     * @param C 输出矩阵 [M, N]
     * @param transposeA 是否转置 A
     * @param transposeB 是否转置 B
     */
    virtual void matmul(
        const Tensor& A,
        const Tensor& B,
        Tensor& C,
        bool transposeA = false,
        bool transposeB = false
    ) = 0;
    
    /**
     * @brief 向量加法: C = A + B
     */
    virtual void add(const Tensor& A, const Tensor& B, Tensor& C) = 0;
    
    /**
     * @brief 逐元素乘法: C = A * B
     */
    virtual void mul(const Tensor& A, const Tensor& B, Tensor& C) = 0;
    
    // ========== 激活函数 ==========
    
    /**
     * @brief SiLU 激活: y = x * sigmoid(x)
     */
    virtual void silu(const Tensor& input, Tensor& output) = 0;
    
    /**
     * @brief Softmax: y = softmax(x)
     * @param input 输入张量
     * @param output 输出张量
     * @param axis 沿哪个轴计算 softmax（默认最后一维）
     */
    virtual void softmax(const Tensor& input, Tensor& output, int axis = -1) = 0;
    
    // ========== 归一化 ==========
    
    /**
     * @brief RMS Normalization
     * @param input 输入张量 [batch, seq, hidden]
     * @param weight 权重 [hidden]
     * @param output 输出张量 [batch, seq, hidden]
     * @param eps epsilon 值
     */
    virtual void rmsNorm(
        const Tensor& input,
        const Tensor& weight,
        Tensor& output,
        float eps = 1e-6f
    ) = 0;
    
    // ========== 位置编码 ==========
    
    /**
     * @brief RoPE (Rotary Position Embedding)
     * @param q 查询张量 [batch, num_heads, seq, head_dim]
     * @param k 键张量 [batch, num_kv_heads, seq, head_dim]
     * @param startPos 起始位置（用于增量推理）
     * @param freqBase RoPE 频率基数
     */
    virtual void rope(
        Tensor& q,
        Tensor& k,
        size_t startPos,
        float freqBase = 10000.0f
    ) = 0;
    
    // ========== 复合算子（可选优化）==========
    
    /**
     * @brief 融合的 QKV 投影
     * @param input 输入 [batch, seq, hidden]
     * @param wq Q 权重 [hidden, q_dim]
     * @param wk K 权重 [hidden, kv_dim]
     * @param wv V 权重 [hidden, kv_dim]
     * @param q 输出 Q [batch, seq, q_dim]
     * @param k 输出 K [batch, seq, kv_dim]
     * @param v 输出 V [batch, seq, kv_dim]
     */
    virtual void qkvProject(
        const Tensor& input,
        const Tensor& wq,
        const Tensor& wk,
        const Tensor& wv,
        Tensor& q,
        Tensor& k,
        Tensor& v
    );
    
    /**
     * @brief 注意力计算
     * @param q 查询 [batch, num_heads, seq, head_dim]
     * @param k 键 [batch, num_kv_heads, ctx, head_dim]
     * @param v 值 [batch, num_kv_heads, ctx, head_dim]
     * @param output 输出 [batch, num_heads, seq, head_dim]
     * @param scale 缩放因子（通常为 1/sqrt(head_dim)）
     * @param causal 是否使用因果 mask
     */
    virtual void attention(
        const Tensor& q,
        const Tensor& k,
        const Tensor& v,
        Tensor& output,
        float scale,
        bool causal = true
    );
    
    /**
     * @brief SwiGLU FFN
     * @param input 输入 [batch, seq, hidden]
     * @param wGate gate 权重 [hidden, intermediate]
     * @param wUp up 权重 [hidden, intermediate]
     * @param wDown down 权重 [intermediate, hidden]
     * @param output 输出 [batch, seq, hidden]
     */
    virtual void swiGLU(
        const Tensor& input,
        const Tensor& wGate,
        const Tensor& wUp,
        const Tensor& wDown,
        Tensor& output
    );
};

/**
 * @brief 算子工厂
 * 
 * 根据配置创建相应的算子实现
 */
class OperatorFactory {
public:
    /**
     * @brief 创建算子实例
     * @param backend 算子后端类型
     * @param deviceBackend 设备后端类型（CPU/Metal/CUDA）
     * @return 算子实例
     */
    static std::unique_ptr<IOperator> create(
        OperatorBackend backend = OperatorBackend::Auto,
        BackendType deviceBackend = BackendType::CPU
    );
    
    /**
     * @brief 检查 GGML 算子是否可用
     */
    static bool isGGMLAvailable();
    
    /**
     * @brief 获取默认算子后端
     */
    static OperatorBackend getDefaultBackend();
    
    /**
     * @brief 设置默认算子后端
     */
    static void setDefaultBackend(OperatorBackend backend);

private:
    static OperatorBackend defaultBackend_;
};

} // namespace kylin
} // namespace cllm
