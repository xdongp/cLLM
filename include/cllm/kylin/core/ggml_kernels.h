/**
 * @file ggml_kernels.h
 * @brief GGML 高性能计算内核封装
 * 
 * 利用 GGML 的 SIMD 优化（AVX/AVX2/AVX512/NEON）加速 HF 模型推理
 * 支持 Metal GPU 加速（macOS）
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include "cllm/kylin/backend/backend_interface.h"

namespace cllm {
namespace kylin {

namespace ggml_kernels {

/**
 * @brief 初始化 GGML 后端
 * 
 * @param device 设备类型（默认 CPU）
 * @return 是否初始化成功
 */
bool initialize(DeviceType device = DeviceType::CPU);

/**
 * @brief 关闭 GGML 后端
 */
void shutdown();

/**
 * @brief 获取当前设备类型
 */
DeviceType getDeviceType();

/**
 * @brief 检查 GPU 是否可用
 */
bool isGPUAvailable();

/**
 * @brief GPU 矩阵向量乘法
 * 
 * 使用 Metal/CUDA 进行 GPU 加速计算
 * 
 * @param weight F32 权重矩阵 [M, K]
 * @param input  F32 输入向量 [K]
 * @param output F32 输出向量 [M]
 * @param M      输出维度
 * @param K      输入维度
 */
void matmul_gpu(
    const float* weight,
    const float* input,
    float* output,
    int M, int K
);

/**
 * @brief BF16 矩阵向量乘法
 * 
 * output = weight @ input
 * 
 * @param weight BF16 权重矩阵 [M, K]，行主序
 * @param input  F32 输入向量 [K]
 * @param output F32 输出向量 [M]
 * @param M      输出维度
 * @param K      输入维度
 */
void matmul_bf16_f32(
    const uint16_t* weight,
    const float* input,
    float* output,
    int M, int K
);

/**
 * @brief F32 矩阵向量乘法（SIMD 优化）
 * 
 * @param weight F32 权重矩阵 [M, K]
 * @param input  F32 输入向量 [K]
 * @param output F32 输出向量 [M]
 * @param M      输出维度
 * @param K      输入维度
 */
void matmul_f32(
    const float* weight,
    const float* input,
    float* output,
    int M, int K
);

/**
 * @brief BF16 批量转 F32（SIMD 优化）
 * 
 * @param src BF16 源数据
 * @param dst F32 目标数据
 * @param count 元素数量
 */
void convert_bf16_to_f32(
    const uint16_t* src,
    float* dst,
    size_t count
);

/**
 * @brief F32 批量转 BF16
 * 
 * @param src F32 源数据
 * @param dst BF16 目标数据
 * @param count 元素数量
 */
void convert_f32_to_bf16(
    const float* src,
    uint16_t* dst,
    size_t count
);

/**
 * @brief RMS Norm（SIMD 优化）
 * 
 * output[i] = input[i] / rms(input) * weight[i]
 * 
 * @param input  输入向量 [size]
 * @param weight 权重向量 [size]
 * @param output 输出向量 [size]
 * @param size   向量大小
 * @param eps    数值稳定性 epsilon
 */
void rms_norm(
    const float* input,
    const float* weight,
    float* output,
    int size,
    float eps
);

/**
 * @brief 向量点积（SIMD 优化）
 * 
 * @param a 向量 a
 * @param b 向量 b
 * @param size 向量大小
 * @return 点积结果
 */
float dot_product(
    const float* a,
    const float* b,
    int size
);

/**
 * @brief SiLU 激活 + 逐元素乘法（SIMD 优化）
 * 
 * output[i] = silu(gate[i]) * up[i]
 * silu(x) = x * sigmoid(x)
 * 
 * @param gate  门控向量 [size]
 * @param up    上投影向量 [size]
 * @param output 输出向量 [size]
 * @param size  向量大小
 */
void silu_mul(
    const float* gate,
    const float* up,
    float* output,
    int size
);

/**
 * @brief SiLU 激活 + 逐元素乘法（原地操作，SIMD 优化）
 * 
 * 融合版本：gate_up 是连续存储的 [gate|up]
 * gate_up[0:size] = silu(gate_up[0:size]) * gate_up[size:2*size]
 * 
 * @param gate_up  连续的 [gate, up] 向量 [2*size]，结果写入 gate 部分
 * @param size     单个向量大小
 */
void silu_mul_fused(
    float* gate_up,
    int size
);

/**
 * @brief Softmax（数值稳定，SIMD 优化）
 * 
 * @param input  输入向量 [size]
 * @param output 输出向量 [size]
 * @param size   向量大小
 */
void softmax(
    const float* input,
    float* output,
    int size
);

/**
 * @brief 向量加法（SIMD 优化）
 * 
 * output[i] = a[i] + b[i]
 */
void vector_add(
    const float* a,
    const float* b,
    float* output,
    int size
);

/**
 * @brief 向量缩放加法
 * 
 * output[i] = a[i] + scale * b[i]
 */
void vector_scale_add(
    const float* a,
    const float* b,
    float* output,
    float scale,
    int size
);

/**
 * @brief 加权向量求和（Attention V weighted sum 专用）
 * 
 * output[d] = sum_t(weights[t] * vectors[t * stride + d])
 * 
 * @param weights  权重向量 [numVectors]
 * @param vectors  向量矩阵 [numVectors * stride]，每行一个向量
 * @param output   输出向量 [vectorSize]
 * @param numVectors 向量数量
 * @param vectorSize 每个向量的维度
 * @param stride     向量间步长（通常 = vectorSize，但可能更大）
 */
void weighted_sum(
    const float* weights,
    const float* vectors,
    float* output,
    int numVectors,
    int vectorSize,
    int stride
);

/**
 * @brief LM Head Top-K 优化矩阵向量乘法
 * 
 * 使用两阶段方法减少计算量：
 * 1. 粗采样：对每个块采样少量行，快速估计块内最大值
 * 2. 精确计算：只对 Top-K 候选块进行完整计算
 * 
 * @param weight F32 权重矩阵 [M, K]
 * @param input  F32 输入向量 [K]
 * @param output F32 输出向量 [M] (完整输出，非候选位置填 -INFINITY)
 * @param M      输出维度（词表大小）
 * @param K      输入维度（隐藏层大小）
 * @param topK   候选块数量（默认 32）
 * @return       是否使用了 Top-K 优化
 */
bool matmul_f32_topk(
    const float* weight,
    const float* input,
    float* output,
    int M, int K,
    int topK = 32
);

} // namespace ggml_kernels
} // namespace kylin
} // namespace cllm
