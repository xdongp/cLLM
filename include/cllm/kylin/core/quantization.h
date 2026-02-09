/**
 * @file quantization.h
 * @brief 统一量化类型定义和接口
 * 
 * 支持多种精度：FP32、FP16、BF16、INT8
 * 可扩展架构，方便后续添加新量化类型
 */
#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include "cllm/kylin/backend/backend_interface.h"

namespace cllm {
namespace kylin {

// QuantType 已在 backend_interface.h 中定义

/**
 * @brief 获取量化类型名称
 */
inline const char* quantTypeName(QuantType type) {
    switch (type) {
        case QuantType::FP32: return "fp32";
        case QuantType::FP16: return "fp16";
        case QuantType::BF16: return "bf16";
        case QuantType::INT8: return "int8";
        case QuantType::Q4_K: return "q4_k";
        case QuantType::Q8_0: return "q8_0";
        default: return "unknown";
    }
}

/**
 * @brief 从字符串解析量化类型
 */
inline QuantType parseQuantType(const std::string& name) {
    if (name == "fp32" || name == "float32") return QuantType::FP32;
    if (name == "fp16" || name == "float16") return QuantType::FP16;
    if (name == "bf16" || name == "bfloat16") return QuantType::BF16;
    if (name == "int8") return QuantType::INT8;
    if (name == "q4_k" || name == "q4_k_m") return QuantType::Q4_K;
    if (name == "q8_0") return QuantType::Q8_0;
    return QuantType::FP32; // 默认
}

/**
 * @brief 获取每个元素的字节数
 */
inline size_t quantTypeBytes(QuantType type) {
    switch (type) {
        case QuantType::FP32: return 4;
        case QuantType::FP16: return 2;
        case QuantType::BF16: return 2;
        case QuantType::INT8: return 1;
        case QuantType::Q4_K: return 1; // 近似，实际是 4.5 bits
        case QuantType::Q8_0: return 1;
        default: return 4;
    }
}

/**
 * @brief 量化权重容器
 * 
 * 统一存储不同精度的权重数据
 */
class QuantizedWeight {
public:
    QuantizedWeight() = default;
    
    /**
     * @brief 从 FP32 数据创建量化权重
     */
    static QuantizedWeight fromFP32(const float* data, size_t count, QuantType targetType);
    
    /**
     * @brief 从 BF16 数据创建量化权重
     */
    static QuantizedWeight fromBF16(const uint16_t* data, size_t count, QuantType targetType);
    
    /**
     * @brief 获取量化类型
     */
    QuantType type() const { return type_; }
    
    /**
     * @brief 获取元素数量
     */
    size_t count() const { return count_; }
    
    /**
     * @brief 获取原始数据指针
     */
    const void* data() const { return data_.data(); }
    void* data() { return data_.data(); }
    
    /**
     * @brief 获取 FP32 数据指针（仅当 type == FP32）
     */
    const float* dataFP32() const { 
        return (type_ == QuantType::FP32) ? reinterpret_cast<const float*>(data_.data()) : nullptr;
    }
    
    /**
     * @brief 获取 FP16 数据指针（仅当 type == FP16）
     */
    const uint16_t* dataFP16() const {
        return (type_ == QuantType::FP16) ? reinterpret_cast<const uint16_t*>(data_.data()) : nullptr;
    }
    
    /**
     * @brief 获取 INT8 数据指针（仅当 type == INT8）
     */
    const int8_t* dataINT8() const {
        return (type_ == QuantType::INT8) ? reinterpret_cast<const int8_t*>(data_.data()) : nullptr;
    }
    
    /**
     * @brief INT8 量化的 scale 值
     */
    float scale() const { return scale_; }
    
    /**
     * @brief INT8 量化的 zero_point 值
     */
    int32_t zeroPoint() const { return zeroPoint_; }
    
    /**
     * @brief 是否为空
     */
    bool empty() const { return data_.empty(); }
    
    /**
     * @brief 获取内存占用（字节）
     */
    size_t memoryBytes() const { return data_.size(); }
    
private:
    QuantType type_ = QuantType::FP32;
    size_t count_ = 0;
    std::vector<uint8_t> data_;
    float scale_ = 1.0f;       // INT8 量化参数
    int32_t zeroPoint_ = 0;    // INT8 量化参数
};

namespace quant_kernels {

// ========== FP16 <==> FP32 转换 ==========

/**
 * @brief FP16 转 FP32（SIMD 优化）
 */
void convert_fp16_to_f32(const uint16_t* src, float* dst, size_t count);

/**
 * @brief FP32 转 FP16（SIMD 优化）
 */
void convert_f32_to_fp16(const float* src, uint16_t* dst, size_t count);

// ========== FP16 矩阵运算 ==========

/**
 * @brief FP16 权重 @ FP32 输入（SIMD 优化）
 * 
 * 权重保持 FP16，输入和输出是 FP32
 * 计算过程：FP16 -> FP32 -> 计算 -> FP32
 * 
 * @param weight FP16 权重矩阵 [M, K]，行主序
 * @param input  FP32 输入向量 [K]
 * @param output FP32 输出向量 [M]
 * @param M      输出维度
 * @param K      输入维度
 */
void matmul_fp16_f32(
    const uint16_t* weight,
    const float* input,
    float* output,
    int M, int K
);

// ========== INT8 矩阵运算 ==========

/**
 * @brief INT8 权重 @ FP32 输入
 * 
 * 使用 per-tensor 量化：output = (weight - zeroPoint) * scale @ input
 * 
 * @param weight INT8 权重矩阵 [M, K]
 * @param input  FP32 输入向量 [K]
 * @param output FP32 输出向量 [M]
 * @param M      输出维度
 * @param K      输入维度
 * @param scale  量化 scale
 * @param zeroPoint 量化 zero point
 */
void matmul_int8_f32(
    const int8_t* weight,
    const float* input,
    float* output,
    int M, int K,
    float scale,
    int32_t zeroPoint
);

// ========== 量化工具 ==========

/**
 * @brief 计算 FP32 数据的 INT8 量化参数
 * 
 * @param data FP32 数据
 * @param count 元素数量
 * @param scale 输出 scale
 * @param zeroPoint 输出 zero point
 */
void compute_int8_params(
    const float* data,
    size_t count,
    float& scale,
    int32_t& zeroPoint
);

/**
 * @brief FP32 量化为 INT8
 */
void quantize_f32_to_int8(
    const float* src,
    int8_t* dst,
    size_t count,
    float scale,
    int32_t zeroPoint
);

} // namespace quant_kernels
} // namespace kylin
} // namespace cllm
