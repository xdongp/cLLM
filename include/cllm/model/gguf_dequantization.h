#pragma once

#include <cstdint>
#include <vector>

namespace cllm {

// GGUF量化类型枚举
enum class GGUFQuantizationType {
    F16,    // 半精度浮点
    Q4_K_M, // 4位K_M量化
    Q5_K_M, // 5位K_M量化
    Q8_0,   // 8位0量化
    UNKNOWN // 未知类型
};

// F16到F32的反量化（SIMD优化）
void dequantizeF16ToF32(const uint16_t* input, float* output, size_t size);

// Q4_K_M到F32的反量化（SIMD优化）
void dequantizeQ4KMF32(const void* input, float* output, const std::vector<size_t>& shape);

// Q5_K_M到F32的反量化（SIMD优化）
void dequantizeQ5KMF32(const void* input, float* output, const std::vector<size_t>& shape);

// Q8_0到F32的反量化（SIMD优化）
void dequantizeQ8ToF32(const void* input, float* output, size_t size);

// 通用反量化接口
void dequantizeTensor(const void* input, float* output, GGUFQuantizationType type, const std::vector<size_t>& shape);

} // namespace cllm