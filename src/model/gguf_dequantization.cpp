#include "cllm/model/gguf_dequantization.h"
#include "cllm/common/logger.h"
#include <cstring>
#include <stdexcept>

// SIMD指令集检测
#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

namespace cllm {

// F16到F32的反量化（标量实现，作为后备）
void dequantizeF16ToF32Scalar(const uint16_t* input, float* output, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        uint16_t f16 = input[i];
        uint32_t sign = (f16 >> 15) & 1;
        uint32_t exponent = (f16 >> 10) & 0x1F;
        uint32_t mantissa = f16 & 0x3FF;
        
        uint32_t f32;
        if (exponent == 0) {
            if (mantissa == 0) {
                // 零
                f32 = sign << 31;
            } else {
                // 非规格化数
                exponent = 1;
                while ((mantissa & 0x400) == 0) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3FF;
                exponent += 127 - 15 - 10;
                f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
            }
        } else if (exponent == 0x1F) {
            // 无穷大或NaN
            f32 = (sign << 31) | (0xFF << 23) | (mantissa << 13);
        } else {
            // 规格化数
            exponent += 127 - 15;
            f32 = (sign << 31) | (exponent << 23) | (mantissa << 13);
        }
        
        memcpy(output + i, &f32, sizeof(float));
    }
}

// F16到F32的反量化（SSE2优化）
#ifdef __SSE2__
void dequantizeF16ToF32SSE2(const uint16_t* input, float* output, size_t size) {
    size_t i = 0;
    
    // 处理多个8元素批次
    for (; i + 8 <= size; i += 8) {
        // 加载16个字节（8个F16值）
        __m128i f16_lo = _mm_load_si128(reinterpret_cast<const __m128i*>(input + i));
        __m128i f16_hi = _mm_load_si128(reinterpret_cast<const __m128i*>(input + i + 4));
        
        // 转换为F32
        __m128 f32_lo = _mm_cvtph_ps(f16_lo);
        __m128 f32_hi = _mm_cvtph_ps(f16_hi);
        
        // 存储结果
        _mm_store_ps(output + i, f32_lo);
        _mm_store_ps(output + i + 4, f32_hi);
    }
    
    // 处理剩余元素
    if (i < size) {
        dequantizeF16ToF32Scalar(input + i, output + i, size - i);
    }
}
#endif

// F16到F32的反量化（AVX2优化）
#ifdef __AVX2__
void dequantizeF16ToF32AVX2(const uint16_t* input, float* output, size_t size) {
    size_t i = 0;
    
    // 处理多个16元素批次
    for (; i + 16 <= size; i += 16) {
        // 加载32个字节（16个F16值）
        __m256i f16_lo = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i));
        __m256i f16_hi = _mm256_load_si256(reinterpret_cast<const __m256i*>(input + i + 8));
        
        // 转换为F32
        __m256 f32_lo = _mm256_cvtph_ps(f16_lo);
        __m256 f32_hi = _mm256_cvtph_ps(f16_hi);
        
        // 存储结果
        _mm256_store_ps(output + i, f32_lo);
        _mm256_store_ps(output + i + 8, f32_hi);
    }
    
    // 处理剩余元素
    if (i < size) {
        dequantizeF16ToF32Scalar(input + i, output + i, size - i);
    }
}
#endif

// F16到F32的反量化（NEON优化，ARM平台）
#ifdef __ARM_NEON
void dequantizeF16ToF32NEON(const uint16_t* input, float* output, size_t size) {
    size_t i = 0;
    
    // 处理多个4元素批次
    for (; i + 4 <= size; i += 4) {
        // 加载8个字节（4个F16值）
        float16x4_t f16 = vld1_f16(reinterpret_cast<const float16_t*>(input + i));
        
        // 转换为F32
        float32x4_t f32 = vcvt_f32_f16(f16);
        
        // 存储结果
        vst1q_f32(output + i, f32);
    }
    
    // 处理剩余元素
    if (i < size) {
        dequantizeF16ToF32Scalar(input + i, output + i, size - i);
    }
}
#endif

// F16到F32的反量化（主入口）
void dequantizeF16ToF32(const uint16_t* input, float* output, size_t size) {
    if (size == 0) {
        return;
    }
    
    // 根据可用的SIMD指令集选择最佳实现
#ifdef __AVX2__
    dequantizeF16ToF32AVX2(input, output, size);
    CLLM_INFO("使用AVX2优化的F16反量化，大小: %zu", size);
#elif __SSE2__
    dequantizeF16ToF32SSE2(input, output, size);
    CLLM_INFO("使用SSE2优化的F16反量化，大小: %zu", size);
#elif __ARM_NEON
    dequantizeF16ToF32NEON(input, output, size);
    CLLM_INFO("使用NEON优化的F16反量化，大小: %zu", size);
#else
    dequantizeF16ToF32Scalar(input, output, size);
    CLLM_INFO("使用标量F16反量化，大小: %zu", size);
#endif
}

// Q8_0到F32的反量化（标量实现）
void dequantizeQ8ToF32Scalar(const void* input, float* output, size_t size) {
    const uint8_t* q8_data = static_cast<const uint8_t*>(input);
    
    for (size_t i = 0; i < size; ++i) {
        // 简化实现，实际需要根据GGUF格式调整
        output[i] = static_cast<float>(q8_data[i]) / 127.0f;
    }
    
    CLLM_INFO("使用Q8_0标量反量化，大小: %zu", size);
}

// Q8_0到F32的反量化（AVX2优化）
#ifdef __AVX2__
void dequantizeQ8ToF32AVX2(const void* input, float* output, size_t size) {
    const uint8_t* q8_data = static_cast<const uint8_t*>(input);
    size_t i = 0;
    
    // AVX2优化：一次处理32个uint8_t值，生成32个float值
    __m256 scale = _mm256_set1_ps(1.0f / 127.0f);
    
    while (i + 32 <= size) {
        // 加载32个uint8_t值
        __m256i q8_vec = _mm256_load_si256(reinterpret_cast<const __m256i*>(q8_data + i));
        
        // 将uint8_t转换为int32_t
        __m256i i32_low = _mm256_cvtepu8_epi16(_mm256_castsi256_si128(q8_vec));
        __m256i i32_high = _mm256_cvtepu8_epi16(_mm256_extracti128_si256(q8_vec, 1));
        
        __m256i i32_lo = _mm256_cvtepi16_epi32(i32_low);
        __m256i i32_hi = _mm256_cvtepi16_epi32(i32_high);
        
        // 将int32_t转换为float
        __m256 f32_lo = _mm256_cvtepi32_ps(i32_lo);
        __m256 f32_hi = _mm256_cvtepi32_ps(i32_hi);
        
        // 应用缩放因子
        __m256 result_lo = _mm256_mul_ps(f32_lo, scale);
        __m256 result_hi = _mm256_mul_ps(f32_hi, scale);
        
        // 存储结果
        _mm256_storeu_ps(output + i, result_lo);
        _mm256_storeu_ps(output + i + 8, result_hi);
        
        i += 32;
    }
    
    // 处理剩余元素
    if (i < size) {
        dequantizeQ8ToF32Scalar(q8_data + i, output + i, size - i);
    }
    
    CLLM_INFO("使用AVX2优化的Q8_0反量化，大小: %zu", size);
}
#endif

// Q8_0到F32的反量化（主入口）
void dequantizeQ8ToF32(const void* input, float* output, size_t size) {
    // 根据可用的SIMD指令集选择最佳实现
#ifdef __AVX2__
    dequantizeQ8ToF32AVX2(input, output, size);
#else
    dequantizeQ8ToF32Scalar(input, output, size);
#endif
}

// Q4_K_M块结构
struct Q4KMBlock {
    float d;              // 缩放因子
    int8_t qs[8];         // 4位量化数据 (8个4位值)
};

// Q4_K_M到F32的反量化（标量实现）
void dequantizeQ4KMF32Scalar(const void* input, float* output, const std::vector<size_t>& shape) {
    // 计算元素数量
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    
    const uint8_t* q4_data = static_cast<const uint8_t*>(input);
    size_t offset = 0;
    size_t output_idx = 0;
    
    // Q4_K_M格式: 每个块包含1个缩放因子 + 8个4位值 (共6字节/块)
    while (output_idx < size) {
        // 读取块数据
        const Q4KMBlock* block = reinterpret_cast<const Q4KMBlock*>(q4_data + offset);
        float scale = block->d;
        
        // 反量化每个值
        for (int i = 0; i < 8 && output_idx < size; ++i) {
            int8_t q_val = block->qs[i];
            // 转换为有符号4位值 (-8 到 7)
            if (q_val >= 8) {
                q_val -= 16;
            }
            output[output_idx++] = scale * static_cast<float>(q_val);
        }
        
        offset += sizeof(Q4KMBlock);
    }
    
    CLLM_INFO("使用Q4_K_M标量反量化，形状: %zu, 元素数: %zu", shape.size(), size);
}

// Q4_K_M到F32的反量化（AVX2优化）
#ifdef __AVX2__
void dequantizeQ4KMF32AVX2(const void* input, float* output, const std::vector<size_t>& shape) {
    // 计算元素数量
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    
    const uint8_t* q4_data = static_cast<const uint8_t*>(input);
    size_t offset = 0;
    size_t output_idx = 0;
    
    // 每个Q4_K_M块包含1个缩放因子 + 8个4位值 (共6字节/块)
    // AVX2优化：一次处理2个块 (12字节)，生成16个float值
    while (output_idx + 16 <= size) {
        // 加载2个块的缩放因子 (2个float)
        __m256 scale1 = _mm256_set1_ps(*reinterpret_cast<const float*>(q4_data + offset));
        __m256 scale2 = _mm256_set1_ps(*reinterpret_cast<const float*>(q4_data + offset + 6));
        
        // 加载2个块的量化数据 (各4字节，包含8个4位值)
        uint32_t qvals1 = *reinterpret_cast<const uint32_t*>(q4_data + offset + 4);
        uint32_t qvals2 = *reinterpret_cast<const uint32_t*>(q4_data + offset + 10);
        
        // 提取4位值并转换为有符号整数
        int8_t q[16];
        for (int i = 0; i < 8; ++i) {
            q[i] = static_cast<int8_t>((qvals1 >> (i * 4)) & 0xF);
            if (q[i] >= 8) q[i] -= 16;
        }
        for (int i = 0; i < 8; ++i) {
            q[i + 8] = static_cast<int8_t>((qvals2 >> (i * 4)) & 0xF);
            if (q[i + 8] >= 8) q[i + 8] -= 16;
        }
        
        // 创建包含16个量化值的AVX2向量
        __m256i q_vec = _mm256_set_epi8(
            q[15], q[14], q[13], q[12], q[11], q[10], q[9], q[8],
            q[7], q[6], q[5], q[4], q[3], q[2], q[1], q[0],
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        );
        
        // 将int8_t转换为float
        __m256 q_float = _mm256_cvtepi8_epi32(q_vec);
        q_float = _mm256_cvtepi32_ps(q_float);
        
        // 分离前8个和后8个值
        __m256 q_float_low = _mm256_castsi256_ps(_mm256_castps_si256(q_float));
        
        // 创建包含前8个量化值的向量
        __m256 q1_float = _mm256_set_m128(_mm_setzero_ps(), _mm256_castps256_ps128(q_float));
        
        // 应用缩放因子
        __m256 result1 = _mm256_mul_ps(q1_float, scale1);
        
        // 创建包含后8个量化值的向量
        __m256i q_vec2 = _mm256_srli_si256(q_vec, 8);
        __m256 q2_float = _mm256_cvtepi8_epi32(q_vec2);
        q2_float = _mm256_cvtepi32_ps(q2_float);
        __m256 q2_float_low = _mm256_set_m128(_mm_setzero_ps(), _mm256_castps256_ps128(q2_float));
        
        // 应用缩放因子
        __m256 result2 = _mm256_mul_ps(q2_float_low, scale2);
        
        // 存储结果
        _mm256_storeu_ps(output + output_idx, result1);
        _mm256_storeu_ps(output + output_idx + 8, result2);
        
        // 更新偏移量和输出索引
        offset += 12; // 2个块 * 6字节/块
        output_idx += 16;
    }
    
    // 处理剩余元素（使用标量实现）
    if (output_idx < size) {
        dequantizeQ4KMF32Scalar(q4_data + offset, output + output_idx, shape);
    }
    
    CLLM_INFO("使用AVX2优化的Q4_K_M反量化，形状: %zu, 元素数: %zu", shape.size(), size);
}
#endif

// Q4_K_M到F32的反量化（主入口）
void dequantizeQ4KMF32(const void* input, float* output, const std::vector<size_t>& shape) {
    // 根据可用的SIMD指令集选择最佳实现
#ifdef __AVX2__
    dequantizeQ4KMF32AVX2(input, output, shape);
#else
    dequantizeQ4KMF32Scalar(input, output, shape);
#endif
}

// Q5_K_M块结构
struct Q5KMBlock {
    float d;              // 缩放因子
    int8_t qh[4];         // 2位量化的高位数据
    int8_t qs[8];         // 4位量化的低位数据
};

// Q5_K_M到F32的反量化（标量实现）
void dequantizeQ5KMF32Scalar(const void* input, float* output, const std::vector<size_t>& shape) {
    // 计算元素数量
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    
    const uint8_t* q5_data = static_cast<const uint8_t*>(input);
    size_t offset = 0;
    size_t output_idx = 0;
    
    // Q5_K_M格式: 每个块包含1个缩放因子 + 2位高位数据 + 4位低位数据 (共7字节/块)
    while (output_idx < size) {
        // 读取块数据
        const Q5KMBlock* block = reinterpret_cast<const Q5KMBlock*>(q5_data + offset);
        float scale = block->d;
        
        // 反量化每个值
        for (int i = 0; i < 8 && output_idx < size; ++i) {
            // 获取2位高位
            int h_val = block->qh[i / 4];
            if (i % 4 == 0) h_val &= 0x03;
            else if (i % 4 == 1) h_val = (h_val >> 2) & 0x03;
            else if (i % 4 == 2) h_val = (h_val >> 4) & 0x03;
            else h_val = (h_val >> 6) & 0x03;
            
            // 获取4位低位
            int l_val = block->qs[i];
            if (l_val >= 8) {
                l_val -= 16;
            }
            
            // 组合高位和低位 (-16 到 15)
            int q_val = (h_val << 3) | (l_val & 0x07);
            if (q_val >= 16) {
                q_val -= 32;
            }
            
            output[output_idx++] = scale * static_cast<float>(q_val);
        }
        
        offset += sizeof(Q5KMBlock);
    }
    
    CLLM_INFO("使用Q5_K_M标量反量化，形状: %zu, 元素数: %zu", shape.size(), size);
}

// Q5_K_M到F32的反量化（AVX2优化）
#ifdef __AVX2__
void dequantizeQ5KMF32AVX2(const void* input, float* output, const std::vector<size_t>& shape) {
    // 计算元素数量
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    
    const uint8_t* q5_data = static_cast<const uint8_t*>(input);
    size_t offset = 0;
    size_t output_idx = 0;
    
    // Q5_K_M格式: 每个块包含1个缩放因子 + 2位高位数据 + 4位低位数据 (共7字节/块)
    // AVX2优化：一次处理1个块 (7字节)，生成8个float值
    while (output_idx + 8 <= size) {
        // 加载块的缩放因子
        __m256 scale = _mm256_set1_ps(*reinterpret_cast<const float*>(q5_data + offset));
        
        // 加载块的高位数据 (4字节，包含8个2位值)
        uint32_t h_vals = *reinterpret_cast<const uint32_t*>(q5_data + offset + 4);
        
        // 加载块的低位数据 (4字节，包含8个4位值)
        uint32_t l_vals = *reinterpret_cast<const uint32_t*>(q5_data + offset + 5);
        
        // 提取2位高位值并转换为有符号整数
        int8_t h[8];
        // 提取4位低位值并转换为有符号整数
        int8_t l[8];
        
        for (int i = 0; i < 8; ++i) {
            // 提取2位高位值
            h[i] = static_cast<int8_t>((h_vals >> (i * 2)) & 0x3);
            
            // 提取4位低位值
            l[i] = static_cast<int8_t>((l_vals >> (i * 4)) & 0xF);
            if (l[i] >= 8) l[i] -= 16;
            
            // 组合高位和低位 (-16 到 15)
            int combined = (h[i] << 3) | (l[i] & 0x7);
            if (combined >= 16) combined -= 32;
            
            l[i] = static_cast<int8_t>(combined);
        }
        
        // 创建包含8个组合值的AVX2向量
        __m256i combined_vec = _mm256_set_epi8(
            0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0,
            l[7], l[6], l[5], l[4], l[3], l[2], l[1], l[0],
            0, 0, 0, 0, 0, 0, 0, 0
        );
        
        // 将int8_t转换为float
        __m256 combined_float = _mm256_cvtepi8_epi32(combined_vec);
        combined_float = _mm256_cvtepi32_ps(combined_float);
        
        // 应用缩放因子
        __m256 result = _mm256_mul_ps(combined_float, scale);
        
        // 存储结果（只使用前8个值）
        _mm_storeu_ps(output + output_idx, _mm256_castps256_ps128(result));
        
        // 更新偏移量和输出索引
        offset += 7; // 1个块 * 7字节/块
        output_idx += 8;
    }
    
    // 处理剩余元素（使用标量实现）
    if (output_idx < size) {
        dequantizeQ5KMF32Scalar(q5_data + offset, output + output_idx, shape);
    }
    
    CLLM_INFO("使用AVX2优化的Q5_K_M反量化，形状: %zu, 元素数: %zu", shape.size(), size);
}
#endif

// Q5_K_M到F32的反量化（主入口）
void dequantizeQ5KMF32(const void* input, float* output, const std::vector<size_t>& shape) {
    // 强制使用标量实现（修复MacOS上的AVX2问题）
    dequantizeQ5KMF32Scalar(input, output, shape);
}

// 通用反量化接口
void dequantizeTensor(const void* input, float* output, GGUFQuantizationType type, const std::vector<size_t>& shape) {
    // 计算元素数量
    size_t size = 1;
    for (size_t dim : shape) {
        size *= dim;
    }
    
    switch (type) {
        case GGUFQuantizationType::F16:
            dequantizeF16ToF32(static_cast<const uint16_t*>(input), output, size);
            break;
        case GGUFQuantizationType::Q4_K_M:
            dequantizeQ4KMF32(input, output, shape);
            break;
        case GGUFQuantizationType::Q5_K_M:
            dequantizeQ5KMF32(input, output, shape);
            break;
        case GGUFQuantizationType::Q8_0:
            dequantizeQ8ToF32(input, output, size);
            break;
        default:
            throw std::runtime_error("不支持的GGUF量化类型");
    }
}

} // namespace cllm