/**
 * @file quantization.h
 * @brief 量化格式数据结构和反量化函数
 * 
 * 参考 docs/research/gguf_q4k_inference_analysis.md 和 llama.cpp 实现
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include <vector>
#include <cstring>

namespace cllm {
namespace kylin {
namespace quantization {

// ==================== 常量定义 ====================

#define QK_K 256              // Q4_K/Q5_K/Q6_K 块大小
#define K_SCALE_SIZE 12       // scales数组大小

// ==================== 数据结构 ====================

// FP16类型（简化实现，实际应使用ggml_half）
typedef uint16_t ggml_half;

// Q4_K块结构（144字节）
struct block_q4_K {
    // 超级块级别的缩放因子
    union {
        struct {
            ggml_half d;    // 超级块scale（用于量化后的scales）
            ggml_half dmin; // 超级块scale（用于量化后的mins）
        };
        struct {
            uint16_t d_raw;
            uint16_t dmin_raw;
        };
    };
    
    // 子块级别的scales和mins（6位量化）
    uint8_t scales[K_SCALE_SIZE];  // 12字节
    
    // 4位量化值（打包存储）
    uint8_t qs[QK_K/2];  // 128字节
};
static_assert(sizeof(block_q4_K) == 144, "block_q4_K size must be 144 bytes");

// Q6_K块结构（210字节）
// 参考 ggml-common.h / llama.cpp：
// - ql: 低4位 (2个值/字节) -> 128字节
// - qh: 高2位 (4个值/字节) -> 64字节
// - scales: 每16个元素一个scale -> 16字节
// - d: 全局scale (FP16) -> 2字节
//
// 重要：字段顺序必须与 GGUF/ggml 的块内存布局一致，否则会把权重错读成极大值。
struct block_q6_K {
    uint8_t ql[QK_K / 2];      // 128字节
    uint8_t qh[QK_K / 4];      // 64字节
    int8_t scales[QK_K / 16];  // 16字节
    ggml_half d;               // 2字节
};
static_assert(sizeof(block_q6_K) == 210, "block_q6_K size must be 210 bytes");

// ==================== 辅助函数 ====================

/**
 * @brief FP16转FP32
 */
inline float fp16_to_fp32(ggml_half h) {
    // 标准 IEEE-754 FP16 到 FP32 转换
    uint32_t sign = (h & 0x8000u) << 16;  // 符号位
    uint32_t exp = (h & 0x7C00u) >> 10;   // 指数 (5 bits)
    uint32_t frac = h & 0x03FFu;          // 尾数 (10 bits)
    
    uint32_t f32;
    
    if (exp == 0) {
        // 零或非规格化数
        if (frac == 0) {
            // 零（正零或负零）
            f32 = sign;
        } else {
            // 非规格化数 -> 规格化
            // 找到第一个 1 的位置
            exp = 1;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                exp--;
            }
            frac &= 0x03FFu;  // 清除隐含的 1
            exp += 127 - 15;  // 调整偏置
            frac <<= 13;      // 对齐到 23 位
            f32 = sign | (exp << 23) | frac;
        }
    } else if (exp == 0x1F) {
        // 无穷大或 NaN
        // 保持 Inf/NaN（不转换为 0）
        f32 = sign | 0x7F800000u | (frac << 13);
    } else {
        // 规格化数
        exp = exp + (127 - 15);  // 调整偏置
        f32 = sign | (exp << 23) | (frac << 13);
    }
    
    float result;
    std::memcpy(&result, &f32, sizeof(float));
    return result;
}

/**
 * @brief 从scales数组中提取子块的scale和min
 * 
 * @param j 子块索引 (0-7)
 * @param scales scales数组
 * @param d 输出的scale值（6位，0-63）
 * @param m 输出的min值（6位，0-63）
 */
inline void get_scale_min_k4(int j, const uint8_t* scales, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        // 前4个子块：直接存储
        *d = scales[j] & 63;        // scale，低6位
        *m = scales[j + 4] & 63;    // min，低6位
    } else {
        // 后4个子块：打包存储
        *d = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4);
        *m = (scales[j+4] >> 4) | ((scales[j-0] >> 6) << 4);  // 修复：应该是 scales[j-0] 即 scales[j]
    }
}

// ==================== 反量化函数 ====================

/**
 * @brief 反量化一行Q4_K数据
 * 
 * @param x 输入的Q4_K块数组
 * @param y 输出的FP32数组
 * @param k 元素数量（必须是QK_K的倍数）
 */
void dequantize_row_q4_K(
    const block_q4_K* x,
    float* y,
    int64_t k
);

/**
 * @brief 反量化Q4_K张量到FP32
 * 
 * @param quantizedData 量化的原始数据（block_q4_K数组）
 * @param output 输出的FP32数组
 * @param elementCount 元素总数
 */
void dequantize_q4_K_to_f32(
    const void* quantizedData,
    float* output,
    size_t elementCount
);

/**
 * @brief 反量化一行Q6_K数据
 *
 * @param x 输入的Q6_K块数组
 * @param y 输出的FP32数组
 * @param k 元素数量（必须是QK_K的倍数）
 */
void dequantize_row_q6_K(
    const block_q6_K* x,
    float* y,
    int64_t k
);

/**
 * @brief 反量化Q6_K张量到FP32
 *
 * @param quantizedData 量化的原始数据（block_q6_K数组）
 * @param output 输出的FP32数组
 * @param elementCount 元素总数
 */
void dequantize_q6_K_to_f32(
    const void* quantizedData,
    float* output,
    size_t elementCount
);

} // namespace quantization
} // namespace kylin
} // namespace cllm
