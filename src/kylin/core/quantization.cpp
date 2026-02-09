/**
 * @file quantization.cpp
 * @brief 量化内核实现
 * 
 * 支持 FP16、INT8 等量化格式的计算内核
 * 使用 SIMD 优化（ARM NEON、x86 SSE/AVX）
 */

#include "cllm/kylin/core/quantization.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <limits>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define USE_NEON 1
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define USE_SSE 1
#if defined(__F16C__)
#define USE_F16C 1
#endif
#endif

// Apple Accelerate 支持
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#define USE_ACCELERATE 1
#endif

// OpenMP 支持
#ifdef _OPENMP
#include <omp.h>
#define USE_OPENMP 1
#endif

namespace cllm {
namespace kylin {

// ========== QuantizedWeight 实现 ==========

QuantizedWeight QuantizedWeight::fromFP32(const float* data, size_t count, QuantType targetType) {
    QuantizedWeight result;
    result.type_ = targetType;
    result.count_ = count;
    
    switch (targetType) {
        case QuantType::FP32: {
            result.data_.resize(count * sizeof(float));
            std::memcpy(result.data_.data(), data, count * sizeof(float));
            break;
        }
        case QuantType::FP16: {
            result.data_.resize(count * sizeof(uint16_t));
            quant_kernels::convert_f32_to_fp16(data, 
                reinterpret_cast<uint16_t*>(result.data_.data()), count);
            break;
        }
        case QuantType::INT8: {
            result.data_.resize(count);
            quant_kernels::compute_int8_params(data, count, result.scale_, result.zeroPoint_);
            quant_kernels::quantize_f32_to_int8(data, 
                reinterpret_cast<int8_t*>(result.data_.data()), 
                count, result.scale_, result.zeroPoint_);
            break;
        }
        default:
            CLLM_WARN("[Quantization] Unsupported target type: %s", quantTypeName(targetType));
            // 回退到 FP32
            result.type_ = QuantType::FP32;
            result.data_.resize(count * sizeof(float));
            std::memcpy(result.data_.data(), data, count * sizeof(float));
            break;
    }
    
    return result;
}

QuantizedWeight QuantizedWeight::fromBF16(const uint16_t* data, size_t count, QuantType targetType) {
    // 先转换为 FP32，再转换为目标类型
    std::vector<float> fp32(count);
    
    // BF16 -> FP32 转换
    for (size_t i = 0; i < count; ++i) {
        uint32_t bits = static_cast<uint32_t>(data[i]) << 16;
        std::memcpy(&fp32[i], &bits, sizeof(float));
    }
    
    return fromFP32(fp32.data(), count, targetType);
}

namespace quant_kernels {

// ========== FP16 <==> FP32 转换 ==========

// IEEE 754 FP16 格式：1 sign + 5 exponent + 10 mantissa
inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            // Zero
            f = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while ((mant & 0x400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x3FF;
            f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f = (sign << 31) | 0x7F800000 | (mant << 13);
    } else {
        // Normal
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    
    float result;
    std::memcpy(&result, &f, sizeof(float));
    return result;
}

inline uint16_t f32_to_fp16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));
    
    uint32_t sign = (bits >> 31) & 0x1;
    int32_t exp = ((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFF;
    
    uint16_t h;
    if (exp <= 0) {
        if (exp < -10) {
            // Too small, underflow to zero
            h = sign << 15;
        } else {
            // Subnormal
            mant = (mant | 0x800000) >> (1 - exp);
            h = (sign << 15) | (mant >> 13);
        }
    } else if (exp >= 31) {
        // Overflow to infinity
        h = (sign << 15) | 0x7C00;
    } else {
        // Normal
        h = (sign << 15) | (exp << 10) | (mant >> 13);
    }
    
    return h;
}

void convert_fp16_to_f32(const uint16_t* src, float* dst, size_t count) {
#if USE_NEON && defined(__ARM_FP16_FORMAT_IEEE)
    // ARM NEON with native FP16 support
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float16x4_t h = vld1_f16(reinterpret_cast<const __fp16*>(src + i));
        float32x4_t f = vcvt_f32_f16(h);
        vst1q_f32(dst + i, f);
    }
    for (; i < count; ++i) {
        dst[i] = fp16_to_f32(src[i]);
    }
#elif USE_F16C
    // x86 with F16C extension
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src + i));
        __m256 f = _mm256_cvtph_ps(h);
        _mm256_storeu_ps(dst + i, f);
    }
    for (; i < count; ++i) {
        dst[i] = fp16_to_f32(src[i]);
    }
#else
    // Scalar fallback
    for (size_t i = 0; i < count; ++i) {
        dst[i] = fp16_to_f32(src[i]);
    }
#endif
}

void convert_f32_to_fp16(const float* src, uint16_t* dst, size_t count) {
#if USE_NEON && defined(__ARM_FP16_FORMAT_IEEE)
    size_t i = 0;
    for (; i + 4 <= count; i += 4) {
        float32x4_t f = vld1q_f32(src + i);
        float16x4_t h = vcvt_f16_f32(f);
        vst1_f16(reinterpret_cast<__fp16*>(dst + i), h);
    }
    for (; i < count; ++i) {
        dst[i] = f32_to_fp16(src[i]);
    }
#elif USE_F16C
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        __m256 f = _mm256_loadu_ps(src + i);
        __m128i h = _mm256_cvtps_ph(f, _MM_FROUND_TO_NEAREST_INT);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + i), h);
    }
    for (; i < count; ++i) {
        dst[i] = f32_to_fp16(src[i]);
    }
#else
    for (size_t i = 0; i < count; ++i) {
        dst[i] = f32_to_fp16(src[i]);
    }
#endif
}

// ========== FP16 矩阵运算 ==========

void matmul_fp16_f32(
    const uint16_t* weight,
    const float* input,
    float* output,
    int M, int K
) {
    // 策略：使用 NEON 原生 FP16 指令进行计算
    // FP16→FP32 转换和乘加在同一个紧凑循环中，最大化缓存效率

#if USE_NEON && defined(__ARM_FP16_FORMAT_IEEE)
    // ========== ARM NEON 优化路径（无 Accelerate） ==========
    // 使用原生 FP16 指令，每次处理 16 个元素
    static bool logged = false;
    if (!logged) {
        CLLM_INFO("[matmul_fp16_f32] Using ARM NEON FP16 optimized path");
        logged = true;
    }
    
    #pragma omp parallel for schedule(static) if(M > 64)
    for (int m = 0; m < M; ++m) {
        const uint16_t* row = weight + static_cast<size_t>(m) * K;
        
        int k = 0;
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        
        // 主循环：每次处理 16 个元素（2x展开）
        for (; k + 16 <= K; k += 16) {
            // 预取下一组数据
            __builtin_prefetch(row + k + 64, 0, 3);
            __builtin_prefetch(input + k + 64, 0, 3);
            
            // 第一组 8 个元素
            float16x8_t h0 = vld1q_f16(reinterpret_cast<const __fp16*>(row + k));
            float32x4_t w0_lo = vcvt_f32_f16(vget_low_f16(h0));
            float32x4_t w0_hi = vcvt_f32_f16(vget_high_f16(h0));
            float32x4_t in0_lo = vld1q_f32(input + k);
            float32x4_t in0_hi = vld1q_f32(input + k + 4);
            vsum0 = vfmaq_f32(vsum0, w0_lo, in0_lo);
            vsum1 = vfmaq_f32(vsum1, w0_hi, in0_hi);
            
            // 第二组 8 个元素
            float16x8_t h1 = vld1q_f16(reinterpret_cast<const __fp16*>(row + k + 8));
            float32x4_t w1_lo = vcvt_f32_f16(vget_low_f16(h1));
            float32x4_t w1_hi = vcvt_f32_f16(vget_high_f16(h1));
            float32x4_t in1_lo = vld1q_f32(input + k + 8);
            float32x4_t in1_hi = vld1q_f32(input + k + 12);
            vsum0 = vfmaq_f32(vsum0, w1_lo, in1_lo);
            vsum1 = vfmaq_f32(vsum1, w1_hi, in1_hi);
        }
        
        // 处理 8 个元素
        for (; k + 8 <= K; k += 8) {
            float16x8_t h = vld1q_f16(reinterpret_cast<const __fp16*>(row + k));
            float32x4_t w_lo = vcvt_f32_f16(vget_low_f16(h));
            float32x4_t w_hi = vcvt_f32_f16(vget_high_f16(h));
            float32x4_t in_lo = vld1q_f32(input + k);
            float32x4_t in_hi = vld1q_f32(input + k + 4);
            vsum0 = vfmaq_f32(vsum0, w_lo, in_lo);
            vsum1 = vfmaq_f32(vsum1, w_hi, in_hi);
        }
        
        // 合并两个累加器
        vsum0 = vaddq_f32(vsum0, vsum1);
        float sum = vaddvq_f32(vsum0);
        
        // 处理剩余
        for (; k < K; ++k) {
            sum += fp16_to_f32(row[k]) * input[k];
        }
        
        output[m] = sum;
    }
    
#elif USE_F16C
    // ========== x86 AVX + F16C 优化路径 ==========
    
    #pragma omp parallel for schedule(static) if(M > 64)
    for (int m = 0; m < M; ++m) {
        const uint16_t* row = weight + static_cast<size_t>(m) * K;
        
        __m256 vsum0 = _mm256_setzero_ps();
        __m256 vsum1 = _mm256_setzero_ps();
        int k = 0;
        
        // 2x 展开
        for (; k + 16 <= K; k += 16) {
            __m128i h0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
            __m128i h1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k + 8));
            __m256 w0 = _mm256_cvtph_ps(h0);
            __m256 w1 = _mm256_cvtph_ps(h1);
            __m256 in0 = _mm256_loadu_ps(input + k);
            __m256 in1 = _mm256_loadu_ps(input + k + 8);
            vsum0 = _mm256_fmadd_ps(w0, in0, vsum0);
            vsum1 = _mm256_fmadd_ps(w1, in1, vsum1);
        }
        
        for (; k + 8 <= K; k += 8) {
            __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
            __m256 w = _mm256_cvtph_ps(h);
            __m256 in = _mm256_loadu_ps(input + k);
            vsum0 = _mm256_fmadd_ps(w, in, vsum0);
        }
        
        // 合并并归约
        vsum0 = _mm256_add_ps(vsum0, vsum1);
        __m128 lo = _mm256_castps256_ps128(vsum0);
        __m128 hi = _mm256_extractf128_ps(vsum0, 1);
        lo = _mm_add_ps(lo, hi);
        lo = _mm_hadd_ps(lo, lo);
        lo = _mm_hadd_ps(lo, lo);
        float sum = _mm_cvtss_f32(lo);
        
        for (; k < K; ++k) {
            sum += fp16_to_f32(row[k]) * input[k];
        }
        
        output[m] = sum;
    }
    
#else
    // ========== 纯 C++ 回退实现 ==========
    // 标量实现，用于不支持 NEON 或 F16C 的平台
    static bool logged = false;
    if (!logged) {
        CLLM_WARN("[matmul_fp16_f32] Using scalar fallback path - performance will be slow!");
        logged = true;
    }

    for (int m = 0; m < M; ++m) {
        const uint16_t* row = weight + static_cast<size_t>(m) * K;
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += fp16_to_f32(row[k]) * input[k];
        }
        output[m] = sum;
    }
#endif
}

// ========== INT8 矩阵运算 ==========

void matmul_int8_f32(
    const int8_t* weight,
    const float* input,
    float* output,
    int M, int K,
    float scale,
    int32_t zeroPoint
) {
    // 预计算 inputSum（只计算一次）
    // output = (weight - zeroPoint) * scale @ input
    // = scale * (weight @ input) - scale * zeroPoint * sum(input)
    float inputSum = 0.0f;
    
#if USE_NEON
    {
        float32x4_t vsum = vdupq_n_f32(0.0f);
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            float32x4_t in0 = vld1q_f32(input + k);
            float32x4_t in1 = vld1q_f32(input + k + 4);
            float32x4_t in2 = vld1q_f32(input + k + 8);
            float32x4_t in3 = vld1q_f32(input + k + 12);
            vsum = vaddq_f32(vsum, in0);
            vsum = vaddq_f32(vsum, in1);
            vsum = vaddq_f32(vsum, in2);
            vsum = vaddq_f32(vsum, in3);
        }
        inputSum = vaddvq_f32(vsum);
        for (; k < K; ++k) {
            inputSum += input[k];
        }
    }
    
    const float zeroPointCorrection = scale * static_cast<float>(zeroPoint) * inputSum;
    
    #pragma omp parallel for schedule(static) if(M > 64)
    for (int m = 0; m < M; ++m) {
        const int8_t* row = weight + static_cast<size_t>(m) * K;
        
        // 预取下一行
        if (m + 1 < M) {
            __builtin_prefetch(row + K, 0, 3);
        }
        
        float32x4_t vsum0 = vdupq_n_f32(0.0f);
        float32x4_t vsum1 = vdupq_n_f32(0.0f);
        int k = 0;
        
        // 主循环：每次处理 32 个元素（2x16 展开）
        for (; k + 32 <= K; k += 32) {
            // 预取数据
            __builtin_prefetch(row + k + 64, 0, 3);
            __builtin_prefetch(input + k + 64, 0, 3);
            
            // 第一组 16 个元素
            int8x16_t w0 = vld1q_s8(row + k);
            float32x4_t in00 = vld1q_f32(input + k);
            float32x4_t in01 = vld1q_f32(input + k + 4);
            float32x4_t in02 = vld1q_f32(input + k + 8);
            float32x4_t in03 = vld1q_f32(input + k + 12);
            
            // int8 -> int16 -> int32 -> float32
            int16x8_t w0_lo = vmovl_s8(vget_low_s8(w0));
            int16x8_t w0_hi = vmovl_s8(vget_high_s8(w0));
            
            float32x4_t fw00 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_lo)));
            float32x4_t fw01 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_lo)));
            float32x4_t fw02 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w0_hi)));
            float32x4_t fw03 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w0_hi)));
            
            vsum0 = vfmaq_f32(vsum0, fw00, in00);
            vsum1 = vfmaq_f32(vsum1, fw01, in01);
            vsum0 = vfmaq_f32(vsum0, fw02, in02);
            vsum1 = vfmaq_f32(vsum1, fw03, in03);
            
            // 第二组 16 个元素
            int8x16_t w1 = vld1q_s8(row + k + 16);
            float32x4_t in10 = vld1q_f32(input + k + 16);
            float32x4_t in11 = vld1q_f32(input + k + 20);
            float32x4_t in12 = vld1q_f32(input + k + 24);
            float32x4_t in13 = vld1q_f32(input + k + 28);
            
            int16x8_t w1_lo = vmovl_s8(vget_low_s8(w1));
            int16x8_t w1_hi = vmovl_s8(vget_high_s8(w1));
            
            float32x4_t fw10 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_lo)));
            float32x4_t fw11 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_lo)));
            float32x4_t fw12 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w1_hi)));
            float32x4_t fw13 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w1_hi)));
            
            vsum0 = vfmaq_f32(vsum0, fw10, in10);
            vsum1 = vfmaq_f32(vsum1, fw11, in11);
            vsum0 = vfmaq_f32(vsum0, fw12, in12);
            vsum1 = vfmaq_f32(vsum1, fw13, in13);
        }
        
        // 处理剩余 16 个元素
        for (; k + 16 <= K; k += 16) {
            int8x16_t w = vld1q_s8(row + k);
            float32x4_t in0 = vld1q_f32(input + k);
            float32x4_t in1 = vld1q_f32(input + k + 4);
            float32x4_t in2 = vld1q_f32(input + k + 8);
            float32x4_t in3 = vld1q_f32(input + k + 12);
            
            int16x8_t w_lo = vmovl_s8(vget_low_s8(w));
            int16x8_t w_hi = vmovl_s8(vget_high_s8(w));
            
            float32x4_t fw0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_lo)));
            float32x4_t fw1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_lo)));
            float32x4_t fw2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_hi)));
            float32x4_t fw3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_hi)));
            
            vsum0 = vfmaq_f32(vsum0, fw0, in0);
            vsum1 = vfmaq_f32(vsum1, fw1, in1);
            vsum0 = vfmaq_f32(vsum0, fw2, in2);
            vsum1 = vfmaq_f32(vsum1, fw3, in3);
        }
        
        // 合并累加器
        vsum0 = vaddq_f32(vsum0, vsum1);
        float fsum = vaddvq_f32(vsum0);
        
        // 处理剩余
        for (; k < K; ++k) {
            fsum += static_cast<float>(row[k]) * input[k];
        }
        
        // 应用 scale 并减去 zeroPoint 校正项
        output[m] = scale * fsum - zeroPointCorrection;
    }
#else
    // 标量实现：预计算 inputSum
    for (int k = 0; k < K; ++k) {
        inputSum += input[k];
    }
    
    const float zeroPointCorrection = scale * static_cast<float>(zeroPoint) * inputSum;
    
    for (int m = 0; m < M; ++m) {
        const int8_t* row = weight + static_cast<size_t>(m) * K;
        
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += static_cast<float>(row[k]) * input[k];
        }
        
        output[m] = scale * sum - zeroPointCorrection;
    }
#endif
}

// ========== 量化工具 ==========

void compute_int8_params(
    const float* data,
    size_t count,
    float& scale,
    int32_t& zeroPoint
) {
    // 找最大最小值
    float minVal = data[0];
    float maxVal = data[0];
    
    for (size_t i = 1; i < count; ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
    }
    
    // 对称量化（假设权重分布对称）
    float absMax = std::max(std::abs(minVal), std::abs(maxVal));
    scale = absMax / 127.0f;
    zeroPoint = 0; // 对称量化 zero_point = 0
    
    if (scale < 1e-10f) {
        scale = 1e-10f; // 避免除零
    }
}

void quantize_f32_to_int8(
    const float* src,
    int8_t* dst,
    size_t count,
    float scale,
    int32_t zeroPoint
) {
    float invScale = 1.0f / scale;
    
#if USE_NEON
    float32x4_t vInvScale = vdupq_n_f32(invScale);
    int32x4_t vZeroPoint = vdupq_n_s32(zeroPoint);
    
    size_t i = 0;
    for (; i + 8 <= count; i += 8) {
        float32x4_t f0 = vld1q_f32(src + i);
        float32x4_t f1 = vld1q_f32(src + i + 4);
        
        // 缩放并转换为 int32
        int32x4_t i0 = vcvtq_s32_f32(vmulq_f32(f0, vInvScale));
        int32x4_t i1 = vcvtq_s32_f32(vmulq_f32(f1, vInvScale));
        
        // 加上 zero point
        i0 = vaddq_s32(i0, vZeroPoint);
        i1 = vaddq_s32(i1, vZeroPoint);
        
        // 饱和转换为 int8
        int16x4_t s0 = vqmovn_s32(i0);
        int16x4_t s1 = vqmovn_s32(i1);
        int16x8_t s = vcombine_s16(s0, s1);
        int8x8_t b = vqmovn_s16(s);
        
        vst1_s8(dst + i, b);
    }
    
    for (; i < count; ++i) {
        int32_t val = static_cast<int32_t>(std::round(src[i] * invScale)) + zeroPoint;
        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
#else
    for (size_t i = 0; i < count; ++i) {
        int32_t val = static_cast<int32_t>(std::round(src[i] * invScale)) + zeroPoint;
        dst[i] = static_cast<int8_t>(std::max(-128, std::min(127, val)));
    }
#endif
}

} // namespace quant_kernels
} // namespace kylin
} // namespace cllm
