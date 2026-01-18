/**
 * @file quantization.cpp
 * @brief 量化格式反量化实现
 */

#include "cllm/kylin/quantization.h"
#include "cllm/common/logger.h"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <string>
#include <cmath>

namespace cllm {
namespace kylin {
namespace quantization {

void dequantize_row_q4_K(
    const block_q4_K* x,
    float* y,
    int64_t k) {
    
    if (k % QK_K != 0) {
        throw std::invalid_argument(
            "dequantize_row_q4_K: k must be multiple of QK_K, got k=" + 
            std::to_string(k) + ", QK_K=" + std::to_string(QK_K)
        );
    }
    const int nb = k / QK_K;
    
    // 移除 static,避免多线程问题
    int debugBlockCount = 0;
    const int maxDebugBlocks = 3;  // 调试前3个块
    
    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        
        // 获取超级块级别的scale
        float d = fp16_to_fp32(x[i].d);
        float min = fp16_to_fp32(x[i].dmin);
        
        // 如果 d 或 min 是 NaN/Inf，将整个块设置为 0
        if (std::isnan(d) || std::isinf(d) || std::isnan(min) || std::isinf(min)) {
            static int warn_count = 0;
            if (warn_count < 10) {
                CLLM_WARN("Block %d has NaN/Inf: d=%.6f (0x%04X), min=%.6f (0x%04X)", 
                         i, d, x[i].d_raw, min, x[i].dmin_raw);
                warn_count++;
            }
            for (int j = 0; j < QK_K; ++j) {
                *y++ = 0.0f;
            }
            continue;
        }
        
        // 调试：打印前几个块的信息
        if (debugBlockCount < maxDebugBlocks) {
            CLLM_INFO("Q4_K Block %d: d=%.6f (0x%04X), min=%.6f (0x%04X)", 
                     debugBlockCount, d, x[i].d_raw, min, x[i].dmin_raw);
            CLLM_INFO("  scales[0-7]: %d,%d,%d,%d,%d,%d,%d,%d", 
                     x[i].scales[0], x[i].scales[1], x[i].scales[2], x[i].scales[3],
                     x[i].scales[4], x[i].scales[5], x[i].scales[6], x[i].scales[7]);
        }
        
        int is = 0;  // scale索引
        uint8_t sc, m;
        
        // 处理每个64元素组（2个子块）
        for (int j = 0; j < QK_K; j += 64) {
            // 获取第一个子块的scale和min
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;   // 实际scale
            const float m1 = min * m;  // 实际min
            const uint8_t sc1 = sc, m1_val = m;  // 保存第一个子块的值
            
            // 获取第二个子块的scale和min
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            const uint8_t sc2 = sc, m2_val = m;  // 保存第二个子块的值
            
            // 调试：打印scale和min信息
            if (debugBlockCount < maxDebugBlocks && j == 0) {
                CLLM_INFO("  Subblock 0: sc=%d, m=%d, d1=%.6f, m1=%.6f", sc1, m1_val, d1, m1);
                CLLM_INFO("  Subblock 1: sc=%d, m=%d, d2=%.6f, m2=%.6f", sc2, m2_val, d2, m2);
            }
            
            // 反量化前32个元素（使用d1和m1）
            for (int l = 0; l < 32; ++l) {
                *y++ = d1 * (q[l] & 0xF) - m1;  // 低4位
            }
            
            // 反量化后32个元素（使用d2和m2）
            for (int l = 0; l < 32; ++l) {
                *y++ = d2 * (q[l] >> 4) - m2;   // 高4位
            }
            
            q += 32;  // 移动到下一组
            is += 2;  // 更新scale索引
        }
        
        // 调试：打印反量化后的前几个值和统计信息
        if (debugBlockCount < maxDebugBlocks) {
            float* y_start = y - QK_K;
            
            // 检查是否有 NaN 或 Inf
            int nan_count = 0, inf_count = 0;
            float min_val = y_start[0], max_val = y_start[0];
            for (int i = 0; i < QK_K; ++i) {
                if (std::isnan(y_start[i])) nan_count++;
                if (std::isinf(y_start[i])) inf_count++;
                if (std::isfinite(y_start[i])) {
                    min_val = std::min(min_val, y_start[i]);
                    max_val = std::max(max_val, y_start[i]);
                }
            }
            
            CLLM_INFO("  Dequantized values (first 10): %.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                     y_start[0], y_start[1], y_start[2], y_start[3], y_start[4],
                     y_start[5], y_start[6], y_start[7], y_start[8], y_start[9]);
            CLLM_INFO("  Block stats: min=%.6f, max=%.6f, NaN=%d, Inf=%d", 
                     min_val, max_val, nan_count, inf_count);
        }
        
        debugBlockCount++;
    }
}

void dequantize_q4_K_to_f32(
    const void* quantizedData,
    float* output,
    size_t elementCount) {
    
    if (quantizedData == nullptr) {
        throw std::invalid_argument("dequantize_q4_K_to_f32: quantizedData is nullptr");
    }
    if (output == nullptr) {
        throw std::invalid_argument("dequantize_q4_K_to_f32: output is nullptr");
    }
    if (elementCount == 0) {
        CLLM_DEBUG("dequantize_q4_K_to_f32: elementCount is 0, nothing to dequantize");
        return;
    }
    
    const block_q4_K* blocks = static_cast<const block_q4_K*>(quantizedData);
    const size_t blockCount = (elementCount + QK_K - 1) / QK_K;
    const size_t actualElementCount = blockCount * QK_K;
    
    // 临时缓冲区存储完整反量化数据
    std::vector<float> tempBuffer(actualElementCount);
    
    // 反量化所有块
    dequantize_row_q4_K(blocks, tempBuffer.data(), actualElementCount);
    
    // 检查是否有 NaN/Inf，并记录位置
    size_t nan_count = 0, inf_count = 0;
    
    for (size_t i = 0; i < actualElementCount; ++i) {
        if (std::isnan(tempBuffer[i])) {
            nan_count++;
        }
        if (std::isinf(tempBuffer[i])) {
            inf_count++;
        }
    }
    
    if (nan_count > 0 || inf_count > 0) {
        // 每次都打印，以便追踪
        CLLM_WARN("dequantize_q4_K_to_f32: Found %zu NaN and %zu Inf values in %zu elements (%zu blocks)",
                 nan_count, inf_count, actualElementCount, blockCount);
    }
    
    // 只复制需要的元素
    std::memcpy(output, tempBuffer.data(), elementCount * sizeof(float));
    
    CLLM_DEBUG("Dequantized Q4_K: %zu elements from %zu blocks", 
              elementCount, blockCount);
}

static inline uint8_t get_nibble_4(const uint8_t* q, int idx) {
    const uint8_t v = q[idx >> 1];
    return (idx & 1) ? (v >> 4) : (v & 0x0F);
}

static inline uint8_t get_bits_2(const uint8_t* qh, int idx) {
    const uint8_t v = qh[idx >> 2];
    const int shift = (idx & 3) * 2;
    return (v >> shift) & 0x03;
}

void dequantize_row_q6_K(
    const block_q6_K* x,
    float* y,
    int64_t k) {

    if (k % QK_K != 0) {
        throw std::invalid_argument(
            "dequantize_row_q6_K: k must be multiple of QK_K, got k=" +
            std::to_string(k) + ", QK_K=" + std::to_string(QK_K)
        );
    }

    const int nb = k / QK_K;

    for (int bi = 0; bi < nb; ++bi) {
        const float d = fp16_to_fp32(x[bi].d);

        // 对于合法模型，d应为有限值；这里做防御，避免传播
        if (std::isnan(d) || std::isinf(d)) {
            for (int j = 0; j < QK_K; ++j) {
                *y++ = 0.0f;
            }
            continue;
        }

        const uint8_t* ql = x[bi].ql;
        const uint8_t* qh = x[bi].qh;
        const int8_t* sc = x[bi].scales;

        for (int j = 0; j < QK_K; ++j) {
            const int g = j >> 4; // 16个元素一组
            const float s = static_cast<float>(sc[g]);

            const uint8_t low4 = get_nibble_4(ql, j);
            const uint8_t hi2 = get_bits_2(qh, j);
            const int q = static_cast<int>(low4 | (hi2 << 4)) - 32; // [-32, 31]

            *y++ = d * s * static_cast<float>(q);
        }
    }
}

void dequantize_q6_K_to_f32(
    const void* quantizedData,
    float* output,
    size_t elementCount) {

    if (quantizedData == nullptr) {
        throw std::invalid_argument("dequantize_q6_K_to_f32: quantizedData is nullptr");
    }
    if (output == nullptr) {
        throw std::invalid_argument("dequantize_q6_K_to_f32: output is nullptr");
    }
    if (elementCount == 0) {
        CLLM_DEBUG("dequantize_q6_K_to_f32: elementCount is 0, nothing to dequantize");
        return;
    }

    const block_q6_K* blocks = static_cast<const block_q6_K*>(quantizedData);
    const size_t blockCount = (elementCount + QK_K - 1) / QK_K;
    const size_t actualElementCount = blockCount * QK_K;

    std::vector<float> tempBuffer(actualElementCount);
    dequantize_row_q6_K(blocks, tempBuffer.data(), actualElementCount);

    // 只复制需要的元素
    std::memcpy(output, tempBuffer.data(), elementCount * sizeof(float));
}


} // namespace quantization
} // namespace kylin
} // namespace cllm
