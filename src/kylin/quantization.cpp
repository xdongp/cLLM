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
    
    for (int i = 0; i < nb; i++) {
        const uint8_t* q = x[i].qs;
        
        // 获取超级块级别的scale
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        
        int is = 0;  // scale索引
        uint8_t sc, m;
        
        // 处理每个64元素组（2个子块）
        for (int j = 0; j < QK_K; j += 64) {
            // 获取第一个子块的scale和min
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc;   // 实际scale
            const float m1 = min * m;  // 实际min
            
            // 获取第二个子块的scale和min
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            
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
    
    // 只复制需要的元素
    std::memcpy(output, tempBuffer.data(), elementCount * sizeof(float));
    
    CLLM_DEBUG("Dequantized Q4_K: %zu elements from %zu blocks", 
               elementCount, blockCount);
}

} // namespace quantization
} // namespace kylin
} // namespace cllm
