#include <iostream>
#include <vector>
#include <cstdint>
#include <climits>
#include <cmath>
#include "cllm/model/gguf_dequantization.h"

// 辅助函数：将float转换为uint16_t的F16表示
uint16_t floatToF16(float f) {
    uint32_t f32 = *reinterpret_cast<uint32_t*>(&f);
    uint32_t sign = (f32 >> 31) & 1;
    uint32_t exponent = (f32 >> 23) & 0xFF;
    uint32_t mantissa = f32 & 0x7FFFFF;
    
    uint16_t f16;
    
    if (exponent == 0) {
        if (mantissa == 0) {
            // 零
            f16 = sign << 15;
        } else {
            // 非规格化数
            exponent = 1;
            while ((mantissa & 0x800000) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x7FFFFF;
            exponent += 15 - 127 + 1;
            f16 = (sign << 15) | ((exponent - 1) << 10) | (mantissa >> 13);
        }
    } else if (exponent == 0xFF) {
        // 无穷大或NaN
        f16 = (sign << 15) | (0x1F << 10) | (mantissa >> 13);
    } else {
        // 规格化数
        exponent += 15 - 127;
        if (exponent <= 0) {
            // 非规格化数
            mantissa = (mantissa | 0x800000) >> (1 - exponent);
            mantissa &= 0x7FFFFF;
            f16 = (sign << 15) | (mantissa >> 13);
        } else if (exponent >= 31) {
            // 无穷大
            f16 = (sign << 15) | (0x1F << 10);
        } else {
            // 正常规格化数
            f16 = (sign << 15) | (exponent << 10) | (mantissa >> 13);
        }
    }
    
    return f16;
}

int main() {
    // 测试F16到F32的反量化
    std::cout << "=== 测试F16到F32的反量化 ===\n";
    
    // 创建测试数据
    std::vector<float> test_floats = {
        0.0f, 1.0f, -1.0f, 0.5f, -0.5f,
        2.0f, -2.0f, 0.1f, -0.1f, 100.0f
    };
    
    // 转换为F16
    std::vector<uint16_t> f16_data(test_floats.size());
    for (size_t i = 0; i < test_floats.size(); ++i) {
        f16_data[i] = floatToF16(test_floats[i]);
    }
    
    // 使用反量化函数转换回F32
    std::vector<float> result_floats(test_floats.size());
    cllm::dequantizeF16ToF32(f16_data.data(), result_floats.data(), test_floats.size());
    
    // 打印结果
    for (size_t i = 0; i < test_floats.size(); ++i) {
        std::cout << "原始值: " << test_floats[i] 
                  << " -> F16: 0x" << std::hex << f16_data[i] << std::dec
                  << " -> 反量化值: " << result_floats[i]
                  << " -> 误差: " << std::abs(test_floats[i] - result_floats[i]) << std::endl;
    }
    
    // 测试Q8_0到F32的反量化
    std::cout << "\n=== 测试Q8_0到F32的反量化 ===\n";
    
    // 创建Q8_0测试数据
    std::vector<int8_t> q8_data = {127, -128, 64, -64, 0, 32, -32};
    std::vector<float> q8_result(q8_data.size());
    
    // 使用反量化函数
    cllm::dequantizeQ8ToF32(q8_data.data(), q8_result.data(), q8_data.size());
    
    // 打印结果
    for (size_t i = 0; i < q8_data.size(); ++i) {
        std::cout << "Q8值: " << static_cast<int>(q8_data[i])
                  << " -> 反量化值: " << q8_result[i] << std::endl;
    }
    
    std::cout << "\n=== 测试完成 ===\n";
    return 0;
}