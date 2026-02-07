/**
 * @file test_ggml_precision.cpp
 * @brief 测试 GGML 的数值精度问题分析
 *
 * 分析 CPU 和 GPU logits 差异的根本原因
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <numeric>

namespace kylin_test {

// ============================================================================
// 测试：GGML 精度分析
// ============================================================================
class GGMLPrecisionTest : public TestCase {
public:
    GGMLPrecisionTest() : TestCase("ggml_precision_analysis", 
        "分析 GGML CPU/GPU 差异的根本原因") {}
    
    void execute() override {
        log(LogLevel::INFO, "========================================");
        log(LogLevel::INFO, "GGML CPU/GPU 差异根本原因分析");
        log(LogLevel::INFO, "========================================");
        
        // 分析 1: 数值精度累积效应
        analyzeAccumulationError();
        
        // 分析 2: Attention 计算的敏感性
        analyzeAttentionSensitivity();
        
        // 分析 3: 可能的根本原因
        analyzeRootCauses();
    }
    
private:
    void analyzeAccumulationError() {
        log(LogLevel::INFO, "\n[分析 1] 数值精度累积效应");
        
        // Transformer 中的典型计算：4096 维度的矩阵乘法
        const int dim = 4096;
        
        // 模拟权重和输入
        std::vector<float> weights(dim);
        std::vector<float> input(dim);
        
        // 填充典型值
        for (int i = 0; i < dim; ++i) {
            weights[i] = ((i % 100) - 50) * 0.01f;  // -0.5 到 0.5
            input[i] = ((i % 20) - 10) * 0.1f;       // -1.0 到 1.0
        }
        
        // CPU 计算（FP32）
        float cpu_result = 0.0f;
        for (int i = 0; i < dim; ++i) {
            cpu_result += weights[i] * input[i];
        }
        
        // 模拟 FP16 计算（降低精度）
        auto to_fp16 = [](float x) -> float {
            // 模拟 FP16：保留约 3-4 位有效数字
            float scale = 1000.0f;
            return std::round(x * scale) / scale;
        };
        
        float fp16_result = 0.0f;
        for (int i = 0; i < dim; ++i) {
            float w_fp16 = to_fp16(weights[i]);
            float i_fp16 = to_fp16(input[i]);
            fp16_result += w_fp16 * i_fp16;
        }
        
        float diff = std::abs(cpu_result - fp16_result);
        float relative_diff = diff / std::abs(cpu_result);
        
        log(LogLevel::INFO, "  矩阵维度: " + std::to_string(dim));
        log(LogLevel::INFO, "  CPU (FP32) 结果: " + std::to_string(cpu_result));
        log(LogLevel::INFO, "  模拟 FP16 结果: " + std::to_string(fp16_result));
        log(LogLevel::INFO, "  绝对差异: " + std::to_string(diff));
        log(LogLevel::INFO, "  相对差异: " + std::to_string(relative_diff * 100) + "%");
        
        // 在 28 层 Transformer 中的累积
        float layer_diff = diff;
        float total_diff = layer_diff * 28;  // 28 层
        log(LogLevel::INFO, "  28 层累积差异估计: " + std::to_string(total_diff));
    }
    
    void analyzeAttentionSensitivity() {
        log(LogLevel::INFO, "\n[分析 2] Attention 计算的敏感性");
        
        // Softmax 对输入值非常敏感
        const int seq_len = 10;
        std::vector<float> logits(seq_len);
        
        // 模拟 Q @ K^T / sqrt(d_k) 的结果
        for (int i = 0; i < seq_len; ++i) {
            logits[i] = (i - seq_len/2) * 2.0f;  // -10, -8, -6, ..., 8
        }
        
        // 原始 Softmax
        auto softmax = [](const std::vector<float>& x) {
            std::vector<float> result(x.size());
            float max_val = *std::max_element(x.begin(), x.end());
            float sum = 0.0f;
            for (size_t i = 0; i < x.size(); ++i) {
                result[i] = std::exp(x[i] - max_val);
                sum += result[i];
            }
            for (auto& v : result) v /= sum;
            return result;
        };
        
        auto probs_original = softmax(logits);
        
        // 添加小噪声（模拟数值误差）
        std::vector<float> logits_noisy(seq_len);
        for (int i = 0; i < seq_len; ++i) {
            logits_noisy[i] = logits[i] + ((i % 2 == 0) ? 0.1f : -0.1f);
        }
        
        auto probs_noisy = softmax(logits_noisy);
        
        // 计算概率分布的差异
        float max_prob_diff = 0.0f;
        for (int i = 0; i < seq_len; ++i) {
            float diff = std::abs(probs_original[i] - probs_noisy[i]);
            max_prob_diff = std::max(max_prob_diff, diff);
        }
        
        log(LogLevel::INFO, "  序列长度: " + std::to_string(seq_len));
        log(LogLevel::INFO, "  原始概率: [" + 
            std::to_string(probs_original[0]) + ", " +
            std::to_string(probs_original[1]) + ", ...]");
        log(LogLevel::INFO, "  加噪声后: [" + 
            std::to_string(probs_noisy[0]) + ", " +
            std::to_string(probs_noisy[1]) + ", ...]");
        log(LogLevel::INFO, "  最大概率差异: " + std::to_string(max_prob_diff));
        
        if (max_prob_diff > 0.01f) {
            log(LogLevel::WARN, "  ⚠ Attention 对数值误差非常敏感！");
        }
    }
    
    void analyzeRootCauses() {
        log(LogLevel::INFO, "\n[分析 3] 可能的根本原因");
        
        log(LogLevel::INFO, "\n  1. GGML Metal 内核实现差异:");
        log(LogLevel::INFO, "     - Metal kernel 可能使用不同的累加顺序");
        log(LogLevel::INFO, "     - SIMD 并行计算导致非确定性累加");
        log(LogLevel::INFO, "     - 可能的 FP16 中间计算");
        
        log(LogLevel::INFO, "\n  2. 矩阵乘法实现差异:");
        log(LogLevel::INFO, "     - CPU 使用 BLAS (OpenBLAS/Accelerate)");
        log(LogLevel::INFO, "     - GPU 使用 Metal Performance Shaders");
        log(LogLevel::INFO, "     - 不同的分块和并行策略");
        
        log(LogLevel::INFO, "\n  3. Attention 计算的放大效应:");
        log(LogLevel::INFO, "     - Softmax 指数运算放大差异");
        log(LogLevel::INFO, "     - 多层 Transformer 累积误差");
        log(LogLevel::INFO, "     - LayerNorm 可能无法完全归一化");
        
        log(LogLevel::INFO, "\n  4. 建议的解决方案:");
        log(LogLevel::INFO, "     a) 检查 GGML Metal kernel 是否使用 FP32 累加");
        log(LogLevel::INFO, "     b) 尝试强制 GGML 使用 CPU 实现对比");
        log(LogLevel::INFO, "     c) 在关键层添加数值稳定性检查");
        log(LogLevel::INFO, "     d) 考虑使用混合精度训练/推理策略");
        
        log(LogLevel::INFO, "\n  5. 验证方法:");
        log(LogLevel::INFO, "     - 对比单层输出（Layer 0）");
        log(LogLevel::INFO, "     - 检查中间层（Layer 13）");
        log(LogLevel::INFO, "     - 对比最终 logits");
        log(LogLevel::INFO, "     - 运行实际文本生成测试");
    }
};

// 注册测试
inline void registerGGMLPrecisionTests(TestSuite& suite) {
    suite.addTest(std::make_shared<GGMLPrecisionTest>());
}

} // namespace kylin_test
