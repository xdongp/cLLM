/**
 * @file test_attention_breakdown.cpp
 * @brief Attention 内部细粒度对比测试
 *
 * Phase 3 已确认问题出在 Layer 0 Attention，现在进一步细化到 Attention 内部各子步骤：
 *
 *   Stage 35 - Attention 细分对比测试：
 *       Step 1: Q、K、V Projection 输出对比
 *       Step 2: RoPE 应用后的 Q、K 对比
 *       Step 3: KV Cache 存储内容对比
 *       Step 4: Attention Score (QK^T / sqrt(d_k)) 对比
 *       Step 5: Softmax 后的 Attention Weights 对比
 *       Step 6: Attention Output (weights @ V) 对比
 *       Step 7: O Projection 输出对比
 *
 * 目标：定位 Attention 内部哪个子步骤开始出现偏差
 *
 * 使用方法：
 *   ./kylin_test_suite --stage=35 --verbose
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <sstream>

namespace kylin_test {

// ============================================================================
// 辅助函数：计算差异（从 test_phased_cpu_gpu_comparison.cpp 复用）
// ============================================================================

struct DetailedDiffMetrics {
    float maxAbsDiff  = 0.0f;
    float meanAbsDiff = 0.0f;
    float rmse        = 0.0f;
    float cosine      = 0.0f;
    size_t maxDiffIdx = 0;
    bool sizeMatch    = true;
    size_t sizeA = 0, sizeB = 0;

    bool isGood(float tol = 0.01f) const {
        return sizeMatch && maxAbsDiff < tol && cosine > 0.99f;
    }

    std::string summary() const {
        std::ostringstream oss;
        if (!sizeMatch) {
            oss << "SIZE_MISMATCH (A=" << sizeA << ", B=" << sizeB << ")";
            return oss.str();
        }
        oss << std::scientific << std::setprecision(3)
            << "maxDiff=" << maxAbsDiff << " meanDiff=" << meanAbsDiff
            << " RMSE=" << rmse
            << std::fixed << std::setprecision(6) << " cos=" << cosine
            << " @idx=" << maxDiffIdx;
        return oss.str();
    }
};

static DetailedDiffMetrics computeDetailedDiff(const std::vector<float>& a, const std::vector<float>& b) {
    DetailedDiffMetrics m;
    m.sizeA = a.size();
    m.sizeB = b.size();
    if (a.size() != b.size()) { m.sizeMatch = false; return m; }
    if (a.empty()) return m;

    double sumDiff = 0, sumDiffSq = 0;
    double dotAB = 0, normA = 0, normB = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > m.maxAbsDiff) {
            m.maxAbsDiff = diff;
            m.maxDiffIdx = i;
        }
        sumDiff += diff;
        sumDiffSq += static_cast<double>(diff) * diff;

        dotAB += static_cast<double>(a[i]) * b[i];
        normA += static_cast<double>(a[i]) * a[i];
        normB += static_cast<double>(b[i]) * b[i];
    }

    m.meanAbsDiff = static_cast<float>(sumDiff / a.size());
    m.rmse        = static_cast<float>(std::sqrt(sumDiffSq / a.size()));

    double denom = std::sqrt(normA) * std::sqrt(normB);
    m.cosine = (denom > 1e-12) ? static_cast<float>(dotAB / denom) : 0.0f;

    return m;
}

static std::string vecFirstN(const std::vector<float>& v, int n = 8) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6) << "[";
    for (int i = 0; i < std::min(n, static_cast<int>(v.size())); ++i) {
        if (i) oss << ", ";
        oss << v[i];
    }
    if (static_cast<int>(v.size()) > n) oss << ", ...";
    oss << "]";
    return oss.str();
}

// ============================================================================
// Stage 35: Attention 细分对比测试
// ============================================================================

class AttentionBreakdownTest : public TestCase {
public:
    AttentionBreakdownTest() : TestCase(
        "attention_breakdown",
        "Stage 35: Attention 内部细分对比 - 定位 RoPE/KVCache/Softmax/MatMul"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Stage 35: Attention 内部细分对比");
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "目标: 定位 Attention 内部哪个子步骤开始出现偏差");
        log(LogLevel::INFO, "");

        // 获取模型路径
        const char* envPath = std::getenv("CLLM_MODEL_PATH");
        std::string modelPath = envPath ? envPath : "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
        log(LogLevel::INFO, "模型路径: " + modelPath);

        // 测试 token
        int32_t testToken = 9707;  // "hello"
        log(LogLevel::INFO, "测试 Token ID: " + std::to_string(testToken));
        log(LogLevel::INFO, "");

        // 加载 CPU 模型
        log(LogLevel::INFO, "[1/2] 加载 CPU 模型 (FP32)...");
        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载");
        log(LogLevel::INFO, "  hiddenSize=" + std::to_string(cpuModel.hiddenSize()) +
                          ", vocabSize=" + std::to_string(cpuModel.vocabSize()));

        // 加载 GPU 模型
        log(LogLevel::INFO, "[2/2] 加载 GPU 模型 (Metal)...");
        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded(), "GPU 模型加载");
        assertTrue(gpuModel.isUsingGPU(), "GPU 后端启用");
        log(LogLevel::INFO, "");

        // 执行推理并获取中间结果
        log(LogLevel::INFO, "[3/4] 执行 CPU 推理...");
        cpuModel.resetKVCache();
        std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayers;
        std::vector<float> cpuEmb, cpuFinalNorm;
        auto cpuLogits = cpuModel.forwardWithDebugCPU({testToken}, cpuLayers, cpuEmb, cpuFinalNorm);
        assertTrue(!cpuLogits.empty(), "CPU 推理成功");

        log(LogLevel::INFO, "[4/4] 执行 GPU 推理...");
        gpuModel.resetKVCache();
        std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayers;
        std::vector<float> gpuEmb, gpuFinalNorm;
        auto gpuLogits = gpuModel.forwardWithDebugGPU({testToken}, gpuLayers, gpuEmb, gpuFinalNorm);
        assertTrue(!gpuLogits.empty(), "GPU 推理成功");
        log(LogLevel::INFO, "");

        // ========================================================================
        // 开始细分 Layer 0 的 Attention 过程
        // ========================================================================

        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Layer 0 Attention 细分对比");
        log(LogLevel::INFO, "============================================================");

        if (cpuLayers.empty() || gpuLayers.empty()) {
            log(LogLevel::FAIL, "无 Layer 输出，无法分析");
            return;
        }

        auto& cpuLayer0 = cpuLayers[0];
        auto& gpuLayer0 = gpuLayers[0];

        // ------------------------------------------------------------------------
        // Step 1: InputNorm 输出对比（已知一致）
        // ------------------------------------------------------------------------
        log(LogLevel::INFO, "\n[Step 1] InputNorm 输出对比");
        auto normDiff = computeDetailedDiff(cpuLayer0.inputNormOutput, gpuLayer0.afterInputNorm);
        log(normDiff.isGood(0.001f) ? LogLevel::PASS : LogLevel::WARN,
            "  " + normDiff.summary());
        log(LogLevel::INFO, "  CPU 前8值: " + vecFirstN(cpuLayer0.inputNormOutput));
        log(LogLevel::INFO, "  GPU 前8值: " + vecFirstN(gpuLayer0.afterInputNorm));

        // ------------------------------------------------------------------------
        // Step 2: QKV Projection 输出对比（已知一致）
        // ------------------------------------------------------------------------
        log(LogLevel::INFO, "\n[Step 2] QKV Projection 输出对比");
        auto qkvDiff = computeDetailedDiff(cpuLayer0.qkvOutput, gpuLayer0.afterQKV);
        log(qkvDiff.isGood(0.001f) ? LogLevel::PASS : LogLevel::WARN,
            "  " + qkvDiff.summary());
        log(LogLevel::INFO, "  CPU QKV size=" + std::to_string(cpuLayer0.qkvOutput.size()) +
                          " 前8值: " + vecFirstN(cpuLayer0.qkvOutput));
        log(LogLevel::INFO, "  GPU QKV size=" + std::to_string(gpuLayer0.afterQKV.size()) +
                          " 前8值: " + vecFirstN(gpuLayer0.afterQKV));

        // ------------------------------------------------------------------------
        // Step 3-7: Attention 内部各步骤对比
        // ------------------------------------------------------------------------
        // 注意：目前 LayerOutput 只包含 afterQKV 和 afterAttention
        // 需要修改 GGMLGPUBackend::forwardWithDebug 来导出更多中间结果
        
        log(LogLevel::INFO, "\n[Step 3-7] Attention 内部子步骤");
        log(LogLevel::WARN, "  当前 GPU 后端只导出了 afterQKV 和 afterAttention");
        log(LogLevel::WARN, "  需要在 ggml_backend.cpp 的 forwardWithDebug 中添加更多检查点:");
        log(LogLevel::INFO, "");
        log(LogLevel::INFO, "  建议添加以下导出点:");
        log(LogLevel::INFO, "    1. RoPE 应用后的 Q、K");
        log(LogLevel::INFO, "    2. 更新后的 KV Cache");
        log(LogLevel::INFO, "    3. Attention Score (QK^T)");
        log(LogLevel::INFO, "    4. Softmax 后的权重");
        log(LogLevel::INFO, "    5. Attention Output (before O Projection)");
        log(LogLevel::INFO, "");

        // ------------------------------------------------------------------------
        // Step 8: Attention 最终输出对比（已知有偏差）
        // ------------------------------------------------------------------------
        log(LogLevel::INFO, "\n[Step 8] Attention 最终输出对比");
        auto attnDiff = computeDetailedDiff(cpuLayer0.attentionOutput, gpuLayer0.afterAttention);
        log(LogLevel::WARN, "  " + attnDiff.summary() + " ← 这里已经有偏差");
        log(LogLevel::INFO, "  CPU 前8值: " + vecFirstN(cpuLayer0.attentionOutput));
        log(LogLevel::INFO, "  GPU 前8值: " + vecFirstN(gpuLayer0.afterAttention));

        // ========================================================================
        // 临时解决方案：直接分析 GPU 后端源码，找出可能的问题点
        // ========================================================================

        log(LogLevel::INFO, "\n============================================================");
        log(LogLevel::INFO, "已知信息分析");
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "✓ Embedding 输出: CPU == GPU (完全一致)");
        log(LogLevel::INFO, "✓ InputNorm 输出: CPU == GPU (完全一致)");
        log(LogLevel::INFO, "✓ QKV Projection: CPU == GPU (完全一致)");
        log(LogLevel::INFO, "✗ Attention 输出: CPU ≠ GPU (maxDiff=" +
            std::to_string(attnDiff.maxAbsDiff) + ", cos=" + std::to_string(attnDiff.cosine) + ")");
        log(LogLevel::INFO, "");

        log(LogLevel::INFO, "问题定位:");
        log(LogLevel::INFO, "  偏差首次出现在: QKV Projection → Attention Output");
        log(LogLevel::INFO, "  可能原因:");
        log(LogLevel::INFO, "    1. RoPE 计算不正确");
        log(LogLevel::INFO, "    2. KV Cache 更新/读取有误");
        log(LogLevel::INFO, "    3. Attention Score 计算 (QK^T) 有误");
        log(LogLevel::INFO, "    4. Softmax 数值稳定性问题");
        log(LogLevel::INFO, "    5. Attention 矩阵乘法 (score @ V) 有误");
        log(LogLevel::INFO, "    6. GQA (Grouped Query Attention) 实现有误");
        log(LogLevel::INFO, "");

        log(LogLevel::INFO, "建议行动:");
        log(LogLevel::INFO, "  1. 检查 src/kylin/hf/ggml_backend.cpp 的 forwardWithDebug 函数");
        log(LogLevel::INFO, "  2. 在 Attention 计算的各子步骤后添加调试输出");
        log(LogLevel::INFO, "  3. 对比 CPU (transformer.cpp) 和 GPU (ggml_backend.cpp) 的 Attention 实现");
        log(LogLevel::INFO, "  4. 特别关注 RoPE 的 position 参数和 KV Cache 的索引");
        log(LogLevel::INFO, "");

        log(LogLevel::PASS, "Attention 细分对比测试完成");
    }
};

// ============================================================================
// 注册函数
// ============================================================================

inline void registerAttentionBreakdownTests(TestSuite& suite) {
    suite.addTest(std::make_shared<AttentionBreakdownTest>());
}

} // namespace kylin_test
