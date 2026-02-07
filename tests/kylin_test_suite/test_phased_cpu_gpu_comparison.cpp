/**
 * @file test_phased_cpu_gpu_comparison.cpp
 * @brief 分阶段 CPU vs GPU 精确对比测试
 *
 * 目标：CPU 结果正确，GPU 结果不对，需要逐步定位 GPU 偏差的源头。
 *
 * 测试阶段设计（每一阶段只有在前一阶段通过后才有意义）：
 *
 *   Phase 1 (Stage 30): 权重一致性验证
 *       - 对比 CPU / GPU 模型加载后，关键权重张量是否一致
 *       - 如果权重就不同，后续计算必然不同
 *
 *   Phase 2 (Stage 31): Embedding 层输出对比
 *       - 喂入同一个 token，对比 CPU / GPU 的 Embedding 输出
 *       - 定位问题是否出在权重查找阶段
 *
 *   Phase 3 (Stage 32): 逐层 Transformer 中间结果对比
 *       - 利用 forwardWithDebugCPU / forwardWithDebugGPU
 *       - 对比每一层的 InputNorm / QKV / Attention / PostNorm / FFN
 *       - 精确找出第一个出现显著偏差的层和子组件
 *
 *   Phase 4 (Stage 33): Logits 与 Top-K 对比
 *       - 对比最终 logits 分布
 *       - 对比 Top-10 token 排名是否一致
 *       - 验证 argmax 是否相同
 *
 *   Phase 5 (Stage 34): 多步生成对比
 *       - 使用贪婪解码，逐 token 对比 CPU / GPU 生成序列
 *       - 记录每步 logits 差异，找到偏差积累点
 *       - 最终对比解码文本
 *
 * 使用方法：
 *   ./kylin_test_suite --stage=30              # 单独运行 Phase 1
 *   ./kylin_test_suite --stage=31              # 单独运行 Phase 2
 *   ./kylin_test_suite --stage=32              # 单独运行 Phase 3
 *   ./kylin_test_suite --stage=33              # 单独运行 Phase 4
 *   ./kylin_test_suite --stage=34              # 单独运行 Phase 5
 *   ./kylin_test_suite --stage=30 --verbose    # 带详细输出
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <sstream>
#include <limits>

namespace kylin_test {

// ============================================================================
// 通用工具函数
// ============================================================================

/// 获取模型路径（优先环境变量）
static std::string getPhasedModelPath() {
    const char* envPath = std::getenv("CLLM_MODEL_PATH");
    if (envPath && std::string(envPath).length() > 0) return envPath;
    return "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
}

/// 计算向量统计
static TensorStats calcStats(const std::vector<float>& data) {
    TensorStats s;
    if (data.empty()) return s;
    s.min = *std::min_element(data.begin(), data.end());
    s.max = *std::max_element(data.begin(), data.end());
    double sum = 0, sumSq = 0;
    int nanCnt = 0, infCnt = 0;
    for (float v : data) {
        if (std::isnan(v)) { nanCnt++; continue; }
        if (std::isinf(v)) { infCnt++; continue; }
        sum += v;
        sumSq += static_cast<double>(v) * v;
    }
    int valid = static_cast<int>(data.size()) - nanCnt - infCnt;
    if (valid > 0) {
        s.mean = static_cast<float>(sum / valid);
        s.std  = static_cast<float>(std::sqrt(std::max(0.0, sumSq / valid - static_cast<double>(s.mean) * s.mean)));
    }
    return s;
}

/// 计算两个向量的差异指标
struct DiffMetrics {
    float maxAbsDiff  = 0.0f;
    float meanAbsDiff = 0.0f;
    float rmse        = 0.0f;
    float cosine      = 0.0f;  // 余弦相似度
    int   nanCountA   = 0;
    int   nanCountB   = 0;
    int   infCountA   = 0;
    int   infCountB   = 0;
    size_t maxDiffIdx = 0;     // 最大差异所在索引
    bool  sizeMatch   = true;
    size_t sizeA = 0, sizeB = 0;

    bool isGood(float tol = 0.01f) const {
        return sizeMatch && nanCountA == 0 && nanCountB == 0 &&
               infCountA == 0 && infCountB == 0 && maxAbsDiff < tol;
    }

    std::string summary() const {
        std::ostringstream oss;
        if (!sizeMatch) {
            oss << "SIZE MISMATCH (A=" << sizeA << ", B=" << sizeB << ")";
            return oss.str();
        }
        oss << "maxDiff=" << std::scientific << std::setprecision(3) << maxAbsDiff
            << " meanDiff=" << meanAbsDiff
            << " RMSE=" << rmse
            << " cos=" << std::fixed << std::setprecision(6) << cosine;
        if (nanCountA || nanCountB) oss << " NaN(A=" << nanCountA << ",B=" << nanCountB << ")";
        if (infCountA || infCountB) oss << " Inf(A=" << infCountA << ",B=" << infCountB << ")";
        oss << " @idx=" << maxDiffIdx;
        return oss.str();
    }
};

static DiffMetrics computeDiff(const std::vector<float>& a, const std::vector<float>& b) {
    DiffMetrics m;
    m.sizeA = a.size();
    m.sizeB = b.size();
    if (a.size() != b.size()) { m.sizeMatch = false; return m; }
    if (a.empty()) return m;

    double sumDiff = 0, sumDiffSq = 0;
    double dotAB = 0, normA = 0, normB = 0;

    for (size_t i = 0; i < a.size(); ++i) {
        if (std::isnan(a[i])) m.nanCountA++;
        if (std::isnan(b[i])) m.nanCountB++;
        if (std::isinf(a[i])) m.infCountA++;
        if (std::isinf(b[i])) m.infCountB++;

        float diff = std::abs(a[i] - b[i]);
        if (diff > m.maxAbsDiff) { m.maxAbsDiff = diff; m.maxDiffIdx = i; }
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

/// 打印向量的前 N 个值
static std::string firstN(const std::vector<float>& v, int n = 8) {
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < std::min(n, static_cast<int>(v.size())); ++i) {
        if (i) oss << ", ";
        oss << std::fixed << std::setprecision(6) << v[i];
    }
    if (static_cast<int>(v.size()) > n) oss << ", ...";
    oss << "]";
    return oss.str();
}

/// 获取 Top-K token IDs
static std::vector<std::pair<int, float>> topK(const std::vector<float>& logits, int k) {
    std::vector<std::pair<int, float>> idx;
    idx.reserve(logits.size());
    for (size_t i = 0; i < logits.size(); ++i)
        idx.push_back({static_cast<int>(i), logits[i]});
    std::partial_sort(idx.begin(),
        idx.begin() + std::min(static_cast<size_t>(k), idx.size()),
        idx.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });
    idx.resize(std::min(static_cast<size_t>(k), idx.size()));
    return idx;
}

/// 贪婪选择
static int argmax(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// ============================================================================
// Phase 1 (Stage 30): 权重一致性验证
// ============================================================================

class Phase1WeightConsistencyTest : public TestCase {
public:
    Phase1WeightConsistencyTest() : TestCase(
        "phase1_weight_consistency",
        "Phase 1: CPU vs GPU 权重一致性验证 - 确认权重上传到 GPU 后未损坏"
    ) {}

    void execute() override {
        std::string modelPath = getPhasedModelPath();
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Phase 1: 权重一致性验证");
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "模型路径: " + modelPath);

        // 1. 加载 CPU 模型 (FP32)
        log(LogLevel::INFO, "\n[Step 1] 加载 CPU 模型 (FP32)...");
        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载成功");
        log(LogLevel::INFO, "  hiddenSize=" + std::to_string(cpuModel.hiddenSize()) +
                          ", vocabSize=" + std::to_string(cpuModel.vocabSize()));

        // 2. 加载 GPU 模型 (Metal)
        log(LogLevel::INFO, "\n[Step 2] 加载 GPU 模型 (Metal)...");
        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded(), "GPU 模型加载成功");
        assertTrue(gpuModel.isUsingGPU(), "GPU 后端已启用");

        // 3. 通过单 token forward 间接验证权重
        //    如果 Embedding 查找结果一致，说明 embed_tokens 权重上传正确
        log(LogLevel::INFO, "\n[Step 3] 通过 Embedding 查找验证权重一致性...");
        
        // 用几个不同的 token ID 测试
        std::vector<int32_t> testTokenIds = {0, 1, 100, 1000, 10000, 50000, 100000, 151644};
        
        for (int32_t tokenId : testTokenIds) {
            if (tokenId >= cpuModel.vocabSize()) continue;
            
            // CPU Embedding
            std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayers;
            std::vector<float> cpuEmb, cpuNorm;
            cpuModel.resetKVCache();
            auto cpuLogits = cpuModel.forwardWithDebugCPU({tokenId}, cpuLayers, cpuEmb, cpuNorm);
            
            // GPU Embedding
            std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayers;
            std::vector<float> gpuEmb, gpuNorm;
            gpuModel.resetKVCache();
            auto gpuLogits = gpuModel.forwardWithDebugGPU({tokenId}, gpuLayers, gpuEmb, gpuNorm);
            
            auto embDiff = computeDiff(cpuEmb, gpuEmb);
            
            std::string status = embDiff.isGood(0.01f) ? "✓" : "✗";
            log(embDiff.isGood(0.01f) ? LogLevel::INFO : LogLevel::WARN,
                "  Token " + std::to_string(tokenId) + ": " + status +
                " embDiff: " + embDiff.summary());
            
            if (!embDiff.isGood(0.1f)) {
                log(LogLevel::WARN, "    CPU Embedding 前8值: " + firstN(cpuEmb));
                log(LogLevel::WARN, "    GPU Embedding 前8值: " + firstN(gpuEmb));
            }
        }
        
        log(LogLevel::PASS, "\nPhase 1 完成: 权重一致性验证结束");
    }
};

// ============================================================================
// Phase 2 (Stage 31): Embedding 层输出对比
// ============================================================================

class Phase2EmbeddingComparisonTest : public TestCase {
public:
    Phase2EmbeddingComparisonTest() : TestCase(
        "phase2_embedding_comparison",
        "Phase 2: CPU vs GPU Embedding 层输出精确对比"
    ) {}

    void execute() override {
        std::string modelPath = getPhasedModelPath();
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Phase 2: Embedding 层输出对比");
        log(LogLevel::INFO, "============================================================");

        // 加载模型
        log(LogLevel::INFO, "\n加载 CPU 模型...");
        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载");

        log(LogLevel::INFO, "加载 GPU 模型...");
        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded() && gpuModel.isUsingGPU(), "GPU 模型加载");

        // 使用有代表性的 token 进行测试
        // 151644 = <|im_start|>, 8948 = system 相关, 198 = 换行
        std::vector<std::pair<int32_t, std::string>> testCases = {
            {151644, "<|im_start|>"},
            {8948,   "system相关token"},
            {198,    "换行token"},
            {9707,   "'hello'"},
            {104169, "'你好'"},
        };

        bool allPassed = true;
        for (const auto& [tokenId, desc] : testCases) {
            if (tokenId >= cpuModel.vocabSize()) continue;
            
            log(LogLevel::INFO, "\n--- 测试 Token " + std::to_string(tokenId) + " (" + desc + ") ---");

            cpuModel.resetKVCache();
            gpuModel.resetKVCache();
            
            std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayers;
            std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayers;
            std::vector<float> cpuEmb, gpuEmb, cpuNorm, gpuNorm;

            cpuModel.forwardWithDebugCPU({tokenId}, cpuLayers, cpuEmb, cpuNorm);
            gpuModel.forwardWithDebugGPU({tokenId}, gpuLayers, gpuEmb, gpuNorm);

            auto embDiff = computeDiff(cpuEmb, gpuEmb);
            auto cpuStats = calcStats(cpuEmb);
            auto gpuStats = calcStats(gpuEmb);

            log(LogLevel::INFO, "  CPU Embedding: size=" + std::to_string(cpuEmb.size()) +
                              " min=" + std::to_string(cpuStats.min) +
                              " max=" + std::to_string(cpuStats.max) +
                              " mean=" + std::to_string(cpuStats.mean));
            log(LogLevel::INFO, "  GPU Embedding: size=" + std::to_string(gpuEmb.size()) +
                              " min=" + std::to_string(gpuStats.min) +
                              " max=" + std::to_string(gpuStats.max) +
                              " mean=" + std::to_string(gpuStats.mean));
            log(LogLevel::INFO, "  差异: " + embDiff.summary());
            log(LogLevel::INFO, "  CPU 前8值: " + firstN(cpuEmb));
            log(LogLevel::INFO, "  GPU 前8值: " + firstN(gpuEmb));

            if (!embDiff.isGood(0.01f)) {
                log(LogLevel::WARN, "  ⚠ Embedding 差异超过阈值!");
                allPassed = false;
            } else {
                log(LogLevel::INFO, "  ✓ Embedding 基本一致 (cos=" +
                    std::to_string(embDiff.cosine) + ")");
            }
        }

        if (allPassed) {
            log(LogLevel::PASS, "\nPhase 2 结论: Embedding 层 CPU/GPU 一致 → 问题在后续层");
        } else {
            log(LogLevel::WARN, "\nPhase 2 结论: Embedding 层已有偏差 → 问题可能在权重上传或精度转换");
        }
    }
};

// ============================================================================
// Phase 3 (Stage 32): 逐层 Transformer 中间结果对比
// ============================================================================

class Phase3LayerByLayerComparisonTest : public TestCase {
public:
    Phase3LayerByLayerComparisonTest() : TestCase(
        "phase3_layer_by_layer_comparison",
        "Phase 3: 逐层 Transformer 中间结果精确对比 - 定位偏差源头"
    ) {}

    void execute() override {
        std::string modelPath = getPhasedModelPath();
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Phase 3: 逐层 Transformer 中间结果对比");
        log(LogLevel::INFO, "============================================================");

        // 加载模型
        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载");

        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded() && gpuModel.isUsingGPU(), "GPU 模型加载");

        // 使用 token 9707 ("hello" 相关) 进行测试
        int32_t testToken = 9707;
        log(LogLevel::INFO, "测试 Token ID: " + std::to_string(testToken));

        cpuModel.resetKVCache();
        gpuModel.resetKVCache();

        std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayers;
        std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayers;
        std::vector<float> cpuEmb, gpuEmb, cpuFinalNorm, gpuFinalNorm;

        log(LogLevel::INFO, "\n执行 CPU forwardWithDebug...");
        auto cpuLogits = cpuModel.forwardWithDebugCPU({testToken}, cpuLayers, cpuEmb, cpuFinalNorm);
        log(LogLevel::INFO, "CPU 完成: " + std::to_string(cpuLayers.size()) + " 层, logits size=" +
            std::to_string(cpuLogits.size()));

        log(LogLevel::INFO, "执行 GPU forwardWithDebug...");
        auto gpuLogits = gpuModel.forwardWithDebugGPU({testToken}, gpuLayers, gpuEmb, gpuFinalNorm);
        log(LogLevel::INFO, "GPU 完成: " + std::to_string(gpuLayers.size()) + " 层, logits size=" +
            std::to_string(gpuLogits.size()));

        assertTrue(!cpuLogits.empty(), "CPU logits 非空");
        assertTrue(!gpuLogits.empty(), "GPU logits 非空");

        // 0. Embedding 对比
        log(LogLevel::INFO, "\n--- Embedding ---");
        auto embDiff = computeDiff(cpuEmb, gpuEmb);
        log(embDiff.isGood(0.01f) ? LogLevel::INFO : LogLevel::WARN,
            "  " + embDiff.summary());

        // 1. 逐层对比
        int numLayers = std::min(cpuLayers.size(), gpuLayers.size());
        int firstBadLayer = -1;
        std::string firstBadComponent;

        log(LogLevel::INFO, "\n--- 逐层对比 (" + std::to_string(numLayers) + " 层) ---");
        log(LogLevel::INFO, std::string(120, '-'));
        log(LogLevel::INFO, "Layer | InputNorm maxDiff | QKV maxDiff      | Attention maxDiff | PostNorm maxDiff | FFN maxDiff");
        log(LogLevel::INFO, std::string(120, '-'));

        for (int i = 0; i < numLayers; ++i) {
            auto normDiff = computeDiff(cpuLayers[i].inputNormOutput, gpuLayers[i].afterInputNorm);
            auto qkvDiff  = computeDiff(cpuLayers[i].qkvOutput,       gpuLayers[i].afterQKV);
            auto attnDiff = computeDiff(cpuLayers[i].attentionOutput,  gpuLayers[i].afterAttention);
            auto postDiff = computeDiff(cpuLayers[i].postNormOutput,   gpuLayers[i].afterPostNorm);
            auto ffnDiff  = computeDiff(cpuLayers[i].ffnOutput,        gpuLayers[i].afterFFN);

            // 格式化一行输出
            std::ostringstream row;
            row << std::setw(5) << i << " | ";

            auto fmtDiff = [](const DiffMetrics& d) -> std::string {
                if (!d.sizeMatch) return "SIZE_MISMATCH    ";
                std::ostringstream o;
                o << std::scientific << std::setprecision(3) << std::setw(12) << d.maxAbsDiff
                  << " cos=" << std::fixed << std::setprecision(4) << d.cosine;
                return o.str();
            };

            row << fmtDiff(normDiff) << " | "
                << fmtDiff(qkvDiff)  << " | "
                << fmtDiff(attnDiff) << " | "
                << fmtDiff(postDiff) << " | "
                << fmtDiff(ffnDiff);

            // 判断是否有显著偏差
            float threshold = 0.1f;  // 对于 FP32 vs Metal FP16 混合计算
            bool layerBad = false;
            
            auto checkComponent = [&](const DiffMetrics& d, const std::string& name) {
                if (!d.sizeMatch || d.maxAbsDiff > threshold || d.cosine < 0.99f) {
                    if (firstBadLayer < 0) {
                        firstBadLayer = i;
                        firstBadComponent = name;
                    }
                    layerBad = true;
                }
            };
            
            checkComponent(normDiff, "InputNorm");
            checkComponent(qkvDiff,  "QKV");
            checkComponent(attnDiff, "Attention");
            checkComponent(postDiff, "PostNorm");
            checkComponent(ffnDiff,  "FFN");

            log(layerBad ? LogLevel::WARN : LogLevel::INFO, row.str());

            // 对于第一个出问题的层，打印详细信息
            if (layerBad && i == firstBadLayer) {
                log(LogLevel::WARN, "  *** 首个偏差层 Layer " + std::to_string(i) +
                    " 组件: " + firstBadComponent + " ***");
                
                // 打印该层各组件的详细对比
                auto printDetail = [this](const std::string& name,
                                         const std::vector<float>& cpu,
                                         const std::vector<float>& gpu) {
                    auto d = computeDiff(cpu, gpu);
                    log(LogLevel::INFO, "    " + name + ": " + d.summary());
                    log(LogLevel::INFO, "      CPU 前8值: " + firstN(cpu));
                    log(LogLevel::INFO, "      GPU 前8值: " + firstN(gpu));
                    auto cs = calcStats(cpu), gs = calcStats(gpu);
                    log(LogLevel::INFO, "      CPU stats: min=" + std::to_string(cs.min) +
                        " max=" + std::to_string(cs.max) + " mean=" + std::to_string(cs.mean));
                    log(LogLevel::INFO, "      GPU stats: min=" + std::to_string(gs.min) +
                        " max=" + std::to_string(gs.max) + " mean=" + std::to_string(gs.mean));
                };
                
                printDetail("InputNorm", cpuLayers[i].inputNormOutput, gpuLayers[i].afterInputNorm);
                printDetail("QKV",       cpuLayers[i].qkvOutput,       gpuLayers[i].afterQKV);
                printDetail("Attention",  cpuLayers[i].attentionOutput, gpuLayers[i].afterAttention);
                printDetail("PostNorm",   cpuLayers[i].postNormOutput,  gpuLayers[i].afterPostNorm);
                printDetail("FFN",        cpuLayers[i].ffnOutput,       gpuLayers[i].afterFFN);
            }
        }

        // 2. Final Norm 对比
        log(LogLevel::INFO, "\n--- Final RMSNorm ---");
        auto fnDiff = computeDiff(cpuFinalNorm, gpuFinalNorm);
        log(fnDiff.isGood(0.1f) ? LogLevel::INFO : LogLevel::WARN,
            "  " + fnDiff.summary());
        log(LogLevel::INFO, "  CPU 前8值: " + firstN(cpuFinalNorm));
        log(LogLevel::INFO, "  GPU 前8值: " + firstN(gpuFinalNorm));

        // 3. 总结
        log(LogLevel::INFO, "\n============================================================");
        log(LogLevel::INFO, "Phase 3 总结");
        log(LogLevel::INFO, "============================================================");
        if (firstBadLayer < 0) {
            log(LogLevel::PASS, "所有 " + std::to_string(numLayers) + " 层中间结果一致!");
        } else {
            log(LogLevel::WARN, "首个偏差出现在: Layer " + std::to_string(firstBadLayer) +
                " -> " + firstBadComponent);
            log(LogLevel::INFO, "建议: 检查 GPU 后端中 Layer " + std::to_string(firstBadLayer) +
                " 的 " + firstBadComponent + " 实现");
        }
    }
};

// ============================================================================
// Phase 4 (Stage 33): Logits 与 Top-K 对比
// ============================================================================

class Phase4LogitsComparisonTest : public TestCase {
public:
    Phase4LogitsComparisonTest() : TestCase(
        "phase4_logits_comparison",
        "Phase 4: Logits 分布与 Top-K 排名对比 - 验证最终输出"
    ) {}

    void execute() override {
        std::string modelPath = getPhasedModelPath();
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Phase 4: Logits 与 Top-K 对比");
        log(LogLevel::INFO, "============================================================");

        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载");

        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded() && gpuModel.isUsingGPU(), "GPU 模型加载");

        // 测试多个不同的 token
        std::vector<std::pair<int32_t, std::string>> testCases = {
            {151644, "<|im_start|>"},
            {9707,   "hello"},
            {104169, "你好"},
            {220,    "空格"},
            {16,     "'1'"},
        };

        for (const auto& [tokenId, desc] : testCases) {
            if (tokenId >= cpuModel.vocabSize()) continue;
            
            log(LogLevel::INFO, "\n--- Token " + std::to_string(tokenId) + " (" + desc + ") ---");

            cpuModel.resetKVCache();
            gpuModel.resetKVCache();

            std::vector<cllm::kylin::HFTransformerModel::LayerDebugOutput> cpuLayersDbg;
            std::vector<cllm::kylin::GGMLGPUBackend::LayerOutput> gpuLayersDbg;
            std::vector<float> cpuEmb, gpuEmb, cpuNorm, gpuNorm;

            auto cpuLogits = cpuModel.forwardWithDebugCPU({tokenId}, cpuLayersDbg, cpuEmb, cpuNorm);
            auto gpuLogits = gpuModel.forwardWithDebugGPU({tokenId}, gpuLayersDbg, gpuEmb, gpuNorm);

            if (cpuLogits.empty() || gpuLogits.empty()) {
                log(LogLevel::WARN, "  logits 为空，跳过");
                continue;
            }

            // Logits 差异
            auto logitsDiff = computeDiff(cpuLogits, gpuLogits);
            log(LogLevel::INFO, "  Logits 差异: " + logitsDiff.summary());

            // argmax 对比
            int cpuArgmax = argmax(cpuLogits);
            int gpuArgmax = argmax(gpuLogits);
            log(cpuArgmax == gpuArgmax ? LogLevel::INFO : LogLevel::WARN,
                "  Argmax: CPU=" + std::to_string(cpuArgmax) +
                " GPU=" + std::to_string(gpuArgmax) +
                (cpuArgmax == gpuArgmax ? " ✓" : " ✗ 不一致!"));

            // Top-10 对比
            auto cpuTop = topK(cpuLogits, 10);
            auto gpuTop = topK(gpuLogits, 10);

            log(LogLevel::INFO, "  CPU Top-10:");
            for (const auto& [id, val] : cpuTop) {
                log(LogLevel::INFO, "    token=" + std::to_string(id) + " logit=" + std::to_string(val));
            }
            log(LogLevel::INFO, "  GPU Top-10:");
            for (const auto& [id, val] : gpuTop) {
                log(LogLevel::INFO, "    token=" + std::to_string(id) + " logit=" + std::to_string(val));
            }

            // 计算 Top-10 排名重叠度
            int overlap = 0;
            for (const auto& [cpuId, _] : cpuTop) {
                for (const auto& [gpuId, __] : gpuTop) {
                    if (cpuId == gpuId) { overlap++; break; }
                }
            }
            log(LogLevel::INFO, "  Top-10 重叠: " + std::to_string(overlap) + "/10");
        }

        log(LogLevel::PASS, "\nPhase 4 完成");
    }
};

// ============================================================================
// Phase 5 (Stage 34): 多步生成对比
// ============================================================================

class Phase5GenerationComparisonTest : public TestCase {
public:
    Phase5GenerationComparisonTest() : TestCase(
        "phase5_generation_comparison",
        "Phase 5: 多步贪婪生成对比 - 对比实际文本输出"
    ) {}

    void execute() override {
        std::string modelPath = getPhasedModelPath();
        log(LogLevel::INFO, "============================================================");
        log(LogLevel::INFO, "Phase 5: 多步贪婪生成对比");
        log(LogLevel::INFO, "============================================================");

        // 加载 tokenizer
        log(LogLevel::INFO, "加载 Tokenizer...");
        cllm::HFTokenizer tokenizer;
        assertTrue(tokenizer.load(modelPath), "Tokenizer 加载");
        log(LogLevel::INFO, "Tokenizer 加载成功, EOS=" + std::to_string(tokenizer.getEosId()));

        // 加载模型
        cllm::kylin::HFTransformerModel cpuModel(modelPath, cllm::kylin::DeviceType::CPU,
                                                   cllm::kylin::QuantType::FP32);
        assertTrue(cpuModel.isLoaded(), "CPU 模型加载");

        cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
        assertTrue(gpuModel.isLoaded() && gpuModel.isUsingGPU(), "GPU 模型加载");

        // 测试 prompt
        std::vector<std::pair<std::string, int>> prompts = {
            {"hello",      20},
            {"你好",       20},
            {"1+1=",       10},
            {"What is AI", 20},
        };

        for (const auto& [prompt, maxTokens] : prompts) {
            log(LogLevel::INFO, "\n============================================================");
            log(LogLevel::INFO, "Prompt: \"" + prompt + "\" | maxTokens=" + std::to_string(maxTokens));
            log(LogLevel::INFO, "============================================================");

            // 编码
            auto inputIds = tokenizer.encode(prompt, true);
            log(LogLevel::INFO, "Input IDs (" + std::to_string(inputIds.size()) + " tokens): " +
                vecToStr(inputIds));

            // CPU 生成
            cpuModel.resetKVCache();
            std::vector<int32_t> cpuInputs(inputIds.begin(), inputIds.end());
            std::vector<int> cpuGenerated;
            std::vector<float> cpuMaxDiffs;  // 记录每步的 logits 差异（如果有）

            log(LogLevel::INFO, "\n--- CPU 贪婪生成 ---");
            // Prefill
            auto cpuLogits = cpuModel.forward(cpuInputs);
            int cpuNext = argmax(cpuLogits);
            cpuGenerated.push_back(cpuNext);

            // Decode
            for (int step = 1; step < maxTokens; ++step) {
                if (cpuNext == tokenizer.getEosId()) break;
                cpuLogits = cpuModel.forward({cpuNext});
                cpuNext = argmax(cpuLogits);
                cpuGenerated.push_back(cpuNext);
            }

            std::vector<int> cpuAllTokens(inputIds.begin(), inputIds.end());
            cpuAllTokens.insert(cpuAllTokens.end(), cpuGenerated.begin(), cpuGenerated.end());
            std::string cpuText = tokenizer.decode(cpuAllTokens, true);
            log(LogLevel::INFO, "CPU tokens: " + vecToStr(cpuGenerated));
            log(LogLevel::INFO, "CPU 生成文本: \"" + cpuText + "\"");

            // GPU 生成
            gpuModel.resetKVCache();
            std::vector<int32_t> gpuInputs(inputIds.begin(), inputIds.end());
            std::vector<int> gpuGenerated;

            log(LogLevel::INFO, "\n--- GPU 贪婪生成 ---");
            // Prefill
            auto gpuLogits = gpuModel.forward(gpuInputs);
            int gpuNext = argmax(gpuLogits);
            gpuGenerated.push_back(gpuNext);

            // 记录首 token 差异
            auto firstDiff = computeDiff(cpuLogits, gpuLogits);
            log(LogLevel::INFO, "Prefill logits 差异: " + firstDiff.summary());

            // Decode
            for (int step = 1; step < maxTokens; ++step) {
                if (gpuNext == tokenizer.getEosId()) break;
                gpuLogits = gpuModel.forward({gpuNext});
                gpuNext = argmax(gpuLogits);
                gpuGenerated.push_back(gpuNext);
            }

            std::vector<int> gpuAllTokens(inputIds.begin(), inputIds.end());
            gpuAllTokens.insert(gpuAllTokens.end(), gpuGenerated.begin(), gpuGenerated.end());
            std::string gpuText = tokenizer.decode(gpuAllTokens, true);
            log(LogLevel::INFO, "GPU tokens: " + vecToStr(gpuGenerated));
            log(LogLevel::INFO, "GPU 生成文本: \"" + gpuText + "\"");

            // 对比
            log(LogLevel::INFO, "\n--- 逐 token 对比 ---");
            int firstDiffStep = -1;
            int minLen = std::min(cpuGenerated.size(), gpuGenerated.size());
            for (int i = 0; i < minLen; ++i) {
                std::string mark = (cpuGenerated[i] == gpuGenerated[i]) ? "✓" : "✗";
                if (cpuGenerated[i] != gpuGenerated[i] && firstDiffStep < 0) {
                    firstDiffStep = i;
                }
                log(cpuGenerated[i] == gpuGenerated[i] ? LogLevel::INFO : LogLevel::WARN,
                    "  Step " + std::to_string(i) + ": CPU=" + std::to_string(cpuGenerated[i]) +
                    " GPU=" + std::to_string(gpuGenerated[i]) + " " + mark);
            }

            // 总结
            log(LogLevel::INFO, "\n--- 总结 ---");
            if (cpuGenerated == gpuGenerated) {
                log(LogLevel::PASS, "✓ CPU 和 GPU 生成序列完全一致!");
            } else {
                log(LogLevel::WARN, "✗ CPU 和 GPU 生成序列不同");
                if (firstDiffStep >= 0) {
                    log(LogLevel::WARN, "  第一个差异出现在 Step " + std::to_string(firstDiffStep));
                }
                log(LogLevel::INFO, "  CPU 文本: \"" + cpuText + "\"");
                log(LogLevel::INFO, "  GPU 文本: \"" + gpuText + "\"");
            }
        }

        log(LogLevel::PASS, "\nPhase 5 完成");
    }

private:
    std::string vecToStr(const std::vector<int>& v, size_t maxN = 15) {
        std::ostringstream oss;
        oss << "[";
        for (size_t i = 0; i < std::min(v.size(), maxN); ++i) {
            if (i) oss << ", ";
            oss << v[i];
        }
        if (v.size() > maxN) oss << ", ... (" << v.size() << " total)";
        oss << "]";
        return oss.str();
    }
};

// ============================================================================
// 注册函数
// ============================================================================

inline void registerPhasedCPUGPUTests_Phase1(TestSuite& suite) {
    suite.addTest(std::make_shared<Phase1WeightConsistencyTest>());
}

inline void registerPhasedCPUGPUTests_Phase2(TestSuite& suite) {
    suite.addTest(std::make_shared<Phase2EmbeddingComparisonTest>());
}

inline void registerPhasedCPUGPUTests_Phase3(TestSuite& suite) {
    suite.addTest(std::make_shared<Phase3LayerByLayerComparisonTest>());
}

inline void registerPhasedCPUGPUTests_Phase4(TestSuite& suite) {
    suite.addTest(std::make_shared<Phase4LogitsComparisonTest>());
}

inline void registerPhasedCPUGPUTests_Phase5(TestSuite& suite) {
    suite.addTest(std::make_shared<Phase5GenerationComparisonTest>());
}

} // namespace kylin_test
