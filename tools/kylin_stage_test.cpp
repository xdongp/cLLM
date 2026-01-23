/**
 * @file kylin_stage_test.cpp
 * @brief Kylin 后端分阶段测试框架
 * 
 * 参考 incremental_benchmark.cpp 的设计，逐步测试 Kylin 后端的每个组件
 */

#include "cllm/inference/kylin_backend.h"
#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/kylin/ggml_transformer.h"
#include "cllm/model/config.h"
#include "cllm/common/config.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace cllm;
using namespace cllm::inference;

// 测试配置
struct TestConfig {
    std::string modelPath;
    std::string prompt;
    size_t maxTokens;
    float temperature;
    bool compareWithLlamaCpp;
};

// 测试结果
struct StageResult {
    std::string stageName;
    bool passed;
    double timeMs;
    std::string errorMsg;
    std::map<std::string, std::string> metrics;
};

// 测试报告
struct TestReport {
    std::vector<StageResult> stages;
    std::string summary;
};

// ========== Stage 0: 基础环境验证 ==========
StageResult testStage0_BasicEnvironment(const TestConfig& config) {
    StageResult result;
    result.stageName = "Stage 0: Basic Environment";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        // 1. 验证配置
        ModelConfig modelConfig;
        modelConfig.vocabSize = 151936;  // Qwen3-0.6B
        modelConfig.hiddenSize = 1024;
        modelConfig.numLayers = 28;
        modelConfig.numAttentionHeads = 16;
        modelConfig.numKeyValueHeads = 8;
        modelConfig.intermediateSize = 3072;
        modelConfig.maxSequenceLength = 40960;
        
        // 2. 创建 KylinBackend
        KylinBackend backend(modelConfig, config.modelPath);
        
        // 3. 验证初始化
        if (!backend.initialize()) {
            result.passed = false;
            result.errorMsg = "Failed to initialize KylinBackend";
            return result;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.passed = true;
        result.metrics["initialization_time_ms"] = std::to_string(result.timeMs);
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.errorMsg = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// ========== Stage 1: 模型加载验证 ==========
StageResult testStage1_ModelLoading(const TestConfig& config) {
    StageResult result;
    result.stageName = "Stage 1: Model Loading";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        ModelConfig modelConfig;
        modelConfig.vocabSize = 151936;
        modelConfig.hiddenSize = 1024;
        modelConfig.numLayers = 28;
        modelConfig.numAttentionHeads = 16;
        modelConfig.numKeyValueHeads = 8;
        modelConfig.intermediateSize = 3072;
        modelConfig.maxSequenceLength = 40960;
        
        KylinBackend backend(modelConfig, config.modelPath);
        
        if (!backend.initialize()) {
            result.passed = false;
            result.errorMsg = "Failed to initialize";
            return result;
        }
        
        // 验证配置
        const auto& config_loaded = backend.getConfig();
        result.metrics["vocab_size"] = std::to_string(config_loaded.vocabSize);
        result.metrics["hidden_size"] = std::to_string(config_loaded.hiddenSize);
        result.metrics["num_layers"] = std::to_string(config_loaded.numLayers);
        result.metrics["num_heads"] = std::to_string(config_loaded.numAttentionHeads);
        result.metrics["num_kv_heads"] = std::to_string(config_loaded.numKeyValueHeads);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.passed = true;
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.errorMsg = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// ========== Stage 2: Token Embedding 验证 ==========
StageResult testStage2_Embedding(const TestConfig& config) {
    StageResult result;
    result.stageName = "Stage 2: Token Embedding";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        ModelConfig modelConfig;
        modelConfig.vocabSize = 151936;
        modelConfig.hiddenSize = 1024;
        modelConfig.numLayers = 28;
        modelConfig.numAttentionHeads = 16;
        modelConfig.numKeyValueHeads = 8;
        modelConfig.intermediateSize = 3072;
        modelConfig.maxSequenceLength = 40960;
        
        KylinBackend backend(modelConfig, config.modelPath);
        
        if (!backend.initialize()) {
            result.passed = false;
            result.errorMsg = "Failed to initialize";
            return result;
        }
        
        // 测试 embedding lookup
        std::vector<int> inputIds = {9707};  // "Hello"
        auto output = backend.forward(inputIds);
        
        // 验证输出形状
        const auto& shape = output.shape();
        if (shape.size() != 2 || shape[0] != 1 || shape[1] != modelConfig.vocabSize) {
            result.passed = false;
            result.errorMsg = "Invalid output shape";
            return result;
        }
        
        // 计算统计信息（从 logits 中提取，实际应该从中间结果获取）
        // 这里只是验证 forward 能正常执行
        result.metrics["output_shape"] = std::to_string(shape[0]) + "x" + std::to_string(shape[1]);
        
        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.passed = true;
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.errorMsg = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// ========== Stage 3: 第一层验证 ==========
StageResult testStage3_Layer0(const TestConfig& config) {
    StageResult result;
    result.stageName = "Stage 3: Layer 0 Output";
    
    // 类似 Stage 2，但需要验证 Layer 0 的输出
    // 实际实现需要从日志中提取或添加回调
    
    result.passed = true;  // 占位
    return result;
}

// ========== Stage 8: 增量推理验证 ==========
StageResult testStage8_IncrementalInference(const TestConfig& config) {
    StageResult result;
    result.stageName = "Stage 8: Incremental Inference (KV Cache)";
    
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        using namespace cllm::kylin;
        
        // 1. 创建并加载模型
        GGMLTransformerModel model(BackendType::CPU);
        
        if (!model.loadFromGGUF(config.modelPath)) {
            result.passed = false;
            result.errorMsg = "Failed to load model from GGUF";
            return result;
        }
        
        const auto& modelConfig = model.getConfig();
        const size_t vocabSize = modelConfig.vocabSize;
        
        // 2. 测试用例 8.1: 首次推理（startPos=0）
        std::cout << "  [8.1] Testing first inference (startPos=0)..." << std::endl;
        
        model.clearKVCache();
        std::vector<int32_t> firstToken = {9707};  // "Hello"
        
        auto firstLogits = model.forward(firstToken);
        
        if (firstLogits.size() != vocabSize) {
            result.passed = false;
            result.errorMsg = "First inference: invalid logits size";
            return result;
        }
        
        size_t firstKVCacheLen = model.getKVCacheLength();
        if (firstKVCacheLen != 1) {
            result.passed = false;
            result.errorMsg = "First inference: KV cache length should be 1, got " + std::to_string(firstKVCacheLen);
            return result;
        }
        
        result.metrics["first_inference_kv_cache_len"] = std::to_string(firstKVCacheLen);
        std::cout << "    ✅ First inference OK: KV cache len=" << firstKVCacheLen << std::endl;
        
        // 3. 测试用例 8.2: 增量推理（逐步添加 token）
        std::cout << "  [8.2] Testing incremental inference..." << std::endl;
        
        model.clearKVCache();
        std::vector<int32_t> tokens = {9707, 11, 1234};  // "Hello world" 的 token IDs
        
        std::vector<std::vector<float>> incrementalLogits;
        
        for (size_t i = 0; i < tokens.size(); ++i) {
            auto tokenLogits = model.forwardOneToken(tokens[i], i);
            
            if (tokenLogits.size() != vocabSize) {
                result.passed = false;
                result.errorMsg = "Incremental step " + std::to_string(i) + ": invalid logits size";
                return result;
            }
            
            incrementalLogits.push_back(tokenLogits);
            
            size_t currentKVCacheLen = model.getKVCacheLength();
            if (currentKVCacheLen != i + 1) {
                result.passed = false;
                result.errorMsg = "Incremental step " + std::to_string(i) + 
                                ": KV cache length should be " + std::to_string(i + 1) + 
                                ", got " + std::to_string(currentKVCacheLen);
                return result;
            }
            
            std::cout << "    Step " << i << ": token=" << tokens[i] 
                      << ", KV cache len=" << currentKVCacheLen << std::endl;
        }
        
        result.metrics["incremental_steps"] = std::to_string(tokens.size());
        result.metrics["final_kv_cache_len"] = std::to_string(model.getKVCacheLength());
        std::cout << "    ✅ Incremental inference OK" << std::endl;
        
        // 4. 测试用例 8.3: 批量推理 vs 增量推理的一致性验证
        std::cout << "  [8.3] Testing batch vs incremental consistency..." << std::endl;
        
        // 4.1 批量推理（全序列）
        model.clearKVCache();
        auto batchLogits = model.forward(tokens);
        
        if (batchLogits.size() != tokens.size() * vocabSize) {
            result.passed = false;
            result.errorMsg = "Batch inference: invalid logits size";
            return result;
        }
        
        // 4.2 提取最后一个位置的 logits（批量推理）
        size_t lastPos = tokens.size() - 1;
        std::vector<float> batchLastLogits(
            batchLogits.begin() + lastPos * vocabSize,
            batchLogits.begin() + (lastPos + 1) * vocabSize
        );
        
        // 4.3 对比最后一个位置的 logits（增量推理的最后一个）
        std::vector<float>& incrLastLogits = incrementalLogits.back();
        
        // 计算差异
        float maxDiff = 0.0f;
        float sumDiff = 0.0f;
        int maxDiffIdx = 0;
        size_t largeDiffCount = 0;  // 差异 > 1e-3 的数量
        
        for (size_t i = 0; i < vocabSize; ++i) {
            float diff = std::abs(batchLastLogits[i] - incrLastLogits[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = static_cast<int>(i);
            }
            sumDiff += diff;
            if (diff > 1e-3f) {
                largeDiffCount++;
            }
        }
        
        float avgDiff = sumDiff / vocabSize;
        
        result.metrics["max_logit_diff"] = std::to_string(maxDiff);
        result.metrics["avg_logit_diff"] = std::to_string(avgDiff);
        result.metrics["max_diff_idx"] = std::to_string(maxDiffIdx);
        result.metrics["large_diff_count"] = std::to_string(largeDiffCount);
        
        std::cout << "    Max logit diff: " << std::fixed << std::setprecision(6) << maxDiff 
                  << " (at idx " << maxDiffIdx << ")" << std::endl;
        std::cout << "    Avg logit diff: " << std::fixed << std::setprecision(6) << avgDiff << std::endl;
        std::cout << "    Large diffs (>1e-3): " << largeDiffCount << " / " << vocabSize << std::endl;
        
        // 4.4 验证一致性（允许小的数值误差）
        // 对于浮点计算，允许一定的误差（1e-2 量级）
        const float tolerance = 1e-2f;
        
        if (maxDiff > tolerance) {
            result.passed = false;
            result.errorMsg = "Batch vs incremental mismatch: max diff=" + 
                            std::to_string(maxDiff) + " > tolerance=" + std::to_string(tolerance) +
                            " (at idx " + std::to_string(maxDiffIdx) + ")";
            
            // 显示具体差异值
            std::cout << "    ❌ Mismatch detected:" << std::endl;
            std::cout << "      Batch[" << maxDiffIdx << "] = " 
                      << std::fixed << std::setprecision(6) << batchLastLogits[maxDiffIdx] << std::endl;
            std::cout << "      Incr[" << maxDiffIdx << "]  = " 
                      << std::fixed << std::setprecision(6) << incrLastLogits[maxDiffIdx] << std::endl;
            
            return result;
        }
        
        std::cout << "    ✅ Batch vs incremental consistency OK (max diff=" 
                  << std::fixed << std::setprecision(6) << maxDiff << " < " << tolerance << ")" << std::endl;
        
        // 5. 测试用例 8.4: 验证中间步骤的一致性
        std::cout << "  [8.4] Testing intermediate step consistency..." << std::endl;
        
        for (size_t step = 1; step < tokens.size(); ++step) {
            // 批量推理：前 step+1 个 token
            model.clearKVCache();
            std::vector<int32_t> partialTokens(tokens.begin(), tokens.begin() + step + 1);
            auto partialBatchLogits = model.forward(partialTokens);
            
            // 提取最后一个位置的 logits
            std::vector<float> partialBatchLast(
                partialBatchLogits.begin() + step * vocabSize,
                partialBatchLogits.begin() + (step + 1) * vocabSize
            );
            
            // 对比增量推理的对应步骤
            std::vector<float>& incrStepLogits = incrementalLogits[step];
            
            float stepMaxDiff = 0.0f;
            for (size_t i = 0; i < vocabSize; ++i) {
                float diff = std::abs(partialBatchLast[i] - incrStepLogits[i]);
                stepMaxDiff = std::max(stepMaxDiff, diff);
            }
            
            if (stepMaxDiff > tolerance) {
                result.passed = false;
                result.errorMsg = "Step " + std::to_string(step) + " consistency failed: max diff=" + 
                                std::to_string(stepMaxDiff);
                return result;
            }
            
            std::cout << "    Step " << step << ": max diff=" 
                      << std::fixed << std::setprecision(6) << stepMaxDiff << " ✅" << std::endl;
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        result.timeMs = std::chrono::duration<double, std::milli>(end - start).count();
        result.passed = true;
        
    } catch (const std::exception& e) {
        result.passed = false;
        result.errorMsg = std::string("Exception: ") + e.what();
    }
    
    return result;
}

// ========== 主测试函数 ==========
TestReport runAllStages(const TestConfig& config) {
    TestReport report;
    
    std::cout << "==========================================" << std::endl;
    std::cout << "Kylin Backend Stage-by-Stage Test" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Model: " << config.modelPath << std::endl;
    std::cout << "Prompt: " << config.prompt << std::endl;
    std::cout << std::endl;
    
    // Stage 0
    std::cout << "Running " << "Stage 0: Basic Environment" << "..." << std::endl;
    auto stage0 = testStage0_BasicEnvironment(config);
    report.stages.push_back(stage0);
    std::cout << (stage0.passed ? "✅ PASSED" : "❌ FAILED") 
              << " (" << std::fixed << std::setprecision(2) << stage0.timeMs << " ms)" << std::endl;
    if (!stage0.passed) {
        std::cout << "  Error: " << stage0.errorMsg << std::endl;
    }
    std::cout << std::endl;
    
    // Stage 1
    std::cout << "Running " << "Stage 1: Model Loading" << "..." << std::endl;
    auto stage1 = testStage1_ModelLoading(config);
    report.stages.push_back(stage1);
    std::cout << (stage1.passed ? "✅ PASSED" : "❌ FAILED") 
              << " (" << std::fixed << std::setprecision(2) << stage1.timeMs << " ms)" << std::endl;
    if (!stage1.passed) {
        std::cout << "  Error: " << stage1.errorMsg << std::endl;
    }
    std::cout << std::endl;
    
    // Stage 2
    std::cout << "Running " << "Stage 2: Token Embedding" << "..." << std::endl;
    auto stage2 = testStage2_Embedding(config);
    report.stages.push_back(stage2);
    std::cout << (stage2.passed ? "✅ PASSED" : "❌ FAILED") 
              << " (" << std::fixed << std::setprecision(2) << stage2.timeMs << " ms)" << std::endl;
    if (!stage2.passed) {
        std::cout << "  Error: " << stage2.errorMsg << std::endl;
    }
    std::cout << std::endl;
    
    // Stage 8
    std::cout << "Running " << "Stage 8: Incremental Inference (KV Cache)" << "..." << std::endl;
    auto stage8 = testStage8_IncrementalInference(config);
    report.stages.push_back(stage8);
    std::cout << (stage8.passed ? "✅ PASSED" : "❌ FAILED") 
              << " (" << std::fixed << std::setprecision(2) << stage8.timeMs << " ms)" << std::endl;
    if (!stage8.passed) {
        std::cout << "  Error: " << stage8.errorMsg << std::endl;
    }
    std::cout << std::endl;
    
    // 生成摘要
    size_t passedCount = 0;
    for (const auto& stage : report.stages) {
        if (stage.passed) passedCount++;
    }
    
    std::ostringstream summary;
    summary << "Test Summary: " << passedCount << "/" << report.stages.size() << " stages passed";
    report.summary = summary.str();
    
    std::cout << "==========================================" << std::endl;
    std::cout << report.summary << std::endl;
    std::cout << "==========================================" << std::endl;
    
    return report;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path> [prompt] [max_tokens] [temperature]" << std::endl;
        return 1;
    }
    
    TestConfig config;
    config.modelPath = argv[1];
    config.prompt = (argc > 2) ? argv[2] : "Hi";
    config.maxTokens = (argc > 3) ? std::stoul(argv[3]) : 5;
    config.temperature = (argc > 4) ? std::stof(argv[4]) : 0.0f;
    config.compareWithLlamaCpp = true;
    
    // 初始化日志
    Logger::instance().setLevel(spdlog::level::info);
    
    // 运行所有阶段
    auto report = runAllStages(config);
    
    // 保存报告
    std::ofstream reportFile("/tmp/kylin_stage_test_report.txt");
    reportFile << "Kylin Backend Stage Test Report\n";
    reportFile << "==============================\n\n";
    reportFile << "Model: " << config.modelPath << "\n";
    reportFile << "Prompt: " << config.prompt << "\n\n";
    
    for (const auto& stage : report.stages) {
        reportFile << stage.stageName << ": " 
                   << (stage.passed ? "PASSED" : "FAILED") << "\n";
        if (!stage.passed) {
            reportFile << "  Error: " << stage.errorMsg << "\n";
        }
        reportFile << "  Time: " << std::fixed << std::setprecision(2) 
                   << stage.timeMs << " ms\n";
        for (const auto& [key, value] : stage.metrics) {
            reportFile << "  " << key << ": " << value << "\n";
        }
        reportFile << "\n";
    }
    
    reportFile << report.summary << "\n";
    reportFile.close();
    
    std::cout << "\nReport saved to: /tmp/kylin_stage_test_report.txt" << std::endl;
    
    return (report.stages.size() == 0 || 
            std::any_of(report.stages.begin(), report.stages.end(), 
                       [](const StageResult& r) { return !r.passed; })) ? 1 : 0;
}
