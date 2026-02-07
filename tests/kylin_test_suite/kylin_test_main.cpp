/**
 * @file kylin_test_main.cpp
 * @brief Kylin Backend 测试主入口
 *
 * 测试阶段按从底层到上层排列：
 * 
 * === 基础层 (模型与组件) ===
 *   Stage 1:  模型配置验证        - 验证模型配置文件
 *   Stage 2:  Tokenizer 验证      - 验证分词器功能
 *   Stage 3:  权重加载验证        - 验证模型权重加载
 *
 * === 输入层 ===
 *   Stage 4:  输入验证测试        - 验证输入tokenization
 *
 * === 模型层 (从浅到深) ===
 *   Stage 5:  Embedding 输出测试  - 验证Embedding层
 *   Stage 6:  单层 Transformer    - 验证第1层Transformer
 *   Stage 7:  中间层 Transformer  - 验证第14层Transformer
 *   Stage 8:  最终层 Transformer  - 验证第28层Transformer
 *   Stage 9:  LM Head 输出测试    - 验证LM Head层
 *
 * === 推理层 ===
 *   Stage 10: KV Cache 测试       - 验证KV缓存机制
 *   Stage 11: 采样策略测试        - 验证各种采样策略
 *
 * === 生成层 ===
 *   Stage 12: 端到端生成测试      - 验证完整生成流程
 *   Stage 13: 输出质量测试        - 验证生成文本质量
 *   Stage 14: 模型输出分析        - 分析模型输出特征
 *
 * === 接口与性能层 ===
 *   Stage 15: HTTP API 集成测试   - 验证HTTP接口
 *   Stage 16: 性能基准测试        - 性能评估
 *   Stage 17: 完整集成测试        - 系统集成验证
 *   Stage 25-26: 逐层调试测试
 *   Stage 27: 真实模型逐层对比    - 使用真实模型对比CPU/GPU
 *   Stage 28: 文本生成对比        - 对比CPU/GPU文本生成结果
 *
 * === CPU vs GPU 对比层 ===
 *   Stage 18: 模型加载对比        - 对比CPU/GPU权重加载
 *   Stage 19: Embedding层对比     - 对比Embedding输出
 *   Stage 20: Attention层对比     - 对比Attention计算
 *   Stage 21: FFN层对比           - 对比前馈网络
 *   Stage 22: Transformer层对比   - 对比单层Transformer
 *   Stage 23: 端到端推理对比      - 对比完整生成流程
 *   Stage 24: 数值精度对比        - 对比数值差异
 *
 * 使用方法:
 *   ./kylin_test_suite --stage=27               # 运行 Stage 27 (真实模型逐层对比)
 *   ./kylin_test_suite --stage=28               # 运行 Stage 28 (文本生成对比)
 *   ./kylin_test_suite --all                    # 运行所有测试
 *   ./kylin_test_suite --stage=1                # 运行 Stage 1 (模型配置验证)
 *   ./kylin_test_suite --stage=2                # 运行 Stage 2 (Tokenizer 验证)
 *   ./kylin_test_suite --stage=3                # 运行 Stage 3 (权重加载验证)
 *   ./kylin_test_suite --stage=4                # 运行 Stage 4 (输入验证测试)
 *   ./kylin_test_suite --stage=5                # 运行 Stage 5 (Embedding 输出测试)
 *   ./kylin_test_suite --stage=6                # 运行 Stage 6 (单层 Transformer)
 *   ./kylin_test_suite --stage=7                # 运行 Stage 7 (中间层 Transformer)
 *   ./kylin_test_suite --stage=8                # 运行 Stage 8 (最终层 Transformer)
 *   ./kylin_test_suite --stage=9                # 运行 Stage 9 (LM Head 输出测试)
 *   ./kylin_test_suite --stage=10               # 运行 Stage 10 (KV Cache 测试)
 *   ./kylin_test_suite --stage=11               # 运行 Stage 11 (采样策略测试)
 *   ./kylin_test_suite --stage=12               # 运行 Stage 12 (端到端生成测试)
 *   ./kylin_test_suite --stage=13               # 运行 Stage 13 (输出质量测试)
 *   ./kylin_test_suite --stage=14               # 运行 Stage 14 (模型输出分析)
 *   ./kylin_test_suite --stage=15               # 运行 Stage 15 (HTTP API 测试)
 *   ./kylin_test_suite --stage=16               # 运行 Stage 16 (性能基准测试)
 *   ./kylin_test_suite --stage=17               # 运行 Stage 17 (完整集成测试)
 *   ./kylin_test_suite --test=<name>            # 运行特定测试
 *   ./kylin_test_suite --verbose                # 详细输出
 *   ./kylin_test_suite --report=html            # 生成 HTML 报告
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <chrono>
#include <memory>
#include <cstring>

// 测试框架
#include "kylin_test_framework.h"

// 各阶段测试 - 按从底层到上层顺序
#include "test_model_loading.cpp"           // Stage 1-3: 基础层
#include "test_tokenizer_integration.cpp"   // Stage 2
#include "test_inference_pipeline.cpp"      // Stage 3
#include "test_layer_by_layer.cpp"          // Stage 4-9: 输入层+模型层
#include "test_sampling_strategies.cpp"     // Stage 10-11: 推理层
#include "test_end_to_end_generation.cpp"   // Stage 12-14: 生成层
#include "test_output_quality.cpp"          // Stage 13
#include "test_model_output_analysis.cpp"   // Stage 14
#include "test_http_api_integration.cpp"    // Stage 15-17: 接口与性能层
#include "test_performance_benchmark.cpp"   // Stage 16
#include "test_output_validation.cpp"       // Stage 17 (集成测试)
#include "test_cpu_gpu_comparison.cpp"      // Stage 18-24: CPU vs GPU对比测试
#include "test_layer_by_layer_debug.cpp"    // Stage 25-26: 逐层调试测试

// 真实模型测试需要的头文件
#include "cllm/kylin/hf/transformer.h"
#include "cllm/tokenizer/tokenizer.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include "test_real_model_comparison.cpp"   // 真实模型对比测试
#include "test_text_generation_comparison.cpp"  // Stage 28: 文本生成对比测试
#include "test_ggml_precision.cpp"          // Stage 29: GGML 精度测试
#include "test_phased_cpu_gpu_comparison.cpp"  // Stage 30-34: 分阶段 CPU vs GPU 精确对比

using namespace kylin_test;
using namespace cllm::kylin;

// 命令行参数
struct TestOptions {
    bool runAll = false;
    int stage = 0;  // 0 = 未指定
    std::string specificTest;
    bool verbose = false;
    std::string reportFormat;  // "html" or ""
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
};

// 解析命令行参数
TestOptions parseArguments(int argc, char** argv) {
    TestOptions options;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--all") {
            options.runAll = true;
        } else if (arg.rfind("--stage=", 0) == 0) {
            options.stage = std::stoi(arg.substr(8));
        } else if (arg.rfind("--test=", 0) == 0) {
            options.specificTest = arg.substr(7);
        } else if (arg == "--verbose") {
            options.verbose = true;
        } else if (arg.rfind("--report=", 0) == 0) {
            options.reportFormat = arg.substr(9);
        } else if (arg.rfind("--model=", 0) == 0) {
            options.modelPath = arg.substr(8);
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Kylin Backend Test Suite\n"
                      << "Usage: " << argv[0] << " [options]\n"
                      << "\nOptions:\n"
                      << "  --all              Run all tests\n"
                      << "  --stage=N          Run specific stage (1-17)\n"
                      << "  --test=NAME        Run specific test\n"
                      << "  --verbose          Enable verbose output\n"
                      << "  --report=FORMAT    Generate report (html)\n"
                      << "  --model=PATH       Set model path\n"
                      << "  --help, -h         Show this help\n"
                      << "\nStages (从底层到上层):\n"
                      << "\n=== 基础层 ===\n"
                      << "  1:  模型配置验证        - 验证模型配置文件\n"
                      << "  2:  Tokenizer 验证      - 验证分词器功能\n"
                      << "  3:  权重加载验证        - 验证模型权重加载\n"
                      << "\n=== 输入层 ===\n"
                      << "  4:  输入验证测试        - 验证输入tokenization\n"
                      << "\n=== 模型层 (从浅到深) ===\n"
                      << "  5:  Embedding 输出测试  - 验证Embedding层\n"
                      << "  6:  单层 Transformer    - 验证第1层Transformer\n"
                      << "  7:  中间层 Transformer  - 验证第14层Transformer\n"
                      << "  8:  最终层 Transformer  - 验证第28层Transformer\n"
                      << "  9:  LM Head 输出测试    - 验证LM Head层\n"
                      << "\n=== 推理层 ===\n"
                      << "  10: KV Cache 测试       - 验证KV缓存机制\n"
                      << "  11: 采样策略测试        - 验证各种采样策略\n"
                      << "\n=== 生成层 ===\n"
                      << "  12: 端到端生成测试      - 验证完整生成流程\n"
                      << "  13: 输出质量测试        - 验证生成文本质量\n"
                      << "  14: 模型输出分析        - 分析模型输出特征\n"
                      << "\n=== 接口与性能层 ===\n"
                      << "  15: HTTP API 集成测试   - 验证HTTP接口\n"
                      << "  16: 性能基准测试        - 性能评估\n"
                      << "  17: 完整集成测试        - 系统集成验证\n"
                      << "\n=== CPU vs GPU 对比层 ===\n"
                      << "  18: 模型加载对比        - 对比CPU/GPU权重加载\n"
                      << "  19: Embedding层对比     - 对比Embedding输出\n"
                      << "  20: Attention层对比     - 对比Attention计算\n"
                      << "  21: FFN层对比           - 对比前馈网络\n"
                      << "  22: Transformer层对比   - 对比单层Transformer\n"
                      << "  23: 端到端推理对比      - 对比完整生成流程\n"
                      << "  24: 数值精度对比        - 对比数值差异\n"
                      << "  25: 逐层输出对比        - 逐层对比CPU/GPU输出\n"
                      << "  26: 内存调试测试        - 内存和数据传输调试\n"
                      << "\n=== 真实模型对比层 ===\n"
                      << "  27: 真实模型对比测试    - 使用forwardWithDebug对比\n"
                      << "\n=== 分阶段 CPU vs GPU 精确对比 ===\n"
                      << "  30: Phase 1 权重一致性    - 验证GPU权重上传是否正确\n"
                      << "  31: Phase 2 Embedding对比  - 对比Embedding层输出\n"
                      << "  32: Phase 3 逐层对比       - 逐层定位偏差源头\n"
                      << "  33: Phase 4 Logits对比     - 对比最终logits与Top-K\n"
                      << "  34: Phase 5 生成对比       - 多步贪婪生成文本对比\n";
            exit(0);
        }
    }
    
    return options;
}

// 打印测试头
void printHeader(const std::string& title) {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "  " << title << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

// 打印阶段分组头
void printStageGroup(const std::string& groupName) {
    std::cout << "\n" << std::string(60, '#') << std::endl;
    std::cout << "## " << groupName << std::endl;
    std::cout << std::string(60, '#') << std::endl;
}

// 主函数
int main(int argc, char** argv) {
    TestOptions options = parseArguments(argc, argv);
    
    // 设置详细模式
    TestCase::setVerbose(options.verbose);
    
    // 如果没有指定任何选项，显示帮助
    if (!options.runAll && options.stage == 0 && options.specificTest.empty()) {
        std::cerr << "Error: No test specified. Use --help for usage information." << std::endl;
        return 1;
    }
    
    // 收集所有测试结果
    std::vector<TestResult> allResults;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // ==================== 基础层 ====================
    if (options.runAll) {
        printStageGroup("基础层测试 (模型与组件)");
    }
    
    // Stage 1: 模型配置验证
    if (options.runAll || options.stage == 1) {
        printHeader("STAGE 1: Model Config Validation");
        auto suite = createModelConfigTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 2: Tokenizer 验证
    if (options.runAll || options.stage == 2) {
        printHeader("STAGE 2: Tokenizer Validation");
        auto suite = createTokenizerValidationTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 3: 权重加载验证
    if (options.runAll || options.stage == 3) {
        printHeader("STAGE 3: Weights Loading Validation");
        auto suite = createWeightsLoadingTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 输入层 ====================
    if (options.runAll) {
        printStageGroup("输入层测试");
    }
    
    // Stage 4: 输入验证测试
    if (options.runAll || options.stage == 4) {
        printHeader("STAGE 4: Input Validation");
        auto suite = createInputValidationTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 模型层 ====================
    if (options.runAll) {
        printStageGroup("模型层测试 (从浅到深)");
    }
    
    // Stage 5: Embedding 输出测试
    if (options.runAll || options.stage == 5) {
        printHeader("STAGE 5: Embedding Output Test");
        auto suite = createEmbeddingOutputTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 6: 单层 Transformer 测试
    if (options.runAll || options.stage == 6) {
        printHeader("STAGE 6: Single Layer Transformer Test");
        auto suite = createSingleLayerTransformerTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 7: 中间层 Transformer 测试
    if (options.runAll || options.stage == 7) {
        printHeader("STAGE 7: Middle Layer Transformer Test");
        auto suite = createMiddleLayerTransformerTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 8: 最终层 Transformer 测试
    if (options.runAll || options.stage == 8) {
        printHeader("STAGE 8: Final Layer Transformer Test");
        auto suite = createFinalLayerTransformerTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 9: LM Head 输出测试
    if (options.runAll || options.stage == 9) {
        printHeader("STAGE 9: LM Head Output Test");
        auto suite = createLMHeadOutputTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 推理层 ====================
    if (options.runAll) {
        printStageGroup("推理层测试");
    }
    
    // Stage 10: KV Cache 测试
    if (options.runAll || options.stage == 10) {
        printHeader("STAGE 10: KV Cache Test");
        auto suite = createKVCacheTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 11: 采样策略测试
    if (options.runAll || options.stage == 11) {
        printHeader("STAGE 11: Sampling Strategies");
        auto suite = createSamplingStrategiesTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 生成层 ====================
    if (options.runAll) {
        printStageGroup("生成层测试");
    }
    
    // Stage 12: 端到端生成测试
    if (options.runAll || options.stage == 12) {
        printHeader("STAGE 12: End-to-End Generation");
        auto suite = createEndToEndGenerationTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 13: 输出质量测试
    if (options.runAll || options.stage == 13) {
        printHeader("STAGE 13: Output Quality Test");
        auto suite = createOutputQualityTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 14: 模型输出分析
    if (options.runAll || options.stage == 14) {
        printHeader("STAGE 14: Model Output Analysis");
        auto suite = createModelOutputAnalysisTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 接口与性能层 ====================
    if (options.runAll) {
        printStageGroup("接口与性能层测试");
    }
    
    // Stage 15: HTTP API 集成测试
    if (options.runAll || options.stage == 15) {
        printHeader("STAGE 15: HTTP API Integration");
        auto suite = createHttpApiIntegrationTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 16: 性能基准测试
    if (options.runAll || options.stage == 16) {
        printHeader("STAGE 16: Performance Benchmark");
        auto suite = createPerformanceBenchmarkTestSuite();
        auto results = suite->runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // Stage 17: 完整集成测试
    if (options.runAll || options.stage == 17) {
        printHeader("STAGE 17: Full Integration Test");
        // 运行所有测试套件
        auto suite1 = createModelConfigTestSuite();
        auto results1 = suite1->runAll();
        allResults.insert(allResults.end(), results1.begin(), results1.end());
        
        auto suite2 = createTokenizerValidationTestSuite();
        auto results2 = suite2->runAll();
        allResults.insert(allResults.end(), results2.begin(), results2.end());
        
        auto suite3 = createWeightsLoadingTestSuite();
        auto results3 = suite3->runAll();
        allResults.insert(allResults.end(), results3.begin(), results3.end());
        
        auto suite4 = createEndToEndGenerationTestSuite();
        auto results4 = suite4->runAll();
        allResults.insert(allResults.end(), results4.begin(), results4.end());
        
        auto suite5 = createHttpApiIntegrationTestSuite();
        auto results5 = suite5->runAll();
        allResults.insert(allResults.end(), results5.begin(), results5.end());
    }
    
    // ==================== CPU vs GPU 对比层 ====================
    if (options.runAll) {
        printStageGroup("CPU vs GPU 对比层测试");
    }
    
    // Stage 18-24: CPU vs GPU 对比测试
    if (options.runAll || (options.stage >= 18 && options.stage <= 24)) {
        printHeader("STAGE 18-24: CPU vs GPU Comparison Tests");
        TestSuite comparisonSuite("CPU vs GPU Comparison");
        registerCPUGPUComparisonTests(comparisonSuite);
        auto results = comparisonSuite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }
    
    // ==================== 逐层调试层 ====================
    if (options.runAll) {
        printStageGroup("逐层调试测试");
    }
    
    // Stage 25-26: 逐层输出对比测试
    if (options.runAll || (options.stage >= 25 && options.stage <= 26)) {
        printHeader("STAGE 25-26: Layer-by-Layer Debug Tests");
        TestSuite debugSuite("Layer-by-Layer Debug");
        registerLayerDebugTests(debugSuite);
        auto results = debugSuite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 27: 真实模型对比测试
    if (options.runAll || options.stage == 27) {
        printHeader("STAGE 27: Real Model Comparison Tests");
        TestSuite realModelSuite("Real Model Comparison");
        registerRealModelTests(realModelSuite);
        auto results = realModelSuite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 28: 文本生成对比测试
    if (options.runAll || options.stage == 28) {
        printHeader("STAGE 28: Text Generation Comparison Tests");
        TestSuite textGenSuite("Text Generation Comparison");
        registerTextGenerationTests(textGenSuite);
        auto results = textGenSuite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 29: GGML 精度测试
    if (options.runAll || options.stage == 29) {
        printHeader("STAGE 29: GGML Precision Tests");
        TestSuite ggmlPrecisionSuite("GGML Precision");
        registerGGMLPrecisionTests(ggmlPrecisionSuite);
        auto results = ggmlPrecisionSuite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // ==================== 分阶段 CPU vs GPU 精确对比 ====================
    
    // Stage 30: Phase 1 - 权重一致性验证
    if (options.runAll || options.stage == 30) {
        printHeader("STAGE 30: Phase 1 - Weight Consistency");
        TestSuite phase1Suite("Phase 1: Weight Consistency");
        registerPhasedCPUGPUTests_Phase1(phase1Suite);
        auto results = phase1Suite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 31: Phase 2 - Embedding 对比
    if (options.runAll || options.stage == 31) {
        printHeader("STAGE 31: Phase 2 - Embedding Comparison");
        TestSuite phase2Suite("Phase 2: Embedding Comparison");
        registerPhasedCPUGPUTests_Phase2(phase2Suite);
        auto results = phase2Suite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 32: Phase 3 - 逐层中间结果对比
    if (options.runAll || options.stage == 32) {
        printHeader("STAGE 32: Phase 3 - Layer-by-Layer Comparison");
        TestSuite phase3Suite("Phase 3: Layer-by-Layer Comparison");
        registerPhasedCPUGPUTests_Phase3(phase3Suite);
        auto results = phase3Suite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 33: Phase 4 - Logits 与 Top-K 对比
    if (options.runAll || options.stage == 33) {
        printHeader("STAGE 33: Phase 4 - Logits & Top-K Comparison");
        TestSuite phase4Suite("Phase 4: Logits & Top-K");
        registerPhasedCPUGPUTests_Phase4(phase4Suite);
        auto results = phase4Suite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // Stage 34: Phase 5 - 多步生成对比
    if (options.runAll || options.stage == 34) {
        printHeader("STAGE 34: Phase 5 - Generation Comparison");
        TestSuite phase5Suite("Phase 5: Generation Comparison");
        registerPhasedCPUGPUTests_Phase5(phase5Suite);
        auto results = phase5Suite.runAll();
        allResults.insert(allResults.end(), results.begin(), results.end());
    }

    // 运行特定测试
    if (!options.specificTest.empty()) {
        printHeader("Running Specific Test: " + options.specificTest);
        // TODO: 实现特定测试查找和运行
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    double totalDuration = std::chrono::duration<double>(endTime - startTime).count();
    
    // 最终汇总
    printHeader("FINAL SUMMARY");
    
    int passed = 0, failed = 0, skipped = 0, errors = 0;
    for (const auto& r : allResults) {
        switch (r.status) {
            case TestStatus::PASSED: passed++; break;
            case TestStatus::FAILED: failed++; break;
            case TestStatus::SKIPPED: skipped++; break;
            case TestStatus::ERROR: errors++; break;
            default: break;
        }
    }
    
    std::cout << "Total Tests: " << allResults.size() << std::endl;
    std::cout << "Passed:      " << passed << std::endl;
    std::cout << "Failed:      " << failed << std::endl;
    std::cout << "Skipped:     " << skipped << std::endl;
    std::cout << "Errors:      " << errors << std::endl;
    std::cout << "Total Time:  " << std::fixed << std::setprecision(2) 
              << totalDuration << " seconds" << std::endl;
    
    // 生成报告
    if (options.reportFormat == "html") {
        std::string reportFile = "kylin_test_report.html";
        TestReport::generateHTML(reportFile, allResults, "Kylin Backend Test Suite");
        std::cout << "\nHTML report generated: " << reportFile << std::endl;
    }
    
    // 返回码
    return (failed > 0 || errors > 0) ? 1 : 0;
}
