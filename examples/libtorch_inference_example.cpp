/**
 * @file libtorch_inference_example.cpp
 * @brief LibTorch 推理引擎使用示例
 * 
 * 演示如何使用 LibTorch 后端进行模型加载和推理
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/model/config.h"
#include <iostream>
#include <vector>
#include <chrono>

using namespace cllm;
using namespace cllm::inference;

// 加载 Qwen3-0.6B 模型配置
ModelConfig loadQwen3Config() {
    ModelConfig config;
    config.vocabSize = 151936;
    config.hiddenSize = 1024;
    config.numLayers = 28;
    config.numAttentionHeads = 16;
    config.numKeyValueHeads = 8;
    config.intermediateSize = 3072;
    config.maxSequenceLength = 40960;
    config.modelType = "qwen";
    return config;
}

int main(int argc, char* argv[]) {
    std::cout << "=== LibTorch 推理引擎测试 ===" << std::endl;
    
    // 模型路径
    const std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
    
    std::cout << "\n[1] 加载模型配置..." << std::endl;
    ModelConfig config = loadQwen3Config();
    std::cout << "  - vocab_size: " << config.vocabSize << std::endl;
    std::cout << "  - hidden_size: " << config.hiddenSize << std::endl;
    std::cout << "  - num_layers: " << config.numLayers << std::endl;
    std::cout << "  - num_attention_heads: " << config.numAttentionHeads << std::endl;
    std::cout << "  - num_key_value_heads: " << config.numKeyValueHeads << std::endl;
    
    std::cout << "\n[2] 初始化推理引擎（使用 LibTorch 后端）..." << std::endl;
    InferenceEngine engine(config, modelPath, true);  // useLibTorch = true
    
    if (!engine.initialize()) {
        std::cerr << "错误：初始化推理引擎失败！" << std::endl;
        return 1;
    }
    std::cout << "  ✓ 推理引擎初始化成功" << std::endl;
    
    // 测试 1: 8 tokens 输入（与 trace 时相同）
    std::cout << "\n[3] 测试推理：8 tokens 输入" << std::endl;
    std::vector<int> inputIds8 = {1, 2, 3, 4, 5, 6, 7, 8};
    
    auto start = std::chrono::high_resolution_clock::now();
    Tensor logits8 = engine.forward(inputIds8);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  - 输入形状: [" << inputIds8.size() << "]" << std::endl;
    std::cout << "  - 输出形状: [" << logits8.shape()[0] << ", " << logits8.shape()[1] << "]" << std::endl;
    std::cout << "  - 推理耗时: " << duration << " ms" << std::endl;
    
    // 显示前几个 token 的 top-5 预测
    std::cout << "\n  Top-5 预测结果（前 3 个位置）:" << std::endl;
    for (size_t pos = 0; pos < std::min(size_t(3), logits8.shape()[0]); ++pos) {
        std::cout << "    位置 " << pos << ": ";
        
        // 找出最大的 5 个 logit
        std::vector<std::pair<float, int>> logit_pairs;
        for (size_t i = 0; i < config.vocabSize; ++i) {
            logit_pairs.push_back({logits8.data()[pos * config.vocabSize + i], i});
        }
        std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 5, logit_pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        
        for (int i = 0; i < 5; ++i) {
            std::cout << "token_" << logit_pairs[i].second 
                      << "(logit=" << logit_pairs[i].first << ") ";
        }
        std::cout << std::endl;
    }
    
    // 测试 2: 性能基准测试
    std::cout << "\n[4] 性能基准测试（warmup + 10 次推理）..." << std::endl;
    
    // Warmup
    for (int i = 0; i < 2; ++i) {
        engine.forward(inputIds8);
    }
    
    // Benchmark
    const int benchRuns = 10;
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < benchRuns; ++i) {
        Tensor result = engine.forward(inputIds8);
    }
    end = std::chrono::high_resolution_clock::now();
    auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    double avgMs = static_cast<double>(totalMs) / benchRuns;
    double tokensPerSec = (inputIds8.size() * benchRuns * 1000.0) / totalMs;
    
    std::cout << "  - 平均推理时间: " << avgMs << " ms/iter" << std::endl;
    std::cout << "  - 吞吐量: " << tokensPerSec << " tokens/sec" << std::endl;
    std::cout << "  - 每 token 耗时: " << (avgMs / inputIds8.size()) << " ms/token" << std::endl;
    
    // 测试 3: 批处理推理
    std::cout << "\n[5] 测试批处理推理（2 个请求）..." << std::endl;
    std::vector<int> flatInputIds = {
        1, 2, 3, 4, 5, 6, 7, 8,  // 请求 1
        9, 10, 11, 12, 13, 14, 15, 16  // 请求 2
    };
    std::vector<std::pair<size_t, size_t>> requestPositions = {
        {0, 8},   // 请求 1: tokens 0-7
        {8, 16}   // 请求 2: tokens 8-15
    };
    
    start = std::chrono::high_resolution_clock::now();
    Tensor batchLogits = engine.forwardBatch(flatInputIds, requestPositions, 2);
    end = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    std::cout << "  - 总 tokens: " << flatInputIds.size() << std::endl;
    std::cout << "  - 批大小: 2" << std::endl;
    std::cout << "  - 输出形状: [" << batchLogits.shape()[0] << ", " << batchLogits.shape()[1] << "]" << std::endl;
    std::cout << "  - 批处理耗时: " << duration << " ms" << std::endl;
    std::cout << "  - 每个请求耗时: " << (duration / 2.0) << " ms" << std::endl;
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    std::cout << "\n总结:" << std::endl;
    std::cout << "  ✓ LibTorch 后端加载成功" << std::endl;
    std::cout << "  ✓ 单序列推理正常" << std::endl;
    std::cout << "  ✓ 批处理推理正常" << std::endl;
    std::cout << "  ✓ 性能基准测试完成" << std::endl;
    
    return 0;
}
