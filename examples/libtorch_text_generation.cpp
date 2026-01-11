/**
 * @file libtorch_text_generation.cpp
 * @brief LibTorch 文本生成示例
 * 
 * 完整演示：输入文本 -> Tokenize -> 推理 -> 采样 -> 解码 -> 输出文本
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/sampler.h"
#include "cllm/memory/float_array.h"
#include <iostream>
#include <vector>
#include <string>
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
    std::cout << "=== LibTorch 文本生成测试 ===" << std::endl;
    
    // 配置路径
    const std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
    const std::string tokenizerPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/Qwen3-0.6B";
    
    // 输入文本
    std::string inputText = "Hello";
    if (argc > 1) {
        inputText = argv[1];
    }
    
    std::cout << "\n📝 输入文本: \"" << inputText << "\"" << std::endl;
    
    // 1. 初始化 Tokenizer
    std::cout << "\n[1] 初始化 Tokenizer..." << std::endl;
    cllm::TokenizerManager tokenizer(tokenizerPath, nullptr);
    if (tokenizer.getTokenizer() == nullptr) {
        std::cerr << "❌ Tokenizer 初始化失败！" << std::endl;
        return 1;
    }
    std::cout << "  ✓ Tokenizer 初始化成功" << std::endl;
    
    // 2. 编码输入文本
    std::cout << "\n[2] 编码输入文本..." << std::endl;
    std::vector<int> inputIds = tokenizer.encode(inputText);
    std::cout << "  - 原始文本: \"" << inputText << "\"" << std::endl;
    std::cout << "  - Token IDs: [";
    for (size_t i = 0; i < inputIds.size(); ++i) {
        std::cout << inputIds[i];
        if (i < inputIds.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  - Token 数量: " << inputIds.size() << std::endl;
    
    // 显示每个 token 的解码
    std::cout << "  - Token 解码:" << std::endl;
    for (size_t i = 0; i < inputIds.size(); ++i) {
        std::string decoded = tokenizer.decode({inputIds[i]});
        std::cout << "    [" << i << "] " << inputIds[i] << " -> \"" << decoded << "\"" << std::endl;
    }
    
    // 如果输入太短，填充到 8 个 tokens（LibTorch trace 固定长度）
    std::vector<int> paddedIds = inputIds;
    if (paddedIds.size() < 8) {
        std::cout << "\n  ⚠️  输入长度 < 8，填充到 8 tokens（LibTorch trace 限制）" << std::endl;
        while (paddedIds.size() < 8) {
            paddedIds.push_back(151643);  // <|endoftext|> token
        }
        std::cout << "  - 填充后: [";
        for (size_t i = 0; i < paddedIds.size(); ++i) {
            std::cout << paddedIds[i];
            if (i < paddedIds.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    } else if (paddedIds.size() > 8) {
        std::cout << "\n  ⚠️  输入长度 > 8，截断到 8 tokens（LibTorch trace 限制）" << std::endl;
        paddedIds.resize(8);
    }
    
    // 3. 初始化推理引擎
    std::cout << "\n[3] 初始化推理引擎（LibTorch 后端）..." << std::endl;
    ModelConfig config = loadQwen3Config();
    InferenceEngine engine(config, modelPath, true);  // useLibTorch = true
    
    if (!engine.initialize()) {
        std::cerr << "❌ 推理引擎初始化失败！" << std::endl;
        return 1;
    }
    std::cout << "  ✓ 推理引擎初始化成功" << std::endl;
    
    // 4. 执行推理
    std::cout << "\n[4] 执行模型推理..." << std::endl;
    auto startInfer = std::chrono::high_resolution_clock::now();
    Tensor logits = engine.forward(paddedIds);
    auto endInfer = std::chrono::high_resolution_clock::now();
    auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - startInfer).count();
    
    std::cout << "  - 输入形状: [" << paddedIds.size() << "]" << std::endl;
    std::cout << "  - 输出形状: [" << logits.shape()[0] << ", " << logits.shape()[1] << "]" << std::endl;
    std::cout << "  - 推理耗时: " << inferMs << " ms" << std::endl;
    
    // 5. 采样下一个 token
    std::cout << "\n[5] 从每个位置采样预测 token..." << std::endl;
    
    // 创建采样器
    SamplerConfig samplerConfig;
    samplerConfig.setTemperature(0.8f);
    samplerConfig.setTopK(50);
    samplerConfig.setTopP(0.9f);
    Sampler sampler(samplerConfig);
    
    std::cout << "  采样配置:" << std::endl;
    std::cout << "    - temperature: " << samplerConfig.getTemperature() << std::endl;
    std::cout << "    - top_k: " << samplerConfig.getTopK() << std::endl;
    std::cout << "    - top_p: " << samplerConfig.getTopP() << std::endl;
    
    // 对原始输入的每个位置进行采样
    std::cout << "\n  预测结果（原始输入的每个位置）:" << std::endl;
    std::vector<int> predictedIds;
    for (size_t pos = 0; pos < inputIds.size(); ++pos) {
        // 获取该位置的 logits
        FloatArray posLogits(config.vocabSize);
        for (size_t i = 0; i < config.vocabSize; ++i) {
            posLogits.data()[i] = logits.data()[pos * config.vocabSize + i];
        }
        
        // 采样
        int nextToken = sampler.sample(posLogits, samplerConfig.getTemperature(), 
                                      samplerConfig.getTopK(), samplerConfig.getTopP());
        predictedIds.push_back(nextToken);
        
        // 解码
        std::string predicted = tokenizer.decode({nextToken});
        std::string current = tokenizer.decode({inputIds[pos]});
        
        std::cout << "    位置 " << pos << ": \"" << current << "\" (token_" << inputIds[pos] 
                  << ") -> 预测下一个: \"" << predicted << "\" (token_" << nextToken << ")" << std::endl;
        
        // 显示 top-5 候选
        std::vector<std::pair<float, int>> logit_pairs;
        for (size_t i = 0; i < config.vocabSize; ++i) {
            logit_pairs.push_back({posLogits.data()[i], static_cast<int>(i)});
        }
        std::partial_sort(logit_pairs.begin(), logit_pairs.begin() + 5, logit_pairs.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "      Top-5: ";
        for (int i = 0; i < 5; ++i) {
            std::string candidateText = tokenizer.decode({logit_pairs[i].second});
            std::cout << "\"" << candidateText << "\"(" << logit_pairs[i].first << ") ";
        }
        std::cout << std::endl;
    }
    
    // 6. 自回归生成（简单版本，只生成几个 token）
    std::cout << "\n[6] 自回归生成（续写 5 个 token）..." << std::endl;
    std::vector<int> generatedIds = inputIds;  // 从原始输入开始
    const int maxNewTokens = 5;
    
    std::cout << "  初始序列: \"" << inputText << "\"" << std::endl;
    
    for (int step = 0; step < maxNewTokens; ++step) {
        // 准备输入（保持 8 tokens）
        std::vector<int> currentInput = generatedIds;
        if (currentInput.size() > 8) {
            // 取最后 8 个 tokens
            currentInput = std::vector<int>(generatedIds.end() - 8, generatedIds.end());
        } else if (currentInput.size() < 8) {
            // 填充到 8
            while (currentInput.size() < 8) {
                currentInput.insert(currentInput.begin(), 151643);  // 前面填充
            }
        }
        
        // 推理
        Tensor stepLogits = engine.forward(currentInput);
        
        // 获取最后一个位置的 logits
        size_t lastPos = currentInput.size() - 1;
        FloatArray lastLogits(config.vocabSize);
        for (size_t i = 0; i < config.vocabSize; ++i) {
            lastLogits.data()[i] = stepLogits.data()[lastPos * config.vocabSize + i];
        }
        
        // 采样
        int nextToken = sampler.sample(lastLogits, samplerConfig.getTemperature(),
                                      samplerConfig.getTopK(), samplerConfig.getTopP());
        generatedIds.push_back(nextToken);
        
        // 解码并显示
        std::string nextText = tokenizer.decode({nextToken});
        std::cout << "  Step " << (step + 1) << ": 生成 token_" << nextToken 
                  << " -> \"" << nextText << "\"" << std::endl;
    }
    
    // 7. 解码完整输出
    std::cout << "\n[7] 解码完整生成结果..." << std::endl;
    std::string generatedText = tokenizer.decode(generatedIds);
    std::cout << "  - 生成的 token 数: " << generatedIds.size() << std::endl;
    std::cout << "  - 完整输出: \"" << generatedText << "\"" << std::endl;
    
    // 8. 性能统计
    std::cout << "\n[8] 性能统计..." << std::endl;
    std::cout << "  - 首次推理: " << inferMs << " ms" << std::endl;
    std::cout << "  - 生成 token 数: " << maxNewTokens << std::endl;
    std::cout << "  - 平均每 token: ~" << (inferMs / 8) << " ms" << std::endl;
    
    std::cout << "\n=== 测试完成 ===" << std::endl;
    
    return 0;
}
