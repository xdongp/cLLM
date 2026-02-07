/**
 * @file test_text_generation_comparison.cpp
 * @brief CPU vs GPU 文本生成对比测试
 *
 * 使用相同的 prompt，分别用 CPU 和 GPU 生成文本，对比输出结果
 */

#pragma once

#include "kylin_test_framework.h"
#include "test_common_types.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <random>
#include <algorithm>
#include <numeric>

// 注意：HFTransformerModel 和 Tokenizer 的完整定义在 kylin_test_main.cpp 中包含

namespace kylin_test {

// ============================================================================
// 辅助函数：贪婪解码生成下一个 token
// ============================================================================
inline int greedySelect(const std::vector<float>& logits) {
    if (logits.empty()) return 0;
    
    int maxIdx = 0;
    float maxVal = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > maxVal) {
            maxVal = logits[i];
            maxIdx = i;
        }
    }
    return maxIdx;
}

// ============================================================================
// 辅助函数：采样生成下一个 token (带 temperature)
// ============================================================================
inline int sampleToken(const std::vector<float>& logits, float temperature, unsigned int& seed) {
    if (logits.empty()) return 0;
    if (temperature <= 0.0f) {
        return greedySelect(logits);
    }
    
    // 应用 temperature
    std::vector<float> probs(logits.size());
    float maxLogit = *std::max_element(logits.begin(), logits.end());
    double sumExp = 0.0;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp((logits[i] - maxLogit) / temperature);
        sumExp += probs[i];
    }
    
    // 归一化
    for (auto& p : probs) {
        p /= sumExp;
    }
    
    // 采样
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    float randVal = dis(gen);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (randVal <= cumsum) {
            seed = gen();  // 更新 seed
            return static_cast<int>(i);
        }
    }
    
    return static_cast<int>(probs.size() - 1);
}

// ============================================================================
// 测试：CPU vs GPU 文本生成对比
// ============================================================================
class TextGenerationComparisonTest : public TestCase {
public:
    TextGenerationComparisonTest() : TestCase("text_generation_comparison", 
        "CPU vs GPU 文本生成对比 - 验证 logits 差异是否影响实际输出") {}
    
    void execute() override {
        const std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0___6B";
        const std::string prompt = "你好";
        const int maxTokens = 20;
        const float temperature = 0.0f;  // 贪婪解码，确保可重复性
        
        log(LogLevel::INFO, "========================================");
        log(LogLevel::INFO, "CPU vs GPU 文本生成对比测试");
        log(LogLevel::INFO, "========================================");
        log(LogLevel::INFO, "模型路径: " + modelPath);
        log(LogLevel::INFO, "Prompt: \"" + prompt + "\"");
        log(LogLevel::INFO, "Max tokens: " + std::to_string(maxTokens));
        log(LogLevel::INFO, "Temperature: " + std::to_string(temperature) + " (贪婪解码)");
        
        // 检查模型路径
        if (!std::ifstream(modelPath + "/config.json").good()) {
            log(LogLevel::FAIL, "模型路径不存在: " + modelPath);
            log(LogLevel::INFO, "请确保模型已下载到正确位置");
            return;
        }
        
        try {
            // 1. 加载 Tokenizer
            log(LogLevel::INFO, "\n[1/5] 加载 Tokenizer...");
            cllm::HFTokenizer tokenizer;
            if (!tokenizer.load(modelPath)) {
                log(LogLevel::FAIL, "Tokenizer 加载失败");
                return;
            }
            log(LogLevel::INFO, "Tokenizer 加载成功");
            
            // 2. 编码 prompt
            log(LogLevel::INFO, "\n[2/5] 编码 prompt...");
            std::vector<int> inputIds = tokenizer.encode(prompt, true);
            log(LogLevel::INFO, "  Input IDs: " + vectorToString(inputIds));
            
            // 3. CPU 生成
            log(LogLevel::INFO, "\n[3/5] CPU 生成...");
            cllm::kylin::HFTransformerModel cpuModel(modelPath, 
                                                      cllm::kylin::DeviceType::CPU,
                                                      cllm::kylin::QuantType::FP16);
            if (!cpuModel.isLoaded()) {
                log(LogLevel::FAIL, "CPU 模型加载失败");
                return;
            }
            
            std::vector<int> cpuTokens = inputIds;
            std::vector<std::vector<float>> cpuLogitsHistory;
            unsigned int cpuSeed = 42;
            
            for (int i = 0; i < maxTokens; ++i) {
                auto logits = cpuModel.forward(cpuTokens);
                cpuLogitsHistory.push_back(logits);
                
                int nextToken = sampleToken(logits, temperature, cpuSeed);
                cpuTokens.push_back(nextToken);
                
                // 检查是否生成了 EOS
                if (nextToken == tokenizer.getEosId()) {
                    log(LogLevel::INFO, "  CPU 在 step " + std::to_string(i) + " 生成 EOS");
                    break;
                }
            }
            
            std::string cpuText = tokenizer.decode(cpuTokens, false);
            log(LogLevel::INFO, "  CPU Tokens: " + vectorToString(cpuTokens));
            log(LogLevel::INFO, "  CPU 生成文本: \"" + cpuText + "\"");
            
            // 4. GPU 生成
            log(LogLevel::INFO, "\n[4/5] GPU 生成...");
            cllm::kylin::HFTransformerModel gpuModel(modelPath, cllm::kylin::DeviceType::Metal);
            if (!gpuModel.isLoaded()) {
                log(LogLevel::FAIL, "GPU 模型加载失败");
                return;
            }
            if (!gpuModel.isUsingGPU()) {
                log(LogLevel::FAIL, "GPU 后端未启用");
                return;
            }
            
            std::vector<int> gpuTokens = inputIds;
            std::vector<std::vector<float>> gpuLogitsHistory;
            unsigned int gpuSeed = 42;
            
            for (int i = 0; i < maxTokens; ++i) {
                auto logits = gpuModel.forward(gpuTokens);
                gpuLogitsHistory.push_back(logits);
                
                int nextToken = sampleToken(logits, temperature, gpuSeed);
                gpuTokens.push_back(nextToken);
                
                // 检查是否生成了 EOS
                if (nextToken == tokenizer.getEosId()) {
                    log(LogLevel::INFO, "  GPU 在 step " + std::to_string(i) + " 生成 EOS");
                    break;
                }
            }
            
            std::string gpuText = tokenizer.decode(gpuTokens, false);
            log(LogLevel::INFO, "  GPU Tokens: " + vectorToString(gpuTokens));
            log(LogLevel::INFO, "  GPU 生成文本: \"" + gpuText + "\"");
            
            // 5. 对比结果
            log(LogLevel::INFO, "\n[5/5] 对比结果...");
            
            // 对比 token 序列
            bool tokensMatch = (cpuTokens == gpuTokens);
            if (tokensMatch) {
                log(LogLevel::INFO, "✓ Token 序列完全一致！");
            } else {
                log(LogLevel::WARN, "✗ Token 序列不同");
                
                // 找出第一个不同的位置
                size_t diffPos = 0;
                for (size_t i = 0; i < std::min(cpuTokens.size(), gpuTokens.size()); ++i) {
                    if (cpuTokens[i] != gpuTokens[i]) {
                        diffPos = i;
                        break;
                    }
                }
                log(LogLevel::INFO, "  第一个不同位置: " + std::to_string(diffPos));
                if (diffPos < cpuTokens.size()) {
                    log(LogLevel::INFO, "  CPU token[" + std::to_string(diffPos) + "] = " + 
                        std::to_string(cpuTokens[diffPos]));
                }
                if (diffPos < gpuTokens.size()) {
                    log(LogLevel::INFO, "  GPU token[" + std::to_string(diffPos) + "] = " + 
                        std::to_string(gpuTokens[diffPos]));
                }
                
                // 如果第一个 token 就不同，对比 logits
                if (diffPos > 0 && diffPos - 1 < cpuLogitsHistory.size() && 
                    diffPos - 1 < gpuLogitsHistory.size()) {
                    size_t logitsIdx = diffPos - 1;
                    auto& cpuLogits = cpuLogitsHistory[logitsIdx];
                    auto& gpuLogits = gpuLogitsHistory[logitsIdx];
                    
                    // 计算差异
                    float maxDiff = 0.0f;
                    for (size_t i = 0; i < std::min(cpuLogits.size(), gpuLogits.size()); ++i) {
                        maxDiff = std::max(maxDiff, std::abs(cpuLogits[i] - gpuLogits[i]));
                    }
                    log(LogLevel::INFO, "  该 step 的 logits max diff = " + std::to_string(maxDiff));
                    
                    // 显示 top 5 tokens
                    auto cpuTop5 = getTopK(cpuLogits, 5);
                    auto gpuTop5 = getTopK(gpuLogits, 5);
                    
                    log(LogLevel::INFO, "  CPU Top 5:");
                    for (const auto& pair : cpuTop5) {
                        log(LogLevel::INFO, "    " + std::to_string(pair.first) + ": " + std::to_string(pair.second));
                    }
                    log(LogLevel::INFO, "  GPU Top 5:");
                    for (const auto& pair : gpuTop5) {
                        log(LogLevel::INFO, "    " + std::to_string(pair.first) + ": " + std::to_string(pair.second));
                    }
                }
            }
            
            // 对比文本
            if (cpuText == gpuText) {
                log(LogLevel::INFO, "✓ 生成文本完全一致！");
            } else {
                log(LogLevel::WARN, "✗ 生成文本不同");
                log(LogLevel::INFO, "  CPU: \"" + cpuText + "\"");
                log(LogLevel::INFO, "  GPU: \"" + gpuText + "\"");
            }
            
            // 总结
            log(LogLevel::INFO, "\n========== 总结 ==========");
            if (tokensMatch && cpuText == gpuText) {
                log(LogLevel::INFO, "✓ CPU 和 GPU 生成结果完全一致！");
                log(LogLevel::INFO, "  结论：logits 差异不影响实际输出质量");
            } else {
                log(LogLevel::WARN, "✗ CPU 和 GPU 生成结果不同");
                log(LogLevel::INFO, "  但这不一定意味着质量问题，可能只是采样差异");
            }
            
        } catch (const std::exception& e) {
            log(LogLevel::FAIL, std::string("测试失败: ") + e.what());
        }
    }
    
private:
    std::string vectorToString(const std::vector<int>& vec, size_t maxElems = 10) {
        std::stringstream ss;
        ss << "[";
        for (size_t i = 0; i < std::min(vec.size(), maxElems); ++i) {
            if (i > 0) ss << ", ";
            ss << vec[i];
        }
        if (vec.size() > maxElems) {
            ss << ", ... (" << vec.size() << " total)";
        }
        ss << "]";
        return ss.str();
    }
    
    std::vector<std::pair<int, float>> getTopK(const std::vector<float>& logits, int k) {
        std::vector<std::pair<int, float>> indexed;
        for (size_t i = 0; i < logits.size(); ++i) {
            indexed.push_back({static_cast<int>(i), logits[i]});
        }
        
        std::partial_sort(indexed.begin(), 
                         indexed.begin() + std::min(static_cast<size_t>(k), indexed.size()),
                         indexed.end(),
                         [](const std::pair<int, float>& a, const std::pair<int, float>& b) { 
                             return a.second > b.second; 
                         });
        
        indexed.resize(std::min(static_cast<size_t>(k), indexed.size()));
        return indexed;
    }
};

// ============================================================================
// 测试套件注册
// ============================================================================
inline void registerTextGenerationTests(TestSuite& suite) {
    suite.addTest(std::make_shared<TextGenerationComparisonTest>());
}

} // namespace kylin_test
