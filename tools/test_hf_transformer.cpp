/**
 * @file test_hf_transformer.cpp
 * @brief 测试 HuggingFace Transformer 模型
 */

#include "cllm/kylin/hf/hf_transformer_model.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

using namespace cllm::kylin;

void printLogitsStats(const std::vector<float>& logits) {
    if (logits.empty()) {
        std::cout << "Logits: empty" << std::endl;
        return;
    }
    
    float minVal = *std::min_element(logits.begin(), logits.end());
    float maxVal = *std::max_element(logits.begin(), logits.end());
    float sum = std::accumulate(logits.begin(), logits.end(), 0.0f);
    float mean = sum / logits.size();
    
    float sumSq = 0.0f;
    for (float v : logits) {
        sumSq += (v - mean) * (v - mean);
    }
    float stddev = std::sqrt(sumSq / logits.size());
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Logits stats:" << std::endl;
    std::cout << "  Size: " << logits.size() << std::endl;
    std::cout << "  Min: " << minVal << std::endl;
    std::cout << "  Max: " << maxVal << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Std: " << stddev << std::endl;
    
    // 找 top-5 tokens
    std::vector<std::pair<float, int>> indexed;
    for (size_t i = 0; i < logits.size(); ++i) {
        indexed.emplace_back(logits[i], static_cast<int>(i));
    }
    std::partial_sort(indexed.begin(), indexed.begin() + 5, indexed.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "  Top-5 tokens:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        std::cout << "    [" << i << "] token=" << indexed[i].second 
                  << ", logit=" << indexed[i].first << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string modelDir = "model/Qwen/Qwen3-0.6B";
    if (argc > 1) {
        modelDir = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Testing HuggingFace Transformer Model" << std::endl;
    std::cout << "Model: " << modelDir << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载模型
    HFTransformerModel model(modelDir);
    if (!model.isLoaded()) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }
    
    std::cout << "\nModel loaded successfully!" << std::endl;
    std::cout << "  Vocab size: " << model.vocabSize() << std::endl;
    std::cout << "  Hidden size: " << model.hiddenSize() << std::endl;
    
    // 测试推理
    // "Hello" 在 Qwen3 tokenizer 中的 ID（需要实际查询）
    // 这里使用一个示例 token ID
    std::vector<int32_t> testTokens = {9707};  // "Hello" 或类似的 token
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Running inference..." << std::endl;
    std::cout << "Input tokens: [";
    for (size_t i = 0; i < testTokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << testTokens[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto logits = model.forward(testTokens);
    
    std::cout << "\n";
    printLogitsStats(logits);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Test completed!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
