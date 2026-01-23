/**
 * @file debug_embedding.cpp
 * @brief 调试 embedding 层输出
 */

#include "cllm/kylin/gguf/transformer.h"
#include <iostream>
#include <iomanip>
#include <cmath>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <gguf_model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "Embedding Debug Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载模型
    cllm::kylin::GGMLTransformerModel model;
    if (!model.loadFromGGUF(modelPath)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    auto config = model.getConfig();
    std::cout << "\nModel config:" << std::endl;
    std::cout << "  vocab_size: " << config.vocabSize << std::endl;
    std::cout << "  hidden_size: " << config.embeddingLength << std::endl;
    std::cout << "  num_layers: " << config.blockCount << std::endl;
    std::cout << "  num_heads: " << config.headCount << std::endl;
    
    // 测试 token 9707 (Hello)
    std::vector<int> inputTokens = {9707};
    std::cout << "\n=== Test 1: Forward token 9707 (Hello) ===" << std::endl;
    
    model.clearKVCache();
    auto logits = model.forward(inputTokens);
    
    if (logits.empty()) {
        std::cerr << "Forward returned empty logits!" << std::endl;
        return 1;
    }
    
    size_t vocabSize = config.vocabSize;
    std::cout << "  Logits size: " << logits.size() << std::endl;
    std::cout << "  Expected size: " << vocabSize << std::endl;
    
    // 找 top-10 token
    std::vector<std::pair<float, int>> scored;
    for (size_t i = 0; i < vocabSize && i < logits.size(); ++i) {
        scored.push_back({logits[i], static_cast<int>(i)});
    }
    std::partial_sort(scored.begin(), scored.begin() + 10, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "\n  Top-10 tokens:" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "    " << std::setw(2) << (i + 1) << ". Token " << std::setw(6) << scored[i].second 
                  << " | logit=" << std::fixed << std::setprecision(4) << scored[i].first << std::endl;
    }
    
    // 检查 logits 分布
    float maxLogit = -std::numeric_limits<float>::infinity();
    float minLogit = std::numeric_limits<float>::infinity();
    float sumLogit = 0.0f;
    int nanCount = 0;
    int infCount = 0;
    
    for (size_t i = 0; i < vocabSize && i < logits.size(); ++i) {
        if (std::isnan(logits[i])) {
            ++nanCount;
        } else if (std::isinf(logits[i])) {
            ++infCount;
        } else {
            maxLogit = std::max(maxLogit, logits[i]);
            minLogit = std::min(minLogit, logits[i]);
            sumLogit += logits[i];
        }
    }
    
    std::cout << "\n  Logits statistics:" << std::endl;
    std::cout << "    Max: " << maxLogit << std::endl;
    std::cout << "    Min: " << minLogit << std::endl;
    std::cout << "    Mean: " << (sumLogit / vocabSize) << std::endl;
    std::cout << "    NaN count: " << nanCount << std::endl;
    std::cout << "    Inf count: " << infCount << std::endl;
    
    // 使用贪婪采样，预测下一个 token
    int nextToken = scored[0].second;
    std::cout << "\n  Greedy next token: " << nextToken << std::endl;
    
    // 比较：直接用 forwardOneToken
    std::cout << "\n=== Test 2: Forward using forwardOneToken ===" << std::endl;
    model.clearKVCache();
    
    // 首先处理 prompt
    auto promptLogits = model.forward(inputTokens);
    
    // 然后增量生成（position 是当前 KV cache 位置，即 prompt 长度）
    int token1 = scored[0].second;
    size_t position = inputTokens.size();  // 下一个位置
    auto incLogits = model.forwardOneToken(token1, position);
    
    if (!incLogits.empty()) {
        scored.clear();
        for (size_t i = 0; i < vocabSize && i < incLogits.size(); ++i) {
            scored.push_back({incLogits[i], static_cast<int>(i)});
        }
        std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        
        std::cout << "  Next token after " << token1 << ":" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "    " << std::setw(2) << (i + 1) << ". Token " << std::setw(6) << scored[i].second 
                      << " | logit=" << std::fixed << std::setprecision(4) << scored[i].first << std::endl;
        }
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
