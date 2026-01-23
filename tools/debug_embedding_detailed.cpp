/**
 * @file debug_embedding_detailed.cpp
 * @brief 详细调试 embedding 层输出
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/kylin/gguf/loader.h"
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
    std::cout << "Detailed Embedding Debug Tool" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 直接使用 GGUFLoader 加载模型
    cllm::kylin::GGUFLoader loader(modelPath);
    if (!loader.isValid()) {
        std::cerr << "Failed to load GGUF model" << std::endl;
        return 1;
    }
    
    auto config = loader.loadConfig();
    std::cout << "\nModel config:" << std::endl;
    std::cout << "  vocab_size: " << config.vocabSize << std::endl;
    std::cout << "  embedding_length: " << config.embeddingLength << std::endl;
    
    // 找到 token embedding 张量
    auto tensorNames = loader.getTensorNames();
    for (const auto& name : tensorNames) {
        if (name == "token_embd.weight" || name == "model.embed_tokens.weight") {
            std::cout << "\nFound token embedding: " << name << std::endl;
            auto type = loader.getTensorType(name);
            auto shape = loader.getTensorShape(name);
            std::cout << "  Type: " << type << std::endl;
            std::cout << "  Shape: [";
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            break;
        }
    }
    
    // 加载完整模型
    cllm::kylin::GGMLTransformerModel model;
    if (!model.loadFromGGUF(modelPath)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    // 检查 token 9707 (Hello) 的 embedding
    std::cout << "\n=== Checking embedding for token 9707 (Hello) ===" << std::endl;
    
    // 执行一次 forward 来获取 embedding
    model.clearKVCache();
    auto logits = model.forward({9707});
    
    // 获取 debug 信息
    auto debugStats = model.getLayer0DebugStats();
    std::cout << "\nDebug stats available:" << std::endl;
    for (const auto& [name, stats] : debugStats) {
        std::cout << "  " << name << ": min=" << stats.min << ", max=" << stats.max 
                  << ", mean=" << stats.mean << std::endl;
    }
    
    // 打印 top-5 logits
    std::cout << "\n=== Top-5 logits ===" << std::endl;
    std::vector<std::pair<float, int>> scored;
    for (size_t i = 0; i < config.vocabSize && i < logits.size(); ++i) {
        scored.push_back({logits[i], static_cast<int>(i)});
    }
    std::partial_sort(scored.begin(), scored.begin() + 5, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    for (int i = 0; i < 5; ++i) {
        std::cout << "  " << std::setw(2) << (i + 1) << ". Token " << std::setw(6) << scored[i].second 
                  << " | logit=" << std::fixed << std::setprecision(4) << scored[i].first << std::endl;
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
