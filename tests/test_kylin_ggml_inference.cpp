/**
 * @file test_kylin_ggml_inference.cpp
 * @brief 测试 Kylin 后端的 GGML 直接推理
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace cllm::kylin;

int main(int argc, char** argv) {
    // 初始化日志
    cllm::Logger::instance().setLevel(spdlog::level::info);
    
    std::string modelPath = "../model/Qwen/qwen3-0.6b-q4_k_m.gguf";
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Kylin GGML Direct Inference Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl << std::endl;
    
    try {
        // 1. 创建并加载模型（使用 Metal GPU）
        std::cout << "1. Loading model with Metal GPU..." << std::endl;
        GGMLTransformerModel model(BackendType::Metal);
        
        if (!model.loadFromGGUF(modelPath)) {
            std::cerr << "❌ Failed to load model!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model loaded successfully!" << std::endl;
        std::cout << "   Vocab size: " << model.getConfig().vocabSize << std::endl;
        std::cout << "   Layers: " << model.getConfig().blockCount << std::endl;
        std::cout << "   Hidden: " << model.getConfig().embeddingLength << std::endl << std::endl;
        
        // 2. 测试简单输入
        std::cout << "2. Testing simple input..." << std::endl;
        
        // 测试 token IDs: [1, 2, 3]（简单测试）
        std::vector<int32_t> testIds = {1, 2, 3};
        
        std::cout << "   Input IDs: [";
        for (size_t i = 0; i < testIds.size(); ++i) {
            std::cout << testIds[i];
            if (i < testIds.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        std::vector<float> logits = model.forward(testIds);
        
        const size_t seqLen = testIds.size();
        const size_t vocabSize = model.getConfig().vocabSize;
        
        std::cout << "   Output shape: [" << seqLen << ", " << vocabSize << "]" << std::endl;
        std::cout << "   Total logits: " << logits.size() << std::endl;
        
        // 3. 验证 logits 数据
        std::cout << "\n3. Verifying logits..." << std::endl;
        
        size_t nanCount = 0, infCount = 0;
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        double sum = 0.0;
        
        for (size_t i = 0; i < logits.size(); ++i) {
            float val = logits[i];
            if (std::isnan(val)) nanCount++;
            else if (std::isinf(val)) infCount++;
            else {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
                sum += val;
            }
        }
        
        std::cout << "   Range: [" << minVal << ", " << maxVal << "]" << std::endl;
        std::cout << "   Mean: " << (sum / logits.size()) << std::endl;
        std::cout << "   NaN count: " << nanCount << std::endl;
        std::cout << "   Inf count: " << infCount << std::endl;
        
        if (nanCount > 0 || infCount > 0) {
            std::cerr << "❌ Logits contain NaN or Inf values!" << std::endl;
            return 1;
        }
        
        // 4. 检查每个位置的最大概率 token
        std::cout << "\n4. Top predictions for each position:" << std::endl;
        
        for (size_t pos = 0; pos < seqLen; ++pos) {
            const float* posLogits = logits.data() + pos * vocabSize;
            
            // 找到 top-5 token IDs
            std::vector<std::pair<float, int32_t>> topTokens;
            for (size_t i = 0; i < vocabSize; ++i) {
                topTokens.push_back({posLogits[i], static_cast<int32_t>(i)});
            }
            
            std::partial_sort(topTokens.begin(), topTokens.begin() + 5, topTokens.end(),
                            [](const auto& a, const auto& b) { return a.first > b.first; });
            
            std::cout << "   Position " << pos << " top-5: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << "(" << topTokens[i].second << ":" << topTokens[i].first << ")";
                if (i < 4) std::cout << ", ";
            }
            std::cout << std::endl;
        }
        
        // 5. 测试增量推理
        std::cout << "\n5. Testing incremental inference (KV cache)..." << std::endl;
        
        model.clearKVCache();
        
        // 首次推理：[1, 2, 3]
        std::vector<float> logits1 = model.forward(testIds);
        std::cout << "   First forward: seq_len=" << testIds.size() 
                 << ", KV cache len=" << model.getKVCacheLength() << std::endl;
        
        // 增量推理：[4]
        std::vector<int32_t> nextId = {4};
        std::vector<float> logits2 = model.forward(nextId);
        std::cout << "   Incremental forward: seq_len=" << nextId.size()
                 << ", KV cache len=" << model.getKVCacheLength() << std::endl;
        
        // 验证 KV cache
        if (model.getKVCacheLength() == testIds.size() + nextId.size()) {
            std::cout << "   ✅ KV cache length correct: " << model.getKVCacheLength() << std::endl;
        } else {
            std::cerr << "   ❌ KV cache length incorrect: expected " 
                     << (testIds.size() + nextId.size()) 
                     << ", got " << model.getKVCacheLength() << std::endl;
        }
        
        std::cout << "\n✅ All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
