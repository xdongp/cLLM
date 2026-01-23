/**
 * @file debug_batch_incr.cpp
 * @brief 调试批量推理 vs 增量推理的差异
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/kylin/core/tensor_stats.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

using namespace cllm::kylin;

void compareLogits(const std::vector<float>& batch, const std::vector<float>& incr, 
                   size_t vocabSize, const char* label) {
    float maxDiff = 0.0f;
    float sumDiff = 0.0f;
    int maxDiffIdx = 0;
    size_t largeDiffCount = 0;
    
    for (size_t i = 0; i < vocabSize; ++i) {
        float diff = std::abs(batch[i] - incr[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
            maxDiffIdx = i;
        }
        sumDiff += diff;
        if (diff > 1e-3f) largeDiffCount++;
    }
    
    float avgDiff = sumDiff / vocabSize;
    
    std::cout << label << ": max=" << std::fixed << std::setprecision(6) << maxDiff 
              << " (idx " << maxDiffIdx << "), avg=" << avgDiff 
              << ", large_diff=" << largeDiffCount << "/" << vocabSize;
    
    if (maxDiff > 0.01f) {
        std::cout << " [FAILED]";
        std::cout << "\n  Batch[" << maxDiffIdx << "]=" << batch[maxDiffIdx];
        std::cout << ", Incr[" << maxDiffIdx << "]=" << incr[maxDiffIdx];
    } else {
        std::cout << " [OK]";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    std::cout << "========================================" << std::endl;
    std::cout << "Debug Batch vs Incremental Inference" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // 加载模型
    GGMLTransformerModel model(BackendType::CPU);
    if (!model.loadFromGGUF(modelPath)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    const size_t vocabSize = model.getConfig().vocabSize;
    std::vector<int32_t> tokens = {9707, 11, 1234};  // "Hello world"
    
    std::cout << "\n=== Step-by-Step Comparison ===" << std::endl;
    
    // ========== 测试 1: 单 token 批量 vs 单 token 增量 ==========
    std::cout << "\n[Test 1] Single token (pos=0):" << std::endl;
    
    model.clearKVCache();
    auto batch1 = model.forward({tokens[0]});  // 批量：1个token
    
    model.clearKVCache();
    auto incr1 = model.forwardOneToken(tokens[0], 0);  // 增量：1个token
    
    compareLogits(
        std::vector<float>(batch1.begin(), batch1.begin() + vocabSize),
        incr1, vocabSize, "  Token 0"
    );
    
    // ========== 测试 2: 2 token 批量 vs 增量 ==========
    std::cout << "\n[Test 2] Two tokens:" << std::endl;
    
    model.clearKVCache();
    auto batch2 = model.forward({tokens[0], tokens[1]});  // 批量：2个token
    auto batch2_last = std::vector<float>(batch2.begin() + vocabSize, batch2.begin() + 2 * vocabSize);
    
    model.clearKVCache();
    auto incr2_0 = model.forwardOneToken(tokens[0], 0);  // 增量：第1个token
    auto incr2_1 = model.forwardOneToken(tokens[1], 1);  // 增量：第2个token
    
    // 比较位置0
    compareLogits(
        std::vector<float>(batch2.begin(), batch2.begin() + vocabSize),
        incr2_0, vocabSize, "  Token 0 (pos 0)"
    );
    
    // 比较位置1
    compareLogits(batch2_last, incr2_1, vocabSize, "  Token 1 (pos 1)");
    
    // ========== 测试 3: 3 token 批量 vs 增量 ==========
    std::cout << "\n[Test 3] Three tokens:" << std::endl;
    
    model.clearKVCache();
    auto batch3 = model.forward(tokens);  // 批量：3个token
    
    model.clearKVCache();
    std::vector<std::vector<float>> incr3;
    for (size_t i = 0; i < tokens.size(); ++i) {
        incr3.push_back(model.forwardOneToken(tokens[i], i));
    }
    
    // 比较每个位置
    for (size_t i = 0; i < 3; ++i) {
        auto batch_pos = std::vector<float>(
            batch3.begin() + i * vocabSize,
            batch3.begin() + (i + 1) * vocabSize
        );
        std::string label = "  Token " + std::to_string(i) + " (pos " + std::to_string(i) + ")";
        compareLogits(batch_pos, incr3[i], vocabSize, label.c_str());
    }
    
    // ========== 测试 4: KV Cache 数据比较 ==========
    std::cout << "\n[Test 4] KV Cache data comparison:" << std::endl;
    
    // 获取增量推理后的 KV
    std::vector<std::vector<float>> incrK(3), incrV(3);
    for (size_t pos = 0; pos < 3; ++pos) {
        model.getKVAtPosition(0, pos, incrK[pos], incrV[pos]);
    }
    
    // 重做批量推理获取 KV
    model.clearKVCache();
    model.forward(tokens);
    
    std::vector<std::vector<float>> batchK(3), batchV(3);
    for (size_t pos = 0; pos < 3; ++pos) {
        model.getKVAtPosition(0, pos, batchK[pos], batchV[pos]);
    }
    
    // 比较 KV 数据
    for (size_t pos = 0; pos < 3; ++pos) {
        float kMaxDiff = 0.0f, vMaxDiff = 0.0f;
        for (size_t i = 0; i < incrK[pos].size(); ++i) {
            kMaxDiff = std::max(kMaxDiff, std::abs(incrK[pos][i] - batchK[pos][i]));
            vMaxDiff = std::max(vMaxDiff, std::abs(incrV[pos][i] - batchV[pos][i]));
        }
        std::cout << "  Pos " << pos << ": K max_diff=" << std::fixed << std::setprecision(9) << kMaxDiff
                  << ", V max_diff=" << vMaxDiff;
        if (kMaxDiff > 1e-6f || vMaxDiff > 1e-6f) {
            std::cout << " [DIFF DETECTED]";
        } else {
            std::cout << " [OK]";
        }
        std::cout << std::endl;
    }
    
    // ========== 测试 5: 单独测试第3个token ===========
    std::cout << "\n[Test 5] Isolated test for token 2 (pos 2):" << std::endl;
    
    // 批量推理：只有前2个token
    model.clearKVCache();
    model.forward({tokens[0], tokens[1]});  // 先处理 token 0, 1
    
    // 然后增量添加 token 2
    auto incr_after_batch = model.forwardOneToken(tokens[2], 2);
    
    // 与纯批量推理的 token 2 比较
    auto batch3_last = std::vector<float>(
        batch3.begin() + 2 * vocabSize,
        batch3.begin() + 3 * vocabSize
    );
    
    compareLogits(batch3_last, incr_after_batch, vocabSize, 
                  "  Batch[0:2]+Incr[2] vs Batch[0:3]");
    
    // ========== 测试 6: 纯批量推理一致性 ==========
    std::cout << "\n[Test 6] Batch consistency check:" << std::endl;
    
    model.clearKVCache();
    auto batch3_run1 = model.forward(tokens);
    
    model.clearKVCache();
    auto batch3_run2 = model.forward(tokens);
    
    auto b1_last = std::vector<float>(batch3_run1.begin() + 2*vocabSize, batch3_run1.begin() + 3*vocabSize);
    auto b2_last = std::vector<float>(batch3_run2.begin() + 2*vocabSize, batch3_run2.begin() + 3*vocabSize);
    
    compareLogits(b1_last, b2_last, vocabSize, "  Batch run1 vs run2");
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Debug Complete" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
