/**
 * @file test_ggml_transformer.cpp
 * @brief 测试基于 GGML 的 Transformer 模型
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <chrono>
#include <iomanip>

using namespace cllm;
using namespace cllm::kylin;

// 模型路径
const std::string MODEL_PATH = "../model/Qwen/qwen3-0.6b-q4_k_m.gguf";

void printTopK(const std::vector<float>& logits, int k = 5) {
    std::vector<std::pair<float, int>> scores;
    for (size_t i = 0; i < logits.size(); ++i) {
        scores.emplace_back(logits[i], i);
    }
    
    std::partial_sort(scores.begin(), scores.begin() + k, scores.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    
    std::cout << "Top " << k << " tokens:" << std::endl;
    for (int i = 0; i < k; ++i) {
        std::cout << "  [" << scores[i].second << "] = " 
                  << std::fixed << std::setprecision(4) << scores[i].first << std::endl;
    }
}

int main(int argc, char* argv[]) {
    std::string modelPath = MODEL_PATH;
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "GGML Transformer Model Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl;
    std::cout << std::endl;
    
    try {
    // 1. 创建模型（使用 CPU 后端）
    GGMLTransformerModel model(BackendType::CPU);
        
        // 2. 加载模型
        std::cout << "Loading model..." << std::endl;
        auto loadStart = std::chrono::high_resolution_clock::now();
        
        if (!model.loadFromGGUF(modelPath)) {
            std::cerr << "Failed to load model!" << std::endl;
            return 1;
        }
        
        auto loadEnd = std::chrono::high_resolution_clock::now();
        auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(loadEnd - loadStart).count();
        std::cout << "Model loaded in " << loadTime << " ms" << std::endl;
        
        // 3. 打印模型配置
        const auto& config = model.getConfig();
        std::cout << std::endl;
        std::cout << "Model Config:" << std::endl;
        std::cout << "  Architecture: " << config.architecture << std::endl;
        std::cout << "  Layers: " << config.blockCount << std::endl;
        std::cout << "  Hidden: " << config.embeddingLength << std::endl;
        std::cout << "  Heads: " << config.headCount << " (KV: " << config.headCountKV << ")" << std::endl;
        std::cout << "  FFN: " << config.feedForwardLength << std::endl;
        std::cout << "  Vocab: " << config.vocabSize << std::endl;
        std::cout << "  Context: " << config.contextLength << std::endl;
        std::cout << std::endl;
        
        // 4. 测试推理
        std::cout << "Testing inference..." << std::endl;
        
        // 简单的输入序列（模拟 "Hello" 的 token IDs）
        std::vector<int32_t> inputIds = {1, 72, 101, 108, 108, 111};  // 示例
        
        auto inferStart = std::chrono::high_resolution_clock::now();
        
        auto logits = model.forward(inputIds);
        
        auto inferEnd = std::chrono::high_resolution_clock::now();
        auto inferTime = std::chrono::duration_cast<std::chrono::milliseconds>(inferEnd - inferStart).count();
        
        std::cout << "Inference completed in " << inferTime << " ms" << std::endl;
        std::cout << "Output shape: [" << inputIds.size() << ", " << config.vocabSize << "]" << std::endl;
        std::cout << "Total logits: " << logits.size() << std::endl;
        std::cout << std::endl;
        
        // 5. 打印最后一个 token 的 top-k logits
        if (logits.size() >= config.vocabSize) {
            size_t lastTokenOffset = (inputIds.size() - 1) * config.vocabSize;
            std::vector<float> lastLogits(logits.begin() + lastTokenOffset, 
                                          logits.begin() + lastTokenOffset + config.vocabSize);
            std::cout << "Last token predictions:" << std::endl;
            printTopK(lastLogits, 10);
        }
        
        // 6. 测试增量推理（验证 KV Cache）
        std::cout << std::endl;
        std::cout << "Testing incremental inference with KV cache..." << std::endl;
        
        model.clearKVCache();
        
        // 逐 token 推理
        std::vector<float> lastTokenLogits;
        auto incrStart = std::chrono::high_resolution_clock::now();
        
        for (size_t i = 0; i < inputIds.size(); ++i) {
            auto tokenLogits = model.forwardOneToken(inputIds[i], i);
            lastTokenLogits = tokenLogits;
            std::cout << "  Token " << i << " (id=" << inputIds[i] << "): "
                      << "KV cache len = " << model.getKVCacheLength() << std::endl;
        }
        
        auto incrEnd = std::chrono::high_resolution_clock::now();
        auto incrTime = std::chrono::duration_cast<std::chrono::milliseconds>(incrEnd - incrStart).count();
        std::cout << "Incremental inference completed in " << incrTime << " ms" << std::endl;
        
        // 比较增量推理和批量推理的结果
        std::cout << std::endl;
        std::cout << "Comparing batch vs incremental results..." << std::endl;
        
        model.clearKVCache();
        auto batchLogits = model.forward(inputIds);
        
        // 取最后一个 token 的 logits 比较
        size_t lastOffset = (inputIds.size() - 1) * config.vocabSize;
        std::vector<float> batchLastLogits(batchLogits.begin() + lastOffset,
                                            batchLogits.begin() + lastOffset + config.vocabSize);
        
        // 计算差异
        float maxDiff = 0.0f;
        float sumDiff = 0.0f;
        int maxDiffIdx = 0;
        for (size_t i = 0; i < config.vocabSize; ++i) {
            float diff = std::abs(batchLastLogits[i] - lastTokenLogits[i]);
            if (diff > maxDiff) {
                maxDiff = diff;
                maxDiffIdx = static_cast<int>(i);
            }
            sumDiff += diff;
        }
        float avgDiff = sumDiff / config.vocabSize;
        
        std::cout << "  Max logit diff: " << std::fixed << std::setprecision(6) << maxDiff 
                  << " (at idx " << maxDiffIdx << ")" << std::endl;
        std::cout << "  Avg logit diff: " << std::fixed << std::setprecision(6) << avgDiff << std::endl;
        
        // 显示具体差异值
        std::cout << "  Sample values at max diff idx:" << std::endl;
        std::cout << "    Batch: " << std::fixed << std::setprecision(6) << batchLastLogits[maxDiffIdx] << std::endl;
        std::cout << "    Incr:  " << std::fixed << std::setprecision(6) << lastTokenLogits[maxDiffIdx] << std::endl;
        
        // 测试不同 token 数量的差异
        std::cout << std::endl;
        std::cout << "Testing diff by token count..." << std::endl;
        
        for (size_t nTokens = 2; nTokens <= inputIds.size(); ++nTokens) {
            std::vector<int32_t> tokens(inputIds.begin(), inputIds.begin() + nTokens);
            
            // 批量推理
            model.clearKVCache();
            auto batchLogitsN = model.forward(tokens);
            std::vector<float> batchLastN(
                batchLogitsN.begin() + (nTokens - 1) * config.vocabSize,
                batchLogitsN.begin() + nTokens * config.vocabSize);
            
            // 增量推理
            model.clearKVCache();
            std::vector<float> incrLastN;
            for (size_t i = 0; i < nTokens; ++i) {
                incrLastN = model.forwardOneToken(tokens[i], i);
            }
            
            // 计算差异
            float maxDiffN = 0.0f;
            float sumDiffN = 0.0f;
            for (size_t i = 0; i < config.vocabSize; ++i) {
                float diff = std::abs(batchLastN[i] - incrLastN[i]);
                maxDiffN = std::max(maxDiffN, diff);
                sumDiffN += diff;
            }
            float avgDiffN = sumDiffN / config.vocabSize;
            
            std::cout << "  " << nTokens << " tokens: max=" << std::fixed << std::setprecision(6) 
                      << maxDiffN << ", avg=" << avgDiffN;
            if (maxDiffN < 1e-3) {
                std::cout << " ✓";
            } else {
                std::cout << " ⚠";
            }
            std::cout << std::endl;
        }
        
        // 测试增量推理的内部一致性（逐 token 叠加 vs 全量）
        std::cout << std::endl;
        std::cout << "Testing incremental consistency (3 tokens)..." << std::endl;
        
        // 方法1: 1+1+1
        model.clearKVCache();
        model.forwardOneToken(inputIds[0], 0);
        model.forwardOneToken(inputIds[1], 1);
        auto incr3a = model.forwardOneToken(inputIds[2], 2);
        
        // 方法2: 2+1
        model.clearKVCache();
        std::vector<int32_t> first2 = {inputIds[0], inputIds[1]};
        model.forward(first2);
        auto incr3b = model.forwardOneToken(inputIds[2], 2);
        
        // 方法3: 3
        model.clearKVCache();
        std::vector<int32_t> first3 = {inputIds[0], inputIds[1], inputIds[2]};
        auto batch3 = model.forward(first3);
        std::vector<float> batch3Last(batch3.begin() + 2 * config.vocabSize,
                                       batch3.begin() + 3 * config.vocabSize);
        
        // 比较
        float diff_1_1_1_vs_batch = 0.0f, diff_2_1_vs_batch = 0.0f, diff_1_1_1_vs_2_1 = 0.0f;
        for (size_t i = 0; i < config.vocabSize; ++i) {
            diff_1_1_1_vs_batch = std::max(diff_1_1_1_vs_batch, std::abs(incr3a[i] - batch3Last[i]));
            diff_2_1_vs_batch = std::max(diff_2_1_vs_batch, std::abs(incr3b[i] - batch3Last[i]));
            diff_1_1_1_vs_2_1 = std::max(diff_1_1_1_vs_2_1, std::abs(incr3a[i] - incr3b[i]));
        }
        
        std::cout << "  1+1+1 vs batch(3): " << std::fixed << std::setprecision(6) << diff_1_1_1_vs_batch << std::endl;
        std::cout << "  2+1 vs batch(3):   " << std::fixed << std::setprecision(6) << diff_2_1_vs_batch << std::endl;
        std::cout << "  1+1+1 vs 2+1:      " << std::fixed << std::setprecision(6) << diff_1_1_1_vs_2_1 << std::endl;
        
        // 7. 性能测试：多次推理取平均
        std::cout << std::endl;
        std::cout << "Performance benchmark (10 iterations)..." << std::endl;
        
        std::vector<long long> times;
        for (int iter = 0; iter < 10; ++iter) {
            model.clearKVCache();
            
            auto start = std::chrono::high_resolution_clock::now();
            auto logitsResult = model.forward(inputIds);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto timeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            times.push_back(timeMs);
        }
        
        // 计算平均和最小
        long long sum = 0, minTime = times[0];
        for (auto t : times) {
            sum += t;
            if (t < minTime) minTime = t;
        }
        double avgTime = static_cast<double>(sum) / times.size();
        
        std::cout << "  Average time: " << std::fixed << std::setprecision(2) << avgTime << " ms" << std::endl;
        std::cout << "  Min time: " << minTime << " ms" << std::endl;
        std::cout << "  Tokens/sec: " << std::fixed << std::setprecision(1) 
                  << (inputIds.size() * 1000.0 / avgTime) << std::endl;
        std::cout << std::endl;
        
        std::cout << "========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
