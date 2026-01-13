#include "cllm/sampler.h"
#include "cllm/memory/float_array.h"
#include <iostream>
#include <vector>

int main() {
    // 创建Sampler实例
    cllm::Sampler sampler;
    
    // 测试数据：模拟logits
    std::vector<float> logits = {1.0f, 2.0f, 0.5f, 3.0f, 0.8f};
    cllm::FloatArray floatArray(logits.size());
    for (size_t i = 0; i < logits.size(); ++i) {
        floatArray[i] = logits[i];
    }
    
    std::cout << "=== Sampler Test ===\n";
    
    // 测试1: 贪心采样
    int greedyResult = sampler.sample(floatArray, 0.0f);
    std::cout << "Greedy sampling result: " << greedyResult << " (expected: 3)\n";
    
    // 测试2: 温度采样
    int tempResult = sampler.sample(floatArray, 0.5f);
    std::cout << "Temperature sampling result (temp=0.5): " << tempResult << "\n";
    
    // 测试3: Top-K采样
    int topKResult = sampler.sample(floatArray, 1.0f, 3);
    std::cout << "Top-K sampling result (k=3): " << topKResult << "\n";
    
    // 测试4: Top-P采样
    int topPResult = sampler.sample(floatArray, 1.0f, -1, 0.9f);
    std::cout << "Top-P sampling result (p=0.9): " << topPResult << "\n";
    
    // 测试5: 混合采样 (Top-K + Top-P)
    int hybridResult = sampler.sample(floatArray, 0.8f, 4, 0.85f);
    std::cout << "Hybrid sampling result (temp=0.8, k=4, p=0.85): " << hybridResult << "\n";
    
    // 测试6: 批处理采样
    std::vector<float> batchLogits = {
        1.0f, 2.0f, 0.5f, 3.0f, 0.8f,  // 批次1
        0.2f, 1.5f, 2.8f, 0.9f, 1.2f   // 批次2
    };
    cllm::FloatArray batchFloatArray(batchLogits.size());
    for (size_t i = 0; i < batchLogits.size(); ++i) {
        batchFloatArray[i] = batchLogits[i];
    }
    
    std::vector<int> batchResults = sampler.sampleBatch(batchFloatArray, 2, 0.5f, 3, 0.9f);
    std::cout << "Batch sampling results: ";
    for (int result : batchResults) {
        std::cout << result << " ";
    }
    std::cout << "\n";
    
    // 测试7: 性能统计
    std::cout << "Sample count after tests: " << sampler.getSampleCount() << "\n";
    
    // 测试8: 重置统计
    sampler.resetStats();
    std::cout << "Sample count after reset: " << sampler.getSampleCount() << "\n";
    
    std::cout << "=== All tests completed ===\n";
    
    return 0;
}