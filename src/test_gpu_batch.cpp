#include <iostream>
#include <vector>
#include <chrono>
#include <thread>
#include <memory>

#include "cllm/kylin/hf/hf_transformer_model.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/hf/config.h"

using namespace cllm::kylin;

void testGPUBatchProcessing() {
    std::cout << "=== Testing GPU Batch Processing ===" << std::endl;
    
    std::string modelPath = "/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B";
    
    std::cout << "Initializing HFTransformerModel..." << std::endl;
    
    try {
        // 创建模型实例，使用 Metal GPU
        auto model = std::make_unique<HFTransformerModel>(modelPath, DeviceType::Metal, QuantType::FP16);
        
        if (!model->isLoaded()) {
            std::cerr << "Failed to load model" << std::endl;
            return;
        }
        
        std::cout << "Model loaded successfully!" << std::endl;
        std::cout << "Vocab size: " << model->vocabSize() << std::endl;
        std::cout << "Hidden size: " << model->hiddenSize() << std::endl;
        
        // 测试单个推理
        std::cout << "\n--- Testing Single Inference ---" << std::endl;
        std::vector<int32_t> singleToken = {1};  // 使用 token ID 1
        auto startTime = std::chrono::high_resolution_clock::now();
        auto singleResult = model->forwardWithRequestId(singleToken, 0);
        auto endTime = std::chrono::high_resolution_clock::now();
        auto singleDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Single inference completed in: " << singleDuration.count() << " ms" << std::endl;
        std::cout << "Single result size: " << singleResult.size() << std::endl;
        
        // 测试标准批量推理（使用 forwardBatch）
        std::cout << "\n--- Testing Standard Batch Processing ---" << std::endl;
        std::vector<std::vector<int32_t>> batchInputIds = {
            {1}, {2}, {3}, {4}  // 4 个不同的单 token 请求
        };
        std::vector<size_t> requestIds = {1, 2, 3, 4};  // 4 个不同的请求 ID
        
        startTime = std::chrono::high_resolution_clock::now();
        auto batchResult = model->forwardBatch(batchInputIds, requestIds);
        endTime = std::chrono::high_resolution_clock::now();
        auto batchDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        
        std::cout << "Batch processing completed in: " << batchDuration.count() << " ms" << std::endl;
        std::cout << "Batch result count: " << batchResult.size() << std::endl;
        for (size_t i = 0; i < batchResult.size(); ++i) {
            std::cout << "  Result " << i << " size: " << batchResult[i].size() << std::endl;
        }
        
        // 计算平均单个处理时间
        double avgBatchTime = static_cast<double>(batchDuration.count()) / batchInputIds.size();
        std::cout << "Average time per request (batch): " << avgBatchTime << " ms" << std::endl;
        std::cout << "Single time per request: " << static_cast<double>(singleDuration.count()) << " ms" << std::endl;
        
        // 检查批处理是否更高效
        if (avgBatchTime < singleDuration.count()) {
            std::cout << "✅ Batch processing is more efficient!" << std::endl;
        } else {
            std::cout << "ℹ️  Batch processing performance similar to single processing" << std::endl;
        }
        
        // 测试获取 KV Cache 长度
        std::cout << "\n--- Testing KV Cache Length Retrieval ---" << std::endl;
        for (size_t i = 0; i < requestIds.size(); ++i) {
            int len = model->getKVCacheCurrentLength(requestIds[i]);
            std::cout << "Request " << requestIds[i] << " KV Cache length: " << len << std::endl;
        }
        
        std::cout << "\n✅ GPU batch processing test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during GPU batch testing: " << e.what() << std::endl;
    }
}

int main() {
    testGPUBatchProcessing();
    return 0;
}