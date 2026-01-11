/**
 * @file test_libtorch_main.cpp
 * @brief LibTorch Backend 主测试程序
 * @author cLLM Team
 * @date 2026-01-10
 */

#include "cllm/inference/libtorch_backend.h"
#include "cllm/model/config.h"
#include "cllm/inference/inference_engine.h"
#include <iostream>
#include <vector>
#include <memory>

using namespace cllm;
using namespace cllm::inference;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "LibTorch Backend Main Test Program" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // 1. 创建模型配置
        ModelConfig config;
        config.hiddenSize = 512;      // 假设隐藏层大小
        config.numAttentionHeads = 8; // 假设注意力头数
        config.numKeyValueHeads = 8;  // 假设KV头数
        config.numLayers = 6;         // 假设层数
        config.vocabSize = 1000;      // 假设词汇表大小
        config.intermediateSize = 2048; // 假设中间层大小

        std::cout << "Created model configuration." << std::endl;

        // 2. 创建LibTorch Backend实例（使用假路径，因为实际模型可能不存在）
        std::string modelPath = "./dummy_model.pt";  // 使用假路径进行测试
        LibTorchBackend backend(modelPath, config);

        std::cout << "Created LibTorchBackend instance." << std::endl;

        // 3. 设置设备为CPU
        backend.setDevice(false, 0); // 使用CPU
        
        std::cout << "Set device to CPU." << std::endl;

        // 4. 尝试初始化（预期会失败，因为模型文件不存在）
        std::cout << "Attempting to initialize backend (this will fail with dummy model path)..." << std::endl;
        bool initSuccess = backend.initialize();
        
        if (!initSuccess) {
            std::cout << "Initialization failed as expected with dummy model path." << std::endl;
        } else {
            std::cout << "Initialization succeeded unexpectedly!" << std::endl;
        }

        // 5. 测试配置获取
        const ModelConfig& retrievedConfig = backend.getConfig();
        std::cout << "Retrieved config - vocab size: " << retrievedConfig.vocabSize 
                  << ", hidden size: " << retrievedConfig.hiddenSize << std::endl;

        // 6. 测试后端名称获取
        std::string backendName = backend.getName();
        std::cout << "Backend name: " << backendName << std::endl;

        // 7. 测试初始化状态
        bool isInit = backend.isInitialized();
        std::cout << "Is initialized: " << (isInit ? "true" : "false") << std::endl;

        // 8. 测试forward方法（预期会抛出异常）
        std::vector<int> inputIds = {1, 2, 3, 4, 5};
        std::cout << "Testing forward method with input: ";
        for (int id : inputIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        try {
            kylin::Tensor result = backend.forward(inputIds);
            std::cout << "Forward succeeded (unexpected!)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Forward correctly threw exception: " << e.what() << std::endl;
        }

        // 9. 测试forwardBatch方法（预期会抛出异常）
        std::vector<int> flatInputIds = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 3}, {3, 5}, {5, 8}};
        size_t batchSize = 3;
        
        std::cout << "Testing forwardBatch method..." << std::endl;
        try {
            kylin::Tensor result = backend.forwardBatch(flatInputIds, requestPositions, batchSize);
            std::cout << "ForwardBatch succeeded (unexpected!)" << std::endl;
        } catch (const std::exception& e) {
            std::cout << "ForwardBatch correctly threw exception: " << e.what() << std::endl;
        }

        std::cout << "===========================================" << std::endl;
        std::cout << "Main test completed successfully!" << std::endl;
        std::cout << "Note: Some operations failed as expected due to dummy model path" << std::endl;
        std::cout << "===========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}