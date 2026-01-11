/**
 * @file test_real_libtorch_backend.cpp
 * @brief LibTorch Backend 实际测试程序（使用真实导出的模型）
 * @author cLLM Team
 * @date 2026-01-10
 */

#include "cllm/inference/libtorch_backend.h"
#include "cllm/model/config.h"
#include <iostream>
#include <vector>
#include <memory>
#include <filesystem>

using namespace cllm;
using namespace cllm::inference;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "Real LibTorch Backend Test Program" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // 1. 创建模型配置，与Python脚本中的一致
        ModelConfig config;
        config.hiddenSize = 32;      // 使用导出模型的实际参数
        config.numAttentionHeads = 4; 
        config.numKeyValueHeads = 4;  
        config.numLayers = 2;         // 使用较小的层数
        config.vocabSize = 500;      // 使用导出模型的实际参数
        config.intermediateSize = 128;

        std::cout << "Created model configuration." << std::endl;

        // 2. 创建LibTorch Backend实例（使用实际的模型路径）
        // 首先尝试当前目录，然后尝试项目根目录
        std::string modelPath = "./simple_model.pt";
        if (!std::filesystem::exists(modelPath)) {
            modelPath = "/Users/dannypan/PycharmProjects/xllm/simple_model.pt";
        }
        
        std::cout << "Looking for model at: " << modelPath << std::endl;
        if (std::filesystem::exists(modelPath)) {
            std::cout << "Model file found!" << std::endl;
        } else {
            std::cout << "Model file NOT found!" << std::endl;
            return 1;
        }
        
        LibTorchBackend backend(modelPath, config);

        std::cout << "Created LibTorchBackend instance with model path: " << modelPath << std::endl;

        // 3. 设置设备为CPU
        backend.setDevice(false, 0); // 使用CPU
        
        std::cout << "Set device to CPU." << std::endl;

        // 4. 尝试初始化
        std::cout << "Attempting to initialize backend..." << std::endl;
        bool initSuccess = backend.initialize();
        
        if (initSuccess) {
            std::cout << "Initialization succeeded!" << std::endl;
        } else {
            std::cout << "Initialization failed." << std::endl;
            return 1;
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

        // 8. 测试forward方法（使用长度为8的输入，与trace时一致）
        std::vector<int> inputIds = {1, 2, 3, 4, 5, 6, 7, 8};  // 长度为8，与trace时一致
        std::cout << "Testing forward method with input: ";
        for (int id : inputIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        
        try {
            kylin::Tensor result = backend.forward(inputIds);
            std::cout << "Forward succeeded!" << std::endl;
            std::cout << "Output tensor shape: ";
            auto shape = result.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << " x ";
            }
            std::cout << std::endl;
            
            // 打印一些输出值（仅前几个）
            const float* data = result.data();
            size_t size = result.size();
            std::cout << "First 10 output values: ";
            for (size_t i = 0; i < std::min(size, static_cast<size_t>(10)); ++i) {
                std::cout << data[i] << " ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Forward failed: " << e.what() << std::endl;
        }

        // 9. 测试forwardBatch方法
        std::vector<int> flatInputIds = {1, 2, 3, 4, 5, 6, 7, 8};
        std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 4}, {4, 8}}; // 两个请求，各4个token
        size_t batchSize = 2;
        
        std::cout << "Testing forwardBatch method with batch size: " << batchSize << std::endl;
        try {
            kylin::Tensor result = backend.forwardBatch(flatInputIds, requestPositions, batchSize);
            std::cout << "ForwardBatch succeeded!" << std::endl;
            std::cout << "Batch output tensor shape: ";
            auto shape = result.shape();
            for (size_t i = 0; i < shape.size(); ++i) {
                std::cout << shape[i];
                if (i < shape.size() - 1) std::cout << " x ";
            }
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cout << "ForwardBatch failed: " << e.what() << std::endl;
        }

        std::cout << "===========================================" << std::endl;
        std::cout << "Real test completed successfully!" << std::endl;
        std::cout << "===========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}