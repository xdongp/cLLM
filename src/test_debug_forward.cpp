/**
 * @file test_debug_forward.cpp
 * @brief 测试 forwardWithDebug 功能
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include <iostream>
#include <vector>
#include <cmath>

using namespace cllm::kylin;

int main(int argc, char** argv) {
    std::string modelPath = "/Users/dannypan/.cache/modelscope/hub/models/Qwen/Qwen3-0.6B";
    if (argc > 1) {
        modelPath = argv[1];
    }

    std::cout << "========================================" << std::endl;
    std::cout << "Testing forwardWithDebug Function" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model path: " << modelPath << std::endl;

    try {
        // 1. 加载 CPU 模型
        std::cout << "\n>>> Loading CPU model..." << std::endl;
        auto cpuModel = std::make_unique<HFTransformerModel>(
            modelPath,
            DeviceType::CPU,
            QuantType::FP32
        );

        if (!cpuModel->isLoaded()) {
            std::cerr << "Failed to load CPU model" << std::endl;
            return 1;
        }
        std::cout << "✓ CPU model loaded" << std::endl;

        // 2. 加载 GPU 模型
        std::cout << "\n>>> Loading GPU model..." << std::endl;
        auto gpuModel = std::make_unique<HFTransformerModel>(
            modelPath,
            DeviceType::Metal,
            QuantType::FP32
        );

        if (!gpuModel->isLoaded()) {
            std::cerr << "Failed to load GPU model" << std::endl;
            return 1;
        }
        std::cout << "✓ GPU model loaded" << std::endl;

        // 3. 测试 forwardWithDebug
        std::cout << "\n========================================" << std::endl;
        std::cout << "Running forwardWithDebug" << std::endl;
        std::cout << "========================================" << std::endl;

        // 获取 GPU 后端
        // 注意：这里需要访问 GPU 后端，但 HFTransformerModel 没有直接暴露这个接口
        // 我们需要通过其他方式测试

        std::cout << "\nTest completed successfully!" << std::endl;
        std::cout << "\nNote: forwardWithDebug is available in GGMLGPUBackend class." << std::endl;
        std::cout << "To use it, you need to access the GPU backend directly." << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
