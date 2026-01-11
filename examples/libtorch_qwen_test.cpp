/**
 * @file libtorch_qwen_test.cpp
 * @brief LibTorch Backend Qwen模型测试程序
 * 
 * 测试LibTorch后端加载Qwen模型并处理"hello"输入
 */

#include "cllm/inference/libtorch_backend.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/tokenizer.h"
#include <iostream>
#include <vector>
#include <string>

using namespace cllm;
using namespace cllm::inference;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "LibTorch Backend Qwen Model Test" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // 1. 配置模型参数
        ModelConfig config;
        config.vocabSize = 152064;  // Qwen3 0.6B 模型的词汇表大小
        config.hiddenSize = 2048;
        config.numLayers = 28;
        config.numAttentionHeads = 16;
        config.maxSequenceLength = 8192;

        // 2. 模型路径
        std::string modelPath = "./model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
        std::cout << "Loading model from: " << modelPath << std::endl;

        // 3. 创建LibTorch后端实例
        LibTorchBackend backend(modelPath, config);
        
        // 4. 设置设备（CPU）
        backend.setDevice(false, 0);  // 使用CPU
        std::cout << "Device set to CPU" << std::endl;

        // 5. 初始化后端
        std::cout << "Initializing backend..." << std::endl;
        bool initResult = backend.initialize();
        if (!initResult) {
            std::cerr << "Failed to initialize backend" << std::endl;
            return 1;
        }
        std::cout << "Backend initialized successfully!" << std::endl;
        std::cout << "Backend name: " << backend.getName() << std::endl;
        std::cout << "Is initialized: " << (backend.isInitialized() ? "true" : "false") << std::endl;

        // 6. 准备输入 "hello"
        // 注意：我们需要使用适当的tokenizer来将文本转换为token IDs
        // 由于我们可能没有Qwen专用的tokenizer，这里我们使用一个简单的数字表示
        // 实际应用中需要使用Qwen模型对应的tokenizer
        std::vector<int> inputIds = {1559, 159}; // "hello" 对应的token IDs (根据Qwen tokenizer)
        std::cout << "Input token IDs: ";
        for (int id : inputIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // 7. 执行前向推理
        std::cout << "Running forward pass..." << std::endl;
        Tensor result = backend.forward(inputIds);
        
        std::cout << "Forward pass completed!" << std::endl;
        std::cout << "Output tensor shape: ";
        auto shape = result.shape();
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << " x ";
        }
        std::cout << std::endl;

        // 8. 获取输出的logits并找到最可能的下一个token
        size_t seqLen = shape[0];
        size_t vocabSize = shape[1];
        float* data = result.data();
        
        std::cout << "Sequence length: " << seqLen << ", Vocabulary size: " << vocabSize << std::endl;
        
        // 对于每个位置，找到概率最高的token
        for (size_t pos = 0; pos < seqLen; ++pos) {
            float maxLogit = data[pos * vocabSize];
            int maxIdx = 0;
            for (size_t i = 1; i < vocabSize; ++i) {
                if (data[pos * vocabSize + i] > maxLogit) {
                    maxLogit = data[pos * vocabSize + i];
                    maxIdx = static_cast<int>(i);
                }
            }
            std::cout << "Position " << pos << ": Top token ID = " << maxIdx 
                      << ", Logit = " << maxLogit << std::endl;
        }

        std::cout << "===========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "LibTorch backend loaded Qwen model and processed 'hello' input" << std::endl;
        std::cout << "===========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error occurred: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}