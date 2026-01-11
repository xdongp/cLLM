#include "cllm/inference/libtorch_backend.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/json_tokenizer.h" // 假设有这个头文件
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Minimal Inference Test (JSON Tokenizer)" << std::endl;
    
    try {
        // 1. 初始化配置
        cllm::ModelConfig config;
        config.hiddenSize = 512;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 8;
        config.numLayers = 6;
        config.vocabSize = 32000;
        config.intermediateSize = 2048;

        // 2. 初始化JSON分词器
        std::string tokenizerPath = "../model/Qwen/Qwen3-0.6B/tokenizer.json";
        auto tokenizer = std::make_shared<cllm::JsonTokenizer>(tokenizerPath);
        
        // 3. 加载模型
        std::string modelPath = "../model/Qwen/Qwen3-0.6B/qwen3_0.6b_torchscript_fp32.pt";
        auto backend = std::make_shared<cllm::inference::LibTorchBackend>(modelPath, config);
        backend->setDevice(false, 0); // 使用CPU

        if (!backend->initialize()) {
            std::cerr << "Failed to initialize model" << std::endl;
            return 1;
        }

        // 4. 处理用户输入
        std::string input;
        std::cout << "Enter your input (type 'exit' to quit): ";
        std::getline(std::cin, input);

        while (input != "exit") {
            // 5. 分词
            auto tokenized = tokenizer->encode(input);
            
            // 6. 推理
            auto logits = backend->forward(tokenized);
            
            // 7. 采样
            int sampledToken = tokenizer->sample(logits);
            
            // 8. 解码输出
            auto output = tokenizer->decode({sampledToken});
            
            std::cout << "Model response: " << output << std::endl;
            
            // 获取下一轮输入
            std::cout << "Enter your input (type 'exit' to quit): ";
            std::getline(std::cin, input);
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}