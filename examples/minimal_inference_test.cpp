#include "cllm/inference/libtorch_backend.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/qwen2_tokenizer.h"
#include <iostream>
#include <vector>
#include <memory>

int main() {
    std::cout << "Minimal Inference Test" << std::endl;
    
    try {
        // 1. 初始化配置
        cllm::ModelConfig config;
        config.hiddenSize = 512;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 8;
        config.numLayers = 6;
        config.vocabSize = 32000;  // 假设词汇表大小
        config.intermediateSize = 2048;
        // 2. 初始化分词器
        std::string tokenizerPath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B";  // 使用Qwen2Tokenizer
        auto tokenizer = std::make_shared<cllm::Qwen2Tokenizer>(tokenizerPath);
        
        // 3. 加载模型
        std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3_0.6b_torchscript_fp32.pt";  // 使用绝对路径
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
            try {
                // 5. 分词
                std::cout << "Encoding input: " << input << std::endl;
                auto tokenized = tokenizer->encode(input);
                
                std::cout << "Tokenized: [";
                for (size_t i = 0; i < tokenized.size(); ++i) {
                    std::cout << tokenized[i];
                    if (i < tokenized.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
                
                // 6. 推理
                std::cout << "Running model inference..." << std::endl;
                auto logits = backend->forward(tokenized);
                
                // 检查logits维度
                std::cout << "Logits shape: [";
                for (size_t i = 0; i < logits.ndim(); ++i) {
                    std::cout << logits.shape()[i];
                    if (i < logits.ndim() - 1) std::cout << " x ";
                }
                std::cout << "]" << std::endl;
                
                // 7. 采样 (简单取概率最高的token)
                const float* logits_data = logits.data();
                size_t vocab_size = logits.size();
                std::cout << "Vocab size: " << vocab_size << std::endl;
                
                size_t sampledToken = 0;
                float maxProb = logits_data[0];
                for (size_t i = 1; i < std::min(vocab_size, size_t(10)); ++i) {
                    if (logits_data[i] > maxProb) {
                        maxProb = logits_data[i];
                        sampledToken = i;
                    }
                }
                
                std::cout << "Sampled token: " << sampledToken << " (probability: " << maxProb << ")" << std::endl;
                
                // 8. 解码输出
                std::vector<int> tokens = {static_cast<int>(sampledToken)};
                auto output = tokenizer->decode(tokens);
                
                std::cout << "Model response: " << output << std::endl;
                
            } catch (const std::exception& e) {
                std::cerr << "Error processing input: " << e.what() << std::endl;
            }
            
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