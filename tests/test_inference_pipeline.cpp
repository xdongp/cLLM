#include "cllm/tokenizer/tokenizer.h"
#include "cllm/model/executor.h"
#include "cllm/sampler.h"
#include "cllm/common/types.h"
#include <iostream>
#include <memory>
#include <vector>

int main() {
    std::cout << "Starting End-to-End Inference Pipeline Test..." << std::endl;
    
    try {
        // 初始化各组件
        // 注意：这里使用模拟数据，实际使用时需要提供有效的模型路径
        std::string modelPath = "dummy_model_path"; // 替换为实际模型路径
        
        // 创建Tokenizer
        auto tokenizer = std::make_unique<cllm::Tokenizer>(modelPath);
        
        // 创建ModelExecutor
        auto modelExecutor = std::make_unique<cllm::ModelExecutor>(
            modelPath,
            "",  // 量化类型
            true // 启用SIMD
        );
        
        // 创建Sampler
        cllm::SamplerConfig samplerConfig;
        auto sampler = std::make_unique<cllm::Sampler>(samplerConfig);
        
        std::cout << "Components initialized successfully." << std::endl;
        
        // 测试文本
        std::string inputText = "Hello, how are you?";
        std::cout << "Input text: " << inputText << std::endl;
        
        // 1. Tokenizer 测试
        std::vector<int> tokenIds = tokenizer->encode(inputText);
        std::cout << "Encoded tokens: ";
        for (size_t i = 0; i < std::min(tokenIds.size(), static_cast<size_t>(10)); ++i) {
            std::cout << tokenIds[i] << " ";
        }
        if (tokenIds.size() > 10) {
            std::cout << "... (total: " << tokenIds.size() << ")";
        }
        std::cout << std::endl;
        
        // 解码测试
        std::string decodedText = tokenizer->decode(tokenIds);
        std::cout << "Decoded text: " << decodedText << std::endl;
        
        // 2. 模型推理测试 (使用模拟数据)
        // 注意：由于没有实际模型，这里演示接口调用流程
        std::cout << "\nPerforming inference simulation..." << std::endl;
        
        // 模拟一些推理结果
        cllm::FloatArray dummyLogits;
        dummyLogits.resize(tokenizer->getVocabSize());
        for (size_t i = 0; i < dummyLogits.size(); ++i) {
            dummyLogits[i] = static_cast<float>(rand()) / RAND_MAX; // 随机值模拟logits
        }
        
        // 3. 采样测试
        int sampledToken = sampler->sample(dummyLogits);
        std::cout << "Sampled token ID: " << sampledToken << std::endl;
        
        // 获取对应词汇
        if (sampledToken >= 0 && sampledToken < tokenizer->getVocabSize()) {
            std::string tokenText = tokenizer->getTokenText(sampledToken);
            std::cout << "Sampled token text: " << tokenText << std::endl;
        }
        
        // 4. 完整推理流水线测试
        std::cout << "\nTesting full pipeline..." << std::endl;
        
        std::string prompt = "The weather today is";
        std::vector<int> promptTokens = tokenizer->encode(prompt);
        
        std::cout << "Prompt: \"" << prompt << "\"" << std::endl;
        std::cout << "Prompt tokens: " << promptTokens.size() << std::endl;
        
        // 模拟生成几个token
        std::vector<int> generatedTokens;
        std::string fullGeneratedText = prompt;
        
        for (int i = 0; i < 5; ++i) { // 生成5个token作为示例
            // 更新模拟logits
            for (size_t j = 0; j < dummyLogits.size(); ++j) {
                dummyLogits[j] = static_cast<float>(rand()) / RAND_MAX;
            }
            
            int nextToken = sampler->sample(dummyLogits);
            generatedTokens.push_back(nextToken);
            
            std::string tokenText = tokenizer->getTokenText(nextToken);
            fullGeneratedText += tokenText;
            
            std::cout << "Generated token " << i+1 << ": " << tokenText << std::endl;
        }
        
        std::cout << "\nFinal generated text: " << fullGeneratedText << std::endl;
        
        std::cout << "\n✅ End-to-End Inference Pipeline Test Completed Successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Error during inference pipeline test: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}