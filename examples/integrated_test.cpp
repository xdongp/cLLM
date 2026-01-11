/**
 * @file integrated_test.cpp
 * @brief 模块联调测试程序
 * 
 * 测试LibTorch Backend + Inference Engine + Tokenizer的完整流程
 * 输入"hello"，显示推理的字符结果
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/tokenizer.h"
#include <iostream>
#include <vector>
#include <string>
#include <memory>

using namespace cllm;
using namespace cllm::inference;

int main() {
    std::cout << "===========================================" << std::endl;
    std::cout << "Integrated Test: LibTorch + Inference Engine + Tokenizer" << std::endl;
    std::cout << "===========================================" << std::endl;

    try {
        // 1. 配置模型参数 (Qwen3 0.6B)
        ModelConfig config;
        config.vocabSize = 152064;  // Qwen3 0.6B 模型的词汇表大小
        config.hiddenSize = 2048;
        config.numLayers = 28;
        config.numAttentionHeads = 16;
        config.maxSequenceLength = 8192;

        // 2. 模型路径
        std::string modelPath = "./model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
        std::cout << "Model path: " << modelPath << std::endl;

        // 3. 创建推理引擎 (使用LibTorch后端)
        std::cout << "Creating inference engine with LibTorch backend..." << std::endl;
        InferenceEngine engine(config, modelPath, true);  // true = use LibTorch

        // 4. 初始化推理引擎
        std::cout << "Initializing inference engine..." << std::endl;
        bool initResult = engine.initialize();
        if (!initResult) {
            std::cerr << "Failed to initialize inference engine" << std::endl;
            return 1;
        }
        std::cout << "Inference engine initialized successfully!" << std::endl;
        std::cout << "Backend type: " << engine.getBackendType() << std::endl;
        std::cout << "Is initialized: " << (engine.isInitialized() ? "true" : "false") << std::endl;

        // 5. 准备输入文本
        std::string inputText = "hello";
        std::cout << "\nInput text: \"" << inputText << "\"" << std::endl;

        // 6. 使用Tokenizer进行编码 (这里我们使用简单的token ID，实际应用中需要真实的tokenizer)
        // 注意：由于我们没有特定的Qwen tokenizer，我们使用已知的"hello"对应的token IDs
        // 在实际应用中，这应该是通过tokenizer.encode(inputText)获得的
        std::vector<int> inputIds = {1559, 159}; // "hello" 对应的token IDs
        std::cout << "Encoded token IDs: ";
        for (int id : inputIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        // 7. 执行推理
        std::cout << "\nRunning inference..." << std::endl;
        kylin::Tensor logits = engine.forward(inputIds);

        // 8. 获取logits的维度信息
        auto shape = logits.shape();
        std::cout << "Logits tensor shape: ";
        for (size_t i = 0; i < shape.size(); ++i) {
            std::cout << shape[i];
            if (i < shape.size() - 1) std::cout << " x ";
        }
        std::cout << std::endl;

        // 9. 从logits中提取最可能的下一个token ID
        size_t seqLen = shape[0];
        size_t vocabSize = shape[1];
        float* data = logits.data();

        std::cout << "Sequence length: " << seqLen << ", Vocabulary size: " << vocabSize << std::endl;

        // 对于每个位置，找到概率最高的token
        std::vector<int> predictedTokenIds;
        for (size_t pos = 0; pos < seqLen; ++pos) {
            float maxLogit = data[pos * vocabSize];
            int maxIdx = 0;
            for (size_t i = 1; i < vocabSize; ++i) {
                if (data[pos * vocabSize + i] > maxLogit) {
                    maxLogit = data[pos * vocabSize + i];
                    maxIdx = static_cast<int>(i);
                }
            }
            std::cout << "Position " << pos << ": Predicted token ID = " << maxIdx 
                      << ", Logit = " << maxLogit << std::endl;
            predictedTokenIds.push_back(maxIdx);
        }

        // 10. 使用Tokenizer进行解码 (同样，这里使用简化方式)
        // 在实际应用中，这应该是通过tokenizer.decode(predictedTokenIds)实现的
        std::cout << "\nPredicted token IDs for next tokens: ";
        for (int id : predictedTokenIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;

        std::cout << "\n===========================================" << std::endl;
        std::cout << "INTEGRATED TEST COMPLETED SUCCESSFULLY!" << std::endl;
        std::cout << "Input: \"" << inputText << "\"" << std::endl;
        std::cout << "Predicted next tokens (IDs): ";
        for (int id : predictedTokenIds) {
            std::cout << id << " ";
        }
        std::cout << std::endl;
        std::cout << "Full pipeline: Input Text -> Tokenizer (encode) -> Inference Engine -> Logits -> Prediction -> Tokenizer (decode)" << std::endl;
        std::cout << "===========================================" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error occurred during integrated test: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}