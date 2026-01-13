/**
 * @file libtorch_api_test.cpp
 * @brief LibTorch 推理引擎 API 集成测试
 * 
 * 模拟完整的 cLLM 生成流程：
 * Tokenizer → InferenceEngine (LibTorch) → Sampler → TokenizerDecode
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/model/config.h"
#include "cllm/tokenizer/manager.h"
#include "cllm/sampler.h"
#include "cllm/memory/float_array.h"
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <thread>

using namespace cllm;
using namespace cllm::inference;

// 加载 Qwen3-0.6B 模型配置
ModelConfig loadQwen3Config() {
    ModelConfig config;
    config.vocabSize = 151936;
    config.hiddenSize = 1024;
    config.numLayers = 28;
    config.numAttentionHeads = 16;
    config.numKeyValueHeads = 8;
    config.intermediateSize = 3072;
    config.maxSequenceLength = 40960;
    config.modelType = "qwen";
    return config;
}

/**
 * @class TextGenerator
 * @brief 文本生成器，集成所有组件
 */
class TextGenerator {
public:
    TextGenerator(const std::string& modelPath, 
                 const std::string& tokenizerPath,
                 const ModelConfig& config)
        : modelPath_(modelPath)
        , tokenizerPath_(tokenizerPath)
        , config_(config)
        , initialized_(false) {}
    
    bool initialize() {
        try {
            std::cout << "\n========================================" << std::endl;
            std::cout << "  LibTorch API 集成测试初始化" << std::endl;
            std::cout << "========================================\n" << std::endl;
            
            // 1. 初始化 Tokenizer
            std::cout << "[1/3] 初始化 Tokenizer..." << std::endl;
            tokenizer_ = std::make_unique<TokenizerManager>(tokenizerPath_, nullptr);
            if (tokenizer_->getTokenizer() == nullptr) {
                std::cerr << "  ✗ Tokenizer 初始化失败！" << std::endl;
                return false;
            }
            std::cout << "  ✓ Tokenizer 初始化成功" << std::endl;
            
            // 2. 初始化 InferenceEngine (LibTorch 后端)
            std::cout << "[2/3] 初始化 InferenceEngine (LibTorch)..." << std::endl;
            engine_ = std::make_unique<InferenceEngine>(config_, modelPath_, true);
            if (!engine_->initialize()) {
                std::cerr << "  ✗ InferenceEngine 初始化失败！" << std::endl;
                return false;
            }
            std::cout << "  ✓ InferenceEngine 初始化成功" << std::endl;
            
            // 3. 初始化 Sampler
            std::cout << "[3/3] 初始化 Sampler..." << std::endl;
            SamplerConfig samplerConfig;
            samplerConfig.setTemperature(0.8f);
            samplerConfig.setTopK(50);
            samplerConfig.setTopP(0.9f);
            sampler_ = std::make_unique<Sampler>(samplerConfig);
            std::cout << "  ✓ Sampler 初始化成功 (temperature=0.8, top_k=50, top_p=0.9)" << std::endl;
            
            initialized_ = true;
            std::cout << "\n✓ 所有组件初始化完成" << std::endl;
            std::cout << "========================================\n" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "✗ 初始化失败: " << e.what() << std::endl;
            return false;
        }
    }
    
    /**
     * @brief 生成文本（自回归生成）
     * @param prompt 输入提示词
     * @param maxNewTokens 最大生成 token 数
     * @return 生成的完整文本
     */
    std::string generate(const std::string& prompt, int maxNewTokens = 20) {
        if (!initialized_) {
            throw std::runtime_error("TextGenerator not initialized");
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "  文本生成测试" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "\n📝 输入提示词: \"" << prompt << "\"" << std::endl;
        std::cout << "🎯 最大生成 tokens: " << maxNewTokens << "\n" << std::endl;
        
        // 1. Tokenize 输入
        auto startEncode = std::chrono::high_resolution_clock::now();
        std::vector<int> tokenIds = tokenizer_->encode(prompt);
        auto endEncode = std::chrono::high_resolution_clock::now();
        auto encodeMs = std::chrono::duration_cast<std::chrono::microseconds>(endEncode - startEncode).count() / 1000.0;
        
        std::cout << "[Step 1] Tokenize 输入" << std::endl;
        std::cout << "  - Token IDs: [";
        for (size_t i = 0; i < tokenIds.size() && i < 10; ++i) {
            std::cout << tokenIds[i];
            if (i < tokenIds.size() - 1 && i < 9) std::cout << ", ";
        }
        if (tokenIds.size() > 10) std::cout << ", ...";
        std::cout << "]" << std::endl;
        std::cout << "  - Token 数量: " << tokenIds.size() << std::endl;
        std::cout << "  - 耗时: " << encodeMs << " ms\n" << std::endl;
        
        // 2. 自回归生成
        std::cout << "[Step 2] 自回归生成" << std::endl;
        std::vector<int> generatedIds = tokenIds;
        std::vector<double> inferenceTime;
        std::vector<double> samplingTime;
        
        for (int step = 0; step < maxNewTokens; ++step) {
            // 准备输入（取最后 8 tokens 或填充到 8）
            std::vector<int> currentInput = prepareInput(generatedIds);
            
            // 推理
            auto startInfer = std::chrono::high_resolution_clock::now();
            Tensor logits = engine_->forward(currentInput);
            auto endInfer = std::chrono::high_resolution_clock::now();
            auto inferMs = std::chrono::duration_cast<std::chrono::milliseconds>(endInfer - startInfer).count();
            inferenceTime.push_back(inferMs);
            
            // 获取最后一个位置的 logits
            size_t lastPos = currentInput.size() - 1;
            FloatArray lastLogits(config_.vocabSize);
            for (size_t i = 0; i < config_.vocabSize; ++i) {
                lastLogits.data()[i] = logits.data()[lastPos * config_.vocabSize + i];
            }
            
            // 采样
            auto startSample = std::chrono::high_resolution_clock::now();
            int nextToken = sampler_->sample(lastLogits, 0.8f, 50, 0.9f);
            auto endSample = std::chrono::high_resolution_clock::now();
            auto sampleUs = std::chrono::duration_cast<std::chrono::microseconds>(endSample - startSample).count() / 1000.0;
            samplingTime.push_back(sampleUs);
            
            generatedIds.push_back(nextToken);
            
            // 解码当前 token
            std::string tokenText = tokenizer_->decode({nextToken});
            
            // 实时输出
            std::cout << "  Step " << std::setw(2) << (step + 1) << ": "
                      << "token_" << std::setw(6) << nextToken 
                      << " → \"" << tokenText << "\" "
                      << "(推理: " << inferMs << " ms, 采样: " << sampleUs << " ms)"
                      << std::endl;
            
            // 检查是否生成了结束 token
            if (nextToken == 151643 || nextToken == 2) {  // <|endoftext|> or EOS
                std::cout << "\n  ⚠️  检测到结束 token，提前停止生成" << std::endl;
                break;
            }
        }
        
        // 3. 解码完整输出
        auto startDecode = std::chrono::high_resolution_clock::now();
        std::string generatedText = tokenizer_->decode(generatedIds);
        auto endDecode = std::chrono::high_resolution_clock::now();
        auto decodeMs = std::chrono::duration_cast<std::chrono::microseconds>(endDecode - startDecode).count() / 1000.0;
        
        std::cout << "\n[Step 3] 解码完整输出" << std::endl;
        std::cout << "  - 生成 token 数: " << (generatedIds.size() - tokenIds.size()) << std::endl;
        std::cout << "  - 总 token 数: " << generatedIds.size() << std::endl;
        std::cout << "  - 解码耗时: " << decodeMs << " ms" << std::endl;
        
        // 4. 性能统计
        std::cout << "\n[Step 4] 性能统计" << std::endl;
        double totalInferMs = 0;
        double totalSampleMs = 0;
        for (double t : inferenceTime) totalInferMs += t;
        for (double t : samplingTime) totalSampleMs += t;
        
        double avgInferMs = inferenceTime.empty() ? 0 : totalInferMs / inferenceTime.size();
        double avgSampleMs = samplingTime.empty() ? 0 : totalSampleMs / samplingTime.size();
        
        std::cout << "  - 总推理时间: " << totalInferMs << " ms" << std::endl;
        std::cout << "  - 平均推理时间: " << avgInferMs << " ms/token" << std::endl;
        std::cout << "  - 总采样时间: " << totalSampleMs << " ms" << std::endl;
        std::cout << "  - 平均采样时间: " << avgSampleMs << " ms/token" << std::endl;
        std::cout << "  - 端到端延迟: " << (encodeMs + totalInferMs + totalSampleMs + decodeMs) << " ms" << std::endl;
        
        if (!inferenceTime.empty()) {
            double tokensPerSec = 1000.0 / avgInferMs;
            std::cout << "  - 吞吐量: " << tokensPerSec << " tokens/sec" << std::endl;
        }
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "📄 生成结果:\n" << std::endl;
        std::cout << generatedText << std::endl;
        std::cout << "\n========================================\n" << std::endl;
        
        return generatedText;
    }
    
private:
    std::vector<int> prepareInput(const std::vector<int>& tokenIds) {
        std::vector<int> input;
        
        if (tokenIds.size() >= 8) {
            // 取最后 8 个 tokens
            input = std::vector<int>(tokenIds.end() - 8, tokenIds.end());
        } else {
            // 填充到 8 个 tokens（前面填充 pad token）
            size_t padSize = 8 - tokenIds.size();
            input.resize(8);
            for (size_t i = 0; i < padSize; ++i) {
                input[i] = 151643;  // <|endoftext|>
            }
            for (size_t i = 0; i < tokenIds.size(); ++i) {
                input[padSize + i] = tokenIds[i];
            }
        }
        
        return input;
    }
    
    std::string modelPath_;
    std::string tokenizerPath_;
    ModelConfig config_;
    bool initialized_;
    
    std::unique_ptr<TokenizerManager> tokenizer_;
    std::unique_ptr<InferenceEngine> engine_;
    std::unique_ptr<Sampler> sampler_;
};

int main(int argc, char* argv[]) {
    std::cout << "╔════════════════════════════════════════╗" << std::endl;
    std::cout << "║   LibTorch 推理引擎 API 集成测试      ║" << std::endl;
    std::cout << "╚════════════════════════════════════════╝\n" << std::endl;
    
    // 配置路径
    const std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/qwen3_0.6b_torchscript_fp32.pt";
    const std::string tokenizerPath = "/Users/dannypan/PycharmProjects/xllm/model/Qwen/Qwen3-0.6B";
    
    // 测试用例
    std::vector<std::string> testPrompts = {
        "Hello",
        "What is AI?",
        "The weather today is"
    };
    
    // 从命令行参数获取提示词
    if (argc > 1) {
        testPrompts = {argv[1]};
    }
    
    try {
        // 初始化生成器
        ModelConfig config = loadQwen3Config();
        TextGenerator generator(modelPath, tokenizerPath, config);
        
        if (!generator.initialize()) {
            std::cerr << "❌ 生成器初始化失败！" << std::endl;
            return 1;
        }
        
        // 运行测试
        for (const auto& prompt : testPrompts) {
            try {
                std::string result = generator.generate(prompt, 10);
                
                // 等待一下再进行下一个测试
                if (&prompt != &testPrompts.back()) {
                    std::cout << "\n等待 2 秒...\n" << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(2));
                }
            } catch (const std::exception& e) {
                std::cerr << "❌ 生成失败: " << e.what() << std::endl;
            }
        }
        
        std::cout << "\n✓ 所有测试完成！" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 错误: " << e.what() << std::endl;
        return 1;
    }
}
