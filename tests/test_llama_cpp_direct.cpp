/**
 * @file test_llama_cpp_direct.cpp
 * @brief 直接测试 LlamaCppBackend（绕过 HTTP 层）
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/sampler.h"
#include "cllm/common/logger.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>

using namespace cllm;
using namespace cllm::inference;

int main(int argc, char** argv) {
    std::cout << "=== 直接测试 LlamaCppBackend ===" << std::endl;

    // 设置日志级别
    {
        auto level = spdlog::level::warn;
        if (const char* envLevel = std::getenv("CLLM_LOG_LEVEL")) {
            std::string s(envLevel);
            std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (s == "debug") level = spdlog::level::debug;
            else if (s == "info") level = spdlog::level::info;
            else if (s == "warn" || s == "warning") level = spdlog::level::warn;
            else if (s == "error") level = spdlog::level::err;
        }
        cllm::Logger::instance().setLevel(level);
    }
    
    // 查找 GGUF 模型路径
    std::string modelPath;
    std::vector<std::string> possiblePaths = {
        "model/Qwen/qwen3-0.6b-q4_k_m.gguf",
        "../model/Qwen/qwen3-0.6b-q4_k_m.gguf",
        "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf",
    };
    
    const char* envPath = std::getenv("CLLM_MODEL_PATH");
    if (envPath && std::filesystem::exists(envPath)) {
        possiblePaths.insert(possiblePaths.begin(), envPath);
    }
    
    for (const auto& path : possiblePaths) {
        if (std::filesystem::exists(path)) {
            modelPath = std::filesystem::absolute(path).string();
            break;
        }
    }
    
    if (modelPath.empty()) {
        std::cerr << "错误: 找不到 GGUF 模型文件。" << std::endl;
        return 1;
    }
    
    CLLM_INFO("使用模型路径: %s", modelPath.c_str());
    
    try {
        // 1. 加载 GGUFTokenizer
        CLLM_INFO("步骤 1: 加载 GGUFTokenizer...");
        auto tokenizer = std::make_unique<GGUFTokenizer>();
        if (!tokenizer->load(modelPath)) {
            std::cerr << "错误: 加载 GGUFTokenizer 失败" << std::endl;
            return 1;
        }
        CLLM_INFO("✅ GGUFTokenizer 加载成功");
        
        int vocabSize = tokenizer->getVocabSize();
        CLLM_INFO("Tokenizer vocab size: %d", vocabSize);
        
        // 2. 创建模型配置
        CLLM_INFO("步骤 2: 创建模型配置...");
        ModelConfig config;
        
        config.vocabSize = static_cast<size_t>(vocabSize);
        config.hiddenSize = 1024;
        config.numLayers = 24;
        config.numAttentionHeads = 8;
        config.numKeyValueHeads = 1;
        config.maxSequenceLength = 4096;
        config.intermediateSize = 2816;
        config.ropeTheta = 1000000.0f;
        config.rmsNormEps = 1e-6f;
        
        config.llamaBatchSize = 512;
        config.llamaNumThreads = 0;
        config.llamaGpuLayers = 0;
        config.llamaUseMmap = true;
        config.llamaUseMlock = false;
        
        CLLM_INFO("模型配置创建完成");
        
        // 3. 创建 InferenceEngine（指定使用 llama_cpp 后端）
        CLLM_INFO("步骤 3: 创建 InferenceEngine (llama.cpp 后端)...");
        InferenceEngine engine(config, modelPath, "llama_cpp");
        
        if (!engine.initialize()) {
            std::cerr << "错误: InferenceEngine 初始化失败" << std::endl;
            return 1;
        }
        CLLM_INFO("✅ InferenceEngine 初始化成功");
        CLLM_INFO("后端类型: %s", engine.getBackendType().c_str());
        
        config = engine.getConfig();
        CLLM_INFO("模型配置已更新:");
        CLLM_INFO("  - vocab_size: %zu", config.vocabSize);
        CLLM_INFO("  - hidden_size: %zu", config.hiddenSize);
        CLLM_INFO("  - num_layers: %zu", config.numLayers);
        
        // 4. 准备测试用例
        std::string inputText = "1+1=";
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "测试用例: '" << inputText << "'" << std::endl;
        std::cout << "========================================" << std::endl;
        
        // 5. 编码输入文本
        std::vector<int> inputIds = tokenizer->encode(inputText, false);
        
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "Prompt: " << inputText << std::endl;
        std::cout << "Prompt tokens: " << inputIds.size() << std::endl;
        
        // 6. 创建采样器
        auto sampler = std::make_unique<Sampler>();
        
        // 7. 测试生成
        size_t maxNewTokens = 16;
        float temperature = 0.7f;
        
        std::cout << "开始生成 " << maxNewTokens << " 个新 tokens (temperature=" << temperature << ")..." << std::endl;
        
        std::vector<int> generatedIds;
        std::vector<int> currentInput = inputIds;
        
        for (size_t i = 0; i < maxNewTokens; ++i) {
            std::vector<int> inputForForward;
            if (i == 0) {
                inputForForward = currentInput;
            } else {
                inputForForward = {currentInput.back()};
            }
            
            Tensor logitsTensor = engine.forward(inputForForward);
            
            size_t seqLen = inputForForward.size();
            size_t vocabSize = config.vocabSize;
            
            const float* logitsPtr = logitsTensor.data();
            const float* lastLogitsPtr = logitsPtr + (seqLen - 1) * vocabSize;
            
            FloatArray logits(vocabSize);
            std::memcpy(logits.data(), lastLogitsPtr, vocabSize * sizeof(float));
            
            int nextToken = sampler->sample(logits, temperature);
            generatedIds.push_back(nextToken);
            currentInput.push_back(nextToken);
            
            if (nextToken == tokenizer->getEosId()) {
                break;
            }
        }
        
        CLLM_INFO("生成了 %zu 个 tokens", generatedIds.size());
        
        // 8. 解码生成的文本
        std::string outputText = tokenizer->decode(generatedIds, true);
        
        // 9. 显示结果
        std::cout << "\n========================================" << std::endl;
        std::cout << "测试结果" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "输入: " << inputText << std::endl;
        std::cout << "输出: " << outputText << std::endl;
        std::cout << "完整: " << inputText << outputText << std::endl;
        std::cout << "生成的 token 数量: " << generatedIds.size() << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
