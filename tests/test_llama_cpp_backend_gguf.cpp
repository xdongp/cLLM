/**
 * @file test_llama_cpp_backend_gguf.cpp
 * @brief 测试 llama.cpp 后端（使用 GGUF 格式模型）
 * 
 * 基于 test_hello_inference.cpp，专门用于测试 llama_cpp_backend
 */

#include "cllm/inference/inference_engine.h"
#ifdef CLLM_USE_LLAMA_CPP
#include "cllm/inference/llama_cpp_backend.h"
#endif
#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/tokenizer/i_tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/sampler.h"
#include "cllm/common/memory_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include <cstdlib>
#include <algorithm>
#include <cstring>

using namespace cllm;
using namespace cllm::inference;

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    std::cout << "=== 开始测试：llama.cpp 后端 (GGUF 格式) ===" << std::endl;

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
        "model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf",
        "../model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf",
        "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/Qwen3-0.6B/qwen3-0.6b-q4_k_m.gguf",
    };
    
    const char* envPath = std::getenv("CLLM_MODEL_PATH");
    if (envPath && fs::exists(envPath)) {
        possiblePaths.insert(possiblePaths.begin(), envPath);
    }
    
    for (const auto& path : possiblePaths) {
        if (fs::exists(path)) {
            modelPath = fs::absolute(path).string();
            break;
        }
    }
    
    if (modelPath.empty()) {
        std::cerr << "错误: 找不到 GGUF 模型文件。请设置 CLLM_MODEL_PATH 环境变量或确保模型文件存在。" << std::endl;
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
        
        // 基本配置（会在加载模型时自动更新）
        config.vocabSize = static_cast<size_t>(vocabSize);
        config.hiddenSize = 1024;  // 默认值，会被模型实际值覆盖
        config.numLayers = 24;      // 默认值，会被模型实际值覆盖
        config.numAttentionHeads = 8;  // 默认值
        config.numKeyValueHeads = 1;   // 默认值
        config.maxSequenceLength = 4096;
        config.intermediateSize = 2816;
        config.ropeTheta = 1000000.0f;
        config.rmsNormEps = 1e-6f;
        
        // llama.cpp 后端参数
        config.llamaBatchSize = 512;
        config.llamaNumThreads = 0;  // 0 表示使用默认值
        config.llamaGpuLayers = 0;   // 0 表示仅使用 CPU
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
        
        // 同步配置（从模型加载后更新）
        config = engine.getConfig();
        CLLM_INFO("模型配置已更新:");
        CLLM_INFO("  - vocab_size: %zu", config.vocabSize);
        CLLM_INFO("  - hidden_size: %zu", config.hiddenSize);
        CLLM_INFO("  - num_layers: %zu", config.numLayers);
        CLLM_INFO("  - max_sequence_length: %zu", config.maxSequenceLength);
        
        // 验证 vocab_size 一致性
        if (config.vocabSize != static_cast<size_t>(vocabSize)) {
            bool allowVocabMismatch = false;
            if (const char* envAllow = std::getenv("CLLM_ALLOW_VOCAB_MISMATCH")) {
                allowVocabMismatch = (std::string(envAllow) == "1");
            }
            if (!allowVocabMismatch) {
                std::cerr << "错误: tokenizer vocab size 与模型 vocab size 不一致。"
                          << " tokenizer=" << vocabSize
                          << ", model=" << config.vocabSize
                          << "。请确认 tokenizer 与模型一致，或设置 CLLM_ALLOW_VOCAB_MISMATCH=1 继续。"
                          << std::endl;
                return 1;
            } else {
                CLLM_WARN("⚠️  vocab size 不一致，但允许继续 (tokenizer=%d, model=%zu)",
                         vocabSize, config.vocabSize);
            }
        }
        
        // 4. 准备测试用例
        std::vector<std::string> testCases = {
            "1+1=",
        };
        
        // 从命令行参数获取输入文本，如果没有则使用测试用例
        std::vector<std::string> testInputs;
        if (argc > 1) {
            for (int i = 1; i < argc; ++i) {
                testInputs.push_back(argv[i]);
            }
        } else {
            testInputs = testCases;
        }
        
        std::cout << "将测试 " << testInputs.size() << " 个用例" << std::endl;
        
        // 5. 创建采样器
        auto sampler = std::make_unique<Sampler>();
        
        // 6. 测试每个输入
        for (size_t testIdx = 0; testIdx < testInputs.size(); ++testIdx) {
            const std::string& inputText = testInputs[testIdx];
            
            CLLM_INFO("\n========================================");
            CLLM_INFO("测试用例 %zu/%zu: '%s'", testIdx + 1, testInputs.size(), inputText.c_str());
            CLLM_INFO("========================================");
            
            // 6.1 编码输入文本
            std::vector<int> inputIds = tokenizer->encode(inputText, false);
            
            std::cout << "\n----------------------------------------" << std::endl;
            std::cout << "Prompt: " << inputText << std::endl;
            std::cout << "Prompt tokens: " << inputIds.size() << std::endl;
            
            // 6.2 执行推理并生成 tokens
            size_t maxNewTokens = 16;
            if (const char* envMaxNew = std::getenv("CLLM_MAX_NEW_TOKENS")) {
                maxNewTokens = static_cast<size_t>(std::strtoul(envMaxNew, nullptr, 10));
            }
            
            float temperature = 0.7f;
            if (const char* envTemp = std::getenv("CLLM_TEMPERATURE")) {
                temperature = std::strtof(envTemp, nullptr);
            }
            
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
                
                kylin::Tensor logitsTensor = engine.forward(inputForForward);
                
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
            
            // 验证生成的 token IDs 是否在有效范围内
            int tokenizerVocabSize = tokenizer->getVocabSize();
            std::vector<int> validGeneratedIds;
            
            for (int id : generatedIds) {
                if (id >= 0 && id < tokenizerVocabSize) {
                    validGeneratedIds.push_back(id);
                } else {
                    CLLM_WARN("警告: 生成的 token ID %d 超出 tokenizer vocab 范围 [0, %d)", id, tokenizerVocabSize);
                }
            }
            
            // 6.3 解码生成的文本
            std::string outputText = tokenizer->decode(validGeneratedIds, true);
            
            // 7. 显示结果
            std::cout << "\n========================================" << std::endl;
            std::cout << "测试用例 " << (testIdx + 1) << "/" << testInputs.size() << std::endl;
            std::cout << "========================================" << std::endl;
            std::cout << "输入: " << inputText << std::endl;
            std::cout << "输出: " << outputText << std::endl;
            std::cout << "完整: " << inputText << outputText << std::endl;
            std::cout << "生成的 token 数量: " << generatedIds.size() << std::endl;
            std::cout << "========================================\n" << std::endl;
        }
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return 1;
    }
}
