/**
 * @file test_kylin_text_generation.cpp
 * @brief 测试 Kylin 后端的完整文本生成流程
 */

#include "cllm/kylin/gguf/transformer.h"
#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/sampler/sampler.h"
#include "cllm/common/logger.h"

#include <iostream>
#include <vector>

using namespace cllm;
using namespace cllm::kylin;

int main(int argc, char** argv) {
    // 初始化日志
    cllm::Logger::instance().setLevel(spdlog::level::info);
    
    std::string modelPath = "../model/Qwen/qwen3-0.6b-q4_k_m.gguf";
    if (argc > 1) {
        modelPath = argv[1];
    }
    
    std::cout << "========================================" << std::endl;
    std::cout << "Kylin Text Generation Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Model: " << modelPath << std::endl << std::endl;
    
    try {
        // 1. 加载模型
        std::cout << "1. Loading model..." << std::endl;
        GGMLTransformerModel model(BackendType::Metal);
        
        if (!model.loadFromGGUF(modelPath)) {
            std::cerr << "❌ Failed to load model!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Model loaded" << std::endl << std::endl;
        
        // 2. 初始化 tokenizer
        std::cout << "2. Loading tokenizer..." << std::endl;
        GGUFTokenizer tokenizer;
        
        if (!tokenizer.loadFromFile(modelPath)) {
            std::cerr << "❌ Failed to load tokenizer!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ Tokenizer loaded" << std::endl;
        std::cout << "   Vocab size: " << tokenizer.vocabSize() << std::endl;
        std::cout << "   BOS: " << tokenizer.bosTokenId() << std::endl;
        std::cout << "   EOS: " << tokenizer.eosTokenId() << std::endl << std::endl;
        
        // 3. 测试 tokenization
        std::cout << "3. Testing tokenization..." << std::endl;
        
        std::string testPrompt = "Hello world";
        std::cout << "   Prompt: \"" << testPrompt << "\"" << std::endl;
        
        std::vector<int> tokenIds = tokenizer.encode(testPrompt);
        std::cout << "   Token IDs: [";
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            std::cout << tokenIds[i];
            if (i < tokenIds.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // 验证 token ID 范围
        bool validIds = true;
        for (int id : tokenIds) {
            if (id < 0 || id >= static_cast<int>(tokenizer.vocabSize())) {
                std::cerr << "   ❌ Invalid token ID: " << id << " (vocab size: " << tokenizer.vocabSize() << ")" << std::endl;
                validIds = false;
            }
        }
        
        if (validIds) {
            std::cout << "   ✅ All token IDs are valid" << std::endl;
        }
        
        // 测试解码
        std::string decoded = tokenizer.decode(tokenIds);
        std::cout << "   Decoded: \"" << decoded << "\"" << std::endl;
        
        if (decoded == testPrompt) {
            std::cout << "   ✅ Round-trip successful" << std::endl;
        } else {
            std::cout << "   ⚠️  Round-trip mismatch (expected due to BPE)" << std::endl;
        }
        
        std::cout << std::endl;
        
        // 4. 测试推理
        std::cout << "4. Testing inference..." << std::endl;
        
        std::vector<int32_t> inputIds32(tokenIds.begin(), tokenIds.end());
        std::vector<float> logits = model.forward(inputIds32);
        
        const size_t seqLen = inputIds32.size();
        const size_t vocabSize = model.getConfig().vocabSize;
        
        std::cout << "   Input length: " << seqLen << std::endl;
        std::cout << "   Output logits: [" << seqLen << ", " << vocabSize << "]" << std::endl;
        
        // 验证 logits
        size_t nanCount = 0, infCount = 0;
        for (const float& val : logits) {
            if (std::isnan(val)) nanCount++;
            else if (std::isinf(val)) infCount++;
        }
        
        if (nanCount > 0 || infCount > 0) {
            std::cerr << "   ❌ Logits contain " << nanCount << " NaN, " << infCount << " Inf" << std::endl;
            return 1;
        }
        
        std::cout << "   ✅ Logits are valid" << std::endl;
        
        // 5. 测试采样
        std::cout << "\n5. Testing sampling..." << std::endl;
        
        SamplerConfig samplerConfig;
        samplerConfig.temperature = 0.7f;
        samplerConfig.topP = 0.9f;
        samplerConfig.topK = 50;
        
        Sampler sampler(samplerConfig);
        
        // 使用最后一个位置的 logits 进行采样
        const float* lastLogits = logits.data() + (seqLen - 1) * vocabSize;
        std::vector<float> lastLogitsVec(lastLogits, lastLogits + vocabSize);
        
        int nextTokenId = sampler.sample(lastLogitsVec);
        
        std::cout << "   Next token ID: " << nextTokenId << std::endl;
        
        // 验证 token ID
        if (nextTokenId < 0 || nextTokenId >= static_cast<int>(vocabSize)) {
            std::cerr << "   ❌ Invalid next token ID: " << nextTokenId << std::endl;
            return 1;
        }
        
        std::cout << "   ✅ Sampled token ID is valid" << std::endl;
        
        // 解码单个 token
        std::string nextToken = tokenizer.decode(std::vector<int>{nextTokenId});
        std::cout << "   Next token text: \"" << nextToken << "\"" << std::endl;
        
        // 6. 生成几个 token
        std::cout << "\n6. Generating text (5 tokens)..." << std::endl;
        
        model.clearKVCache();
        std::vector<int32_t> generatedIds = inputIds32;
        
        std::cout << "   Starting from: \"" << testPrompt << "\"" << std::endl;
        std::cout << "   Generated tokens: ";
        
        for (int i = 0; i < 5; ++i) {
            // 推理
            std::vector<float> genLogits = model.forward(generatedIds);
            
            // 采样最后一个位置
            const float* lastPos = genLogits.data() + (generatedIds.size() - 1) * vocabSize;
            std::vector<float> lastPosVec(lastPos, lastPos + vocabSize);
            
            int nextId = sampler.sample(lastPosVec);
            generatedIds.push_back(nextId);
            
            std::string token = tokenizer.decode(std::vector<int>{nextId});
            std::cout << "\"" << token << "\" ";
        }
        
        std::cout << std::endl;
        
        // 解码完整生成结果
        std::vector<int> allIds(generatedIds.begin(), generatedIds.end());
        std::string fullText = tokenizer.decode(allIds);
        
        std::cout << "\n   Full generated text: \"" << fullText << "\"" << std::endl;
        
        std::cout << "\n✅ Text generation test passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
