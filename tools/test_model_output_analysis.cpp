/**
 * @file test_model_output_analysis.cpp
 * @brief Stage 5: 模型输出分析测试 - 直接推理测试
 *
 * 测试内容：
 * - 直接调用 Kylin backend 进行推理
 * - 检查 logits 输出分布
 * - 验证 token 生成过程
 * - 显示实际生成的文本
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <cllm/kylin/hf/transformer.h>
#include <cllm/tokenizer/hf_tokenizer.h>

using namespace cllm;
using namespace cllm::kylin;

void printUsage() {
    std::cout << "用法: test_model_output_analysis <model_path> [prompt]" << std::endl;
    std::cout << "  model_path: 模型目录路径" << std::endl;
    std::cout << "  prompt: 可选，默认使用 'hello'" << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printUsage();
        return 1;
    }

    std::string modelPath = argv[1];
    std::string prompt = (argc > 2) ? argv[2] : "hello";

    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║        Stage 5: 模型输出分析测试 - 直接推理测试                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    // Step 1: 加载模型配置
    std::cout << "【Step 1】加载模型配置..." << std::endl;

    std::ifstream configFile(modelPath + "/config.json");
    if (!configFile.is_open()) {
        std::cerr << "❌ 无法打开 config.json" << std::endl;
        return 1;
    }

    try {
        nlohmann::json configJson = nlohmann::json::parse(configFile);
        std::cout << "✅ 模型配置加载成功" << std::endl;
        std::cout << "   - Vocab Size: " << configJson.value("vocab_size", 0) << std::endl;
        std::cout << "   - Hidden Size: " << configJson.value("hidden_size", 0) << std::endl;
        std::cout << "   - Layers: " << configJson.value("num_hidden_layers", 0) << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "❌ 配置解析失败: " << e.what() << std::endl;
        return 1;
    }

    // Step 2: 加载 tokenizer
    std::cout << std::endl << "【Step 2】加载 Tokenizer..." << std::endl;
    HFTokenizer tokenizer(ModelType::QWEN);
    if (!tokenizer.load(modelPath)) {
        std::cerr << "❌ Tokenizer 加载失败" << std::endl;
        return 1;
    }

    std::cout << "✅ Tokenizer 加载成功" << std::endl;
    std::cout << "   - Vocab Size: " << tokenizer.getVocabSize() << std::endl;
    std::cout << "   - BOS ID: " << tokenizer.getBosId() << std::endl;
    std::cout << "   - EOS ID: " << tokenizer.getEosId() << std::endl;

    // Step 3: 加载 Transformer 模型
    std::cout << std::endl << "【Step 3】加载 Transformer 模型..." << std::endl;
    HFTransformerModel transformer(modelPath, DeviceType::CPU, QuantType::INT8);
    if (!transformer.loadWeights()) {
        std::cerr << "❌ Transformer 模型加载失败" << std::endl;
        return 1;
    }
    std::cout << "✅ Transformer 模型加载成功" << std::endl;

    // Step 4: 编码提示词
    std::cout << std::endl << "【Step 4】编码提示词..." << std::endl;
    std::vector<int> inputIds = tokenizer.encode(prompt, false);
    std::cout << "✅ 提示词编码成功" << std::endl;
    std::cout << "   - 提示词: \"" << prompt << "\"" << std::endl;
    std::cout << "   - Token 数量: " << inputIds.size() << std::endl;
    std::cout << "   - Tokens: [";
    for (size_t i = 0; i < std::min(inputIds.size(), (size_t)10); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << inputIds[i];
    }
    if (inputIds.size() > 10) std::cout << " ...";
    std::cout << "]" << std::endl;

    // Step 5: 直接推理测试
    std::cout << std::endl << "【Step 5】执行直接推理测试..." << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;

    auto startTime = std::chrono::high_resolution_clock::now();

    // 使用第一个 token 进行前向传播
    int firstToken = inputIds[0];
    std::vector<float> logits = transformer.forward({firstToken});

    auto endTime = std::chrono::high_resolution_clock::now();
    float inferenceTime = std::chrono::duration<float>(endTime - startTime).count();

    std::cout << "✅ 推理完成 (耗时: " << inferenceTime * 1000 << " ms)" << std::endl;

    // 分析 logits
    std::cout << std::endl << "【Logits 分析】" << std::endl;

    if (logits.empty()) {
        std::cerr << "❌ Logits 为空！" << std::endl;
        return 1;
    }

    float minLogit = logits[0], maxLogit = logits[0];
    double sum = 0;
    size_t nanCount = 0, infCount = 0;

    for (size_t i = 0; i < logits.size(); ++i) {
        float v = logits[i];
        if (std::isnan(v)) { nanCount++; continue; }
        if (std::isinf(v)) { infCount++; continue; }
        if (v < minLogit) minLogit = v;
        if (v > maxLogit) maxLogit = v;
        sum += v;
    }

    std::cout << "   - Vocab Size: " << logits.size() << std::endl;
    std::cout << "   - Min Logit: " << minLogit << std::endl;
    std::cout << "   - Max Logit: " << maxLogit << std::endl;
    std::cout << "   - Mean Logit: " << (sum / logits.size()) << std::endl;
    std::cout << "   - NaN Count: " << nanCount << std::endl;
    std::cout << "   - Inf Count: " << infCount << std::endl;

    // 检查特殊 tokens 的 logit
    std::cout << std::endl << "【特殊 Token Logits】" << std::endl;
    std::cout << "   - BOS (" << tokenizer.getBosId() << "): " << logits[tokenizer.getBosId()] << std::endl;
    std::cout << "   - EOS (" << tokenizer.getEosId() << "): " << logits[tokenizer.getEosId()] << std::endl;

    // 检查 token 151668
    if (151668 < (int)logits.size()) {
        std::cout << "   - Token 151668 (<|im_end|>): " << logits[151668] << std::endl;
        std::cout << "     Token 名称: " << tokenizer.idToToken(151668) << std::endl;
    }

    // 找到 Top 10 tokens
    std::cout << std::endl << "【Top 10 Tokens】" << std::endl;
    std::vector<std::pair<float, int>> topTokens;
    for (int i = 0; i < (int)std::min(logits.size(), (size_t)200000); ++i) {
        topTokens.push_back({logits[i], i});
    }
    std::partial_sort(topTokens.begin(), topTokens.begin() + 10, topTokens.end(),
                     [](const auto& a, const auto& b) { return a.first > b.first; });

    for (int i = 0; i < 10; ++i) {
        int tokenId = topTokens[i].second;
        float logit = topTokens[i].first;
        std::string tokenName = tokenizer.idToToken(tokenId);
        bool isSpecial = tokenizer.isSpecialToken(tokenId);
        std::cout << "   " << (i + 1) << ". [ID=" << tokenId << "] logit=" << logit
                  << " name=\"" << tokenName << "\" special=" << (isSpecial ? "YES" : "NO") << std::endl;
    }

    // Step 6: 贪婪解码测试
    std::cout << std::endl << "【Step 6】贪婪解码测试..." << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;

    int greedyToken = 0;
    float maxLogitVal = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > maxLogitVal) {
            maxLogitVal = logits[i];
            greedyToken = i;
        }
    }

    std::cout << "   - 贪婪选择的 Token: " << greedyToken << std::endl;
    std::cout << "   - Token 名称: " << tokenizer.idToToken(greedyToken) << std::endl;
    std::cout << "   - Logit 值: " << maxLogitVal << std::endl;

    // 尝试解码这个 token
    std::string decoded = tokenizer.decode({greedyToken}, true);
    std::cout << "   - 解码结果 (skipSpecial=true): \"" << decoded << "\"" << std::endl;

    std::string decodedWithSpecial = tokenizer.decode({greedyToken}, false);
    std::cout << "   - 解码结果 (skipSpecial=false): \"" << decodedWithSpecial << "\"" << std::endl;

    // Step 7: 完整生成测试
    std::cout << std::endl << "【Step 7】完整生成测试..." << std::endl;
    std::cout << "═══════════════════════════════════════════════════════════════════════════" << std::endl;

    std::vector<int> generatedTokens;
    int currentToken = inputIds.back();
    std::vector<float> currentLogits;

    for (int step = 0; step < 10; ++step) {
        currentLogits = transformer.forward({currentToken});

        // 贪婪选择
        int nextToken = 0;
        float maxVal = currentLogits[0];
        for (size_t i = 1; i < currentLogits.size(); ++i) {
            if (currentLogits[i] > maxVal) {
                maxVal = currentLogits[i];
                nextToken = i;
            }
        }

        generatedTokens.push_back(nextToken);
        std::cout << "   Step " << (step + 1) << ": Token " << nextToken
                  << " (\"" << tokenizer.idToToken(nextToken) << "\")"
                  << " logit=" << maxVal << std::endl;

        if (nextToken == tokenizer.getEosId()) {
            std::cout << "   → 遇到 EOS，停止生成" << std::endl;
            break;
        }

        currentToken = nextToken;
    }

    // 解码生成的 tokens
    std::cout << std::endl << "【生成结果】" << std::endl;
    std::cout << "   - 生成的 Tokens: [";
    for (size_t i = 0; i < generatedTokens.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << generatedTokens[i];
    }
    std::cout << "]" << std::endl;

    // 使用 skipSpecialTokens=true 解码
    std::string generatedText = tokenizer.decode(generatedTokens, true);
    std::cout << "   - 解码文本 (skipSpecial=true): \"" << generatedText << "\"" << std::endl;

    // 使用 skipSpecialTokens=false 解码
    std::string generatedTextWithSpecial = tokenizer.decode(generatedTokens, false);
    std::cout << "   - 解码文本 (skipSpecial=false): \"" << generatedTextWithSpecial << "\"" << std::endl;

    std::cout << std::endl;
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                         测试完成                                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝" << std::endl;

    return 0;
}
