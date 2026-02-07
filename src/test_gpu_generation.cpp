#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <memory>
#include <iomanip>
#include <sstream>

#include "cllm/kylin/hf/hf_transformer_model.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/hf/config.h"
#include "cllm/tokenizer/tokenizer.h"

using namespace cllm::kylin;

int main() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║     cLLM Kylin + HF Model + GPU 生成测试                    ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
    std::cout << std::endl;

    std::string modelPath = "/Users/dannypan/PycharmProjects/cLLM/model/Qwen/Qwen3-0.6B";

    std::cout << "🔧 初始化模型 (Metal GPU)..." << std::endl;
    std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;

    try {
        auto model = std::make_unique<HFTransformerModel>(modelPath, DeviceType::Metal, QuantType::FP16);

        if (!model->isLoaded()) {
            std::cerr << "❌ 模型加载失败" << std::endl;
            return 1;
        }

        std::cout << "✅ 模型加载成功!" << std::endl;
        std::cout << "   • 词表大小: " << model->vocabSize() << std::endl;
        std::cout << "   • 隐藏层大小: " << model->hiddenSize() << std::endl;
        std::cout << "   • 层数: " << model->config().numLayers << std::endl;
        std::cout << "   • 设备: Metal GPU" << std::endl;
        std::cout << std::endl;

        // 加载 tokenizer
        cllm::Tokenizer tokenizer(modelPath);
        if (!tokenizer.load(modelPath)) {
            std::cerr << "❌ Tokenizer 加载失败" << std::endl;
            return 1;
        }
        std::cout << "✅ Tokenizer 加载成功" << std::endl;
        std::cout << "   • EOS Token ID: " << tokenizer.getEosId() << std::endl;
        std::cout << std::endl;

        // 测试用例
        struct TestCase {
            std::string name;
            std::string prompt;
            int maxTokens;
        };

        std::vector<TestCase> testCases = {
            {"基础问候", "hello", 50},
            {"数学计算", "1+1=", 30},
            {"知识问答", "介绍一下人工智能", 100}
        };

        for (const auto& test : testCases) {
            std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
            std::cout << "║ 测试: " << test.name << std::endl;
            std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
            std::cout << std::endl;
            std::cout << "📝 输入: \"" << test.prompt << "\"" << std::endl;
            std::cout << std::endl;
            std::cout << "🚀 生成中..." << std::endl;

            auto startTime = std::chrono::high_resolution_clock::now();

            // 编码输入
            std::vector<int> inputIds = tokenizer.encode(test.prompt);
            std::cout << "   📊 输入 tokens: " << inputIds.size() << std::endl;

            // 生成 tokens
            std::vector<int> generatedTokens;

            for (int i = 0; i < test.maxTokens; ++i) {
                auto logits = model->forwardWithRequestId(inputIds, 0);

                // 贪婪解码
                int nextToken = 0;
                float maxProb = -1.0f;
                for (size_t j = 0; j < logits.size(); ++j) {
                    if (logits[j] > maxProb) {
                        maxProb = logits[j];
                        nextToken = static_cast<int>(j);
                    }
                }

                // 检查结束条件
                if (nextToken == tokenizer.getEosId()) {
                    break;
                }

                generatedTokens.push_back(nextToken);
                inputIds.push_back(nextToken);

                // 限制输出长度
                if (generatedTokens.size() >= 50) {
                    break;
                }
            }

            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);

            // 解码输出
            std::string output = tokenizer.decode(generatedTokens);

            std::cout << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << "📤 输出: " << output << std::endl;
            std::cout << std::endl;
            std::cout << "📊 统计:" << std::endl;
            std::cout << "   • 生成 tokens: " << generatedTokens.size() << std::endl;
            std::cout << "   • 耗时: " << duration.count() << " ms" << std::endl;
            if (duration.count() > 0) {
                std::cout << "   • 吞吐量: " << (generatedTokens.size() * 1000.0 / duration.count()) << " tokens/s" << std::endl;
            }
            std::cout << std::endl;
            std::cout << "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" << std::endl;
            std::cout << std::endl;
        }

        std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║                    所有测试完成!                             ║" << std::endl;
        std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "❌ 异常: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
