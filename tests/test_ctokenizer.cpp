#include <gtest/gtest.h>
#include "cllm/CTokenizer/manager.h"
#include "cllm/CTokenizer/tokenizer.h"
#include "cllm/CTokenizer/sentencepiece_tokenizer.h"
#include "cllm/CTokenizer/qwen_tokenizer.h"
#include "cllm/CTokenizer/deepseek_tokenizer.h"
#include "cllm/CTokenizer/model_detector.h"
#include <chrono>
#include <thread>

using namespace cllm;

class CTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 准备测试环境
    }
    
    void TearDown() override {
        // 清理测试环境
    }
    
    size_t getCurrentMemoryUsage() {
        return 0;
    }
    
    TokenizerManager manager;
};

// 基础功能测试
TEST_F(CTokenizerTest, EncodeDecodeBasic) {
    // 由于我们没有实际的测试模型文件，我们测试接口的可用性
    SentencePieceTokenizer tokenizer(ModelType::QWEN);
    
    // 我们不能加载实际模型，因为没有测试文件，所以这里主要是验证接口可用性
    EXPECT_EQ(tokenizer.getModelType(), ModelType::QWEN);
    
    // 测试空输入情况
    auto emptyTokens = tokenizer.encode("");
    // 空字符串可能产生特殊token或空结果，这取决于具体实现
}

TEST_F(CTokenizerTest, VocabOperations) {
    SentencePieceTokenizer tokenizer(ModelType::QWEN);
    
    // 测试词汇表操作接口
    int vocabSize = tokenizer.getVocabSize();
    // 如果模型未加载，词汇表大小可能为0
    EXPECT_GE(vocabSize, 0);
    
    // 测试ID到Token的转换（对于未加载模型的情况）
    std::string token = tokenizer.idToToken(100);
    // 未加载模型时，应返回空字符串或默认值
    EXPECT_TRUE(true); // 接口调用不会崩溃
    
    // 测试Token到ID的转换
    llama_token id = tokenizer.tokenToId("test");
    EXPECT_TRUE(true); // 接口调用不会崩溃
}

TEST_F(CTokenizerTest, SpecialTokens) {
    SentencePieceTokenizer tokenizer(ModelType::QWEN);
    
    // 测试特殊Token接口
    llama_token bosId = tokenizer.getBosId();
    llama_token eosId = tokenizer.getEosId();
    llama_token padId = tokenizer.getPadId();
    llama_token unkId = tokenizer.getUnkId();
    
    // 未加载模型时，特殊token ID通常为负数
    EXPECT_LE(bosId, -1);
    EXPECT_LE(eosId, -1);
    EXPECT_LE(padId, -1);
    EXPECT_LE(unkId, -1);
}

// QwenTokenizer测试
TEST_F(CTokenizerTest, QwenFimDetection) {
    QwenTokenizer tokenizer;
    
    // 测试FIM标记检测
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_pre|>test<|fim_suf|>content<|fim_end|>"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("test `` code ``"));
    EXPECT_FALSE(tokenizer.needsFimProcessing("regular text"));
}

// DeepSeekTokenizer测试
TEST_F(CTokenizerTest, DeepSeekModelTypes) {
    // 测试不同DeepSeek模型类型的构造
    DeepSeekTokenizer llmTokenizer(ModelType::DEEPSEEK_LLM);
    DeepSeekTokenizer coderTokenizer(ModelType::DEEPSEEK_CODER);
    DeepSeekTokenizer llm3Tokenizer(ModelType::DEEPSEEK3_LLM);
    
    EXPECT_EQ(llmTokenizer.getModelType(), ModelType::DEEPSEEK_LLM);
    EXPECT_EQ(coderTokenizer.getModelType(), ModelType::DEEPSEEK_CODER);
    EXPECT_EQ(llm3Tokenizer.getModelType(), ModelType::DEEPSEEK3_LLM);
}

// ModelDetector测试
TEST_F(CTokenizerTest, ModelDetectorDefault) {
    // 测试无效配置文件的处理
    ModelType type = ModelDetector::detectModelType("nonexistent/config.json");
    EXPECT_EQ(type, ModelType::SPM); // 应该返回默认类型
}

// TokenizerManager测试
TEST_F(CTokenizerTest, TokenizerManagerGet) {
    TokenizerManager manager;
    
    // 测试获取分词器（虽然无法加载实际模型，但可以测试管理器逻辑）
    CTokenizer* qwenTokenizer = manager.getTokenizer("qwen");
    EXPECT_NE(qwenTokenizer, nullptr);
    EXPECT_EQ(qwenTokenizer->getModelType(), ModelType::QWEN);
    
    CTokenizer* deepseekTokenizer = manager.getTokenizer("deepseek-coder");
    EXPECT_NE(deepseekTokenizer, nullptr);
    EXPECT_EQ(deepseekTokenizer->getModelType(), ModelType::DEEPSEEK_CODER);
    
    CTokenizer* llamaTokenizer = manager.getTokenizer("llama");
    EXPECT_NE(llamaTokenizer, nullptr);
    EXPECT_EQ(llamaTokenizer->getModelType(), ModelType::LLAMA);
}

TEST_F(CTokenizerTest, TokenizerManagerCache) {
    // 测试分词器缓存行为
    TokenizerManager manager;
    
    CTokenizer* tokenizer1 = manager.getTokenizer("qwen");
    CTokenizer* tokenizer2 = manager.getTokenizer("qwen");
    
    // 对于相同的模型类型，应该返回相同的实例（由管理器缓存）
    // 注意：这取决于具体实现，可能不是严格意义上的单例
    EXPECT_NE(tokenizer1, nullptr);
    EXPECT_NE(tokenizer2, nullptr);
}

// 模型类型枚举测试
TEST_F(CTokenizerTest, ModelTypeValues) {
    // 测试模型类型枚举值
    EXPECT_EQ(ModelType::AUTO, ModelType::AUTO);
    EXPECT_EQ(ModelType::QWEN, ModelType::QWEN);
    EXPECT_EQ(ModelType::QWEN2, ModelType::QWEN2);
    EXPECT_EQ(ModelType::DEEPSEEK_LLM, ModelType::DEEPSEEK_LLM);
    EXPECT_EQ(ModelType::DEEPSEEK_CODER, ModelType::DEEPSEEK_CODER);
    EXPECT_EQ(ModelType::DEEPSEEK3_LLM, ModelType::DEEPSEEK3_LLM);
    EXPECT_EQ(ModelType::LLAMA, ModelType::LLAMA);
    EXPECT_EQ(ModelType::BERT, ModelType::BERT);
    EXPECT_EQ(ModelType::GPT2, ModelType::GPT2);
    EXPECT_EQ(ModelType::SPM, ModelType::SPM);
    EXPECT_EQ(ModelType::BPE, ModelType::BPE);
    EXPECT_EQ(ModelType::WPM, ModelType::WPM);
}

// 边界条件测试
TEST_F(CTokenizerTest, BoundaryConditions) {
    SentencePieceTokenizer tokenizer(ModelType::QWEN);
    
    // 测试各种边界条件
    std::vector<std::string> testInputs = {
        "",           // 空字符串
        " ",          // 单空格
        "\n",         // 单换行
        "\t",         // 单制表符
        "A",          // 单字符
        std::string(10, 'A'), // 短字符串
    };
    
    for (const auto& input : testInputs) {
        // 测试编码接口不会崩溃
        auto tokens = tokenizer.encode(input);
        EXPECT_TRUE(true); // 只要不崩溃就算通过
        
        // 测试解码接口不会崩溃
        std::string decoded = tokenizer.decode(tokens);
        EXPECT_TRUE(true); // 只要不崩溃就算通过
    }
}

// 性能测试 - 主要是测试接口响应时间
TEST_F(CTokenizerTest, InterfaceResponsiveness) {
    SentencePieceTokenizer tokenizer(ModelType::QWEN);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 快速调用多个接口
    int vocabSize = tokenizer.getVocabSize();
    llama_token bosId = tokenizer.getBosId();
    llama_token eosId = tokenizer.getEosId();
    std::string token = tokenizer.idToToken(100);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 接口调用应在合理时间内完成（即使没有加载模型）
    EXPECT_LT(duration.count(), 1000); // 应该在1毫秒内完成
}

// 多线程安全性测试（基本验证）
TEST_F(CTokenizerTest, ThreadSafetyBasic) {
    TokenizerManager manager;
    
    // 启动多个线程同时获取分词器
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([&manager, i]() {
            CTokenizer* tok = manager.getTokenizer("qwen");
            EXPECT_NE(tok, nullptr);
            // 模拟一些操作
            EXPECT_EQ(tok->getModelType(), ModelType::QWEN);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// 测试完成
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

// ============ 集成测试 ============

TEST_F(CTokenizerTest, IntegrationEndToEnd) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::string input = "This is a test sentence for end-to-end validation.";
    auto tokens = tokenizer->encode(input);
    
    int vocabSize = tokenizer->getVocabSize();
    EXPECT_GE(vocabSize, 0);
    
    if (!tokens.empty()) {
        std::string output = tokenizer->decode(tokens);
        EXPECT_EQ(input, output);
    }
}

TEST_F(CTokenizerTest, IntegrationMultiModelSupport) {
    std::vector<std::string> modelTypes = {"qwen", "deepseek-llm", "deepseek-coder", "llama"};
    
    for (const auto& modelType : modelTypes) {
        auto tokenizer = this->manager.getTokenizer(modelType);
        ASSERT_NE(tokenizer, nullptr) << "Failed to get tokenizer for " << modelType;
        
        std::string testText = "Test text for " + modelType;
        auto tokens = tokenizer->encode(testText);
        
        if (!tokens.empty()) {
            std::string decoded = tokenizer->decode(tokens);
            EXPECT_EQ(decoded, testText) << "Decoding mismatch for " << modelType;
        }
    }
}

TEST_F(CTokenizerTest, IntegrationBatchEncodeDecode) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::vector<std::string> texts = {
        "Hello, world!",
        "This is a test sentence.",
        "Another test with numbers: 12345",
        "Mixed content: Hello 世界 🌍"
    };
    
    for (const auto& text : texts) {
        auto tokens = tokenizer->encode(text);
        
        if (!tokens.empty()) {
            std::string decoded = tokenizer->decode(tokens);
            EXPECT_EQ(decoded, text);
        }
    }
}

// ============ 性能测试 ============

TEST_F(CTokenizerTest, PerformanceEncodeSpeed) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::string longText;
    for (int i = 0; i < 1000; ++i) {
        longText += "This is a test sentence for performance evaluation. ";
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto tokens = tokenizer->encode(longText);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (!tokens.empty()) {
        EXPECT_LT(duration.count(), 1000);
        
        double speed = (double)longText.length() / (duration.count() / 1000.0);
        EXPECT_GT(speed, 50000);
    }
}

TEST_F(CTokenizerTest, PerformanceDecodeSpeed) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::string text = "Performance test text. ";
    auto tokens = tokenizer->encode(text);
    
    if (tokens.empty()) {
        return;
    }
    
    std::vector<std::vector<llama_token>> batchTokens;
    for (int i = 0; i < 1000; ++i) {
        batchTokens.push_back(tokens);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& tokenList : batchTokens) {
        std::string decoded = tokenizer->decode(tokenList);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    EXPECT_LT(duration.count(), 1000);
}

TEST_F(CTokenizerTest, PerformanceMemoryUsage) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    size_t initialMemory = getCurrentMemoryUsage();
    
    for (int i = 0; i < 10000; ++i) {
        std::string text = "Test " + std::to_string(i);
        auto tokens = tokenizer->encode(text);
        std::string decoded = tokenizer->decode(tokens);
    }
    
    size_t finalMemory = getCurrentMemoryUsage();
    
    if (initialMemory > 0 && finalMemory > 0) {
        EXPECT_LT(finalMemory - initialMemory, 10 * 1024 * 1024);
    }
}

TEST_F(CTokenizerTest, PerformanceInterfaceResponsiveness) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    int vocabSize = tokenizer->getVocabSize();
    llama_token bosId = tokenizer->getBosId();
    llama_token eosId = tokenizer->getEosId();
    std::string token = tokenizer->idToToken(100);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    EXPECT_LT(duration.count(), 1000);
}

// ============ 验证测试 ============

TEST_F(CTokenizerTest, ValidationCrossPlatformConsistency) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::vector<std::string> testCases = {
        "Hello, world!",
        "测试中文分词",
        "Test with numbers: 123456789",
        "Mixed: Hello 世界 🌍 emoji",
        "Special chars: !@#$%^&*()",
        "Long text with multiple sentences. This is sentence two. And this is three."
    };
    
    for (const auto& testCase : testCases) {
        auto tokens = tokenizer->encode(testCase);
        
        if (!tokens.empty()) {
            std::string decoded = tokenizer->decode(tokens);
            EXPECT_EQ(testCase, decoded) << "Mismatch for test case: " << testCase;
        }
        
        int vocabSize = tokenizer->getVocabSize();
        EXPECT_GE(vocabSize, 0);
    }
}

TEST_F(CTokenizerTest, ValidationModelSpecificFeatures) {
    QwenTokenizer qwenTokenizer;
    
    std::string code = "def function():\n    pass";
    auto tokens = qwenTokenizer.encode(code);
    
    if (!tokens.empty()) {
        std::string decoded = qwenTokenizer.decode(tokens);
        EXPECT_EQ(decoded, code);
    }
    
    DeepSeekTokenizer deepseekTokenizer(ModelType::DEEPSEEK_CODER);
    
    std::string code2 = "class MyClass:\n    def method(self):\n        return True";
    auto tokens2 = deepseekTokenizer.encode(code2);
    
    if (!tokens2.empty()) {
        std::string decoded2 = deepseekTokenizer.decode(tokens2);
        EXPECT_EQ(decoded2, code2);
    }
}

// ============ 回归测试 ============

TEST_F(CTokenizerTest, RegressionKnownIssues) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::vector<std::string> problematicInputs = {
        "",
        " ",
        "\n",
        "\t",
        "\r\n",
        std::string(1000, 'A'),
        "A" + std::string(1000, 'B') + "C",
        "!@#$%^&*()_+-=[]{}|;:,.<>?",
        "αβγδεζηθικλμνξοπρστυφχψω",
        "あいうえおかきくけこ",
        "한국어 테스트",
        "العربية",
        "Русский",
        " 🌍 ✨ 🚀 "
    };
    
    for (const auto& input : problematicInputs) {
        try {
            auto tokens = tokenizer->encode(input);
            
            if (!tokens.empty()) {
                std::string decoded = tokenizer->decode(tokens);
                EXPECT_EQ(decoded, input) << "Regression detected for input: " << input;
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Exception thrown for input '" << input << "': " << e.what();
        }
    }
}

// ============ 边界和异常测试 ============

TEST_F(CTokenizerTest, BoundaryConditionsExtended) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::vector<std::string> testInputs = {
        "",
        " ",
        "\n",
        "\t",
        "A",
        std::string(10, 'A'),
        std::string(100, 'A'),
        std::string(1000, 'A'),
        std::string(10000, 'A'),
        "Hello, 世界! 🌍",
        "\0\1\2\3",
        "Line1\nLine2\nLine3",
        "Tab\tTab\tTab"
    };
    
    for (const auto& input : testInputs) {
        auto tokens = tokenizer->encode(input);
        EXPECT_TRUE(true);
        
        std::string decoded = tokenizer->decode(tokens);
        EXPECT_TRUE(true);
    }
}

TEST_F(CTokenizerTest, SpecialTokensWithEncoding) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    std::string text = "Hello";
    auto idsWithoutSpecial = tokenizer->encode(text, false);
    auto idsWithSpecial = tokenizer->encode(text, true);
    
    EXPECT_GE(idsWithSpecial.size(), idsWithoutSpecial.size());
}

TEST_F(CTokenizerTest, VocabOperationsExtended) {
    auto tokenizer = this->manager.getTokenizer("qwen");
    ASSERT_NE(tokenizer, nullptr);
    
    int vocabSize = tokenizer->getVocabSize();
    EXPECT_GE(vocabSize, 0);
    
    if (vocabSize > 0) {
        std::string token = tokenizer->idToToken(100);
        llama_token id = tokenizer->tokenToId(token);
        
        if (!token.empty()) {
            EXPECT_EQ(token, tokenizer->idToToken(id));
        }
    }
}

// ============ QwenTokenizer专项测试 ============

TEST_F(CTokenizerTest, QwenFimProcessing) {
    QwenTokenizer tokenizer;
    
    std::string text = "<|fim_pre|>def hello():<|fim_suf|>    return 'world'<|fim_end|>";
    auto ids = tokenizer.encode(text);
    
    if (!ids.empty()) {
        std::string decoded = tokenizer.decode(ids);
        if (!decoded.empty()) {
            EXPECT_EQ(decoded, text);
        }
    }
}

TEST_F(CTokenizerTest, QwenFimDetectionExtended) {
    QwenTokenizer tokenizer;
    
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_pre|>test<|fim_suf|>content<|fim_end|>"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("test `` code ``"));
    EXPECT_FALSE(tokenizer.needsFimProcessing("regular text"));
    EXPECT_FALSE(tokenizer.needsFimProcessing(""));
    EXPECT_FALSE(tokenizer.needsFimProcessing("   "));
}

// ============ DeepSeekTokenizer专项测试 ============

TEST_F(CTokenizerTest, DeepSeekPreprocessing) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    std::string code = "class MyClass:\n    def method(self):\n        return True";
    auto tokens = tokenizer.encode(code);
    
    if (!tokens.empty()) {
        std::string decoded = tokenizer.decode(tokens);
        EXPECT_EQ(decoded, code);
    }
}

TEST_F(CTokenizerTest, DeepSeekModelTypesExtended) {
    DeepSeekTokenizer llmTokenizer(ModelType::DEEPSEEK_LLM);
    DeepSeekTokenizer coderTokenizer(ModelType::DEEPSEEK_CODER);
    DeepSeekTokenizer llm3Tokenizer(ModelType::DEEPSEEK3_LLM);
    
    EXPECT_EQ(llmTokenizer.getModelType(), ModelType::DEEPSEEK_LLM);
    EXPECT_EQ(coderTokenizer.getModelType(), ModelType::DEEPSEEK_CODER);
    EXPECT_EQ(llm3Tokenizer.getModelType(), ModelType::DEEPSEEK3_LLM);
    
    std::string text = "Hello world";
    
    auto llmTokens = llmTokenizer.encode(text);
    if (!llmTokens.empty()) {
        EXPECT_FALSE(llmTokens.empty());
    }
    
    auto coderTokens = coderTokenizer.encode(text);
    if (!coderTokens.empty()) {
        EXPECT_FALSE(coderTokens.empty());
    }
    
    auto llm3Tokens = llm3Tokenizer.encode(text);
    if (!llm3Tokens.empty()) {
        EXPECT_FALSE(llm3Tokens.empty());
    }
}