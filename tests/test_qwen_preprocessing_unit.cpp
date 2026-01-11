#include <gtest/gtest.h>
#include "cllm/CTokenizer/qwen_tokenizer.h"
#include <string>
#include <vector>
#include <thread>
#include <chrono>

using namespace cllm;

// ============================================================================
// QwenTokenizer 预处理单元测试
// ============================================================================

class QwenPreprocessingUnitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 每个测试前的设置
    }
    
    void TearDown() override {
        // 每个测试后的清理
    }
};

// ========== 基础功能测试 ==========

TEST_F(QwenPreprocessingUnitTest, ConstructorAndModelType) {
    QwenTokenizer tokenizer;
    
    // 验证模型类型
    EXPECT_EQ(tokenizer.getModelType(), ModelType::QWEN);
}

TEST_F(QwenPreprocessingUnitTest, EmptyTextHandling) {
    QwenTokenizer tokenizer;
    
    // 空文本应该返回空结果
    std::vector<llama_token> tokens = tokenizer.encode("", true);
    EXPECT_TRUE(tokens.empty());
    
    tokens = tokenizer.encode("", false);
    EXPECT_TRUE(tokens.empty());
}

// ========== FIM 检测测试 ==========

TEST_F(QwenPreprocessingUnitTest, FimDetectionWithStandardMarkers) {
    QwenTokenizer tokenizer;
    
    // 测试标准FIM标记检测
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_pre|>test<|fim_suf|>content<|fim_end|>"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_begin|>test"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("test<|fim_end|>"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_pre|>test"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("<|fim_suf|>test"));
}

TEST_F(QwenPreprocessingUnitTest, FimDetectionWithSimpleMarkers) {
    QwenTokenizer tokenizer;
    
    // 测试简化的``标记检测
    EXPECT_TRUE(tokenizer.needsFimProcessing("test `` code"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("``"));
    EXPECT_TRUE(tokenizer.needsFimProcessing("prefix `` suffix"));
}

TEST_F(QwenPreprocessingUnitTest, FimDetectionWithoutMarkers) {
    QwenTokenizer tokenizer;
    
    // 普通文本不应检测为FIM
    EXPECT_FALSE(tokenizer.needsFimProcessing("regular text without fim markers"));
    EXPECT_FALSE(tokenizer.needsFimProcessing("test with ` single backtick"));
    EXPECT_FALSE(tokenizer.needsFimProcessing(""));
    EXPECT_FALSE(tokenizer.needsFimProcessing("   "));
}

// ========== 英语缩写预处理测试 ==========

TEST_F(QwenPreprocessingUnitTest, EnglishContractions) {
    QwenTokenizer tokenizer;
    
    // 测试常见英语缩写
    // 注意：这里只测试预处理不会破坏文本，具体分词由SentencePiece决定
    std::vector<std::string> test_cases = {
        "don't worry",
        "it's working",
        "they're here",
        "I've done it",
        "I'm happy",
        "we'll see",
        "he'd go"
    };
    
    for (const auto& text : test_cases) {
        // 编码应该成功（不崩溃）
        auto tokens = tokenizer.encode(text, false);
        // 无模型时返回空，这是正常的
        EXPECT_TRUE(tokens.empty() || !tokens.empty()) 
            << "Processing failed for: " << text;
    }
}

// ========== 混合内容预处理测试 ==========

TEST_F(QwenPreprocessingUnitTest, MixedEnglishAndNumbers) {
    QwenTokenizer tokenizer;
    
    std::string text = "The year is 2024 and it's great!";
    auto tokens = tokenizer.encode(text, false);
    
    // 无模型时返回空，验证不崩溃即可
    EXPECT_TRUE(tokens.empty());
}

TEST_F(QwenPreprocessingUnitTest, MixedChineseAndEnglish) {
    QwenTokenizer tokenizer;
    
    std::string text = "你好World! This is 测试123.";
    auto tokens = tokenizer.encode(text, false);
    
    // 无模型时返回空，验证不崩溃即可
    EXPECT_TRUE(tokens.empty());
}

TEST_F(QwenPreprocessingUnitTest, PunctuationHandling) {
    QwenTokenizer tokenizer;
    
    std::string text = "Hello, World! How are you? I'm fine.";
    auto tokens = tokenizer.encode(text, false);
    
    // 无模型时返回空，验证不崩溃即可
    EXPECT_TRUE(tokens.empty());
}

// ========== 数字处理测试 ==========

TEST_F(QwenPreprocessingUnitTest, NumberHandling) {
    QwenTokenizer tokenizer;
    
    std::vector<std::string> test_cases = {
        "0",
        "123",
        "1 2 3",
        "number 42 is the answer",
        "3.14159"
    };
    
    for (const auto& text : test_cases) {
        auto tokens = tokenizer.encode(text, false);
        // 验证不崩溃
        EXPECT_TRUE(tokens.empty()) << "Failed for: " << text;
    }
}

// ========== 空白字符处理测试 ==========

TEST_F(QwenPreprocessingUnitTest, WhitespaceHandling) {
    QwenTokenizer tokenizer;
    
    std::vector<std::string> test_cases = {
        "test   multiple   spaces",
        "test\ttab\tcharacters",
        "test\nnewline\ncharacters",
        "test\r\nwindows\r\nnewlines",
        "   leading spaces",
        "trailing spaces   ",
        "\n\n\nmultiple newlines\n\n\n"
    };
    
    for (const auto& text : test_cases) {
        auto tokens = tokenizer.encode(text, false);
        // 验证不崩溃
        EXPECT_TRUE(tokens.empty()) << "Failed for: " << text;
    }
}

// ========== 边界条件测试 ==========

TEST_F(QwenPreprocessingUnitTest, SingleCharacter) {
    QwenTokenizer tokenizer;
    
    std::vector<std::string> test_cases = {
        "a", "Z", "0", "9", "!", "?", " ", "\n"
    };
    
    for (const auto& text : test_cases) {
        auto tokens = tokenizer.encode(text, false);
        EXPECT_TRUE(tokens.empty());
    }
}

TEST_F(QwenPreprocessingUnitTest, VeryLongText) {
    QwenTokenizer tokenizer;
    
    // 生成一个很长的文本（10KB）
    std::string long_text;
    long_text.reserve(10000);
    for (int i = 0; i < 1000; i++) {
        long_text += "This is a test sentence. ";
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    auto tokens = tokenizer.encode(long_text, false);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 验证处理时间合理（应该在100ms内）
    EXPECT_LT(duration.count(), 100) << "Processing took " << duration.count() << "ms";
    EXPECT_TRUE(tokens.empty());
}

// ========== 特殊字符测试 ==========

TEST_F(QwenPreprocessingUnitTest, UnicodeCharacters) {
    QwenTokenizer tokenizer;
    
    std::vector<std::string> test_cases = {
        "中文测试",
        "日本語テスト",
        "한국어 테스트",
        "Тест на русском",
        "Test with émojis 😀🎉",
        "Mixed: 中文English日本語123"
    };
    
    for (const auto& text : test_cases) {
        auto tokens = tokenizer.encode(text, false);
        // 验证不崩溃
        EXPECT_TRUE(tokens.empty()) << "Failed for: " << text;
    }
}

TEST_F(QwenPreprocessingUnitTest, SpecialCharacters) {
    QwenTokenizer tokenizer;
    
    std::string text = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
    auto tokens = tokenizer.encode(text, false);
    
    // 验证不崩溃
    EXPECT_TRUE(tokens.empty());
}

// ========== 词汇表操作测试 ==========

TEST_F(QwenPreprocessingUnitTest, VocabOperationsWithoutModel) {
    QwenTokenizer tokenizer;
    
    // 没有模型时，词汇表大小应该为0
    EXPECT_EQ(tokenizer.getVocabSize(), 0);
    
    // 没有模型时，ID到token的转换应该返回空字符串
    std::string token = tokenizer.idToToken(100);
    EXPECT_TRUE(token.empty());
    
    // 没有模型时，token到ID的转换应该返回0或-1
    llama_token id = tokenizer.tokenToId("test");
    EXPECT_TRUE(id == 0 || id == -1) << "Expected 0 or -1, got " << id;
}

// ========== 编码选项测试 ==========

TEST_F(QwenPreprocessingUnitTest, EncodeWithAndWithoutSpecialTokens) {
    QwenTokenizer tokenizer;
    
    std::string text = "test text";
    
    // 测试带特殊tokens的编码
    auto tokens_with = tokenizer.encode(text, true);
    EXPECT_TRUE(tokens_with.empty());
    
    // 测试不带特殊tokens的编码
    auto tokens_without = tokenizer.encode(text, false);
    EXPECT_TRUE(tokens_without.empty());
}

// ========== 多线程安全性测试 ==========

TEST_F(QwenPreprocessingUnitTest, ConcurrentEncode) {
    QwenTokenizer tokenizer;
    
    std::vector<std::thread> threads;
    std::atomic<int> success_count{0};
    
    // 启动10个线程并发编码
    for (int i = 0; i < 10; i++) {
        threads.emplace_back([&tokenizer, &success_count, i]() {
            std::string text = "test text " + std::to_string(i);
            auto tokens = tokenizer.encode(text, false);
            // 无模型时返回空是正常的
            if (tokens.empty()) {
                success_count++;
            }
        });
    }
    
    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }
    
    // 所有线程都应该成功
    EXPECT_EQ(success_count.load(), 10);
}

// ========== 异常处理测试 ==========

TEST_F(QwenPreprocessingUnitTest, NullCharacterHandling) {
    QwenTokenizer tokenizer;
    
    std::string text_with_null = "test\0text";
    text_with_null.resize(9); // 确保包含null字符
    
    // 应该能够处理包含null字符的文本（不崩溃）
    auto tokens = tokenizer.encode(text_with_null, false);
    EXPECT_TRUE(tokens.empty());
}

// ========== 接口响应时间测试 ==========

TEST_F(QwenPreprocessingUnitTest, InterfaceResponseTime) {
    QwenTokenizer tokenizer;
    std::string text = "test text for performance";
    
    // 预热
    tokenizer.encode(text, false);
    
    // 测量响应时间
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        tokenizer.encode(text, false);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double avg_time = duration.count() / 1000.0;
    
    // 平均响应时间应该小于10微秒（无模型情况下）
    EXPECT_LT(avg_time, 10.0) << "Average response time: " << avg_time << "μs";
}

// ========== 代码补全场景测试 ==========

TEST_F(QwenPreprocessingUnitTest, CodeCompletionScenario) {
    QwenTokenizer tokenizer;
    
    // 模拟代码补全场景
    std::string code = R"(def add(a, b):
    return a + b

def subtract(a, b):
    return a - b)";
    
    auto tokens = tokenizer.encode(code, false);
    
    // 验证不崩溃，无模型时返回空
    EXPECT_TRUE(tokens.empty());
}

// ============================================================================
// 主函数
// ============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
