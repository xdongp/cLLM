#include <gtest/gtest.h>
#include "cllm/CTokenizer/deepseek_tokenizer.h"
#include <string>
#include <vector>
#include <thread>
#include <chrono>

using namespace cllm;

/**
 * DeepSeek分词器预处理功能单元测试
 * 
 * 注意: 这些测试不需要实际的模型文件
 * 主要测试预处理逻辑的正确性和鲁棒性
 */

class DeepSeekPreprocessingUnitTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 准备测试环境
    }
    
    void TearDown() override {
        // 清理测试环境
    }
};

// ========== 基础接口测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, TokenizerConstruction) {
    // 测试不同模型类型的构造
    EXPECT_NO_THROW({
        DeepSeekTokenizer llmTokenizer(ModelType::DEEPSEEK_LLM);
        EXPECT_EQ(llmTokenizer.getModelType(), ModelType::DEEPSEEK_LLM);
    });
    
    EXPECT_NO_THROW({
        DeepSeekTokenizer coderTokenizer(ModelType::DEEPSEEK_CODER);
        EXPECT_EQ(coderTokenizer.getModelType(), ModelType::DEEPSEEK_CODER);
    });
    
    EXPECT_NO_THROW({
        DeepSeekTokenizer llm3Tokenizer(ModelType::DEEPSEEK3_LLM);
        EXPECT_EQ(llm3Tokenizer.getModelType(), ModelType::DEEPSEEK3_LLM);
    });
}

TEST_F(DeepSeekPreprocessingUnitTest, EncodeWithoutModelNoCrash) {
    // 测试在没有加载模型的情况下调用encode不会崩溃
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    std::vector<std::string> testCases = {
        "",
        "Hello World",
        "你好世界",
        "123456",
        "!@#$%"
    };
    
    for (const auto& testCase : testCases) {
        EXPECT_NO_THROW({
            auto tokens = tokenizer.encode(testCase);
            // 没有模型时应该返回空向量
            EXPECT_TRUE(tokens.empty());
        }) << "Crashed for input: " << testCase;
    }
}

TEST_F(DeepSeekPreprocessingUnitTest, DecodeWithoutModelNoCrash) {
    // 测试在没有加载模型的情况下调用decode不会崩溃
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    std::vector<std::vector<llama_token>> testCases = {
        {},
        {1, 2, 3},
        {100, 200, 300}
    };
    
    for (const auto& testCase : testCases) {
        EXPECT_NO_THROW({
            std::string decoded = tokenizer.decode(testCase);
            // 没有模型时应该返回空字符串
            EXPECT_TRUE(decoded.empty());
        });
    }
}

// ========== 特殊Token测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, SpecialTokensWithoutModel) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    // 没有模型时,特殊token应该返回默认值
    EXPECT_LE(tokenizer.getBosId(), -1);
    EXPECT_LE(tokenizer.getEosId(), -1);
    EXPECT_LE(tokenizer.getPadId(), -1);
    EXPECT_LE(tokenizer.getUnkId(), -1);
}

// ========== 词汇表操作测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, VocabOperationsWithoutModel) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    // 没有模型时,词汇表大小应该为0
    EXPECT_EQ(tokenizer.getVocabSize(), 0);
    
    // 没有模型时,ID到token的转换应该返回空字符串
    std::string token = tokenizer.idToToken(100);
    EXPECT_TRUE(token.empty());
    
    // 没有模型时,token到ID的转换应该返回0或-1（取决于SentencePiece实现）
    llama_token id = tokenizer.tokenToId("test");
    EXPECT_TRUE(id == 0 || id == -1) << "Expected 0 or -1, got " << id;
}

// ========== 模型类型验证测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, ModelTypeConsistency) {
    std::vector<std::pair<ModelType, std::string>> testCases = {
        {ModelType::DEEPSEEK_LLM, "DeepSeek LLM"},
        {ModelType::DEEPSEEK_CODER, "DeepSeek Coder"},
        {ModelType::DEEPSEEK3_LLM, "DeepSeek3 LLM"}
    };
    
    for (const auto& [modelType, name] : testCases) {
        DeepSeekTokenizer tokenizer(modelType);
        EXPECT_EQ(tokenizer.getModelType(), modelType) 
            << "Model type mismatch for: " << name;
    }
}

// ========== 边界条件测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, BoundaryEmptyInput) {
    std::vector<ModelType> models = {
        ModelType::DEEPSEEK_LLM,
        ModelType::DEEPSEEK_CODER,
        ModelType::DEEPSEEK3_LLM
    };
    
    for (auto modelType : models) {
        DeepSeekTokenizer tokenizer(modelType);
        
        // 空字符串输入
        EXPECT_NO_THROW({
            auto tokens = tokenizer.encode("");
            EXPECT_TRUE(tokens.empty());
        });
        
        // 空向量解码
        EXPECT_NO_THROW({
            std::string decoded = tokenizer.decode({});
            EXPECT_TRUE(decoded.empty());
        });
    }
}

TEST_F(DeepSeekPreprocessingUnitTest, BoundarySingleCharacter) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    std::vector<std::string> testCases = {"A", "1", " ", "中", "\n", "\t"};
    
    for (const auto& testCase : testCases) {
        EXPECT_NO_THROW({
            auto tokens = tokenizer.encode(testCase);
            // 没有模型时返回空向量
            EXPECT_TRUE(tokens.empty());
        }) << "Failed for: " << testCase;
    }
}

TEST_F(DeepSeekPreprocessingUnitTest, BoundaryVeryLongText) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    // 非常长的文本
    std::string longText(100000, 'A');
    
    EXPECT_NO_THROW({
        auto tokens = tokenizer.encode(longText);
        EXPECT_TRUE(tokens.empty()); // 没有模型时返回空向量
    });
}

// ========== 特殊字符测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, SpecialCharactersHandling) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    std::vector<std::string> testCases = {
        "!@#$%^&*()",
        "<>[]{}()",
        "\\n\\t\\r",
        "αβγδεζ",
        "あいうえお",
        "한국어",
        "العربية",
        "🌍✨🚀"
    };
    
    for (const auto& testCase : testCases) {
        EXPECT_NO_THROW({
            auto tokens = tokenizer.encode(testCase);
            EXPECT_TRUE(tokens.empty());
        }) << "Failed for: " << testCase;
    }
}

// ========== 编码/解码选项测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, EncodeWithSpecialTokensOption) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    std::string text = "Hello World";
    
    // 测试带特殊token的编码
    EXPECT_NO_THROW({
        auto tokensWithSpecial = tokenizer.encode(text, true);
        EXPECT_TRUE(tokensWithSpecial.empty());
    });
    
    // 测试不带特殊token的编码
    EXPECT_NO_THROW({
        auto tokensWithoutSpecial = tokenizer.encode(text, false);
        EXPECT_TRUE(tokensWithoutSpecial.empty());
    });
}

TEST_F(DeepSeekPreprocessingUnitTest, DecodeWithSpecialTokensOption) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK3_LLM);
    
    std::vector<llama_token> tokens = {1, 2, 3};
    
    // 测试跳过特殊token的解码
    EXPECT_NO_THROW({
        std::string decoded1 = tokenizer.decode(tokens, true);
        EXPECT_TRUE(decoded1.empty());
    });
    
    // 测试不跳过特殊token的解码
    EXPECT_NO_THROW({
        std::string decoded2 = tokenizer.decode(tokens, false);
        EXPECT_TRUE(decoded2.empty());
    });
}

// ========== 多线程安全性基础测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, ThreadSafetyBasic) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 10; ++i) {
        threads.emplace_back([&tokenizer, i]() {
            std::string text = "Test " + std::to_string(i);
            auto tokens = tokenizer.encode(text);
            EXPECT_TRUE(tokens.empty());
            
            int vocabSize = tokenizer.getVocabSize();
            EXPECT_EQ(vocabSize, 0);
        });
    }
    
    for (auto& t : threads) {
        t.join();
    }
}

// ========== 异常处理测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, InvalidTokenIds) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_CODER);
    
    std::vector<llama_token> invalidTokens = {-1, -100, 999999};
    
    EXPECT_NO_THROW({
        std::string decoded = tokenizer.decode(invalidTokens);
        EXPECT_TRUE(decoded.empty());
    });
}

TEST_F(DeepSeekPreprocessingUnitTest, LoadNonExistentModel) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    // 尝试加载不存在的模型文件
    EXPECT_NO_THROW({
        bool loaded = tokenizer.load("nonexistent/model/path");
        EXPECT_FALSE(loaded);
    });
    
    // 加载失败后,分词器应该仍然可用
    EXPECT_NO_THROW({
        auto tokens = tokenizer.encode("test");
        EXPECT_TRUE(tokens.empty());
    });
}

// ========== 性能测试（接口响应时间） ==========

TEST_F(DeepSeekPreprocessingUnitTest, InterfaceResponsiveness) {
    DeepSeekTokenizer tokenizer(ModelType::DEEPSEEK_LLM);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 快速调用多个接口
    int vocabSize = tokenizer.getVocabSize();
    llama_token bosId = tokenizer.getBosId();
    llama_token eosId = tokenizer.getEosId();
    std::string token = tokenizer.idToToken(100);
    llama_token id = tokenizer.tokenToId("test");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // 接口调用应在合理时间内完成（即使没有加载模型）
    EXPECT_LT(duration.count(), 1000); // 应该在1毫秒内完成
}

// ========== 模型类型枚举测试 ==========

TEST_F(DeepSeekPreprocessingUnitTest, ModelTypeEnumValues) {
    // 验证模型类型枚举值的有效性
    EXPECT_NE(ModelType::DEEPSEEK_LLM, ModelType::DEEPSEEK_CODER);
    EXPECT_NE(ModelType::DEEPSEEK_LLM, ModelType::DEEPSEEK3_LLM);
    EXPECT_NE(ModelType::DEEPSEEK_CODER, ModelType::DEEPSEEK3_LLM);
}

// 测试入口
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
