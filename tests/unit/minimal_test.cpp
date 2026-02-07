/**
 * @file minimal_test.cpp
 * @brief 最小测试 - 验证测试框架和工具库可以正常编译
 */

#include <gtest/gtest.h>
#include "utils/mock_tokenizer.h"
#include "utils/test_data_helpers.h"

using namespace cllm;
using namespace cllm::test;

// 基础测试 - 不依赖任何其他组件
TEST(MinimalTest, MockTokenizerBasic) {
    MockTokenizer tokenizer;
    
    EXPECT_EQ(tokenizer.getVocabSize(), 1000);
    EXPECT_EQ(tokenizer.getBosId(), 1);
    EXPECT_EQ(tokenizer.getEosId(), 2);
    EXPECT_EQ(tokenizer.getPadId(), 0);
    EXPECT_EQ(tokenizer.getUnkId(), 3);
}

TEST(MinimalTest, MockTokenizerEncode) {
    MockTokenizer tokenizer;
    
    std::string text = "Hello";
    auto tokens = tokenizer.encode(text, false);
    
    EXPECT_FALSE(tokens.empty());
    EXPECT_EQ(tokens.size(), text.length());
}

TEST(MinimalTest, MockTokenizerEncodeWithSpecialTokens) {
    MockTokenizer tokenizer;
    
    std::string text = "Hello";
    auto tokens = tokenizer.encode(text, true);
    
    EXPECT_FALSE(tokens.empty());
    EXPECT_EQ(tokens[0], tokenizer.getBosId());
}

TEST(MinimalTest, MockTokenizerDecode) {
    MockTokenizer tokenizer;
    
    std::vector<int> tokens = {72, 101, 108, 108, 111};
    auto text = tokenizer.decode(tokens, false);
    
    EXPECT_FALSE(text.empty());
}

TEST(MinimalTest, SimpleMockTokenizer) {
    SimpleMockTokenizer tokenizer;
    
    auto tokens = tokenizer.encode("test", true);
    EXPECT_FALSE(tokens.empty());
    
    auto text = tokenizer.decode(tokens, true);
    EXPECT_FALSE(text.empty());
}

TEST(MinimalTest, TestDataHelpersRandomString) {
    auto text = TestDataHelpers::generateRandomString(10);
    EXPECT_EQ(text.length(), 10);
}

TEST(MinimalTest, TestDataHelpersTestPrompts) {
    auto prompts = TestDataHelpers::generateTestPrompts();
    EXPECT_FALSE(prompts.empty());
    EXPECT_GT(prompts.size(), 0);
}

TEST(MinimalTest, TestDataHelpersRandomTokens) {
    auto tokens = TestDataHelpers::generateRandomTokens(10, 1000);
    EXPECT_EQ(tokens.size(), 10);
    
    for (int token : tokens) {
        EXPECT_GE(token, 0);
        EXPECT_LT(token, 1000);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
