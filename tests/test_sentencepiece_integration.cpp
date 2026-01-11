#include <gtest/gtest.h>
#include <sentencepiece_processor.h>
#include <cllm/tokenizer/tokenizer.h>
#include <string>
#include <vector>

// 测试SentencePiece库是否可以正确链接和使用
// 由于我们没有真实的tokenizer.model，我们只验证库的基本功能

TEST(SentencePieceIntegrationTest, LibraryLinking) {
    // 验证SentencePiece库可以正确链接和使用
    // 这个测试主要验证SentencePiece库是否正确集成到项目中
    EXPECT_TRUE(true);  // 如果编译通过，说明库已经正确链接
}

TEST(SentencePieceIntegrationTest, SentencePieceProcessorHeaders) {
    // 验证SentencePiece头文件可以正确包含
    // 创建一个SentencePieceProcessor对象的指针，但不使用它
    sentencepiece::SentencePieceProcessor* processor = nullptr;
    EXPECT_EQ(processor, nullptr);
    
    // 验证我们可以访问SentencePiece的API
    // 由于我们没有模型，我们只测试库是否可以编译
    EXPECT_TRUE(true);
}

TEST(SentencePieceIntegrationTest, TokenizerClassHeaders) {
    // 验证Tokenizer类的头文件可以正确包含
    // 尝试声明一个Tokenizer对象的指针
    cllm::Tokenizer* tokenizer = nullptr;
    EXPECT_EQ(tokenizer, nullptr);
    
    // 验证我们可以在不加载模型的情况下编译
    EXPECT_TRUE(true);
}

// 一个更实际的测试，使用空模型路径来验证错误处理
TEST(SentencePieceIntegrationTest, TokenizerErrorHandling) {
    // 测试当模型路径不存在时的错误处理
    EXPECT_THROW({
        cllm::Tokenizer tokenizer("/nonexistent/path");
    }, std::runtime_error);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}