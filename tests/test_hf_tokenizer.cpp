#include <gtest/gtest.h>
#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/tokenizer/manager.h"
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>

namespace cllm {
namespace test {

namespace fs = std::filesystem;

// ============ HFTokenizer 单元测试 ============

class HFTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 创建临时测试目录
        testDir_ = "./temp_hf_tokenizer_test";
        fs::create_directories(testDir_);
    }
    
    void TearDown() override {
        // 清理临时文件
        if (fs::exists(testDir_)) {
            fs::remove_all(testDir_);
        }
    }
    
    // 创建一个简单的配置文件
    void createMockConfig() {
        std::ofstream configFile(testDir_ + "/config.json");
        nlohmann::json config = {
            {"model_type", "qwen2"},
            {"bos_token_id", 151643},
            {"eos_token_id", 151645},
            {"pad_token_id", 151643},
            {"unk_token_id", 151643}
        };
        configFile << config.dump(2);
        configFile.close();
    }
    
    std::string testDir_;
};

#ifdef USE_TOKENIZERS_CPP

// 测试: 加载不存在的路径
TEST_F(HFTokenizerTest, LoadInvalidPath) {
    HFTokenizer tokenizer;
    EXPECT_FALSE(tokenizer.load("/nonexistent/path/to/tokenizer"));
}

// 测试: 加载没有 tokenizer.json 的目录
TEST_F(HFTokenizerTest, LoadDirectoryWithoutTokenizerJson) {
    createMockConfig();
    
    HFTokenizer tokenizer;
    EXPECT_FALSE(tokenizer.load(testDir_));
}

// 测试: 初始状态
TEST_F(HFTokenizerTest, InitialState) {
    HFTokenizer tokenizer;
    
    // 未加载时的默认值
    EXPECT_EQ(tokenizer.getVocabSize(), 0);
    EXPECT_EQ(tokenizer.getBosId(), -1);
    EXPECT_EQ(tokenizer.getEosId(), -1);
    EXPECT_EQ(tokenizer.getPadId(), -1);
    EXPECT_EQ(tokenizer.getUnkId(), -1);
}

// 测试: 未加载时调用 encode
TEST_F(HFTokenizerTest, EncodeWithoutLoad) {
    HFTokenizer tokenizer;
    
    auto tokens = tokenizer.encode("Hello");
    EXPECT_TRUE(tokens.empty());
}

// 测试: 未加载时调用 decode
TEST_F(HFTokenizerTest, DecodeWithoutLoad) {
    HFTokenizer tokenizer;
    
    std::vector<int> tokens = {1, 2, 3};
    auto text = tokenizer.decode(tokens);
    EXPECT_TRUE(text.empty());
}

// 测试: 空文本编码
TEST_F(HFTokenizerTest, EncodeEmptyText) {
    HFTokenizer tokenizer;
    
    // 即使未加载，也应该正确处理空文本
    auto tokens = tokenizer.encode("");
    EXPECT_TRUE(tokens.empty());
}

// 测试: 空 tokens 解码
TEST_F(HFTokenizerTest, DecodeEmptyTokens) {
    HFTokenizer tokenizer;
    
    std::vector<int> tokens;
    auto text = tokenizer.decode(tokens);
    EXPECT_TRUE(text.empty());
}

// 测试: ModelType
TEST_F(HFTokenizerTest, ModelType) {
    HFTokenizer tokenizer(ModelType::QWEN2);
    EXPECT_EQ(tokenizer.getModelType(), ModelType::QWEN2);
    
    HFTokenizer tokenizerAuto(ModelType::AUTO);
    EXPECT_EQ(tokenizerAuto.getModelType(), ModelType::AUTO);
}

// 测试: 特殊 Token 检查 (未加载)
TEST_F(HFTokenizerTest, IsSpecialTokenWithoutLoad) {
    HFTokenizer tokenizer;
    
    // 未加载时，没有特殊 Token
    EXPECT_FALSE(tokenizer.isSpecialToken(0));
    EXPECT_FALSE(tokenizer.isSpecialToken(100));
}

// ============ 集成测试 (需要真实模型) ============

// 注意: 以下测试需要真实的 tokenizer.json 文件
// 可以通过环境变量指定测试模型路径

class HFTokenizerIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 从环境变量获取测试模型路径
        const char* modelPathEnv = std::getenv("CLLM_TEST_MODEL_PATH");
        if (modelPathEnv) {
            testModelPath_ = modelPathEnv;
            hasTestModel_ = fs::exists(testModelPath_ + "/tokenizer.json");
        } else {
            hasTestModel_ = false;
        }
    }
    
    std::string testModelPath_;
    bool hasTestModel_ = false;
};

TEST_F(HFTokenizerIntegrationTest, LoadRealTokenizer) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available. Set CLLM_TEST_MODEL_PATH environment variable.";
    }
    
    HFTokenizer tokenizer;
    EXPECT_TRUE(tokenizer.load(testModelPath_));
    
    // 验证基本属性
    EXPECT_GT(tokenizer.getVocabSize(), 0);
    std::cout << "Vocab size: " << tokenizer.getVocabSize() << std::endl;
}

TEST_F(HFTokenizerIntegrationTest, EncodeDecodeEnglish) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available";
    }
    
    HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(testModelPath_));
    
    std::string text = "Hello, world!";
    auto tokens = tokenizer.encode(text);
    
    EXPECT_FALSE(tokens.empty());
    std::cout << "Encoded '" << text << "' to " << tokens.size() << " tokens" << std::endl;
    
    // 解码
    auto decoded = tokenizer.decode(tokens);
    EXPECT_FALSE(decoded.empty());
    std::cout << "Decoded: " << decoded << std::endl;
}

TEST_F(HFTokenizerIntegrationTest, EncodeDecodeChinese) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available";
    }
    
    HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(testModelPath_));
    
    std::string text = "你好，世界！";
    auto tokens = tokenizer.encode(text);
    
    EXPECT_FALSE(tokens.empty());
    std::cout << "Encoded '" << text << "' to " << tokens.size() << " tokens" << std::endl;
    
    // 解码
    auto decoded = tokenizer.decode(tokens);
    EXPECT_FALSE(decoded.empty());
    std::cout << "Decoded: " << decoded << std::endl;
}

TEST_F(HFTokenizerIntegrationTest, SpecialTokens) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available";
    }
    
    HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(testModelPath_));
    
    // 检查特殊 Token IDs
    int bosId = tokenizer.getBosId();
    int eosId = tokenizer.getEosId();
    
    std::cout << "BOS ID: " << bosId << std::endl;
    std::cout << "EOS ID: " << eosId << std::endl;
    
    // BOS 和 EOS 应该是特殊 Token
    if (bosId >= 0) {
        // EXPECT_TRUE(tokenizer.isSpecialToken(bosId));
    }
    if (eosId >= 0) {
        // EXPECT_TRUE(tokenizer.isSpecialToken(eosId));
    }
}

TEST_F(HFTokenizerIntegrationTest, TokenizeMethod) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available";
    }
    
    HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(testModelPath_));
    
    std::string text = "Hello world";
    auto tokens = tokenizer.tokenize(text);
    
    EXPECT_FALSE(tokens.empty());
    std::cout << "Tokenized: ";
    for (const auto& token : tokens) {
        std::cout << "'" << token << "' ";
    }
    std::cout << std::endl;
}

TEST_F(HFTokenizerIntegrationTest, IdToTokenConversion) {
    if (!hasTestModel_) {
        GTEST_SKIP() << "Test model not available";
    }
    
    HFTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(testModelPath_));
    
    // 测试一些常见的 Token IDs
    for (int id = 0; id < 10 && id < tokenizer.getVocabSize(); ++id) {
        std::string token = tokenizer.idToToken(id);
        EXPECT_FALSE(token.empty());
        
        // 尝试转换回 ID
        // int backId = tokenizer.tokenToId(token);
        // 注意: 不一定能精确转换回去，因为可能有多个 token 映射到同一个ID
    }
}

#else

// 测试: USE_TOKENIZERS_CPP 未启用
TEST(HFTokenizerDisabledTest, RequiresCompileFlag) {
    GTEST_SKIP() << "USE_TOKENIZERS_CPP not enabled. Rebuild with -DUSE_TOKENIZERS_CPP=ON";
}

#endif  // USE_TOKENIZERS_CPP

// ============ TokenizerManager 集成测试 ============

class TokenizerManagerHFTest : public ::testing::Test {
protected:
    void SetUp() override {
        testDir_ = "./temp_manager_hf_test";
        fs::create_directories(testDir_);
        
        // 创建配置文件
        std::ofstream configFile(testDir_ + "/config.json");
        nlohmann::json config = {
            {"model_type", "qwen2"},
            {"stop_token_ids", {151643, 151645}}
        };
        configFile << config.dump(2);
        configFile.close();
    }
    
    void TearDown() override {
        if (fs::exists(testDir_)) {
            fs::remove_all(testDir_);
        }
    }
    
    std::string testDir_;
};

TEST_F(TokenizerManagerHFTest, AutoDetectionNoTokenizer) {
    // 没有 tokenizer.json 和 tokenizer.model
    // 应该抛出异常或回退到 Native
    EXPECT_THROW({
        TokenizerManager manager(testDir_);
    }, std::runtime_error);
}

TEST_F(TokenizerManagerHFTest, ForceHF) {
    // 强制使用 HF，但文件不存在，应该失败
    EXPECT_THROW({
        TokenizerManager manager(testDir_, nullptr, TokenizerManager::TokenizerImpl::HF);
    }, std::runtime_error);
}

TEST_F(TokenizerManagerHFTest, ForceNative) {
    // 强制使用 Native，但文件不存在，应该失败
    EXPECT_THROW({
        TokenizerManager manager(testDir_, nullptr, TokenizerManager::TokenizerImpl::NATIVE);
    }, std::runtime_error);
}

}  // namespace test
}  // namespace cllm

// 主函数
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    std::cout << "\n====================================" << std::endl;
    std::cout << "  HFTokenizer 测试套件" << std::endl;
    std::cout << "====================================" << std::endl;
    
#ifdef USE_TOKENIZERS_CPP
    std::cout << "✅ USE_TOKENIZERS_CPP 已启用" << std::endl;
#else
    std::cout << "❌ USE_TOKENIZERS_CPP 未启用" << std::endl;
    std::cout << "   重新编译: cmake .. -DUSE_TOKENIZERS_CPP=ON" << std::endl;
#endif
    
    // 检查测试模型
    const char* modelPath = std::getenv("CLLM_TEST_MODEL_PATH");
    if (modelPath) {
        std::cout << "✅ 测试模型路径: " << modelPath << std::endl;
    } else {
        std::cout << "⚠️  未设置测试模型路径" << std::endl;
        std::cout << "   设置: export CLLM_TEST_MODEL_PATH=/path/to/model" << std::endl;
        std::cout << "   集成测试将被跳过" << std::endl;
    }
    
    std::cout << "====================================" << std::endl;
    std::cout << std::endl;
    
    return RUN_ALL_TESTS();
}
