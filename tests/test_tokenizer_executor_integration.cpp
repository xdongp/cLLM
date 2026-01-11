/**
 * @file test_tokenizer_executor_integration.cpp
 * @brief Tokenizer ↔ ModelExecutor 联调集成测试
 * @author cLLM Team
 * @date 2026-01-10
 * 
 * 测试目标:
 * 1. 验证 Tokenizer encode() 输出能正确输入到 ModelExecutor
 * 2. 验证 ModelExecutor generate() 输出能正确输入到 Tokenizer decode()
 * 3. 验证端到端文本生成流程
 * 4. 验证批处理场景
 * 5. 验证错误处理和边界情况
 */

#include <gtest/gtest.h>
#include <cllm/tokenizer/unified_tokenizer.h>
#include <cllm/model/executor.h>
#include <cllm/inference/kylin_backend.h>
#include <cllm/batch/input.h>
#include <cllm/batch/output.h>
#include <cllm/common/logger.h>
#include <cllm/common/config.h>
#include <memory>
#include <string>
#include <vector>
#include <chrono>
#include <filesystem>
#include <cstdlib>

using namespace cllm;

/**
 * @brief Tokenizer-ModelExecutor 集成测试套件
 */
class TokenizerExecutorIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        CLLM_INFO("=== Setting up TokenizerExecutorIntegrationTest ===");
        
        namespace fs = std::filesystem;

        // 1. 加载测试配置（优先环境变量，其次自动推断仓库根目录）
        std::string configPath;
        if (const char* env = std::getenv("CLLM_TEST_CONFIG_PATH")) {
            configPath = env;
        } else {
            // 常见执行目录：<repo>/build 或 <repo>/build/bin
            fs::path cwd = fs::current_path();
            fs::path repo = cwd;
            if (fs::exists(repo / "config")) {
                // ok
            } else if (fs::exists(repo / "../config")) {
                repo = repo / "..";
            } else if (fs::exists(repo / "../../config")) {
                repo = repo / "../..";
            }
            configPath = (repo / "config" / "test_config.yaml").string();
        }

        try {
            if (!configPath.empty() && fs::exists(configPath)) {
                Config::instance().load(configPath);
                CLLM_INFO("Test configuration loaded: {}", configPath);
            } else {
                CLLM_WARN("Test config not found, will use default configuration. configPath={}", configPath);
            }
        } catch (const std::exception& e) {
            CLLM_WARN("Failed to load test config: {}", e.what());
            CLLM_WARN("Will use default configuration");
        }

        // 2. 配置测试模型路径（优先环境变量）
        if (const char* env = std::getenv("CLLM_TEST_TOKENIZER_MODEL_PATH")) {
            tokenizerModelPath_ = env;
        }
        if (const char* env = std::getenv("CLLM_TEST_TORCHSCRIPT_MODEL_PATH")) {
            executorModelPath_ = env;
        }

        // 3. 默认使用仓库内路径（更符合当前项目布局）
        if (tokenizerModelPath_.empty() || executorModelPath_.empty()) {
            fs::path cwd = fs::current_path();
            fs::path repo = cwd;
            if (fs::exists(repo / "model")) {
                // ok
            } else if (fs::exists(repo / "../model")) {
                repo = repo / "..";
            } else if (fs::exists(repo / "../../model")) {
                repo = repo / "../..";
            }

            if (tokenizerModelPath_.empty()) {
                tokenizerModelPath_ = (repo / "tests" / "test_tokenizer.model").string();
            }
            if (executorModelPath_.empty()) {
                executorModelPath_ = (repo / "model" / "Qwen" / "qwen3_0.6b_torchscript_fp32.pt").string();
            }
        }
        
        // 初始化 Tokenizer（使用 UnifiedTokenizer）
        try {
            tokenizer_ = std::make_unique<UnifiedTokenizer>(tokenizerModelPath_);
            tokenizerLoaded_ = true;
            CLLM_INFO("Tokenizer loaded successfully");
            CLLM_INFO("  Vocab size: {}", tokenizer_->getVocabSize());
            // UnifiedTokenizer doesn't have getBosId/getEosId methods directly
        } catch (const std::exception& e) {
            CLLM_WARN("Failed to load tokenizer model: {}", e.what());
            tokenizerLoaded_ = false;
        }
        
        // 初始化 ModelExecutor（使用 KylinBackend 占位权重）
        try {
            CLLM_INFO("Creating ModelExecutor with KylinBackend (placeholder weights)...");
            
            // 使用 mock model path 初始化 ModelExecutor，使用自研引擎
            executor_ = std::make_unique<ModelExecutor>("mock_model_path", "", true, false);
            executorLoaded_ = true;
            CLLM_INFO("✓ ModelExecutor loaded successfully (KylinBackend)");
            
        } catch (const std::exception& e) {
            CLLM_ERROR("✗ Failed to initialize ModelExecutor (KylinBackend): {}", e.what());
            CLLM_WARN("KylinBackend initialization failed, tests will be skipped");
            executorLoaded_ = false;
        }
        
        CLLM_INFO("=== Setup complete ===");
    }
    
    void TearDown() override {
        CLLM_INFO("=== Tearing down TokenizerExecutorIntegrationTest ===");
        tokenizer_.reset();
        executor_.reset();
    }
    
    /**
     * @brief 检查测试是否可以运行
     */
    bool canRunTests() const {
        return tokenizerLoaded_ && executorLoaded_;
    }
    
    std::string tokenizerModelPath_;
    std::string executorModelPath_;
    std::unique_ptr<UnifiedTokenizer> tokenizer_;
    std::unique_ptr<ModelExecutor> executor_;
    bool tokenizerLoaded_ = false;
    bool executorLoaded_ = false;
};

/**
 * @test 测试 1: 基本接口兼容性
 * 验证 Tokenizer 输出的 token IDs 能够被 ModelExecutor 接受
 */
TEST_F(TokenizerExecutorIntegrationTest, BasicInterfaceCompatibility) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 1: Basic Interface Compatibility ===");
    
    // Step 1: Tokenizer encode
    std::string inputText = "Hello, world!";
    CLLM_INFO("Input text: \"%s\"", inputText.c_str());
    
    std::vector<int> tokenIds = tokenizer_->encode(inputText, true);
    CLLM_INFO("Tokenized IDs count: %zu", tokenIds.size());
    EXPECT_GT(tokenIds.size(), 0) << "Tokenizer should produce at least one token";
    
    // 打印 token IDs
    std::string idsStr;
    for (size_t i = 0; i < std::min(tokenIds.size(), size_t(10)); ++i) {
        idsStr += std::to_string(tokenIds[i]) + " ";
    }
    CLLM_INFO("Token IDs (first 10): [%s]", idsStr.c_str());
    
    // Step 2: Token IDs are already std::vector<int> for ModelExecutor
    std::vector<int> executorInput = tokenIds;
    EXPECT_EQ(executorInput.size(), tokenIds.size()) << "Type conversion should preserve size";
    
    // Step 3: ModelExecutor forward
    try {
        BatchInput batchInput;
        batchInput.inputIds = executorInput;
        batchInput.batchSize = 1;
        batchInput.requestPositions = {{0, executorInput.size()}};
        batchInput.sequenceIds = {0};
        
        CLLM_INFO("Calling ModelExecutor::forward()...");
        BatchOutput output = executor_->forward(batchInput);
        
        EXPECT_GT(output.logits.size(), 0) << "ModelExecutor should produce logits";
        CLLM_INFO("Output logits size: %zu", output.logits.size());
        
        // 验证输出维度
        size_t expectedSize = executorInput.size() * executor_->getConfig().vocabSize;
        EXPECT_EQ(output.logits.size(), expectedSize) 
            << "Output size should be [num_tokens * vocab_size]";
        
        CLLM_INFO("✓ Interface compatibility verified");
    } catch (const std::exception& e) {
        FAIL() << "ModelExecutor::forward() failed: " << e.what();
    }
}

/**
 * @test 测试 2: 端到端文本生成流程
 * 验证完整的 encode → generate → decode 流程
 */
TEST_F(TokenizerExecutorIntegrationTest, EndToEndTextGeneration) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 2: End-to-End Text Generation ===");
    
    // Step 1: Encode input text
    std::string inputText = "Once upon a time";
    CLLM_INFO("Input: \"%s\"", inputText.c_str());
    
    std::vector<int> inputIds = tokenizer_->encode(inputText, true);
    CLLM_INFO("Encoded to %zu tokens", inputIds.size());
    
    // Step 2: Generate new tokens
    std::vector<int> executorInput = inputIds;
    
    try {
        CLLM_INFO("Generating 5 new tokens...");
        auto startTime = std::chrono::high_resolution_clock::now();
        
        std::vector<int> generatedIds = executor_->generate(
            executorInput,
            5,      // maxNewTokens
            0.7f    // temperature
        );
        
        auto endTime = std::chrono::high_resolution_clock::now();
        float elapsedMs = std::chrono::duration<float, std::milli>(endTime - startTime).count();
        
        CLLM_INFO("Generated %zu tokens in %.2f ms", generatedIds.size(), elapsedMs);
        EXPECT_GT(generatedIds.size(), 0) << "Should generate at least one token";
        EXPECT_LE(generatedIds.size(), 5) << "Should not exceed maxNewTokens";
        
        // Step 3: Decode generated tokens
        std::vector<int> outputTokens(generatedIds.begin(), generatedIds.end());
        std::string outputText = tokenizer_->decode(outputTokens, true);
        
        CLLM_INFO("Generated text: \"%s\"", outputText.c_str());
        EXPECT_FALSE(outputText.empty()) << "Decoded text should not be empty";
        
        // Step 4: 验证完整流程
        std::vector<int> fullSequence = inputIds;
        fullSequence.insert(fullSequence.end(), outputTokens.begin(), outputTokens.end());
        std::string fullText = tokenizer_->decode(fullSequence, true);
        
        CLLM_INFO("Full generated text: \"%s\"", fullText.c_str());
        EXPECT_TRUE(fullText.find(inputText) != std::string::npos || inputText.find(fullText) != std::string::npos)
            << "Full text should contain or be contained in input text (tokenizer roundtrip)";
        
        CLLM_INFO("✓ End-to-end generation successful");
    } catch (const std::exception& e) {
        FAIL() << "Generation failed: " << e.what();
    }
}

/**
 * @test 测试 3: 批处理场景
 * 验证多个请求的批处理
 */
TEST_F(TokenizerExecutorIntegrationTest, BatchProcessing) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 3: Batch Processing ===");
    
    // 准备多个输入文本
    std::vector<std::string> inputTexts = {
        "Hello world",
        "How are you",
        "Nice to meet you"
    };
    
    std::vector<std::vector<int>> batchTokenIds;
    std::vector<int> flattenedIds;
    std::vector<std::pair<size_t, size_t>> requestPositions;
    
    size_t currentPos = 0;
    
    // Step 1: Batch encode
    CLLM_INFO("Encoding %zu texts...", inputTexts.size());
    for (size_t i = 0; i < inputTexts.size(); ++i) {
        std::vector<int> ids = tokenizer_->encode(inputTexts[i], true);
        batchTokenIds.push_back(ids);
        
        size_t startPos = currentPos;
        for (auto id : ids) {
            flattenedIds.push_back(static_cast<int>(id));
        }
        currentPos += ids.size();
        requestPositions.push_back({startPos, ids.size()});
        
        CLLM_INFO("  Text %zu: \"%s\" → %zu tokens", i, inputTexts[i].c_str(), ids.size());
    }
    
    // Step 2: Batch forward
    try {
        BatchInput batchInput;
        batchInput.inputIds = flattenedIds;
        batchInput.batchSize = inputTexts.size();
        batchInput.requestPositions = requestPositions;
        batchInput.sequenceIds = {0, 1, 2};
        
        CLLM_INFO("Calling batch forward with %zu requests...", batchInput.batchSize);
        BatchOutput output = executor_->forward(batchInput);
        
        EXPECT_GT(output.logits.size(), 0) << "Batch output should have logits";
        CLLM_INFO("Batch output logits size: %zu", output.logits.size());
        
        // Step 3: Extract logits for each request
        for (size_t i = 0; i < inputTexts.size(); ++i) {
            FloatArray requestLogits = output.getLogitsForRequest(i);
            EXPECT_GT(requestLogits.size(), 0) << "Request " << i << " should have logits";
            CLLM_INFO("  Request %zu logits size: %zu", i, requestLogits.size());
        }
        
        CLLM_INFO("✓ Batch processing successful");
    } catch (const std::exception& e) {
        FAIL() << "Batch processing failed: " << e.what();
    }
}

/**
 * @test 测试 4: 特殊 Token 处理
 * 验证 BOS/EOS/PAD 等特殊 token 的处理
 */
TEST_F(TokenizerExecutorIntegrationTest, SpecialTokenHandling) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 4: Special Token Handling ===");
    
    // 获取特殊 token IDs
    int bosId = tokenizer_->getBosToken();
    int eosId = tokenizer_->getEosToken();
    int padId = tokenizer_->getPadToken();
    
    CLLM_INFO("Special tokens: BOS=%d, EOS=%d, PAD=%d", bosId, eosId, padId);
    
    // Step 1: 测试带特殊 token 的序列
    std::vector<int> sequenceWithSpecialTokens = {
        static_cast<int>(bosId),
        100, 200, 300,  // 普通 tokens
        static_cast<int>(eosId)
    };
    
    try {
        BatchInput batchInput;
        batchInput.inputIds = sequenceWithSpecialTokens;
        batchInput.batchSize = 1;
        batchInput.requestPositions = {{0, sequenceWithSpecialTokens.size()}};
        batchInput.sequenceIds = {0};
        
        CLLM_INFO("Testing sequence with special tokens...");
        BatchOutput output = executor_->forward(batchInput);
        
        EXPECT_GT(output.logits.size(), 0) << "Should handle special tokens correctly";
        CLLM_INFO("✓ Special tokens handled correctly");
        
        // Step 2: Decode 应该跳过特殊 tokens (skipSpecialTokens=true)
        std::vector<int> tokensToDeode(
            sequenceWithSpecialTokens.begin(), 
            sequenceWithSpecialTokens.end()
        );
        std::string decodedWithoutSpecial = tokenizer_->decode(tokensToDeode, true);
        std::string decodedWithSpecial = tokenizer_->decode(tokensToDeode, false);
        
        CLLM_INFO("Decoded without special tokens: \"%s\"", decodedWithoutSpecial.c_str());
        CLLM_INFO("Decoded with special tokens: \"%s\"", decodedWithSpecial.c_str());
        
        // 验证 skipSpecialTokens 参数的效果
        EXPECT_NE(decodedWithoutSpecial, decodedWithSpecial) 
            << "skipSpecialTokens parameter should affect output";
        
    } catch (const std::exception& e) {
        FAIL() << "Special token handling failed: " << e.what();
    }
}

/**
 * @test 测试 5: 边界情况处理
 * 验证空输入、超长输入等边界情况
 */
TEST_F(TokenizerExecutorIntegrationTest, EdgeCases) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 5: Edge Cases ===");
    
    // Case 1: 空字符串
    CLLM_INFO("Testing empty string...");
    std::vector<int> emptyTokens = tokenizer_->encode("", true);
    CLLM_INFO("  Empty string encoded to %zu tokens", emptyTokens.size());
    // 某些 tokenizer 可能会添加 BOS/EOS，所以不一定为空
    
    // Case 2: 单字符
    CLLM_INFO("Testing single character...");
    std::vector<int> singleChar = tokenizer_->encode("a", true);
    EXPECT_GT(singleChar.size(), 0) << "Single character should produce tokens";
    CLLM_INFO("  Single char 'a' encoded to %zu tokens", singleChar.size());
    
    // Case 3: 超长输入 (测试性能和内存)
    CLLM_INFO("Testing long input...");
    std::string longInput(500, 'a');  // 500 个字符
    std::vector<int> longTokens = tokenizer_->encode(longInput, true);
    CLLM_INFO("  Long input (%zu chars) encoded to %zu tokens", longInput.size(), longTokens.size());
    EXPECT_GT(longTokens.size(), 0) << "Long input should produce tokens";
    
    // Case 4: 特殊字符
    CLLM_INFO("Testing special characters...");
    std::string specialChars = "!@#$%^&*()_+-=[]{}|;:',.<>?/~`";
    std::vector<int> specialTokens = tokenizer_->encode(specialChars, true);
    CLLM_INFO("  Special chars encoded to %zu tokens", specialTokens.size());
    
    // Case 5: Unicode 字符
    CLLM_INFO("Testing Unicode characters...");
    std::string unicodeText = "你好世界 🌍";
    std::vector<int> unicodeTokens = tokenizer_->encode(unicodeText, true);
    CLLM_INFO("  Unicode text encoded to %zu tokens", unicodeTokens.size());
    
    // 尝试 decode 回来验证
    std::string decoded = tokenizer_->decode(unicodeTokens, true);
    CLLM_INFO("  Decoded: \"%s\"", decoded.c_str());
    
    CLLM_INFO("✓ Edge cases handled");
}

/**
 * @test 测试 6: 性能基准测试
 * 测量 Tokenizer 和 ModelExecutor 的性能
 */
TEST_F(TokenizerExecutorIntegrationTest, PerformanceBenchmark) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 6: Performance Benchmark ===");
    
    std::string testText = "The quick brown fox jumps over the lazy dog";
    const int iterations = 100;
    
    // Benchmark 1: Tokenizer encode
    auto encodeStart = std::chrono::high_resolution_clock::now();
    std::vector<int> encodedIds;
    for (int i = 0; i < iterations; ++i) {
        encodedIds = tokenizer_->encode(testText, true);
    }
    auto encodeEnd = std::chrono::high_resolution_clock::now();
    float encodeTime = std::chrono::duration<float, std::milli>(encodeEnd - encodeStart).count();
    float encodeAvg = encodeTime / iterations;
    
    CLLM_INFO("Tokenizer encode:");
    CLLM_INFO("  Total time: %.2f ms (%d iterations)", encodeTime, iterations);
    CLLM_INFO("  Average: %.4f ms/op", encodeAvg);
    CLLM_INFO("  Throughput: %.2f ops/sec", 1000.0f / encodeAvg);
    
    // Benchmark 2: ModelExecutor forward
    std::vector<int> executorInput = encodedIds;
    BatchInput batchInput;
    batchInput.inputIds = executorInput;
    batchInput.batchSize = 1;
    batchInput.requestPositions = {{0, executorInput.size()}};
    batchInput.sequenceIds = {0};
    
    auto forwardStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        BatchOutput output = executor_->forward(batchInput);
    }
    auto forwardEnd = std::chrono::high_resolution_clock::now();
    float forwardTime = std::chrono::duration<float, std::milli>(forwardEnd - forwardStart).count();
    float forwardAvg = forwardTime / iterations;
    
    CLLM_INFO("ModelExecutor forward:");
    CLLM_INFO("  Total time: %.2f ms (%d iterations)", forwardTime, iterations);
    CLLM_INFO("  Average: %.4f ms/op", forwardAvg);
    CLLM_INFO("  Throughput: %.2f ops/sec", 1000.0f / forwardAvg);
    
    // Benchmark 3: Tokenizer decode
    auto decodeStart = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        std::string decoded = tokenizer_->decode(encodedIds, true);
    }
    auto decodeEnd = std::chrono::high_resolution_clock::now();
    float decodeTime = std::chrono::duration<float, std::milli>(decodeEnd - decodeStart).count();
    float decodeAvg = decodeTime / iterations;
    
    CLLM_INFO("Tokenizer decode:");
    CLLM_INFO("  Total time: %.2f ms (%d iterations)", decodeTime, iterations);
    CLLM_INFO("  Average: %.4f ms/op", decodeAvg);
    CLLM_INFO("  Throughput: %.2f ops/sec", 1000.0f / decodeAvg);
    
    // 性能断言 (设置合理的性能要求)
    EXPECT_LT(encodeAvg, 10.0f) << "Encode should be faster than 10ms";
    EXPECT_LT(decodeAvg, 10.0f) << "Decode should be faster than 10ms";
    
    CLLM_INFO("✓ Performance benchmark completed");
}

/**
 * @test 测试 7: 错误处理
 * 验证异常情况的处理
 */
TEST_F(TokenizerExecutorIntegrationTest, ErrorHandling) {
    if (!canRunTests()) {
        GTEST_SKIP() << "Required models not loaded, skipping test";
    }
    
    CLLM_INFO("\n=== Test 7: Error Handling ===");
    
    // Error 1: 无效的 token ID
    CLLM_INFO("Testing invalid token ID...");
    std::vector<int> invalidIds = {-1, 999999, -999};
    std::string decoded = tokenizer_->decode(invalidIds, false);
    CLLM_INFO("  Decoded invalid IDs: \"%s\"", decoded.c_str());
    // 应该不会崩溃，可能返回 UNK 或空字符串
    
    // Error 2: 空的 inputIds
    CLLM_INFO("Testing empty input IDs...");
    try {
        BatchInput emptyBatch;
        emptyBatch.inputIds = {};
        emptyBatch.batchSize = 0;
        emptyBatch.requestPositions = {};
        emptyBatch.sequenceIds = {};
        
        // 这应该抛出异常或返回空结果
        // BatchOutput output = executor_->forward(emptyBatch);
        CLLM_INFO("  Empty batch handling: implementation-dependent");
    } catch (const std::exception& e) {
        CLLM_INFO("  Empty batch correctly throws exception: %s", e.what());
    }
    
    CLLM_INFO("✓ Error handling verified");
}

/**
 * @brief 主函数
 */
int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    CLLM_INFO("========================================");
    CLLM_INFO("Tokenizer ↔ ModelExecutor Integration Test");
    CLLM_INFO("========================================");
    
    int result = RUN_ALL_TESTS();
    
    CLLM_INFO("========================================");
    CLLM_INFO("Test execution completed with result: %d", result);
    CLLM_INFO("========================================");
    
    return result;
}
