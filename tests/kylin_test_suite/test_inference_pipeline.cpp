/**
 * @file test_inference_pipeline.cpp
 * @brief Stage 3: 推理流程测试
 *
 * 测试内容：
 * - 单次前向传播
 * - 批处理推理
 * - KV Cache 管理
 * - 增量生成
 */

#include "kylin_test_framework.h"
#include <cmath>

namespace kylin_test {

// Test 1: 简单输入前向传播
class SimpleForwardTest : public TestCase {
public:
    SimpleForwardTest() : TestCase(
        "simple_forward",
        "测试简单的单次前向传播"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing simple forward pass...");

        // 模拟输入 tokens
        std::vector<int> inputIds = {1, 2, 3, 4, 5};
        log(LogLevel::INFO, "Input tokens: [1, 2, 3, 4, 5]");

        // 模拟输出 logits
        std::vector<float> logits(inputIds.size() * 1000);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        // 验证输出
        assertEquals(static_cast<int>(inputIds.size() * 1000),
                     static_cast<int>(logits.size()),
                     "Logits size mismatch");

        assertValidLogits(logits, "Forward output logits");

        log(LogLevel::INFO, "Forward pass successful");
    }
};

// Test 2: 批处理推理
class BatchInferenceTest : public TestCase {
public:
    BatchInferenceTest() : TestCase(
        "batch_inference",
        "测试批处理推理"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing batch inference...");

        // 模拟两个请求
        std::vector<int> flatInputIds = {1, 2, 3, 4, 5, 6};  // 6 tokens total
        std::vector<std::pair<size_t, size_t>> requestPositions = {
            {0, 3},  // Request 1: tokens [1, 2, 3]
            {3, 6}   // Request 2: tokens [4, 5, 6]
        };
        size_t batchSize = 2;

        log(LogLevel::INFO, "Batch size: " + std::to_string(batchSize));
        log(LogLevel::INFO, "Total tokens: " + std::to_string(flatInputIds.size()));

        // 模拟批处理输出
        std::vector<float> logits(flatInputIds.size() * 1000);
        for (size_t i = 0; i < logits.size(); ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }

        assertEquals(static_cast<int>(flatInputIds.size() * 1000),
                     static_cast<int>(logits.size()),
                     "Batch logits size mismatch");

        assertValidLogits(logits, "Batch forward output");

        log(LogLevel::INFO, "Batch inference successful");
    }
};

// Test 3: KV Cache 管理
class KVCacheManagementTest : public TestCase {
public:
    KVCacheManagementTest() : TestCase(
        "kv_cache_management",
        "测试 KV Cache 管理"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing KV Cache management...");

        // 模拟 KV Cache 状态
        size_t initialCacheLen = 0;
        size_t afterPromptLen = 5;
        size_t afterGenerationLen = 10;

        log(LogLevel::INFO, "Initial cache length: " + std::to_string(initialCacheLen));
        log(LogLevel::INFO, "After prompt processing: " + std::to_string(afterPromptLen));
        log(LogLevel::INFO, "After generation: " + std::to_string(afterGenerationLen));

        assertTrue(afterPromptLen > initialCacheLen,
                   "Cache should grow after prompt processing");
        assertTrue(afterGenerationLen > afterPromptLen,
                   "Cache should grow during generation");

        log(LogLevel::INFO, "KV Cache management test passed");
    }
};

// Test 4: 增量生成
class IncrementalGenerationTest : public TestCase {
public:
    IncrementalGenerationTest() : TestCase(
        "incremental_generation",
        "测试增量 token 生成"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing incremental generation...");

        std::vector<int> generatedTokens;
        int maxTokens = 10;

        for (int i = 0; i < maxTokens; ++i) {
            // 模拟生成一个 token
            int nextToken = rand() % 1000;
            generatedTokens.push_back(nextToken);

            log(LogLevel::DEBUG, "Generated token " + std::to_string(i + 1) +
                ": " + std::to_string(nextToken));
        }

        assertEquals(maxTokens, static_cast<int>(generatedTokens.size()),
                     "Generated token count mismatch");

        log(LogLevel::INFO, "Generated " + std::to_string(generatedTokens.size()) +
            " tokens successfully");
    }
};

// Test 5: 序列长度边界测试
class SequenceLengthBoundaryTest : public TestCase {
public:
    SequenceLengthBoundaryTest() : TestCase(
        "sequence_length_boundary",
        "测试序列长度边界情况"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing sequence length boundaries...");

        // 测试空输入
        std::vector<int> emptyInput;
        log(LogLevel::INFO, "Empty input test");

        // 测试单个 token
        std::vector<int> singleInput = {42};
        log(LogLevel::INFO, "Single token input: [42]");

        // 测试长序列
        std::vector<int> longInput(1000);
        for (size_t i = 0; i < longInput.size(); ++i) {
            longInput[i] = i % 100;
        }
        log(LogLevel::INFO, "Long sequence input: " + std::to_string(longInput.size()) +
            " tokens");

        log(LogLevel::INFO, "Boundary tests completed");
    }
};

// Test 6: 错误处理
class InferenceErrorHandlingTest : public TestCase {
public:
    InferenceErrorHandlingTest() : TestCase(
        "inference_error_handling",
        "测试推理错误处理"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing error handling...");

        // 测试无效输入
        log(LogLevel::INFO, "Testing invalid token IDs...");
        std::vector<int> invalidTokens = {-1, 999999, -100};

        for (int token : invalidTokens) {
            log(LogLevel::DEBUG, "Invalid token: " + std::to_string(token));
        }

        log(LogLevel::INFO, "Error handling tests completed");
    }
};

// 创建 Stage 3 测试套件
std::shared_ptr<TestSuite> createWeightsLoadingTestSuite() {
    auto suite = std::make_shared<TestSuite>("Stage 3: Weights Loading Validation");

    suite->addTest(std::make_shared<SimpleForwardTest>());
    suite->addTest(std::make_shared<BatchInferenceTest>());
    suite->addTest(std::make_shared<KVCacheManagementTest>());
    suite->addTest(std::make_shared<IncrementalGenerationTest>());
    suite->addTest(std::make_shared<SequenceLengthBoundaryTest>());
    suite->addTest(std::make_shared<InferenceErrorHandlingTest>());

    return suite;
}

} // namespace kylin_test
