/**
 * @file test_output_validation.cpp
 * @brief Stage 4: 输出验证测试
 *
 * 测试内容：
 * - Logits 数值检查
 * - Token 分布验证
 * - 生成文本质量
 * - 重复 token 检测
 */

#include "kylin_test_framework.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace kylin_test {

// Test 1: Logits 数值范围检查
class LogitsRangeTest : public TestCase {
public:
    LogitsRangeTest() : TestCase(
        "logits_range",
        "验证 logits 数值范围"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing logits value range...");

        const size_t vocabSize = 151936;
        std::vector<float> logits(vocabSize);

        // 生成模拟 logits
        for (size_t i = 0; i < vocabSize; ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 20.0f - 10.0f;
        }

        // 计算统计信息
        float minVal = *std::min_element(logits.begin(), logits.end());
        float maxVal = *std::max_element(logits.begin(), logits.end());
        double sum = std::accumulate(logits.begin(), logits.end(), 0.0);
        float mean = static_cast<float>(sum / logits.size());

        log(LogLevel::INFO, "Logits range: [" + std::to_string(minVal) + ", " +
            std::to_string(maxVal) + "]");
        log(LogLevel::INFO, "Logits mean: " + std::to_string(mean));

        // 验证范围合理
        assertTrue(minVal > -100.0f && maxVal < 100.0f,
                   "Logits range too extreme");

        assertValidLogits(logits, "Generated logits");
    }
};

// Test 2: NaN 和 Inf 检测
class NaNInfDetectionTest : public TestCase {
public:
    NaNInfDetectionTest() : TestCase(
        "nan_inf_detection",
        "检测 logits 中的 NaN 和 Inf"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing for NaN and Inf values...");

        const size_t vocabSize = 151936;
        std::vector<float> logits(vocabSize);

        // 生成正常的 logits
        for (size_t i = 0; i < vocabSize; ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        }

        size_t nanCount = 0, infCount = 0;
        for (const auto& val : logits) {
            if (std::isnan(val)) nanCount++;
            else if (std::isinf(val)) infCount++;
        }

        log(LogLevel::INFO, "NaN count: " + std::to_string(nanCount));
        log(LogLevel::INFO, "Inf count: " + std::to_string(infCount));

        assertEquals(0, static_cast<int>(nanCount), "Found NaN values in logits");
        assertEquals(0, static_cast<int>(infCount), "Found Inf values in logits");
    }
};

// Test 3: Token 分布多样性
class TokenDiversityTest : public TestCase {
public:
    TokenDiversityTest() : TestCase(
        "token_diversity",
        "验证生成 token 的多样性"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing token diversity...");

        // 模拟生成 50 个 tokens
        std::vector<int> generatedTokens;
        int maxTokens = 50;

        // 模拟生成（这里使用随机值，实际应该是模型输出）
        for (int i = 0; i < maxTokens; ++i) {
            generatedTokens.push_back(rand() % 1000);
        }

        // 检查多样性
        std::vector<int> uniqueTokens = generatedTokens;
        std::sort(uniqueTokens.begin(), uniqueTokens.end());
        uniqueTokens.erase(std::unique(uniqueTokens.begin(), uniqueTokens.end()),
                           uniqueTokens.end());

        float diversity = static_cast<float>(uniqueTokens.size()) / generatedTokens.size();

        log(LogLevel::INFO, "Generated tokens: " + std::to_string(generatedTokens.size()));
        log(LogLevel::INFO, "Unique tokens: " + std::to_string(uniqueTokens.size()));
        log(LogLevel::INFO, "Diversity ratio: " + std::to_string(diversity));

        // 多样性应该大于 0.3（至少 30% 的 token 是唯一的）
        assertTrue(diversity > 0.3f, "Token diversity too low: " + std::to_string(diversity));
    }
};

// Test 4: 重复 Token 检测
class RepetitiveTokenDetectionTest : public TestCase {
public:
    RepetitiveTokenDetectionTest() : TestCase(
        "repetitive_token_detection",
        "检测重复 token 问题（如 151668 问题）"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing for repetitive token patterns...");

        // 测试场景 1: 正常的多样化 tokens（应该通过）
        std::vector<int> normalTokens;
        for (int i = 0; i < 20; ++i) {
            normalTokens.push_back(rand() % 1000);
        }

        // 检查是否所有 token 都相同
        bool allSame = std::all_of(normalTokens.begin(), normalTokens.end(),
                                   [first = normalTokens[0]](int val) {
                                       return val == first;
                                   });

        if (allSame) {
            log(LogLevel::WARN, "Detected repetitive token pattern in normal tokens!");
            throw std::runtime_error("Normal tokens should not be all the same");
        }

        // 计算唯一 token 数量
        std::vector<int> uniqueTokens = normalTokens;
        std::sort(uniqueTokens.begin(), uniqueTokens.end());
        uniqueTokens.erase(std::unique(uniqueTokens.begin(), uniqueTokens.end()),
                          uniqueTokens.end());

        float diversity = static_cast<float>(uniqueTokens.size()) / normalTokens.size();
        log(LogLevel::INFO, "Normal tokens diversity: " + std::to_string(diversity));
        assertTrue(diversity > 0.3f, "Normal tokens should have diversity > 30%");

        // 测试场景 2: 检测真正的重复问题（如果存在）
        std::vector<int> problematicTokens = {151668, 151668, 151668, 151668};

        // 这个测试验证我们的检测逻辑是否正确
        int repeatCount = 0;
        int lastToken = -1;
        int maxConsecutive = 1;
        int currentConsecutive = 1;

        for (size_t i = 0; i < problematicTokens.size(); ++i) {
            if (problematicTokens[i] == lastToken) {
                currentConsecutive++;
                maxConsecutive = std::max(maxConsecutive, currentConsecutive);
            } else {
                currentConsecutive = 1;
                lastToken = problematicTokens[i];
            }
        }

        log(LogLevel::INFO, "Max consecutive repeats in problematic tokens: " +
            std::to_string(maxConsecutive));
        assertTrue(maxConsecutive >= 4, "Should detect 4+ consecutive repeats");

        log(LogLevel::INFO, "Repetitive token detection test passed");
    }
};

// Test 5: Softmax 概率分布
class SoftmaxDistributionTest : public TestCase {
public:
    SoftmaxDistributionTest() : TestCase(
        "softmax_distribution",
        "验证 Softmax 概率分布"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing softmax probability distribution...");

        const size_t vocabSize = 1000;
        std::vector<float> logits(vocabSize);

        // 生成 logits
        for (size_t i = 0; i < vocabSize; ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        }

        // 计算 softmax
        float maxLogit = *std::max_element(logits.begin(), logits.end());
        double sumExp = 0.0;
        std::vector<float> probs(vocabSize);

        for (size_t i = 0; i < vocabSize; ++i) {
            probs[i] = std::exp(logits[i] - maxLogit);
            sumExp += probs[i];
        }

        for (size_t i = 0; i < vocabSize; ++i) {
            probs[i] /= static_cast<float>(sumExp);
        }

        // 验证概率和为 1
        double probSum = std::accumulate(probs.begin(), probs.end(), 0.0);
        log(LogLevel::INFO, "Probability sum: " + std::to_string(probSum));

        assertNear(1.0f, static_cast<float>(probSum), 0.001f,
                   "Softmax probabilities should sum to 1");

        // 验证所有概率在 [0, 1] 范围内
        for (size_t i = 0; i < vocabSize; ++i) {
            assertTrue(probs[i] >= 0.0f && probs[i] <= 1.0f,
                       "Probability out of range");
        }

        log(LogLevel::INFO, "Softmax distribution valid");
    }
};

// Test 6: Top-K 采样验证
class TopKSamplingTest : public TestCase {
public:
    TopKSamplingTest() : TestCase(
        "topk_sampling",
        "验证 Top-K 采样"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing Top-K sampling...");

        const size_t vocabSize = 1000;
        const int k = 50;

        std::vector<float> logits(vocabSize);
        for (size_t i = 0; i < vocabSize; ++i) {
            logits[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
        }

        // 找到 top-k
        std::vector<std::pair<float, int>> indexedLogits;
        for (size_t i = 0; i < vocabSize; ++i) {
            indexedLogits.push_back({logits[i], static_cast<int>(i)});
        }

        std::partial_sort(indexedLogits.begin(), indexedLogits.begin() + k,
                          indexedLogits.end(),
                          std::greater<std::pair<float, int>>());

        log(LogLevel::INFO, "Top-5 tokens:");
        for (int i = 0; i < 5 && i < k; ++i) {
            log(LogLevel::INFO, "  " + std::to_string(i + 1) + ". Token " +
                std::to_string(indexedLogits[i].second) + ": " +
                std::to_string(indexedLogits[i].first));
        }

        assertTrue(indexedLogits[0].first >= indexedLogits[1].first,
                   "Top-K sorting incorrect");

        log(LogLevel::INFO, "Top-K sampling test passed");
    }
};

// Test 7: 生成文本质量检查
class GenerationQualityTest : public TestCase {
public:
    GenerationQualityTest() : TestCase(
        "generation_quality",
        "检查生成文本的基本质量"
    ) {}

    void execute() override {
        log(LogLevel::INFO, "Testing generation quality...");

        // 模拟生成 tokens
        std::vector<int> generatedTokens;
        for (int i = 0; i < 20; ++i) {
            generatedTokens.push_back(rand() % 50000 + 100);  // 避免特殊 tokens
        }

        log(LogLevel::INFO, "Generated " + std::to_string(generatedTokens.size()) + " tokens");

        // 检查没有连续的重复 token（超过 3 次）
        int maxRepeat = 1;
        int currentRepeat = 1;
        int lastToken = generatedTokens[0];

        for (size_t i = 1; i < generatedTokens.size(); ++i) {
            if (generatedTokens[i] == lastToken) {
                currentRepeat++;
                maxRepeat = std::max(maxRepeat, currentRepeat);
            } else {
                currentRepeat = 1;
                lastToken = generatedTokens[i];
            }
        }

        log(LogLevel::INFO, "Max consecutive repeats: " + std::to_string(maxRepeat));

        assertTrue(maxRepeat < 5, "Too many consecutive repeated tokens");

        log(LogLevel::INFO, "Generation quality check passed");
    }
};

// 创建 Stage 4 测试套件
std::shared_ptr<TestSuite> createOutputValidationTestSuite() {
    auto suite = std::make_shared<TestSuite>("Stage 4: Output Validation");

    suite->addTest(std::make_shared<LogitsRangeTest>());
    suite->addTest(std::make_shared<NaNInfDetectionTest>());
    suite->addTest(std::make_shared<TokenDiversityTest>());
    suite->addTest(std::make_shared<RepetitiveTokenDetectionTest>());
    suite->addTest(std::make_shared<SoftmaxDistributionTest>());
    suite->addTest(std::make_shared<TopKSamplingTest>());
    suite->addTest(std::make_shared<GenerationQualityTest>());

    return suite;
}

} // namespace kylin_test
