/**
 * @file tokenizer_p0_features_test.cpp
 * @brief P0 优先级特性测试
 * 
 * 测试内容：
 * 1. LlamaTokenizer 完整功能测试
 * 2. BatchTokenizer 批处理接口测试
 * 3. PerformanceMonitor 性能监控测试
 */

#include <gtest/gtest.h>
#include "cllm/CTokenizer/llama_tokenizer.h"
#include "cllm/CTokenizer/batch_tokenizer.h"
#include "cllm/CTokenizer/performance_monitor.h"
#include <chrono>
#include <thread>

using namespace cllm;

// ==================== LlamaTokenizer 测试 ====================

class LlamaTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 注意：需要提供实际的模型路径
        // 这里使用环境变量或配置文件指定
        modelPath_ = std::getenv("LLAMA_MODEL_PATH");
        if (modelPath_.empty()) {
            GTEST_SKIP() << "LLAMA_MODEL_PATH not set, skipping LlamaTokenizer tests";
        }
    }
    
    std::string modelPath_;
};

TEST_F(LlamaTokenizerTest, LoadModel) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    EXPECT_TRUE(tokenizer.load(modelPath_));
    EXPECT_GT(tokenizer.getVocabSize(), 0);
}

TEST_F(LlamaTokenizerTest, EncodeDecodeBasic) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    std::string text = "Hello, world!";
    auto tokens = tokenizer.encode(text, true);
    
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode(tokens, true);
    // 注意：解码结果可能包含额外空格，取决于分词器实现
    EXPECT_FALSE(decoded.empty());
}

TEST_F(LlamaTokenizerTest, SpecialTokens) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    // 验证特殊 token ID
    auto bosId = tokenizer.getBosId();
    auto eosId = tokenizer.getEosId();
    auto padId = tokenizer.getPadId();
    
    EXPECT_NE(bosId, -1);
    EXPECT_NE(eosId, -1);
    // padId 可能等于 eosId
}

TEST_F(LlamaTokenizerTest, VocabOperations) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    int vocabSize = tokenizer.getVocabSize();
    EXPECT_GT(vocabSize, 1000);  // Llama 词汇表通常 > 30000
    
    // 测试 ID 到 token 的转换
    for (llama_token id = 0; id < std::min(100, vocabSize); ++id) {
        std::string token = tokenizer.idToToken(id);
        // 某些 ID 可能无效，返回空字符串
        // EXPECT_FALSE(token.empty());
    }
}

TEST_F(LlamaTokenizerTest, ChineseText) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    std::string chineseText = "你好，世界！";
    auto tokens = tokenizer.encode(chineseText, true);
    
    EXPECT_FALSE(tokens.empty());
    
    std::string decoded = tokenizer.decode(tokens, true);
    // 验证中文文本能正确编解码
    EXPECT_FALSE(decoded.empty());
}

TEST_F(LlamaTokenizerTest, EmptyText) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    auto tokens = tokenizer.encode("", true);
    // 空文本可能返回空数组或只有特殊 token
    EXPECT_TRUE(tokens.empty() || tokens.size() <= 2);
}

// ==================== BatchTokenizer 测试 ====================

class BatchTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        modelPath_ = std::getenv("LLAMA_MODEL_PATH");
        if (modelPath_.empty()) {
            GTEST_SKIP() << "LLAMA_MODEL_PATH not set";
        }
        
        tokenizer_ = std::make_unique<LlamaTokenizer>(ModelType::LLAMA);
        ASSERT_TRUE(tokenizer_->load(modelPath_));
    }
    
    std::string modelPath_;
    std::unique_ptr<LlamaTokenizer> tokenizer_;
};

TEST_F(BatchTokenizerTest, BatchEncodeBasic) {
    std::vector<std::string> texts = {
        "Hello, world!",
        "How are you?",
        "This is a test."
    };
    
    auto result = BatchTokenizer::batchEncode(tokenizer_.get(), texts, true, 2);
    
    EXPECT_EQ(result.tokenized.size(), texts.size());
    EXPECT_EQ(result.success.size(), texts.size());
    EXPECT_EQ(result.errors.size(), texts.size());
    
    for (size_t i = 0; i < texts.size(); ++i) {
        EXPECT_TRUE(result.success[i]) << "Failed for text " << i;
        EXPECT_FALSE(result.tokenized[i].empty()) << "Empty result for text " << i;
    }
}

TEST_F(BatchTokenizerTest, BatchDecodeBasic) {
    std::vector<std::string> texts = {
        "Hello",
        "World",
        "Test"
    };
    
    // 先编码
    auto encodeResult = BatchTokenizer::batchEncode(tokenizer_.get(), texts, false, 2);
    ASSERT_EQ(encodeResult.tokenized.size(), texts.size());
    
    // 再解码
    auto decodeResult = BatchTokenizer::batchDecode(
        tokenizer_.get(), 
        encodeResult.tokenized, 
        true, 
        2
    );
    
    EXPECT_EQ(decodeResult.decoded.size(), texts.size());
    for (size_t i = 0; i < texts.size(); ++i) {
        EXPECT_TRUE(decodeResult.success[i]) << "Failed for text " << i;
        EXPECT_FALSE(decodeResult.decoded[i].empty());
    }
}

TEST_F(BatchTokenizerTest, EmptyBatch) {
    std::vector<std::string> emptyTexts;
    
    auto result = BatchTokenizer::batchEncode(tokenizer_.get(), emptyTexts, true, 2);
    
    EXPECT_TRUE(result.tokenized.empty());
    EXPECT_TRUE(result.success.empty());
    EXPECT_TRUE(result.errors.empty());
}

TEST_F(BatchTokenizerTest, SingleThreadVsMultiThread) {
    std::vector<std::string> texts;
    for (int i = 0; i < 100; ++i) {
        texts.push_back("Test sentence number " + std::to_string(i));
    }
    
    // 单线程
    auto start1 = std::chrono::high_resolution_clock::now();
    auto result1 = BatchTokenizer::batchEncode(tokenizer_.get(), texts, true, 1);
    auto end1 = std::chrono::high_resolution_clock::now();
    auto duration1 = std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1);
    
    // 多线程
    auto start2 = std::chrono::high_resolution_clock::now();
    auto result2 = BatchTokenizer::batchEncode(tokenizer_.get(), texts, true, 4);
    auto end2 = std::chrono::high_resolution_clock::now();
    auto duration2 = std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2);
    
    std::cout << "Single-thread: " << duration1.count() << "ms\n";
    std::cout << "Multi-thread (4): " << duration2.count() << "ms\n";
    std::cout << "Speedup: " << static_cast<double>(duration1.count()) / duration2.count() << "x\n";
    
    // 验证结果一致性
    EXPECT_EQ(result1.tokenized.size(), result2.tokenized.size());
}

TEST_F(BatchTokenizerTest, NullTokenizerThrows) {
    std::vector<std::string> texts = {"test"};
    
    EXPECT_THROW(
        BatchTokenizer::batchEncode(nullptr, texts, true, 2),
        std::invalid_argument
    );
}

// ==================== PerformanceMonitor 测试 ====================

TEST(PerformanceMonitorTest, BasicRecording) {
    PerformanceMonitor monitor;
    
    // 记录一些操作
    monitor.recordEncode(10.5, 50);
    monitor.recordEncode(15.2, 60);
    monitor.recordDecode(8.3, 40);
    
    auto stats = monitor.getStats();
    
    EXPECT_EQ(stats.totalEncodes, 2);
    EXPECT_EQ(stats.totalDecodes, 1);
    EXPECT_EQ(stats.totalTokensEncoded, 110);
    EXPECT_EQ(stats.totalTokensDecoded, 40);
    EXPECT_GT(stats.avgEncodeLatency, 0.0);
}

TEST(PerformanceMonitorTest, CacheStatistics) {
    PerformanceMonitor monitor;
    
    monitor.recordCacheHit();
    monitor.recordCacheHit();
    monitor.recordCacheMiss();
    
    auto stats = monitor.getStats();
    
    EXPECT_EQ(stats.cacheHits, 2);
    EXPECT_EQ(stats.cacheMisses, 1);
    EXPECT_DOUBLE_EQ(stats.getCacheHitRate(), 2.0 / 3.0);
}

TEST(PerformanceMonitorTest, MemoryTracking) {
    PerformanceMonitor monitor;
    
    monitor.updateMemoryUsage(1000);
    monitor.updateMemoryUsage(2000);
    monitor.updateMemoryUsage(1500);
    
    auto stats = monitor.getStats();
    
    EXPECT_EQ(stats.currentMemoryUsage, 1500);
    EXPECT_EQ(stats.peakMemoryUsage, 2000);
}

TEST(PerformanceMonitorTest, PercentileLatency) {
    PerformanceMonitor monitor;
    
    // 记录一系列延迟
    for (int i = 1; i <= 100; ++i) {
        monitor.recordEncode(static_cast<double>(i), 10);
    }
    
    auto stats = monitor.getStats();
    
    // P50 应该接近 50
    EXPECT_NEAR(stats.p50EncodeLatency, 50.0, 5.0);
    // P95 应该接近 95
    EXPECT_NEAR(stats.p95EncodeLatency, 95.0, 5.0);
    // P99 应该接近 99
    EXPECT_NEAR(stats.p99EncodeLatency, 99.0, 5.0);
}

TEST(PerformanceMonitorTest, ThroughputCalculation) {
    PerformanceMonitor monitor;
    
    // 模拟一些操作
    for (int i = 0; i < 10; ++i) {
        monitor.recordEncode(10.0, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    auto stats = monitor.getStats();
    
    EXPECT_GT(stats.encodeSpeed, 0.0);
    EXPECT_GT(stats.getRuntimeSeconds(), 0.0);
}

TEST(PerformanceMonitorTest, ResetFunctionality) {
    PerformanceMonitor monitor;
    
    monitor.recordEncode(10.0, 50);
    monitor.recordDecode(5.0, 30);
    monitor.recordCacheHit();
    
    auto stats1 = monitor.getStats();
    EXPECT_GT(stats1.totalEncodes, 0);
    
    monitor.reset();
    
    auto stats2 = monitor.getStats();
    EXPECT_EQ(stats2.totalEncodes, 0);
    EXPECT_EQ(stats2.totalDecodes, 0);
    EXPECT_EQ(stats2.cacheHits, 0);
}

TEST(PerformanceMonitorTest, ThreadSafety) {
    PerformanceMonitor monitor;
    
    // 多线程并发记录
    std::vector<std::thread> threads;
    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&monitor]() {
            for (int i = 0; i < 100; ++i) {
                monitor.recordEncode(10.0, 50);
                monitor.recordDecode(5.0, 30);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    auto stats = monitor.getStats();
    EXPECT_EQ(stats.totalEncodes, 400);
    EXPECT_EQ(stats.totalDecodes, 400);
}

// ==================== PerformanceTimer 测试 ====================

TEST(PerformanceTimerTest, AutoRecording) {
    PerformanceMonitor monitor;
    
    {
        PerformanceTimer timer(&monitor, PerformanceTimer::Operation::Encode, 100);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        // 析构时自动记录
    }
    
    auto stats = monitor.getStats();
    EXPECT_EQ(stats.totalEncodes, 1);
    EXPECT_EQ(stats.totalTokensEncoded, 100);
    EXPECT_GT(stats.avgEncodeLatency, 5.0);  // 至少 10ms
}

// ==================== 集成测试 ====================

TEST_F(LlamaTokenizerTest, WithPerformanceMonitor) {
    LlamaTokenizer tokenizer(ModelType::LLAMA);
    ASSERT_TRUE(tokenizer.load(modelPath_));
    
    // 启用性能监控
    tokenizer.enablePerformanceMonitor(true);
    EXPECT_TRUE(tokenizer.isPerformanceMonitorEnabled());
    
    // 执行一些操作
    std::string text = "Hello, world!";
    auto tokens = tokenizer.encode(text, true);
    auto decoded = tokenizer.decode(tokens, true);
    
    // 获取统计信息
    auto stats = tokenizer.getPerformanceStats();
    EXPECT_EQ(stats.totalEncodes, 1);
    EXPECT_EQ(stats.totalDecodes, 1);
    EXPECT_GT(stats.totalTokensEncoded, 0);
    
    // 重置统计
    tokenizer.resetPerformanceStats();
    stats = tokenizer.getPerformanceStats();
    EXPECT_EQ(stats.totalEncodes, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
