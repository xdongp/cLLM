#include <gtest/gtest.h>
#include "cllm/CTokenizer/token_cache.h"
#include "cllm/CTokenizer/performance_config.h"
#include "cllm/CTokenizer/sentencepiece_tokenizer.h"
#include "cllm/CTokenizer/batch_tokenizer.h"

using namespace cllm;

// ===== TokenCache 测试 =====

TEST(TokenCacheTest, BasicEncodeCache) {
    TokenCache cache;
    
    std::string text = "Hello, world!";
    std::vector<int> tokens = {72, 101, 108, 108, 111};
    
    // 写入缓存
    cache.putEncode(text, tokens);
    
    // 读取缓存
    auto cached = cache.getEncode(text);
    ASSERT_TRUE(cached.has_value());
    EXPECT_EQ(cached->size(), tokens.size());
    EXPECT_EQ(*cached, tokens);
}

TEST(TokenCacheTest, BasicDecodeCache) {
    TokenCache cache;
    
    std::vector<int> tokens = {72, 101, 108, 108, 111};
    std::string text = "Hello";
    
    // 写入缓存
    cache.putDecode(tokens, text);
    
    // 读取缓存
    auto cached = cache.getDecode(tokens);
    ASSERT_TRUE(cached.has_value());
    EXPECT_EQ(*cached, text);
}

TEST(TokenCacheTest, CacheMiss) {
    TokenCache cache;
    
    // 读取不存在的缓存
    auto cachedEncode = cache.getEncode("nonexistent");
    EXPECT_FALSE(cachedEncode.has_value());
    
    std::vector<int> tokens = {1, 2, 3};
    auto cachedDecode = cache.getDecode(tokens);
    EXPECT_FALSE(cachedDecode.has_value());
}

TEST(TokenCacheTest, CacheEviction) {
    TokenCache cache(5); // 最大 5 个条目
    
    // 添加 6 个条目（应该淘汰最早的）
    for (int i = 0; i < 6; i++) {
        std::string text = "text" + std::to_string(i);
        std::vector<int> tokens = {i};
        cache.putEncode(text, tokens);
    }
    
    // 缓存大小应该被限制
    EXPECT_LE(cache.size(), 5);
    
    // 最早的条目应该被淘汰
    auto cached = cache.getEncode("text0");
    EXPECT_FALSE(cached.has_value());
    
    // 最新的条目应该存在
    auto cached5 = cache.getEncode("text5");
    EXPECT_TRUE(cached5.has_value());
}

TEST(TokenCacheTest, CacheClear) {
    TokenCache cache;
    
    cache.putEncode("text1", {1, 2, 3});
    cache.putEncode("text2", {4, 5, 6});
    
    EXPECT_GT(cache.size(), 0);
    
    cache.clear();
    
    EXPECT_EQ(cache.size(), 0);
    EXPECT_FALSE(cache.getEncode("text1").has_value());
    EXPECT_FALSE(cache.getEncode("text2").has_value());
}

TEST(TokenCacheTest, CacheResizing) {
    TokenCache cache(100);
    
    // 添加一些条目
    for (int i = 0; i < 50; i++) {
        cache.putEncode("text" + std::to_string(i), {i});
    }
    
    EXPECT_EQ(cache.size(), 50);
    
    // 缩小缓存大小
    cache.setMaxSize(20);
    
    // 缓存应该被压缩
    EXPECT_LE(cache.size(), 20);
}

// ===== PerformanceConfig 测试 =====

TEST(PerformanceConfigTest, DefaultConfig) {
    auto config = TokenizerPerformanceConfig::getDefault();
    
    EXPECT_TRUE(config.cacheEnabled);
    EXPECT_EQ(config.cacheMaxSize, 10000);
    EXPECT_TRUE(config.batchEnabled);
    EXPECT_EQ(config.batchSize, 32);
    EXPECT_TRUE(config.metricsEnabled);
}

TEST(PerformanceConfigTest, HighPerformanceConfig) {
    auto config = TokenizerPerformanceConfig::getHighPerformance();
    
    EXPECT_TRUE(config.cacheEnabled);
    EXPECT_GT(config.cacheMaxSize, 10000);
    EXPECT_GT(config.batchSize, 32);
    EXPECT_GT(config.numThreads, 0);
}

TEST(PerformanceConfigTest, LowMemoryConfig) {
    auto config = TokenizerPerformanceConfig::getLowMemory();
    
    EXPECT_TRUE(config.cacheEnabled);
    EXPECT_LT(config.cacheMaxSize, 10000);
    EXPECT_LT(config.batchSize, 32);
    EXPECT_GT(config.memoryLimit, 0);
}

TEST(PerformanceConfigTest, ConfigValidation) {
    TokenizerPerformanceConfig config;
    
    // 默认配置应该有效
    EXPECT_TRUE(config.validate());
    
    // 无效配置：缓存启用但大小为 0
    config.cacheEnabled = true;
    config.cacheMaxSize = 0;
    EXPECT_FALSE(config.validate());
    
    // 修正
    config.cacheMaxSize = 100;
    EXPECT_TRUE(config.validate());
    
    // 无效策略
    config.cacheEvictionPolicy = "invalid";
    EXPECT_FALSE(config.validate());
}

TEST(PerformanceConfigTest, JsonLoading) {
    nlohmann::json json = {
        {"cache_enabled", false},
        {"cache_size", 5000},
        {"batch_size", 64},
        {"num_threads", 8},
        {"enable_metrics", false}
    };
    
    TokenizerPerformanceConfig config;
    config.loadFromJson(&json);
    
    EXPECT_FALSE(config.cacheEnabled);
    EXPECT_EQ(config.cacheMaxSize, 5000);
    EXPECT_EQ(config.batchSize, 64);
    EXPECT_EQ(config.numThreads, 8);
    EXPECT_FALSE(config.metricsEnabled);
}

// ===== 集成测试：缓存 + 配置 =====

class MockTokenizer : public CTokenizer {
public:
    int encodeCallCount = 0;
    int decodeCallCount = 0;
    
    std::vector<llama_token> encode(const std::string& text, bool addSpecial) override {
        encodeCallCount++;
        std::vector<llama_token> result;
        for (char c : text) {
            result.push_back(static_cast<llama_token>(c));
        }
        return result;
    }
    
    std::string decode(const std::vector<llama_token>& ids, bool skipSpecial) override {
        decodeCallCount++;
        std::string result;
        for (auto id : ids) {
            result += static_cast<char>(id);
        }
        return result;
    }
    
    int getVocabSize() const override { return 10000; }
    std::string idToToken(llama_token id) const override { return std::string(1, static_cast<char>(id)); }
    llama_token tokenToId(const std::string& token) const override { return token.empty() ? 0 : static_cast<llama_token>(token[0]); }
    
    llama_token getBosId() const override { return 1; }
    llama_token getEosId() const override { return 2; }
    llama_token getPadId() const override { return 0; }
    llama_token getUnkId() const override { return 3; }
    
    cllm::ModelType getModelType() const override { return cllm::ModelType::QWEN; }
    bool load(const std::string& path) override { return true; }
};

TEST(TokenizerIntegrationTest, CacheReducesCalls) {
    TokenCache cache;
    MockTokenizer tokenizer;
    
    std::string text = "test";
    
    // 第一次调用
    auto tokens1 = tokenizer.encode(text, true);
    cache.putEncode(text, std::vector<int>(tokens1.begin(), tokens1.end()));
    EXPECT_EQ(tokenizer.encodeCallCount, 1);
    
    // 第二次从缓存读取
    auto cached = cache.getEncode(text);
    ASSERT_TRUE(cached.has_value());
    EXPECT_EQ(tokenizer.encodeCallCount, 1); // 没有增加
    
    // 验证缓存结果正确
    EXPECT_EQ(cached->size(), tokens1.size());
}

TEST(TokenizerIntegrationTest, ConfigAppliedToTokenizer) {
    MockTokenizer tokenizer;
    
    // 设置性能配置
    auto config = TokenizerPerformanceConfig::getLowMemory();
    tokenizer.setPerformanceConfig(config);
    
    auto appliedConfig = tokenizer.getPerformanceConfig();
    EXPECT_EQ(appliedConfig.cacheMaxSize, config.cacheMaxSize);
    EXPECT_EQ(appliedConfig.batchSize, config.batchSize);
}

TEST(TokenizerIntegrationTest, BatchProcessingWithConfig) {
    MockTokenizer tokenizer;
    
    std::vector<std::string> texts = {"hello", "world", "test"};
    
    // 使用配置的批处理
    auto config = TokenizerPerformanceConfig::getHighPerformance();
    auto result = BatchTokenizer::batchEncode(&tokenizer, texts, config, true);
    
    EXPECT_EQ(result.tokenized.size(), 3);
    EXPECT_EQ(result.success[0], true);
    EXPECT_EQ(result.success[1], true);
    EXPECT_EQ(result.success[2], true);
}

// ===== 性能测试 =====

TEST(TokenizerPerformanceTest, CacheHitVsMiss) {
    TokenCache cache;
    MockTokenizer tokenizer;
    
    std::string text = "performance test";
    auto tokens = tokenizer.encode(text, true);
    cache.putEncode(text, std::vector<int>(tokens.begin(), tokens.end()));
    
    // 测试缓存命中率
    int hits = 0;
    int misses = 0;
    
    for (int i = 0; i < 100; i++) {
        std::string testText = (i % 2 == 0) ? text : ("other" + std::to_string(i));
        auto cached = cache.getEncode(testText);
        if (cached.has_value()) {
            hits++;
        } else {
            misses++;
        }
    }
    
    // 应该有 50 个命中（重复的 text）
    EXPECT_EQ(hits, 50);
    EXPECT_EQ(misses, 50);
}
