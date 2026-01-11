#include <gtest/gtest.h>
#include "cllm/kv_cache/cache.h"
#include "cllm/kv_cache/entry.h"
#include "cllm/kv_cache/stats.h"
#include "cllm/memory/float_array.h"
#include "cllm/common/config.h"
#include <thread>
#include <vector>

using namespace cllm;

class KVCacheTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("../config/cache_config.yaml");
        cache = std::make_unique<KVCache>(10, 100);
    }
    
    void TearDown() override {
        cache.reset();
    }
    
    FloatArray createTestKeyCache(size_t size) {
        FloatArray keyCache(size);
        for (size_t i = 0; i < size; ++i) {
            keyCache[i] = static_cast<float>(i);
        }
        return keyCache;
    }
    
    FloatArray createTestValueCache(size_t size) {
        FloatArray valueCache(size);
        for (size_t i = 0; i < size; ++i) {
            valueCache[i] = static_cast<float>(i * 2);
        }
        return valueCache;
    }
    
    std::unique_ptr<KVCache> cache;
};

TEST_F(KVCacheTest, PutAndGet) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    cache->put(1, keyCache, valueCache);
    
    KVCacheEntry entry;
    bool found = cache->get(1, entry);
    
    EXPECT_TRUE(found);
    EXPECT_EQ(entry.sequenceId, 1);
    EXPECT_EQ(entry.keyCache.size(), 100);
    EXPECT_EQ(entry.valueCache.size(), 100);
}

TEST_F(KVCacheTest, GetNonExistent) {
    KVCacheEntry entry;
    bool found = cache->get(999, entry);
    
    EXPECT_FALSE(found);
}

TEST_F(KVCacheTest, PutMultipleEntries) {
    for (size_t i = 0; i < 5; ++i) {
        FloatArray keyCache = createTestKeyCache(100);
        FloatArray valueCache = createTestValueCache(100);
        cache->put(i + 1, keyCache, valueCache);
    }
    
    EXPECT_EQ(cache->size(), 5);
    
    for (size_t i = 0; i < 5; ++i) {
        KVCacheEntry entry;
        bool found = cache->get(i + 1, entry);
        EXPECT_TRUE(found);
        EXPECT_EQ(entry.sequenceId, i + 1);
    }
}

TEST_F(KVCacheTest, RemoveEntry) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    cache->put(1, keyCache, valueCache);
    EXPECT_EQ(cache->size(), 1);
    
    cache->remove(1);
    EXPECT_EQ(cache->size(), 0);
    
    KVCacheEntry entry;
    bool found = cache->get(1, entry);
    EXPECT_FALSE(found);
}

TEST_F(KVCacheTest, RemoveNonExistent) {
    cache->remove(999);
    EXPECT_EQ(cache->size(), 0);
}

TEST_F(KVCacheTest, ClearCache) {
    for (size_t i = 0; i < 5; ++i) {
        FloatArray keyCache = createTestKeyCache(100);
        FloatArray valueCache = createTestValueCache(100);
        cache->put(i + 1, keyCache, valueCache);
    }
    
    EXPECT_EQ(cache->size(), 5);
    
    cache->clear();
    
    EXPECT_EQ(cache->size(), 0);
    EXPECT_EQ(cache->getMemoryUsageMB(), 0.0f);
}

TEST_F(KVCacheTest, Contains) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    EXPECT_FALSE(cache->contains(1));
    
    cache->put(1, keyCache, valueCache);
    
    EXPECT_TRUE(cache->contains(1));
    EXPECT_FALSE(cache->contains(2));
}

TEST_F(KVCacheTest, EvictionWhenMaxSizeReached) {
    KVCache smallCache(3, 0);
    
    for (size_t i = 0; i < 5; ++i) {
        FloatArray keyCache = createTestKeyCache(100);
        FloatArray valueCache = createTestValueCache(100);
        smallCache.put(i + 1, keyCache, valueCache);
    }
    
    EXPECT_EQ(smallCache.size(), 3);
    EXPECT_FALSE(smallCache.contains(1));
    EXPECT_FALSE(smallCache.contains(2));
    EXPECT_TRUE(smallCache.contains(3));
    EXPECT_TRUE(smallCache.contains(4));
    EXPECT_TRUE(smallCache.contains(5));
}

TEST_F(KVCacheTest, EvictionWhenMemoryLimitReached) {
    KVCache memoryCache(10, 1);
    
    for (size_t i = 0; i < 5; ++i) {
        FloatArray keyCache = createTestKeyCache(100000);
        FloatArray valueCache = createTestValueCache(100000);
        memoryCache.put(i + 1, keyCache, valueCache);
    }
    
    EXPECT_LE(memoryCache.getMemoryUsageMB(), 1.0f);
}

TEST_F(KVCacheTest, UpdateExistingEntry) {
    FloatArray keyCache1 = createTestKeyCache(100);
    FloatArray valueCache1 = createTestValueCache(100);
    
    cache->put(1, keyCache1, valueCache1);
    
    FloatArray keyCache2 = createTestKeyCache(200);
    FloatArray valueCache2 = createTestValueCache(200);
    
    cache->put(1, keyCache2, valueCache2);
    
    KVCacheEntry entry;
    cache->get(1, entry);
    
    EXPECT_EQ(entry.keyCache.size(), 200);
    EXPECT_EQ(entry.valueCache.size(), 200);
    EXPECT_EQ(cache->size(), 1);
}

TEST_F(KVCacheTest, UpdateIncremental) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    cache->put(1, keyCache, valueCache);
    
    FloatArray newKeyPart = createTestKeyCache(50);
    FloatArray newValuePart = createTestValueCache(50);
    
    cache->updateIncremental(1, newKeyPart, newValuePart);
    
    KVCacheEntry entry;
    cache->get(1, entry);
    
    EXPECT_EQ(entry.keyCache.size(), 150);
    EXPECT_EQ(entry.valueCache.size(), 150);
}

TEST_F(KVCacheTest, UpdateIncrementalNonExistent) {
    FloatArray newKeyPart = createTestKeyCache(50);
    FloatArray newValuePart = createTestValueCache(50);
    
    cache->updateIncremental(999, newKeyPart, newValuePart);
    
    EXPECT_EQ(cache->size(), 0);
}

TEST_F(KVCacheTest, GetStats) {
    CacheStats stats = cache->getStats();
    
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.evictions.load(), 0);
    EXPECT_EQ(stats.memoryReclaims.load(), 0);
}

TEST_F(KVCacheTest, UpdateStatsOnHit) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    cache->put(1, keyCache, valueCache);
    
    KVCacheEntry entry;
    cache->get(1, entry);
    
    CacheStats stats = cache->getStats();
    EXPECT_EQ(stats.hits.load(), 1);
    EXPECT_EQ(stats.misses.load(), 0);
}

TEST_F(KVCacheTest, UpdateStatsOnMiss) {
    KVCacheEntry entry;
    cache->get(999, entry);
    
    CacheStats stats = cache->getStats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 1);
}

TEST_F(KVCacheTest, UpdateStatsOnEviction) {
    KVCache smallCache(2, 0);
    
    for (size_t i = 0; i < 3; ++i) {
        FloatArray keyCache = createTestKeyCache(100);
        FloatArray valueCache = createTestValueCache(100);
        smallCache.put(i + 1, keyCache, valueCache);
    }
    
    CacheStats stats = smallCache.getStats();
    EXPECT_EQ(stats.evictions.load(), 1);
}

TEST_F(KVCacheTest, ResetStats) {
    FloatArray keyCache = createTestKeyCache(100);
    FloatArray valueCache = createTestValueCache(100);
    
    cache->put(1, keyCache, valueCache);
    
    KVCacheEntry entry;
    cache->get(1, entry);
    cache->get(999, entry);
    
    cache->resetStats();
    
    CacheStats stats = cache->getStats();
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
}

TEST_F(KVCacheTest, GetMemoryUsageMB) {
    EXPECT_EQ(cache->getMemoryUsageMB(), 0.0f);
    
    FloatArray keyCache = createTestKeyCache(1000000);  // 1M elements
    FloatArray valueCache = createTestValueCache(1000000);
    
    cache->put(1, keyCache, valueCache);
    
    double expected = (1000000 * 2 * sizeof(float)) / (1024.0 * 1024.0);
    double actual = cache->getMemoryUsageMB();
    
    EXPECT_NEAR(actual, expected, 0.01);  // 允许0.01MB误差
    EXPECT_GT(actual, 0.0f);
}

TEST_F(KVCacheTest, MemoryLimitPrecision) {
    KVCache preciseCache(100, 100);  // 100MB limit, 100 entries
    
    // 添加刚好100MB的数据
    size_t elements = (100 * 1024 * 1024) / (2 * sizeof(float));
    FloatArray keyCache = createTestKeyCache(elements);
    FloatArray valueCache = createTestValueCache(elements);
    
    preciseCache.put(1, keyCache, valueCache);
    EXPECT_NEAR(preciseCache.getMemoryUsageMB(), 100.0, 0.01);
    
    // 尝试添加更多数据应该触发淘汰
    FloatArray smallKey = createTestKeyCache(100);
    FloatArray smallValue = createTestValueCache(100);
    preciseCache.put(2, smallKey, smallValue);
    
    // 第一个条目应该被淘汰，内存使用量应该是小条目的大小
    EXPECT_FALSE(preciseCache.contains(1));  // 第一个条目应该被淘汰
    EXPECT_TRUE(preciseCache.contains(2));   // 第二个条目应该存在
    
    // 计算小条目的预期内存使用量
    float smallEntryMemory = (smallKey.size() + smallValue.size()) * sizeof(float) / (1024.0f * 1024.0f);
    EXPECT_NEAR(preciseCache.getMemoryUsageMB(), smallEntryMemory, 0.01);
}

TEST_F(KVCacheTest, GetMaxSize) {
    EXPECT_EQ(cache->getMaxSize(), 10);
    
    KVCache customCache(20, 100);
    EXPECT_EQ(customCache.getMaxSize(), 20);
}

TEST_F(KVCacheTest, GetMaxMemoryMB) {
    EXPECT_EQ(cache->getMaxMemoryMB(), 100);
    
    KVCache customCache(10, 200);
    EXPECT_EQ(customCache.getMaxMemoryMB(), 200);
}

TEST_F(KVCacheTest, ThreadSafety) {
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    KVCache threadSafeCache(numThreads * operationsPerThread, 0);  // 足够大的缓存
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, &threadSafeCache, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t id = i * operationsPerThread + j;
                FloatArray keyCache = createTestKeyCache(100);
                FloatArray valueCache = createTestValueCache(100);
                threadSafeCache.put(id, keyCache, valueCache);
                
                KVCacheEntry entry;
                threadSafeCache.get(id, entry);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // 由于maxSize足够大(numThreads * operationsPerThread)，所有条目都应该被保留
    EXPECT_EQ(threadSafeCache.size(), numThreads * operationsPerThread);
}

TEST_F(KVCacheTest, ConcurrentPutAndGet) {
    const int numThreads = 5;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i]() {
            for (int j = 0; j < 10; ++j) {
                size_t id = i * 10 + j;
                FloatArray keyCache = createTestKeyCache(100);
                FloatArray valueCache = createTestValueCache(100);
                cache->put(id, keyCache, valueCache);
            }
        });
    }
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i]() {
            for (int j = 0; j < 10; ++j) {
                size_t id = i * 10 + j;
                KVCacheEntry entry;
                cache->get(id, entry);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    CacheStats stats = cache->getStats();
    EXPECT_GT(stats.hits.load(), 0);
}

class KVCacheEntryTest : public ::testing::Test {
protected:
    void SetUp() override {
        keyCache = FloatArray(100);
        valueCache = FloatArray(100);
        for (size_t i = 0; i < 100; ++i) {
            keyCache[i] = static_cast<float>(i);
            valueCache[i] = static_cast<float>(i * 2);
        }
        
        entry = KVCacheEntry(keyCache, valueCache, 1);
    }
    
    FloatArray keyCache;
    FloatArray valueCache;
    KVCacheEntry entry;
};

TEST_F(KVCacheEntryTest, DefaultConstructor) {
    KVCacheEntry defaultEntry;
    
    EXPECT_EQ(defaultEntry.sequenceId, 0);
    EXPECT_EQ(defaultEntry.lastAccessTime, 0);
    EXPECT_EQ(defaultEntry.hitCount, 0);
    EXPECT_EQ(defaultEntry.createdTime, 0);
    EXPECT_EQ(defaultEntry.lastUpdateTime, 0);
    EXPECT_EQ(defaultEntry.memoryUsage, 0.0f);
    EXPECT_EQ(defaultEntry.accessCount, 0);
}

TEST_F(KVCacheEntryTest, ParameterizedConstructor) {
    EXPECT_EQ(entry.sequenceId, 1);
    EXPECT_EQ(entry.keyCache.size(), 100);
    EXPECT_EQ(entry.valueCache.size(), 100);
    EXPECT_GT(entry.memoryUsage, 0.0f);
}

TEST_F(KVCacheEntryTest, UpdateAccess) {
    size_t initialAccessCount = entry.accessCount;
    size_t initialHitCount = entry.hitCount;
    
    entry.updateAccess();
    
    EXPECT_GT(entry.lastAccessTime, 0);
    EXPECT_EQ(entry.accessCount, initialAccessCount + 1);
    EXPECT_EQ(entry.hitCount, initialHitCount + 1);
}

TEST_F(KVCacheEntryTest, UpdateMemoryUsage) {
    entry.keyCache = FloatArray(200);
    entry.valueCache = FloatArray(200);
    
    entry.updateMemoryUsage();
    
    EXPECT_GT(entry.memoryUsage, 0.0f);
}

class CacheStatsTest : public ::testing::Test {
protected:
    void SetUp() override {
        stats = CacheStats();
    }
    
    CacheStats stats;
};

TEST_F(CacheStatsTest, DefaultConstructor) {
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.evictions.load(), 0);
    EXPECT_EQ(stats.memoryReclaims.load(), 0);
}

TEST_F(CacheStatsTest, GetHitRateNoAccesses) {
    EXPECT_FLOAT_EQ(stats.getHitRate(), 0.0f);
}

TEST_F(CacheStatsTest, GetHitRateWithHits) {
    stats.updateHit();
    stats.updateHit();
    stats.updateHit();
    stats.updateMiss();
    
    EXPECT_FLOAT_EQ(stats.getHitRate(), 0.75f);
}

TEST_F(CacheStatsTest, GetHitRateOnlyMisses) {
    stats.updateMiss();
    stats.updateMiss();
    
    EXPECT_FLOAT_EQ(stats.getHitRate(), 0.0f);
}

TEST_F(CacheStatsTest, UpdateHit) {
    stats.updateHit();
    EXPECT_EQ(stats.hits.load(), 1);
    
    stats.updateHit();
    EXPECT_EQ(stats.hits.load(), 2);
}

TEST_F(CacheStatsTest, UpdateMiss) {
    stats.updateMiss();
    EXPECT_EQ(stats.misses.load(), 1);
    
    stats.updateMiss();
    EXPECT_EQ(stats.misses.load(), 2);
}

TEST_F(CacheStatsTest, UpdateEviction) {
    stats.updateEviction();
    EXPECT_EQ(stats.evictions.load(), 1);
    
    stats.updateEviction();
    EXPECT_EQ(stats.evictions.load(), 2);
}

TEST_F(CacheStatsTest, UpdateMemoryReclaim) {
    stats.updateMemoryReclaim();
    EXPECT_EQ(stats.memoryReclaims.load(), 1);
    
    stats.updateMemoryReclaim();
    EXPECT_EQ(stats.memoryReclaims.load(), 2);
}

TEST_F(CacheStatsTest, Reset) {
    stats.updateHit();
    stats.updateMiss();
    stats.updateEviction();
    stats.updateMemoryReclaim();
    
    stats.reset();
    
    EXPECT_EQ(stats.hits.load(), 0);
    EXPECT_EQ(stats.misses.load(), 0);
    EXPECT_EQ(stats.evictions.load(), 0);
    EXPECT_EQ(stats.memoryReclaims.load(), 0);
}

TEST_F(CacheStatsTest, ToString) {
    stats.updateHit();
    stats.updateMiss();
    
    std::string str = stats.toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("hits"), std::string::npos);
    EXPECT_NE(str.find("misses"), std::string::npos);
    EXPECT_NE(str.find("evictions"), std::string::npos);
    EXPECT_NE(str.find("memoryReclaims"), std::string::npos);
    EXPECT_NE(str.find("hitRate"), std::string::npos);
}
