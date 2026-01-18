#include <gtest/gtest.h>
#include "cllm/inference/kv_cache_manager.h"
#include <thread>
#include <vector>
#include <chrono>

using namespace cllm::inference;

class KVCacheManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        kvCacheManager = std::make_unique<KVCacheManager>(1000, 100);  // maxItems=1000, maxMemoryMb=100
    }
    
    void TearDown() override {
        kvCacheManager.reset();
    }
    
    std::unique_ptr<KVCacheManager> kvCacheManager;
};

TEST_F(KVCacheManagerTest, UpdateKVCacheStatsSingleRequest) {
    size_t requestId = 1;
    size_t sequenceLength = 10;
    
    kvCacheManager->updateKVCacheStats(requestId, sequenceLength);
    
    EXPECT_TRUE(kvCacheManager->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager->getTotalItems(), sequenceLength);
    EXPECT_EQ(kvCacheManager->getCacheCount(), 1);
    
    KVCacheStats stats = kvCacheManager->getKVCacheStats(requestId);
    EXPECT_EQ(stats.requestId, requestId);
    EXPECT_EQ(stats.itemCount, sequenceLength);
    EXPECT_GT(stats.memoryMb, 0);
}

TEST_F(KVCacheManagerTest, UpdateKVCacheStatsMultipleRequests) {
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    kvCacheManager->updateKVCacheStats(request1, 10);
    kvCacheManager->updateKVCacheStats(request2, 20);
    kvCacheManager->updateKVCacheStats(request3, 30);
    
    EXPECT_EQ(kvCacheManager->getTotalItems(), 60);
    EXPECT_EQ(kvCacheManager->getCacheCount(), 3);
    
    KVCacheStats stats1 = kvCacheManager->getKVCacheStats(request1);
    EXPECT_EQ(stats1.itemCount, 10);
    
    KVCacheStats stats2 = kvCacheManager->getKVCacheStats(request2);
    EXPECT_EQ(stats2.itemCount, 20);
    
    KVCacheStats stats3 = kvCacheManager->getKVCacheStats(request3);
    EXPECT_EQ(stats3.itemCount, 30);
}

TEST_F(KVCacheManagerTest, UpdateKVCacheStatsExistingRequest) {
    size_t requestId = 1;
    
    kvCacheManager->updateKVCacheStats(requestId, 10);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 10);
    
    kvCacheManager->updateKVCacheStats(requestId, 20);  // 更新为20
    EXPECT_EQ(kvCacheManager->getTotalItems(), 20);  // 应该是20，不是30
    
    KVCacheStats stats = kvCacheManager->getKVCacheStats(requestId);
    EXPECT_EQ(stats.itemCount, 20);
}

TEST_F(KVCacheManagerTest, GetKVCacheStatsNonExistent) {
    size_t requestId = 999;
    
    KVCacheStats stats = kvCacheManager->getKVCacheStats(requestId);
    EXPECT_EQ(stats.requestId, 0);
    EXPECT_EQ(stats.itemCount, 0);
    EXPECT_EQ(stats.memoryMb, 0);
}

TEST_F(KVCacheManagerTest, HasKVCacheStats) {
    size_t requestId = 1;
    
    EXPECT_FALSE(kvCacheManager->hasKVCacheStats(requestId));
    
    kvCacheManager->updateKVCacheStats(requestId, 10);
    EXPECT_TRUE(kvCacheManager->hasKVCacheStats(requestId));
}

TEST_F(KVCacheManagerTest, RemoveKVCacheWithoutContext) {
    size_t requestId = 1;
    size_t seqId = 0;
    
    kvCacheManager->updateKVCacheStats(requestId, 10);
    EXPECT_TRUE(kvCacheManager->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager->getTotalItems(), 10);
    
    bool success = kvCacheManager->removeKVCache(nullptr, requestId, seqId);
    
    EXPECT_TRUE(success);
    EXPECT_FALSE(kvCacheManager->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager->getTotalItems(), 0);
}

TEST_F(KVCacheManagerTest, RemoveKVCacheNonExistent) {
    size_t requestId = 999;
    size_t seqId = 0;
    
    bool success = kvCacheManager->removeKVCache(nullptr, requestId, seqId);
    
    EXPECT_FALSE(success);
}

TEST_F(KVCacheManagerTest, TotalMemoryCalculation) {
    size_t requestId = 1;
    size_t sequenceLength = 10;
    
    kvCacheManager->updateKVCacheStats(requestId, sequenceLength);
    
    size_t totalMemory = kvCacheManager->getTotalMemoryMb();
    EXPECT_GT(totalMemory, 0);
    
    KVCacheStats stats = kvCacheManager->getKVCacheStats(requestId);
    EXPECT_EQ(totalMemory, stats.memoryMb);
}

TEST_F(KVCacheManagerTest, RequestStatusUpdate) {
    size_t requestId = 1;
    
    EXPECT_EQ(kvCacheManager->getRequestStatus(requestId), RequestStatus::PENDING);
    
    kvCacheManager->updateRequestStatus(requestId, RequestStatus::PROCESSING);
    EXPECT_EQ(kvCacheManager->getRequestStatus(requestId), RequestStatus::PROCESSING);
    
    kvCacheManager->updateRequestStatus(requestId, RequestStatus::COMPLETED);
    EXPECT_EQ(kvCacheManager->getRequestStatus(requestId), RequestStatus::COMPLETED);
}

TEST_F(KVCacheManagerTest, MultipleRequestsIndependentStats) {
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    kvCacheManager->updateKVCacheStats(request1, 10);
    kvCacheManager->updateKVCacheStats(request2, 20);
    kvCacheManager->updateKVCacheStats(request3, 30);
    
    kvCacheManager->updateRequestStatus(request1, RequestStatus::PROCESSING);
    kvCacheManager->updateRequestStatus(request2, RequestStatus::COMPLETED);
    kvCacheManager->updateRequestStatus(request3, RequestStatus::PENDING);
    
    EXPECT_EQ(kvCacheManager->getRequestStatus(request1), RequestStatus::PROCESSING);
    EXPECT_EQ(kvCacheManager->getRequestStatus(request2), RequestStatus::COMPLETED);
    EXPECT_EQ(kvCacheManager->getRequestStatus(request3), RequestStatus::PENDING);
    
    EXPECT_EQ(kvCacheManager->getKVCacheStats(request1).itemCount, 10);
    EXPECT_EQ(kvCacheManager->getKVCacheStats(request2).itemCount, 20);
    EXPECT_EQ(kvCacheManager->getKVCacheStats(request3).itemCount, 30);
}

TEST_F(KVCacheManagerTest, ConcurrentUpdateStats) {
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager->updateKVCacheStats(requestId, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(kvCacheManager->getCacheCount(), numThreads * operationsPerThread);
    EXPECT_EQ(kvCacheManager->getTotalItems(), numThreads * operationsPerThread * 10);
}

TEST_F(KVCacheManagerTest, ConcurrentQueryStats) {
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager->updateKVCacheStats(requestId, 10);
                KVCacheStats stats = kvCacheManager->getKVCacheStats(requestId);
                EXPECT_EQ(stats.itemCount, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

TEST_F(KVCacheManagerTest, ConcurrentRemoveStats) {
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager->updateKVCacheStats(requestId, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(kvCacheManager->getCacheCount(), numThreads * operationsPerThread);
    
    threads.clear();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager->removeKVCache(nullptr, requestId, 0);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(kvCacheManager->getCacheCount(), 0);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 0);
}

TEST_F(KVCacheManagerTest, LastAccessTimeUpdate) {
    size_t requestId = 1;
    
    kvCacheManager->updateKVCacheStats(requestId, 10);
    
    KVCacheStats stats1 = kvCacheManager->getKVCacheStats(requestId);
    auto time1 = stats1.lastAccessTime;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    KVCacheStats stats2 = kvCacheManager->getKVCacheStats(requestId);
    auto time2 = stats2.lastAccessTime;
    
    EXPECT_GT(time2, time1);
}

TEST_F(KVCacheManagerTest, MaxItemsAndMaxMemory) {
    EXPECT_EQ(kvCacheManager->getMaxItems(), 1000);
    EXPECT_EQ(kvCacheManager->getMaxMemoryMb(), 100);
    
    KVCacheManager customManager(2000, 200);
    EXPECT_EQ(customManager.getMaxItems(), 2000);
    EXPECT_EQ(customManager.getMaxMemoryMb(), 200);
}

TEST_F(KVCacheManagerTest, EstimateMemoryPerItem) {
    size_t memoryPerItem = KVCacheManager::estimateMemoryPerItem();
    EXPECT_GT(memoryPerItem, 0);
    
    size_t memoryPerItemCustom = KVCacheManager::estimateMemoryPerItem(151936, 4096);
    EXPECT_GT(memoryPerItemCustom, 0);
}
