/**
 * @file kv_cache_manager_test.cpp
 * @brief KV缓存管理器单元测试
 * 
 * 测试KVCacheManager的统计管理、并发安全等功能
 * 迁移自：tests/test_kv_cache_manager.cpp
 * 改进：使用TestBase基类，增强了测试命名和文档
 */

#include <gtest/gtest.h>
#include "utils/test_base.h"
#include "cllm/inference/kv_cache_manager.h"
#include <thread>
#include <vector>
#include <chrono>

using namespace cllm;
using namespace cllm::inference;
using namespace cllm::test;

/**
 * @brief KV缓存管理器测试类
 * 
 * 使用TestBase作为基类，提供通用的测试环境
 */
class KVCacheManagerTest : public TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
        // 创建KVCacheManager实例：maxItems=1000, maxMemoryMb=100
        kvCacheManager_ = std::make_unique<KVCacheManager>(1000, 100);
    }
    
    void TearDown() override {
        kvCacheManager_.reset();
        TestBase::TearDown();
    }
    
    std::unique_ptr<KVCacheManager> kvCacheManager_;
};

// ========== 单个请求统计测试 ==========

TEST_F(KVCacheManagerTest, UpdateStats_SingleRequest_CreatesEntry) {
    // Arrange
    size_t requestId = 1;
    size_t sequenceLength = 10;
    
    // Act
    kvCacheManager_->updateKVCacheStats(requestId, sequenceLength);
    
    // Assert
    EXPECT_TRUE(kvCacheManager_->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager_->getTotalItems(), sequenceLength);
    EXPECT_EQ(kvCacheManager_->getCacheCount(), 1);
    
    KVCacheStats stats = kvCacheManager_->getKVCacheStats(requestId);
    EXPECT_EQ(stats.requestId, requestId);
    EXPECT_EQ(stats.itemCount, sequenceLength);
    EXPECT_GT(stats.memoryMb, 0);
}

// ========== 多个请求统计测试 ==========

TEST_F(KVCacheManagerTest, UpdateStats_MultipleRequests_TracksCorrectly) {
    // Arrange
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    // Act
    kvCacheManager_->updateKVCacheStats(request1, 10);
    kvCacheManager_->updateKVCacheStats(request2, 20);
    kvCacheManager_->updateKVCacheStats(request3, 30);
    
    // Assert
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 60);
    EXPECT_EQ(kvCacheManager_->getCacheCount(), 3);
    
    KVCacheStats stats1 = kvCacheManager_->getKVCacheStats(request1);
    EXPECT_EQ(stats1.itemCount, 10);
    
    KVCacheStats stats2 = kvCacheManager_->getKVCacheStats(request2);
    EXPECT_EQ(stats2.itemCount, 20);
    
    KVCacheStats stats3 = kvCacheManager_->getKVCacheStats(request3);
    EXPECT_EQ(stats3.itemCount, 30);
}

// ========== 更新现有请求测试 ==========

TEST_F(KVCacheManagerTest, UpdateStats_ExistingRequest_ReplacesOldValue) {
    // Arrange
    size_t requestId = 1;
    
    // Act
    kvCacheManager_->updateKVCacheStats(requestId, 10);
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 10);
    
    kvCacheManager_->updateKVCacheStats(requestId, 20);  // 更新为20
    
    // Assert
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 20);  // 应该是20，不是30
    
    KVCacheStats stats = kvCacheManager_->getKVCacheStats(requestId);
    EXPECT_EQ(stats.itemCount, 20);
}

// ========== 查询不存在的请求测试 ==========

TEST_F(KVCacheManagerTest, GetStats_NonExistentRequest_ReturnsEmptyStats) {
    // Arrange
    size_t requestId = 999;
    
    // Act
    KVCacheStats stats = kvCacheManager_->getKVCacheStats(requestId);
    
    // Assert
    EXPECT_EQ(stats.requestId, 0);
    EXPECT_EQ(stats.itemCount, 0);
    EXPECT_EQ(stats.memoryMb, 0);
}

// ========== 检查统计存在性测试 ==========

TEST_F(KVCacheManagerTest, HasStats_BeforeAndAfterUpdate_ReturnsCorrectly) {
    // Arrange
    size_t requestId = 1;
    
    // Act & Assert - before
    EXPECT_FALSE(kvCacheManager_->hasKVCacheStats(requestId));
    
    // Act - update
    kvCacheManager_->updateKVCacheStats(requestId, 10);
    
    // Assert - after
    EXPECT_TRUE(kvCacheManager_->hasKVCacheStats(requestId));
}

// ========== 删除缓存测试 ==========

TEST_F(KVCacheManagerTest, RemoveCache_ExistingRequest_RemovesSuccessfully) {
    // Arrange
    size_t requestId = 1;
    size_t seqId = 0;
    
    kvCacheManager_->updateKVCacheStats(requestId, 10);
    EXPECT_TRUE(kvCacheManager_->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 10);
    
    // Act
    bool success = kvCacheManager_->removeKVCache(nullptr, requestId, seqId);
    
    // Assert
    EXPECT_TRUE(success);
    EXPECT_FALSE(kvCacheManager_->hasKVCacheStats(requestId));
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 0);
}

TEST_F(KVCacheManagerTest, RemoveCache_NonExistentRequest_ReturnsSuccess) {
    // Arrange - 不存在的请求，但ctx为nullptr时仍可能返回true
    size_t requestId = 999;
    size_t seqId = 0;
    
    // Act
    bool success = kvCacheManager_->removeKVCache(nullptr, requestId, seqId);
    
    // Assert - 当ctx为nullptr时，实现可能直接返回true
    // 这是合理的，因为没有实际的缓存需要清理
    EXPECT_TRUE(success);
}

// ========== 内存计算测试 ==========

TEST_F(KVCacheManagerTest, GetTotalMemory_AfterUpdate_ReturnsPositiveValue) {
    // Arrange
    size_t requestId = 1;
    size_t sequenceLength = 10;
    
    // Act
    kvCacheManager_->updateKVCacheStats(requestId, sequenceLength);
    
    // Assert
    size_t totalMemory = kvCacheManager_->getTotalMemoryMb();
    EXPECT_GT(totalMemory, 0);
    
    KVCacheStats stats = kvCacheManager_->getKVCacheStats(requestId);
    EXPECT_EQ(totalMemory, stats.memoryMb);
}

// ========== 请求状态更新测试 ==========

TEST_F(KVCacheManagerTest, UpdateStatus_TransitionsCorrectly) {
    // Arrange
    size_t requestId = 1;
    
    // Act & Assert
    EXPECT_EQ(kvCacheManager_->getRequestStatus(requestId), RequestStatus::PENDING);
    
    kvCacheManager_->updateRequestStatus(requestId, RequestStatus::PROCESSING);
    EXPECT_EQ(kvCacheManager_->getRequestStatus(requestId), RequestStatus::PROCESSING);
    
    kvCacheManager_->updateRequestStatus(requestId, RequestStatus::COMPLETED);
    EXPECT_EQ(kvCacheManager_->getRequestStatus(requestId), RequestStatus::COMPLETED);
}

// ========== 多请求独立性测试 ==========

TEST_F(KVCacheManagerTest, MultipleRequests_IndependentStats_MaintainedCorrectly) {
    // Arrange
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    // Act
    kvCacheManager_->updateKVCacheStats(request1, 10);
    kvCacheManager_->updateKVCacheStats(request2, 20);
    kvCacheManager_->updateKVCacheStats(request3, 30);
    
    kvCacheManager_->updateRequestStatus(request1, RequestStatus::PROCESSING);
    kvCacheManager_->updateRequestStatus(request2, RequestStatus::COMPLETED);
    kvCacheManager_->updateRequestStatus(request3, RequestStatus::PENDING);
    
    // Assert
    EXPECT_EQ(kvCacheManager_->getRequestStatus(request1), RequestStatus::PROCESSING);
    EXPECT_EQ(kvCacheManager_->getRequestStatus(request2), RequestStatus::COMPLETED);
    EXPECT_EQ(kvCacheManager_->getRequestStatus(request3), RequestStatus::PENDING);
    
    EXPECT_EQ(kvCacheManager_->getKVCacheStats(request1).itemCount, 10);
    EXPECT_EQ(kvCacheManager_->getKVCacheStats(request2).itemCount, 20);
    EXPECT_EQ(kvCacheManager_->getKVCacheStats(request3).itemCount, 30);
}

// ========== 并发统计更新测试 ==========

TEST_F(KVCacheManagerTest, ConcurrentUpdate_ThreadSafe_AllUpdatesSucceed) {
    // Arrange
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    // Act
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager_->updateKVCacheStats(requestId, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Assert
    EXPECT_EQ(kvCacheManager_->getCacheCount(), numThreads * operationsPerThread);
    EXPECT_EQ(kvCacheManager_->getTotalItems(), numThreads * operationsPerThread * 10);
}

// ========== 并发查询测试 ==========

TEST_F(KVCacheManagerTest, ConcurrentQuery_ThreadSafe_AllQueriesSucceed) {
    // Arrange
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    // Act
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager_->updateKVCacheStats(requestId, 10);
                KVCacheStats stats = kvCacheManager_->getKVCacheStats(requestId);
                EXPECT_EQ(stats.itemCount, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
}

// ========== 并发删除测试 ==========

TEST_F(KVCacheManagerTest, ConcurrentRemove_ThreadSafe_AllRemovesSucceed) {
    // Arrange
    const int numThreads = 10;
    const int operationsPerThread = 10;
    std::vector<std::thread> threads;
    
    // 先添加所有缓存
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager_->updateKVCacheStats(requestId, 10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(kvCacheManager_->getCacheCount(), numThreads * operationsPerThread);
    
    // 并发删除
    threads.clear();
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, operationsPerThread]() {
            for (int j = 0; j < operationsPerThread; ++j) {
                size_t requestId = i * operationsPerThread + j;
                kvCacheManager_->removeKVCache(nullptr, requestId, 0);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Assert
    EXPECT_EQ(kvCacheManager_->getCacheCount(), 0);
    EXPECT_EQ(kvCacheManager_->getTotalItems(), 0);
}

// ========== 最后访问时间更新测试 ==========

TEST_F(KVCacheManagerTest, LastAccessTime_UpdatesOnQuery) {
    // Arrange
    size_t requestId = 1;
    
    kvCacheManager_->updateKVCacheStats(requestId, 10);
    
    // Act
    KVCacheStats stats1 = kvCacheManager_->getKVCacheStats(requestId);
    auto time1 = stats1.lastAccessTime;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    KVCacheStats stats2 = kvCacheManager_->getKVCacheStats(requestId);
    auto time2 = stats2.lastAccessTime;
    
    // Assert
    EXPECT_GT(time2, time1);
}

// ========== 最大容量测试 ==========

TEST_F(KVCacheManagerTest, MaxItemsAndMemory_ReturnsConfiguredValues) {
    // Assert
    EXPECT_EQ(kvCacheManager_->getMaxItems(), 1000);
    EXPECT_EQ(kvCacheManager_->getMaxMemoryMb(), 100);
    
    // 测试自定义值
    KVCacheManager customManager(2000, 200);
    EXPECT_EQ(customManager.getMaxItems(), 2000);
    EXPECT_EQ(customManager.getMaxMemoryMb(), 200);
}

// ========== 内存估算测试 ==========

TEST_F(KVCacheManagerTest, EstimateMemory_ReturnsPositiveValue) {
    // Act
    size_t memoryPerItem = KVCacheManager::estimateMemoryPerItem();
    
    // Assert
    EXPECT_GT(memoryPerItem, 0);
    
    // 测试自定义参数
    size_t memoryPerItemCustom = KVCacheManager::estimateMemoryPerItem(151936, 4096);
    EXPECT_GT(memoryPerItemCustom, 0);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
