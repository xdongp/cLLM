#include <gtest/gtest.h>
#include "cllm/inference/kv_cache_manager.h"
#include <thread>
#include <vector>
#include <chrono>

using namespace cllm::inference;

class KVCacheLRUEvictionTest : public ::testing::Test {
protected:
    void SetUp() override {
        kvCacheManager = std::make_unique<KVCacheManager>(100, 100);  // maxItems=100, maxMemoryMb=100
    }
    
    void TearDown() override {
        kvCacheManager.reset();
    }
    
    std::unique_ptr<KVCacheManager> kvCacheManager;
};

TEST_F(KVCacheLRUEvictionTest, ShouldEvictItemsThreshold) {
    size_t requestId = 1;
    
    // 添加统计信息，总条目数 = 90
    kvCacheManager->updateKVCacheStats(requestId, 90);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 90);
    
    // 检查是否需要淘汰（阈值 = 0.8 * 100 = 80）
    // 90 > 80，应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
}

TEST_F(KVCacheLRUEvictionTest, ShouldEvictMemoryThreshold) {
    size_t requestId = 1;
    
    // 添加统计信息，总内存 = 90MB
    kvCacheManager->updateKVCacheStats(requestId, 45);  // 45 * 2MB = 90MB
    EXPECT_EQ(kvCacheManager->getTotalMemoryMb(), 90);
    
    // 检查是否需要淘汰（阈值 = 0.8 * 100 = 80MB）
    // 90 > 80，应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
}

TEST_F(KVCacheLRUEvictionTest, ShouldNotEvictBelowThreshold) {
    size_t requestId = 1;
    
    // 添加统计信息，总条目数 = 30，总内存 = 60MB
    kvCacheManager->updateKVCacheStats(requestId, 30);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 30);
    EXPECT_EQ(kvCacheManager->getTotalMemoryMb(), 60);
    
    // 检查是否需要淘汰（阈值 = 0.8 * 100 = 80 items, 80MB）
    // 30 < 80, 60 < 80，不应该需要淘汰
    EXPECT_FALSE(kvCacheManager->shouldEvict(0.8));
}

TEST_F(KVCacheLRUEvictionTest, LRUEvictionByAccessTime) {
    // 创建多个请求，并设置不同的访问时间
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    // 添加请求1（最久未使用）
    kvCacheManager->updateKVCacheStats(request1, 10);
    kvCacheManager->updateRequestStatus(request1, RequestStatus::COMPLETED);
    
    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // 添加请求2（中间）
    kvCacheManager->updateKVCacheStats(request2, 10);
    kvCacheManager->updateRequestStatus(request2, RequestStatus::COMPLETED);
    
    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // 添加请求3（最近使用）
    kvCacheManager->updateKVCacheStats(request3, 10);
    kvCacheManager->updateRequestStatus(request3, RequestStatus::COMPLETED);
    
    // 总条目数 = 30，阈值 = 0.8 * 100 = 80
    // 不需要淘汰
    EXPECT_FALSE(kvCacheManager->shouldEvict(0.8));
    
    // 手动添加更多请求，使总条目数超过阈值
    for (size_t i = 4; i <= 10; ++i) {
        kvCacheManager->updateKVCacheStats(i, 10);
        kvCacheManager->updateRequestStatus(i, RequestStatus::COMPLETED);
    }
    
    // 总条目数 = 100，阈值 = 80
    // 应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
}

TEST_F(KVCacheLRUEvictionTest, EvictionProtectionProcessing) {
    // 创建多个请求，其中一些是 PROCESSING 状态
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    // 添加请求1（PROCESSING 状态，应该被保护）
    kvCacheManager->updateKVCacheStats(request1, 30);
    kvCacheManager->updateRequestStatus(request1, RequestStatus::PROCESSING);
    
    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // 添加请求2（COMPLETED 状态，可以被淘汰）
    kvCacheManager->updateKVCacheStats(request2, 30);
    kvCacheManager->updateRequestStatus(request2, RequestStatus::COMPLETED);
    
    // 等待一段时间
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    
    // 添加请求3（COMPLETED 状态，可以被淘汰）
    kvCacheManager->updateKVCacheStats(request3, 30);
    kvCacheManager->updateRequestStatus(request3, RequestStatus::COMPLETED);
    
    // 总条目数 = 90，阈值 = 0.8 * 100 = 80
    // 应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
    
    // 执行淘汰（无上下文，只测试统计信息清理）
    size_t evictedCount = kvCacheManager->evictLRUCache(nullptr, 0.8, nullptr);
    
    // 应该淘汰 2 个请求（request2 和 request3）
    // 因为 request1 是 PROCESSING 状态，被保护
    // 淘汰 request2 后，总条目数 = 60，仍然低于阈值 80
    // 所以会继续淘汰 request3
    EXPECT_EQ(evictedCount, 2);
    
    // request1 应该仍然存在（PROCESSING 状态，被保护）
    EXPECT_TRUE(kvCacheManager->hasKVCacheStats(request1));
    EXPECT_EQ(kvCacheManager->getRequestStatus(request1), RequestStatus::PROCESSING);
}

TEST_F(KVCacheLRUEvictionTest, BatchEviction) {
    // 创建多个请求，使总条目数远超阈值
    for (size_t i = 1; i <= 20; ++i) {
        kvCacheManager->updateKVCacheStats(i, 10);
        kvCacheManager->updateRequestStatus(i, RequestStatus::COMPLETED);
    }
    
    // 总条目数 = 200，阈值 = 0.8 * 100 = 80
    // 应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
    
    // 执行淘汰（无上下文，只测试统计信息清理）
    size_t evictedCount = kvCacheManager->evictLRUCache(nullptr, 0.8, nullptr);
    
    // 应该淘汰多个请求，直到总条目数低于阈值
    EXPECT_GT(evictedCount, 0);
    EXPECT_LE(kvCacheManager->getTotalItems(), 80);
}

TEST_F(KVCacheLRUEvictionTest, EvictionWithSeqIdCallback) {
    // 创建多个请求
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    kvCacheManager->updateKVCacheStats(request1, 30);
    kvCacheManager->updateRequestStatus(request1, RequestStatus::COMPLETED);
    
    kvCacheManager->updateKVCacheStats(request2, 30);
    kvCacheManager->updateRequestStatus(request2, RequestStatus::COMPLETED);
    
    kvCacheManager->updateKVCacheStats(request3, 30);
    kvCacheManager->updateRequestStatus(request3, RequestStatus::COMPLETED);
    
    // 总条目数 = 90，阈值 = 0.8 * 100 = 80
    // 应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
    
    // 提供回调函数来获取 seqId
    std::map<size_t, int32_t> seqIdMap = {
        {request1, 10},
        {request2, 20},
        {request3, 30}
    };
    
    auto getSeqIdCallback = [&seqIdMap](size_t requestId) -> int32_t {
        auto it = seqIdMap.find(requestId);
        return (it != seqIdMap.end()) ? it->second : -1;
    };
    
    // 执行淘汰（无上下文，只测试统计信息清理）
    size_t evictedCount = kvCacheManager->evictLRUCache(nullptr, 0.8, getSeqIdCallback);
    
    // 应该淘汰 2 个请求（request1 和 request2）
    // 因为淘汰 request1 后，总条目数 = 60，仍然低于阈值 80
    // 所以会继续淘汰 request2
    EXPECT_EQ(evictedCount, 2);
}

TEST_F(KVCacheLRUEvictionTest, NoEvictableRequests) {
    // 创建多个请求，全部是 PROCESSING 状态
    for (size_t i = 1; i <= 10; ++i) {
        kvCacheManager->updateKVCacheStats(i, 15);
        kvCacheManager->updateRequestStatus(i, RequestStatus::PROCESSING);
    }
    
    // 总条目数 = 150，阈值 = 0.8 * 100 = 80
    // 应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
    
    // 执行淘汰（无上下文，只测试统计信息清理）
    size_t evictedCount = kvCacheManager->evictLRUCache(nullptr, 0.8, nullptr);
    
    // 由于所有请求都是 PROCESSING 状态，不应该淘汰任何请求
    EXPECT_EQ(evictedCount, 0);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 150);
}

TEST_F(KVCacheLRUEvictionTest, EvictionUpdatesStats) {
    // 创建多个请求
    size_t request1 = 1;
    size_t request2 = 2;
    size_t request3 = 3;
    
    kvCacheManager->updateKVCacheStats(request1, 30);
    kvCacheManager->updateRequestStatus(request1, RequestStatus::COMPLETED);
    
    kvCacheManager->updateKVCacheStats(request2, 30);
    kvCacheManager->updateRequestStatus(request2, RequestStatus::COMPLETED);
    
    kvCacheManager->updateKVCacheStats(request3, 30);
    kvCacheManager->updateRequestStatus(request3, RequestStatus::COMPLETED);
    
    // 总条目数 = 90，总内存 = 180MB
    EXPECT_EQ(kvCacheManager->getTotalItems(), 90);
    EXPECT_EQ(kvCacheManager->getTotalMemoryMb(), 180);
    
    // 执行淘汰（无上下文，只测试统计信息清理）
    size_t evictedCount = kvCacheManager->evictLRUCache(nullptr, 0.8, nullptr);
    
    // 淘汰后，统计信息应该更新
    EXPECT_GT(evictedCount, 0);
    EXPECT_LE(kvCacheManager->getTotalItems(), 80);
    EXPECT_LE(kvCacheManager->getTotalMemoryMb(), 160);
}

TEST_F(KVCacheLRUEvictionTest, EvictionWithDifferentThresholds) {
    size_t requestId = 1;
    
    // 添加统计信息，总条目数 = 85，总内存 = 170MB
    kvCacheManager->updateKVCacheStats(requestId, 85);
    EXPECT_EQ(kvCacheManager->getTotalItems(), 85);
    EXPECT_EQ(kvCacheManager->getTotalMemoryMb(), 170);
    
    // 阈值 = 0.9 * 100 = 90 items, 90MB
    // 85 <= 90 (items), 170 > 90 (memory)，应该需要淘汰（因为内存超过阈值）
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.9));
    
    // 阈值 = 0.8 * 100 = 80 items, 80MB
    // 85 > 80 (items), 170 > 80 (memory)，应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.8));
    
    // 阈值 = 0.7 * 100 = 70 items, 70MB
    // 85 > 70 (items), 170 > 70 (memory)，应该需要淘汰
    EXPECT_TRUE(kvCacheManager->shouldEvict(0.7));
}
