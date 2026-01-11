#include <gtest/gtest.h>
#include "cllm/memory/float_array.h"
#include "cllm/memory/cache_manager.h"
#include "cllm/memory/executor_manager.h"
#include <thread>
#include <vector>
#include <stdexcept>

using namespace cllm;

class FloatArrayTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(FloatArrayTest, DefaultConstructor) {
    FloatArray arr;
    EXPECT_EQ(arr.size(), 0);
    EXPECT_NE(arr.data(), nullptr);
}

TEST_F(FloatArrayTest, ConstructorWithSize) {
    FloatArray arr(100);
    EXPECT_EQ(arr.size(), 100);
    EXPECT_NE(arr.data(), nullptr);
}

TEST_F(FloatArrayTest, AccessElements) {
    FloatArray arr(10);
    
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<float>(i) * 1.5f;
    }
    
    for (size_t i = 0; i < arr.size(); ++i) {
        EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i) * 1.5f);
    }
}

TEST_F(FloatArrayTest, ConstAccess) {
    FloatArray arr(10);
    
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<float>(i);
    }
    
    const FloatArray& constArr = arr;
    for (size_t i = 0; i < constArr.size(); ++i) {
        EXPECT_FLOAT_EQ(constArr[i], static_cast<float>(i));
    }
}

TEST_F(FloatArrayTest, Resize) {
    FloatArray arr(10);
    
    for (size_t i = 0; i < arr.size(); ++i) {
        arr[i] = static_cast<float>(i);
    }
    
    arr.resize(20);
    EXPECT_EQ(arr.size(), 20);
    
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i));
    }
    
    arr.resize(5);
    EXPECT_EQ(arr.size(), 5);
    
    for (size_t i = 0; i < 5; ++i) {
        EXPECT_FLOAT_EQ(arr[i], static_cast<float>(i));
    }
}

TEST_F(FloatArrayTest, CopyConstructor) {
    FloatArray arr1(10);
    
    for (size_t i = 0; i < arr1.size(); ++i) {
        arr1[i] = static_cast<float>(i);
    }
    
    FloatArray arr2(arr1);
    
    EXPECT_EQ(arr2.size(), arr1.size());
    for (size_t i = 0; i < arr2.size(); ++i) {
        EXPECT_FLOAT_EQ(arr2[i], arr1[i]);
    }
    
    arr1[0] = 999.0f;
    EXPECT_NE(arr2[0], arr1[0]);
}

TEST_F(FloatArrayTest, CopyAssignment) {
    FloatArray arr1(10);
    FloatArray arr2(5);
    
    for (size_t i = 0; i < arr1.size(); ++i) {
        arr1[i] = static_cast<float>(i);
    }
    
    arr2 = arr1;
    
    EXPECT_EQ(arr2.size(), arr1.size());
    for (size_t i = 0; i < arr2.size(); ++i) {
        EXPECT_FLOAT_EQ(arr2[i], arr1[i]);
    }
}

TEST_F(FloatArrayTest, MoveConstructor) {
    FloatArray arr1(10);
    
    for (size_t i = 0; i < arr1.size(); ++i) {
        arr1[i] = static_cast<float>(i);
    }
    
    float* originalData = arr1.data();
    size_t originalSize = arr1.size();
    
    FloatArray arr2(std::move(arr1));
    
    EXPECT_EQ(arr2.data(), originalData);
    EXPECT_EQ(arr2.size(), originalSize);
    EXPECT_FLOAT_EQ(arr2[0], 0.0f);
    EXPECT_FLOAT_EQ(arr2[9], 9.0f);
}

TEST_F(FloatArrayTest, MoveAssignment) {
    FloatArray arr1(10);
    FloatArray arr2(5);
    
    for (size_t i = 0; i < arr1.size(); ++i) {
        arr1[i] = static_cast<float>(i);
    }
    
    float* originalData = arr1.data();
    size_t originalSize = arr1.size();
    
    arr2 = std::move(arr1);
    
    EXPECT_EQ(arr2.data(), originalData);
    EXPECT_EQ(arr2.size(), originalSize);
    EXPECT_FLOAT_EQ(arr2[0], 0.0f);
    EXPECT_FLOAT_EQ(arr2[9], 9.0f);
}

TEST_F(FloatArrayTest, DataPointer) {
    FloatArray arr(10);
    
    float* ptr = arr.data();
    EXPECT_NE(ptr, nullptr);
    
    const float* constPtr = arr.data();
    EXPECT_NE(constPtr, nullptr);
}

class KVCacheMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<KVCacheMemoryManager>(10);
    }

    void TearDown() override {
        manager.reset();
    }

    std::unique_ptr<KVCacheMemoryManager> manager;
};

TEST_F(KVCacheMemoryManagerTest, Constructor) {
    EXPECT_EQ(manager->getTotalMemory(), 10 * 1024 * 1024);
    EXPECT_EQ(manager->getUsedMemory(), 0);
}

TEST_F(KVCacheMemoryManagerTest, InsertAndGet) {
    std::vector<float> keyCache = {1.0f, 2.0f, 3.0f};
    std::vector<float> valueCache = {4.0f, 5.0f, 6.0f};
    
    bool success = manager->insert("request1", keyCache, valueCache);
    EXPECT_TRUE(success);
    EXPECT_GT(manager->getUsedMemory(), 0);
    
    std::vector<float> retrievedKey, retrievedValue;
    success = manager->get("request1", retrievedKey, retrievedValue);
    EXPECT_TRUE(success);
    EXPECT_EQ(retrievedKey, keyCache);
    EXPECT_EQ(retrievedValue, valueCache);
}

TEST_F(KVCacheMemoryManagerTest, GetNonExistent) {
    std::vector<float> keyCache, valueCache;
    bool success = manager->get("nonexistent", keyCache, valueCache);
    EXPECT_FALSE(success);
}

TEST_F(KVCacheMemoryManagerTest, Evict) {
    std::vector<float> keyCache = {1.0f, 2.0f, 3.0f};
    std::vector<float> valueCache = {4.0f, 5.0f, 6.0f};
    
    manager->insert("request1", keyCache, valueCache);
    size_t usedBefore = manager->getUsedMemory();
    
    manager->evict("request1");
    size_t usedAfter = manager->getUsedMemory();
    
    EXPECT_LT(usedAfter, usedBefore);
    
    std::vector<float> retrievedKey, retrievedValue;
    bool success = manager->get("request1", retrievedKey, retrievedValue);
    EXPECT_FALSE(success);
}

TEST_F(KVCacheMemoryManagerTest, MultipleInserts) {
    std::vector<float> keyCache1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> valueCache1 = {4.0f, 5.0f, 6.0f};
    
    std::vector<float> keyCache2 = {7.0f, 8.0f, 9.0f};
    std::vector<float> valueCache2 = {10.0f, 11.0f, 12.0f};
    
    manager->insert("request1", keyCache1, valueCache1);
    manager->insert("request2", keyCache2, valueCache2);
    
    std::vector<float> retrievedKey, retrievedValue;
    EXPECT_TRUE(manager->get("request1", retrievedKey, retrievedValue));
    EXPECT_EQ(retrievedKey, keyCache1);
    
    EXPECT_TRUE(manager->get("request2", retrievedKey, retrievedValue));
    EXPECT_EQ(retrievedKey, keyCache2);
}

TEST_F(KVCacheMemoryManagerTest, EvictionCallback) {
    std::vector<float> keyCache = {1.0f, 2.0f, 3.0f};
    std::vector<float> valueCache = {4.0f, 5.0f, 6.0f};
    
    std::string evictedRequest;
    manager->setEvictionCallback([&evictedRequest](const std::string& requestId) {
        evictedRequest = requestId;
    });
    
    manager->insert("request1", keyCache, valueCache);
    manager->evict("request1");
    
    EXPECT_EQ(evictedRequest, "request1");
}

TEST_F(KVCacheMemoryManagerTest, MemoryLimit) {
    KVCacheMemoryManager smallManager(1);
    
    std::vector<float> largeKey(1000000, 1.0f);
    std::vector<float> largeValue(1000000, 2.0f);
    
    bool success = smallManager.insert("request1", largeKey, largeValue);
    EXPECT_FALSE(success);
}

TEST_F(KVCacheMemoryManagerTest, ThreadSafety) {
    const int numThreads = 10;
    const int insertsPerThread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, insertsPerThread]() {
            for (int j = 0; j < insertsPerThread; ++j) {
                std::string requestId = "request_" + std::to_string(i) + "_" + std::to_string(j);
                std::vector<float> keyCache = {static_cast<float>(j)};
                std::vector<float> valueCache = {static_cast<float>(j + 1)};
                
                manager->insert(requestId, keyCache, valueCache);
                
                std::vector<float> retrievedKey, retrievedValue;
                manager->get(requestId, retrievedKey, retrievedValue);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_GT(manager->getUsedMemory(), 0);
}

class ModelExecutorMemoryManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<ModelExecutorMemoryManager>(100);
    }

    void TearDown() override {
        manager.reset();
    }

    std::unique_ptr<ModelExecutorMemoryManager> manager;
};

TEST_F(ModelExecutorMemoryManagerTest, Constructor) {
    EXPECT_EQ(manager->getTempMemoryUsed(), 0);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 0);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}

TEST_F(ModelExecutorMemoryManagerTest, AllocateTempBuffer) {
    void* ptr = manager->allocateTempBuffer(1024);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(manager->getTempMemoryUsed(), 1024);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 1024);
}

TEST_F(ModelExecutorMemoryManagerTest, DeallocateTempBuffer) {
    void* ptr = manager->allocateTempBuffer(1024);
    EXPECT_EQ(manager->getTempMemoryUsed(), 1024);
    
    manager->deallocateTempBuffer(ptr);
    EXPECT_EQ(manager->getTempMemoryUsed(), 0);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}

TEST_F(ModelExecutorMemoryManagerTest, AllocateWeightsCache) {
    void* ptr = manager->allocateWeightsCache(2048);
    EXPECT_NE(ptr, nullptr);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 2048);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 2048);
}

TEST_F(ModelExecutorMemoryManagerTest, DeallocateWeightsCache) {
    void* ptr = manager->allocateWeightsCache(2048);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 2048);
    
    manager->deallocateWeightsCache(ptr);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 0);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}

TEST_F(ModelExecutorMemoryManagerTest, MultipleBuffers) {
    void* ptr1 = manager->allocateTempBuffer(1024);
    void* ptr2 = manager->allocateTempBuffer(2048);
    void* ptr3 = manager->allocateWeightsCache(4096);
    
    EXPECT_EQ(manager->getTempMemoryUsed(), 3072);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 4096);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 7168);
    
    manager->deallocateTempBuffer(ptr1);
    EXPECT_EQ(manager->getTempMemoryUsed(), 2048);
    
    manager->deallocateWeightsCache(ptr3);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 0);
    
    manager->deallocateTempBuffer(ptr2);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}

TEST_F(ModelExecutorMemoryManagerTest, ClearAll) {
    manager->allocateTempBuffer(1024);
    manager->allocateTempBuffer(2048);
    manager->allocateWeightsCache(4096);
    
    EXPECT_GT(manager->getTotalMemoryUsed(), 0);
    
    manager->clearAll();
    
    EXPECT_EQ(manager->getTempMemoryUsed(), 0);
    EXPECT_EQ(manager->getWeightsMemoryUsed(), 0);
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}

TEST_F(ModelExecutorMemoryManagerTest, ThreadSafety) {
    const int numThreads = 10;
    const int allocationsPerThread = 100;
    std::vector<std::thread> threads;
    std::vector<void*> pointers;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, allocationsPerThread, &pointers]() {
            for (int j = 0; j < allocationsPerThread; ++j) {
                void* ptr = manager->allocateTempBuffer(1024);
                if (ptr) {
                    pointers.push_back(ptr);
                    std::this_thread::sleep_for(std::chrono::microseconds(1));
                    manager->deallocateTempBuffer(ptr);
                }
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(manager->getTotalMemoryUsed(), 0);
}
