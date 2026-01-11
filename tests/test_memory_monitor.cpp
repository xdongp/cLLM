#include <gtest/gtest.h>
#include "cllm/memory/monitor.h"
#include <thread>
#include <vector>
#include <stdexcept>

using namespace cllm;

class MemoryMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        monitor = &MemoryMonitor::instance();
        monitor->setLimit(0);
        monitor->resetAll();
    }

    void TearDown() override {
        monitor->setLimit(0);
        monitor->resetAll();
    }

    MemoryMonitor* monitor;
};

TEST_F(MemoryMonitorTest, SingletonInstance) {
    MemoryMonitor& instance1 = MemoryMonitor::instance();
    MemoryMonitor& instance2 = MemoryMonitor::instance();
    EXPECT_EQ(&instance1, &instance2);
}

TEST_F(MemoryMonitorTest, SetAndGetLimit) {
    monitor->setLimit(1024);
    EXPECT_EQ(monitor->getLimit(), 1024);
    
    monitor->setLimit(2048);
    EXPECT_EQ(monitor->getLimit(), 2048);
}

TEST_F(MemoryMonitorTest, AllocateWithoutLimit) {
    monitor->setLimit(0);
    
    monitor->allocate(100);
    EXPECT_EQ(monitor->getUsed(), 100);
    
    monitor->allocate(200);
    EXPECT_EQ(monitor->getUsed(), 300);
}

TEST_F(MemoryMonitorTest, Deallocate) {
    monitor->setLimit(0);
    
    monitor->allocate(100);
    monitor->allocate(200);
    EXPECT_EQ(monitor->getUsed(), 300);
    
    monitor->deallocate(50);
    EXPECT_EQ(monitor->getUsed(), 250);
    
    monitor->deallocate(250);
    EXPECT_EQ(monitor->getUsed(), 0);
}

TEST_F(MemoryMonitorTest, PeakMemoryTracking) {
    monitor->setLimit(0);
    
    monitor->allocate(100);
    EXPECT_EQ(monitor->getPeak(), 100);
    
    monitor->allocate(200);
    EXPECT_EQ(monitor->getPeak(), 300);
    
    monitor->deallocate(100);
    EXPECT_EQ(monitor->getPeak(), 300);
    
    monitor->allocate(50);
    EXPECT_EQ(monitor->getPeak(), 300);
    
    monitor->allocate(100);
    EXPECT_EQ(monitor->getPeak(), 350);
}

TEST_F(MemoryMonitorTest, ResetPeak) {
    monitor->setLimit(0);
    monitor->resetAll();
    
    monitor->allocate(100);
    monitor->allocate(200);
    EXPECT_EQ(monitor->getPeak(), 300);
    
    monitor->resetPeak();
    EXPECT_EQ(monitor->getPeak(), 0);
    
    monitor->allocate(50);
    EXPECT_EQ(monitor->getPeak(), 350);
}

TEST_F(MemoryMonitorTest, MemoryLimitEnforcement) {
    monitor->setLimit(1000);
    
    monitor->allocate(500);
    EXPECT_EQ(monitor->getUsed(), 500);
    
    monitor->allocate(400);
    EXPECT_EQ(monitor->getUsed(), 900);
    
    EXPECT_THROW(monitor->allocate(200), std::runtime_error);
    EXPECT_EQ(monitor->getUsed(), 900);
}

TEST_F(MemoryMonitorTest, LimitCallback) {
    monitor->setLimit(1000);
    
    bool callbackCalled = false;
    size_t callbackUsed = 0;
    size_t callbackLimit = 0;
    
    monitor->setLimitCallback([&callbackCalled, &callbackUsed, &callbackLimit](size_t used, size_t limit) {
        callbackCalled = true;
        callbackUsed = used;
        callbackLimit = limit;
    });
    
    monitor->allocate(900);
    EXPECT_FALSE(callbackCalled);
    
    EXPECT_THROW(monitor->allocate(200), std::runtime_error);
    EXPECT_TRUE(callbackCalled);
    EXPECT_EQ(callbackUsed, 900);
    EXPECT_EQ(callbackLimit, 1000);
}

TEST_F(MemoryMonitorTest, NoLimitCallbackWhenDisabled) {
    monitor->setLimit(0);
    
    bool callbackCalled = false;
    monitor->setLimitCallback([&callbackCalled](size_t, size_t) {
        callbackCalled = true;
    });
    
    monitor->allocate(1000);
    EXPECT_FALSE(callbackCalled);
}

TEST_F(MemoryMonitorTest, ThreadSafety) {
    monitor->setLimit(10000);
    
    const int numThreads = 10;
    const int allocationsPerThread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, allocationsPerThread]() {
            for (int j = 0; j < allocationsPerThread; ++j) {
                monitor->allocate(10);
                std::this_thread::sleep_for(std::chrono::microseconds(1));
                monitor->deallocate(10);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(monitor->getUsed(), 0);
}

TEST_F(MemoryMonitorTest, LargeAllocations) {
    monitor->setLimit(0);
    
    size_t largeSize = 1024 * 1024 * 1024;
    monitor->allocate(largeSize);
    EXPECT_EQ(monitor->getUsed(), largeSize);
    EXPECT_EQ(monitor->getPeak(), largeSize);
    
    monitor->deallocate(largeSize);
    EXPECT_EQ(monitor->getUsed(), 0);
}

TEST_F(MemoryMonitorTest, ZeroAllocation) {
    monitor->setLimit(0);
    
    monitor->allocate(0);
    EXPECT_EQ(monitor->getUsed(), 0);
    
    monitor->deallocate(0);
    EXPECT_EQ(monitor->getUsed(), 0);
}

TEST_F(MemoryMonitorTest, ExactLimitBoundary) {
    monitor->setLimit(1000);
    
    monitor->allocate(500);
    monitor->allocate(500);
    EXPECT_EQ(monitor->getUsed(), 1000);
    
    EXPECT_THROW(monitor->allocate(1), std::runtime_error);
}
