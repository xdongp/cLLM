#include <gtest/gtest.h>
#include "cllm/thread_pool/manager.h"
#include "cllm/thread_pool/monitor.h"
#include <thread>
#include <vector>
#include <atomic>
#include <chrono>

using namespace cllm;

class ThreadPoolManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<ThreadPoolManager>(4);
    }

    void TearDown() override {
        if (manager) {
            manager->shutdown();
        }
        manager.reset();
    }

    std::unique_ptr<ThreadPoolManager> manager;
};

TEST_F(ThreadPoolManagerTest, Constructor) {
    EXPECT_EQ(manager->getThreadCount(), 4);
    // BS::thread_pool的get_tasks_total()返回tasks_running + tasks.size()
    // worker线程启动后会立即将tasks_running减为0，所以初始值是0
    EXPECT_EQ(manager->getTasksTotal(), 0);
    EXPECT_EQ(manager->getTasksRunning(), 0);
    EXPECT_EQ(manager->getTasksQueued(), 0);
}

TEST_F(ThreadPoolManagerTest, SubmitTask) {
    std::atomic<int> counter{0};
    
    manager->submitTask([&counter]() {
        counter.fetch_add(1);
    });
    
    manager->waitForAll();
    
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(ThreadPoolManagerTest, SubmitMultipleTasks) {
    std::atomic<int> counter{0};
    const int numTasks = 100;
    
    for (int i = 0; i < numTasks; ++i) {
        manager->submitTask([&counter]() {
            counter.fetch_add(1);
        });
    }
    
    manager->waitForAll();
    
    EXPECT_EQ(counter.load(), numTasks);
}

TEST_F(ThreadPoolManagerTest, SubmitTaskWithResult) {
    auto future = manager->submitTaskWithResult([]() -> int {
        return 42;
    });
    
    int result = future.get();
    EXPECT_EQ(result, 42);
}

TEST_F(ThreadPoolManagerTest, SubmitTaskWithResultMultiple) {
    std::vector<std::future<int>> futures;
    
    for (int i = 0; i < 10; ++i) {
        futures.push_back(manager->submitTaskWithResult([i]() -> int {
            return i * 2;
        }));
    }
    
    for (size_t i = 0; i < futures.size(); ++i) {
        int result = futures[i].get();
        EXPECT_EQ(result, static_cast<int>(i * 2));
    }
}

TEST_F(ThreadPoolManagerTest, WaitForAll) {
    std::atomic<int> counter{0};
    const int numTasks = 50;
    
    for (int i = 0; i < numTasks; ++i) {
        manager->submitTask([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }
    
    manager->waitForAll();
    
    EXPECT_EQ(counter.load(), numTasks);
}

TEST_F(ThreadPoolManagerTest, PauseAndResume) {
    std::atomic<int> counter{0};
    
    manager->pause();
    EXPECT_TRUE(manager->isPaused());
    
    manager->submitTask([&counter]() {
        counter.fetch_add(1);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_EQ(counter.load(), 0);
    
    manager->resume();
    EXPECT_FALSE(manager->isPaused());
    
    manager->waitForAll();
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(ThreadPoolManagerTest, MultiplePauseResume) {
    std::atomic<int> counter{0};
    
    for (int i = 0; i < 3; ++i) {
        manager->pause();
        EXPECT_TRUE(manager->isPaused());
        
        manager->submitTask([&counter]() {
            counter.fetch_add(1);
        });
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        EXPECT_EQ(counter.load(), i);
        
        manager->resume();
        EXPECT_FALSE(manager->isPaused());
        
        manager->waitForAll();
        EXPECT_EQ(counter.load(), i + 1);
    }
}

TEST_F(ThreadPoolManagerTest, TaskExecutionOrder) {
    std::vector<int> executionOrder;
    std::mutex orderMutex;
    
    for (int i = 0; i < 10; ++i) {
        manager->submitTask([&executionOrder, &orderMutex, i]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10 * (10 - i)));
            std::lock_guard<std::mutex> lock(orderMutex);
            executionOrder.push_back(i);
        });
    }
    
    manager->waitForAll();
    
    EXPECT_EQ(executionOrder.size(), 10);
}

TEST_F(ThreadPoolManagerTest, ExceptionInTask) {
    auto future = manager->submitTaskWithResult([]() -> int {
        throw std::runtime_error("Test exception");
        return 42;
    });
    
    EXPECT_THROW(future.get(), std::runtime_error);
}

TEST_F(ThreadPoolManagerTest, Shutdown) {
    std::atomic<int> counter{0};
    
    for (int i = 0; i < 10; ++i) {
        manager->submitTask([&counter]() {
            counter.fetch_add(1);
        });
    }
    
    manager->shutdown();
    
    EXPECT_EQ(counter.load(), 10);
}

TEST_F(ThreadPoolManagerTest, ThreadSafety) {
    std::atomic<int> counter{0};
    const int numThreads = 10;
    const int tasksPerThread = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, &counter, tasksPerThread]() {
            for (int j = 0; j < tasksPerThread; ++j) {
                manager->submitTask([&counter]() {
                    counter.fetch_add(1);
                });
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    manager->waitForAll();
    
    EXPECT_EQ(counter.load(), numThreads * tasksPerThread);
}

TEST_F(ThreadPoolManagerTest, TaskStats) {
    std::atomic<int> counter{0};
    
    manager->submitTask([&counter]() {
        counter.fetch_add(1);
    });
    
    manager->waitForAll();
    
    EXPECT_EQ(counter.load(), 1);
}

TEST_F(ThreadPoolManagerTest, LongRunningTask) {
    std::atomic<bool> taskStarted{false};
    std::atomic<bool> taskCompleted{false};
    
    manager->submitTask([&taskStarted, &taskCompleted]() {
        taskStarted.store(true);
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        taskCompleted.store(true);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_TRUE(taskStarted.load());
    EXPECT_FALSE(taskCompleted.load());
    
    manager->waitForAll();
    
    EXPECT_TRUE(taskCompleted.load());
}

TEST_F(ThreadPoolManagerTest, ZeroThreadCount) {
    // BS::thread_pool接受0线程数，但会使用硬件并发数
    ThreadPoolManager zeroPool(0);
    EXPECT_GT(zeroPool.getThreadCount(), 0);
    zeroPool.shutdown();
}

TEST_F(ThreadPoolManagerTest, LargeThreadPool) {
    ThreadPoolManager largePool(20);
    
    std::atomic<int> counter{0};
    const int numTasks = 100;
    
    for (int i = 0; i < numTasks; ++i) {
        largePool.submitTask([&counter]() {
            counter.fetch_add(1);
        });
    }
    
    largePool.waitForAll();
    
    EXPECT_EQ(counter.load(), numTasks);
    
    largePool.shutdown();
}

class ThreadPoolMonitorTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager = std::make_unique<ThreadPoolManager>(4);
        monitor = std::make_unique<ThreadPoolMonitor>(*manager);
    }

    void TearDown() override {
        if (monitor) {
            monitor->stopMonitoring();
        }
        if (manager) {
            manager->shutdown();
        }
        monitor.reset();
        manager.reset();
    }

    std::unique_ptr<ThreadPoolManager> manager;
    std::unique_ptr<ThreadPoolMonitor> monitor;
};

TEST_F(ThreadPoolMonitorTest, Constructor) {
    EXPECT_NE(monitor, nullptr);
}

TEST_F(ThreadPoolMonitorTest, StartStopMonitoring) {
    monitor->startMonitoring(100);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    monitor->stopMonitoring();
}

TEST_F(ThreadPoolMonitorTest, GetStats) {
    monitor->startMonitoring(100);
    
    std::atomic<int> counter{0};
    
    for (int i = 0; i < 10; ++i) {
        manager->submitTask([&counter]() {
            counter.fetch_add(1);
        });
    }
    
    manager->waitForAll();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    ThreadPoolStats stats = monitor->getStats();
    
    EXPECT_EQ(stats.total_threads, 4);
    EXPECT_EQ(stats.completed_tasks, 10);
    
    monitor->stopMonitoring();
}

TEST_F(ThreadPoolMonitorTest, ContinuousMonitoring) {
    monitor->startMonitoring(50);
    
    std::atomic<int> counter{0};
    
    for (int i = 0; i < 20; ++i) {
        manager->submitTask([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }
    
    manager->waitForAll();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    ThreadPoolStats stats = monitor->getStats();
    
    EXPECT_EQ(stats.completed_tasks, 20);
    
    monitor->stopMonitoring();
}

TEST_F(ThreadPoolMonitorTest, StatsAccuracy) {
    monitor->startMonitoring(100);
    
    std::atomic<int> counter{0};
    const int numTasks = 50;
    
    for (int i = 0; i < numTasks; ++i) {
        manager->submitTask([&counter]() {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            counter.fetch_add(1);
        });
    }
    
    manager->waitForAll();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    ThreadPoolStats stats = monitor->getStats();
    
    EXPECT_EQ(stats.total_threads, 4);
    EXPECT_EQ(stats.completed_tasks, numTasks);
    EXPECT_GE(stats.avg_task_time_ms, 0);
    
    monitor->stopMonitoring();
}
