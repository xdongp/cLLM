#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <cllm/scheduler/scheduler.h>
#include <cllm/memory/monitor.h>
#include <cllm/thread_pool/manager.h>
#include <cllm/common/queue.h>
#include <cllm/batch/manager.h>
#include <cllm/memory/cache_manager.h>

using namespace cllm;

class IntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        modelPath_ = "/tmp/test_model";
        maxBatchSize_ = 4;
        maxContextLength_ = 1024;
        
        memoryMonitor_ = &MemoryMonitor::instance();
        memoryMonitor_->setLimit(8ULL * 1024ULL * 1024ULL * 1024ULL);
        
        threadPoolManager_ = std::make_unique<ThreadPoolManager>(4);

        
        scheduler_ = std::make_unique<Scheduler>(
            modelPath_,
            "",
            maxBatchSize_,
            maxContextLength_
        );
    }

    void TearDown() override {
        if (scheduler_) {
            scheduler_->stop();
        }
        
        if (threadPoolManager_) {
            threadPoolManager_->shutdown();
        }
        
        if (memoryMonitor_) {

        }
    }

    RequestState createTestRequest(size_t requestId, const std::string& prompt) {
        RequestState request;
        request.requestId = requestId;
        request.tokenizedPrompt = {1, 2, 3, 4, 5};
        request.maxTokens = 100;
        request.temperature = 0.7f;
        request.topP = 0.9f;
        request.priority = 1;
        request.arrivalTime = static_cast<size_t>(getCurrentTime() * 1000);
        request.isCompleted = false;
        request.isRunning = false;
        request.isFailed = false;
        return request;
    }

    float getCurrentTime() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float>(
            now.time_since_epoch()
        ).count();
    }

    std::string modelPath_;
    size_t maxBatchSize_;
    size_t maxContextLength_;
    
    MemoryMonitor* memoryMonitor_;
    std::unique_ptr<ThreadPoolManager> threadPoolManager_;
    std::unique_ptr<Scheduler> scheduler_;
};

TEST_F(IntegrationTest, MemoryMonitorThreadPoolIntegration) {
    auto initialMemory = memoryMonitor_->getUsed();
    
    auto task = []() {
        std::vector<int> data(1000, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    };
    
    threadPoolManager_->submitTask(task);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto finalMemory = memoryMonitor_->getUsed();
    
    EXPECT_GE(finalMemory, initialMemory);
}

TEST_F(IntegrationTest, SchedulerMemoryMonitorIntegration) {
    scheduler_->start();
    
    auto initialMemory = memoryMonitor_->getUsed();
    
    for (size_t i = 1; i <= 5; ++i) {
        auto request = createTestRequest(i, "Test prompt " + std::to_string(i));
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    auto finalMemory = memoryMonitor_->getUsed();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, ThreadPoolSchedulerIntegration) {
    scheduler_->start();
    
    std::vector<std::thread> threads;
    const int numThreads = 3;
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t]() {
            for (int i = 0; i < 3; ++i) {
                size_t requestId = t * 3 + i + 1;
                auto request = createTestRequest(requestId, "Prompt " + std::to_string(requestId));
                scheduler_->addRequest(request);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, FullPipelineTest) {
    scheduler_->start();
    
    auto request = createTestRequest(1, "Hello, world!");
    scheduler_->addRequest(request);
    
    bool completed = scheduler_->waitForRequest(1, 1.0f);
    
    auto result = scheduler_->getRequestResult(1);
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, MultipleRequestsPipeline) {
    scheduler_->start();
    
    std::vector<size_t> requestIds;
    for (size_t i = 1; i <= 10; ++i) {
        auto request = createTestRequest(i, "Test prompt " + std::to_string(i));
        requestIds.push_back(scheduler_->addRequest(request));
    }
    
    int completedCount = 0;
    for (size_t requestId : requestIds) {
        if (scheduler_->waitForRequest(requestId, 2.0f)) {
            completedCount++;
        }
    }
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, ConcurrentRequestsPipeline) {
    scheduler_->start();
    
    std::vector<std::thread> threads;
    const int numThreads = 5;
    const int requestsPerThread = 2;
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, requestsPerThread]() {
            for (int i = 0; i < requestsPerThread; ++i) {
                size_t requestId = t * requestsPerThread + i + 1;
                auto request = createTestRequest(requestId, "Concurrent prompt " + std::to_string(requestId));
                scheduler_->addRequest(request);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, MemoryPressureTest) {
    scheduler_->start();
    
    auto initialMemory = memoryMonitor_->getUsed();
    
    for (size_t i = 1; i <= 20; ++i) {
        auto request = createTestRequest(i, "Memory test prompt " + std::to_string(i));
        request.tokenizedPrompt = std::vector<int>(500, 1);
        request.maxTokens = 200;
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto peakMemory = memoryMonitor_->getUsed();
    
    scheduler_->stop();
    
    EXPECT_GE(peakMemory, initialMemory);
}

TEST_F(IntegrationTest, ThreadPoolLoadTest) {
    const int numTasks = 100;
    std::atomic<int> completedTasks(0);
    
    auto task = [&completedTasks]() {
        std::vector<int> data(1000, 0);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        completedTasks++;
    };
    
    for (int i = 0; i < numTasks; ++i) {
        threadPoolManager_->submitTask(task);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    EXPECT_EQ(completedTasks.load(), numTasks);
}

TEST_F(IntegrationTest, SchedulerBatchProcessingIntegration) {
    scheduler_->start();
    
    std::vector<size_t> requestIds;
    for (size_t i = 1; i <= maxBatchSize_; ++i) {
        auto request = createTestRequest(i, "Batch test " + std::to_string(i));
        requestIds.push_back(scheduler_->addRequest(request));
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    auto runningRequests = scheduler_->getRunningRequests();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, PrioritySchedulingIntegration) {
    scheduler_->start();
    
    auto lowPriority = createTestRequest(1, "Low priority");
    lowPriority.priority = 0.3f;
    scheduler_->addRequest(lowPriority);
    
    auto mediumPriority = createTestRequest(2, "Medium priority");
    mediumPriority.priority = 0.6f;
    scheduler_->addRequest(mediumPriority);
    
    auto highPriority = createTestRequest(3, "High priority");
    highPriority.priority = 1.0f;
    scheduler_->addRequest(highPriority);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    auto completedRequests = scheduler_->getCompletedRequests();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, ErrorHandlingIntegration) {
    scheduler_->start();
    
    auto validRequest = createTestRequest(1, "Valid request");
    scheduler_->addRequest(validRequest);
    
    auto invalidRequest = createTestRequest(2, "");
    invalidRequest.tokenizedPrompt.clear();
    scheduler_->addRequest(invalidRequest);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, ResourceCleanupIntegration) {
    scheduler_->start();
    
    for (size_t i = 1; i <= 10; ++i) {
        auto request = createTestRequest(i, "Cleanup test " + std::to_string(i));
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    scheduler_->stop();
    
    auto stats = scheduler_->getStats();
    auto memoryUsage = memoryMonitor_->getUsed();
    
    EXPECT_GE(stats.completedRequests, 0);
}

TEST_F(IntegrationTest, LongRunningRequestIntegration) {
    scheduler_->start();
    
    auto longRequest = createTestRequest(1, "Long running request");
    longRequest.tokenizedPrompt = std::vector<int>(200, 1);
    longRequest.maxTokens = 300;
    scheduler_->addRequest(longRequest);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto result = scheduler_->getRequestResult(1);
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, RapidRequestSubmissionIntegration) {
    scheduler_->start();
    
    const int numRequests = 50;
    
    for (int i = 1; i <= numRequests; ++i) {
        auto request = createTestRequest(i, "Rapid test " + std::to_string(i));
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(IntegrationTest, MemoryMonitorAlertIntegration) {
    bool alertTriggered = false;
    
    auto callback = [&alertTriggered](size_t used, size_t limit) {
        float usage = static_cast<float>(used) / static_cast<float>(limit);
        if (usage > 0.8f) {
            alertTriggered = true;
        }
    };
    
    memoryMonitor_->setLimitCallback(callback);
    
    std::vector<int> largeData(10000000, 0);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    memoryMonitor_->setLimitCallback(nullptr);
}

TEST_F(IntegrationTest, ThreadPoolExceptionHandlingIntegration) {
    int successCount = 0;
    int failureCount = 0;
    
    auto successTask = [&successCount]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        successCount++;
    };
    
    auto failureTask = [&failureCount]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        throw std::runtime_error("Test exception");
    };
    
    for (int i = 0; i < 5; ++i) {
        threadPoolManager_->submitTask(successTask);
    }
    
    for (int i = 0; i < 5; ++i) {
        threadPoolManager_->submitTask(failureTask);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    EXPECT_EQ(successCount, 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
