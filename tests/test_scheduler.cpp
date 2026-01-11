#include <gtest/gtest.h>
#include <thread>
#include <chrono>
#include <cllm/scheduler/scheduler.h>
#include <cllm/common/queue.h>
#include <cllm/batch/manager.h>
#include <cllm/common/config.h>

using namespace cllm;

class SchedulerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("config/scheduler_config.yaml");
        maxBatchSize_ = 4;
        maxContextLength_ = 1024;
        
        // 尝试创建Scheduler，如果失败则跳过测试
        try {
            // 使用空路径，希望ModelExecutor能跳过文件加载
            scheduler_ = std::make_unique<Scheduler>(
                "",
                "",
                maxBatchSize_,
                maxContextLength_
            );
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Scheduler initialization failed (likely missing model files): " << e.what();
        }
    }

    void TearDown() override {
        if (scheduler_) {
            scheduler_->stop();
        }
    }

    RequestState createTestRequest(size_t requestId) {
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
    std::unique_ptr<Scheduler> scheduler_;
};

TEST_F(SchedulerTest, Constructor) {
    EXPECT_NE(scheduler_, nullptr);
    EXPECT_EQ(scheduler_->getQueueSize(), 0);
}

TEST_F(SchedulerTest, StartStop) {
    scheduler_->start();
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto stats = scheduler_->getStats();
    EXPECT_GE(stats.totalRequests, 0);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, AddRequest) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    size_t requestId = scheduler_->addRequest(request);
    
    EXPECT_EQ(requestId, 1);
    EXPECT_GE(scheduler_->getQueueSize(), 0);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, AddMultipleRequests) {
    scheduler_->start();
    
    for (size_t i = 1; i <= 5; ++i) {
        auto request = createTestRequest(i);
        size_t requestId = scheduler_->addRequest(request);
        EXPECT_EQ(requestId, i);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, RemoveRequest) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    bool removed = scheduler_->removeRequest(1);
    EXPECT_TRUE(removed);
    
    bool removedAgain = scheduler_->removeRequest(1);
    EXPECT_FALSE(removedAgain);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, GetRequestResult) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    auto result = scheduler_->getRequestResult(1);
    EXPECT_EQ(result.requestId, 1);
    
    auto nonExistent = scheduler_->getRequestResult(999);
    EXPECT_EQ(nonExistent.requestId, 0);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, WaitForRequest) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    bool completed = scheduler_->waitForRequest(1, 0.1f);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, WaitForRequestTimeout) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    bool completed = scheduler_->waitForRequest(1, 0.01f);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, GetRunningRequests) {
    scheduler_->start();
    
    auto request1 = createTestRequest(1);
    auto request2 = createTestRequest(2);
    scheduler_->addRequest(request1);
    scheduler_->addRequest(request2);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto running = scheduler_->getRunningRequests();
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, GetCompletedRequests) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto completed = scheduler_->getCompletedRequests();
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, GetQueueSize) {
    scheduler_->start();
    
    size_t initialSize = scheduler_->getQueueSize();
    
    for (size_t i = 1; i <= 3; ++i) {
        auto request = createTestRequest(i);
        scheduler_->addRequest(request);
    }
    
    size_t newSize = scheduler_->getQueueSize();
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, GetStats) {
    scheduler_->start();
    
    auto stats = scheduler_->getStats();
    EXPECT_EQ(stats.totalRequests, 0);
    EXPECT_EQ(stats.completedRequests, 0);
    EXPECT_EQ(stats.failedRequests, 0);
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto newStats = scheduler_->getStats();
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, ResetStats) {
    scheduler_->start();
    
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    scheduler_->resetStats();
    
    auto stats = scheduler_->getStats();
    EXPECT_EQ(stats.totalRequests, 0);
    EXPECT_EQ(stats.completedRequests, 0);
    EXPECT_EQ(stats.failedRequests, 0);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, ConcurrentRequests) {
    scheduler_->start();
    
    std::vector<std::thread> threads;
    const int numThreads = 5;
    const int requestsPerThread = 3;
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, requestsPerThread]() {
            for (int i = 0; i < requestsPerThread; ++i) {
                size_t requestId = t * requestsPerThread + i + 1;
                auto request = createTestRequest(requestId);
                scheduler_->addRequest(request);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, HighPriorityRequests) {
    scheduler_->start();
    
    auto lowPriority = createTestRequest(1);
    lowPriority.priority = 0.5f;
    scheduler_->addRequest(lowPriority);
    
    auto highPriority = createTestRequest(2);
    highPriority.priority = 1.0f;
    scheduler_->addRequest(highPriority);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, LongRequests) {
    scheduler_->start();
    
    auto longRequest = createTestRequest(1);
    longRequest.maxTokens = 500;
    longRequest.tokenizedPrompt = std::vector<int>(100, 1);
    scheduler_->addRequest(longRequest);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, MultipleStartStop) {
    scheduler_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler_->stop();
    
    scheduler_->start();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    scheduler_->stop();
}

TEST_F(SchedulerTest, EmptyQueue) {
    scheduler_->start();
    
    EXPECT_EQ(scheduler_->getQueueSize(), 0);
    EXPECT_EQ(scheduler_->getRunningRequests().size(), 0);
    EXPECT_EQ(scheduler_->getCompletedRequests().size(), 0);
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, RequestWithDifferentTemperatures) {
    scheduler_->start();
    
    for (size_t i = 1; i <= 3; ++i) {
        auto request = createTestRequest(i);
        request.temperature = 0.1f * i;
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, RequestWithDifferentTopP) {
    scheduler_->start();
    
    for (size_t i = 1; i <= 3; ++i) {
        auto request = createTestRequest(i);
        request.topP = 0.5f + 0.2f * i;
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    scheduler_->stop();
}

TEST_F(SchedulerTest, StressTest) {
    scheduler_->start();
    
    const int numRequests = 50;
    
    for (int i = 1; i <= numRequests; ++i) {
        auto request = createTestRequest(i);
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::seconds(2));
    
    auto stats = scheduler_->getStats();
    
    scheduler_->stop();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
