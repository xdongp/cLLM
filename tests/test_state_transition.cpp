#include <gtest/gtest.h>
#include <cllm/scheduler/scheduler.h>
#include <cllm/common/request_state.h>
#include <cllm/common/config.h>
#include <thread>
#include <chrono>
#include <filesystem>

using namespace cllm;
namespace fs = std::filesystem;

static std::string resolveSchedulerConfigPath() {
    const std::vector<std::string> candidates = {
        "config/scheduler_config.yaml",
        "../config/scheduler_config.yaml",
        "../../config/scheduler_config.yaml"
    };
    for (const auto& path : candidates) {
        if (fs::exists(path)) {
            return fs::absolute(path).string();
        }
    }
    return "config/scheduler_config.yaml";
}

class StateTransitionTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load(resolveSchedulerConfigPath());
        
        try {
            scheduler_ = std::make_unique<Scheduler>(
                "",
                "",
                4,
                1024
            );
            scheduler_->start();
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
        request.maxTokens = 10;
        request.temperature = 0.7f;
        request.topP = 0.9f;
        request.priority = 1.0f;
        request.isCompleted = false;
        request.isRunning = false;
        request.isFailed = false;
        return request;
    }

    size_t getCurrentTime() {
        auto now = std::chrono::high_resolution_clock::now();
        return static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()
            ).count()
        );
    }

    std::unique_ptr<Scheduler> scheduler_;
    
    bool waitForRequestDone(size_t requestId, int timeoutMs, RequestState* out = nullptr) {
        const float timeoutSec = static_cast<float>(timeoutMs) / 1000.0f;
        if (!scheduler_->waitForRequest(requestId, timeoutSec)) {
            return false;
        }
        if (out) {
            *out = scheduler_->getRequestResult(requestId);
        }
        return true;
    }
    
    bool waitForRequestVisible(size_t requestId, int timeoutMs, RequestState* out = nullptr, bool* inCompleted = nullptr) {
        auto start = std::chrono::steady_clock::now();
        while (std::chrono::steady_clock::now() - start < std::chrono::milliseconds(timeoutMs)) {
            auto completedRequests = scheduler_->getCompletedRequests();
            for (const auto& req : completedRequests) {
                if (req.requestId == requestId) {
                    if (out) *out = req;
                    if (inCompleted) *inCompleted = true;
                    return true;
                }
            }
            auto runningRequests = scheduler_->getRunningRequests();
            for (const auto& req : runningRequests) {
                if (req.requestId == requestId) {
                    if (out) *out = req;
                    if (inCompleted) *inCompleted = false;
                    return true;
                }
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        return false;
    }
};

TEST_F(StateTransitionTest, PendingToProcessingToCompleted) {
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    RequestState result;
    ASSERT_TRUE(waitForRequestDone(1, 2000, &result)) << "Request should complete within timeout";
    EXPECT_TRUE(result.isCompleted);
    EXPECT_FALSE(result.isFailed);
    EXPECT_GT(result.generatedTokens.size(), 0);
}

TEST_F(StateTransitionTest, PendingToProcessingToFailed) {
    auto request = createTestRequest(1);
    request.maxTokens = 0;
    scheduler_->addRequest(request);
    RequestState result;
    ASSERT_TRUE(waitForRequestDone(1, 2000, &result)) << "Request should finish within timeout";
    EXPECT_TRUE(result.isCompleted || result.isFailed);
}

TEST_F(StateTransitionTest, GetRunningRequestsFiltersCompleted) {
    auto request1 = createTestRequest(1);
    auto request2 = createTestRequest(2);
    auto request3 = createTestRequest(3);
    
    scheduler_->addRequest(request1);
    scheduler_->addRequest(request2);
    scheduler_->addRequest(request3);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    auto runningRequests = scheduler_->getRunningRequests();
    
    for (const auto& req : runningRequests) {
        EXPECT_FALSE(req.isCompleted) << "Running requests should not include completed requests";
        EXPECT_FALSE(req.isFailed) << "Running requests should not include failed requests";
    }
}

TEST_F(StateTransitionTest, RequestStateIsPending) {
    RequestState request = createTestRequest(1);
    
    EXPECT_TRUE(request.isPending()) << "New request should be PENDING";
    EXPECT_FALSE(request.isProcessing()) << "New request should not be PROCESSING";
    EXPECT_FALSE(request.isCompletedState()) << "New request should not be COMPLETED";
    EXPECT_FALSE(request.isFailedState()) << "New request should not be FAILED";
}

TEST_F(StateTransitionTest, RequestStateIsProcessing) {
    RequestState request = createTestRequest(1);
    request.isRunning = true;
    
    EXPECT_FALSE(request.isPending()) << "Running request should not be PENDING";
    EXPECT_TRUE(request.isProcessing()) << "Running request should be PROCESSING";
    EXPECT_FALSE(request.isCompletedState()) << "Running request should not be COMPLETED";
    EXPECT_FALSE(request.isFailedState()) << "Running request should not be FAILED";
}

TEST_F(StateTransitionTest, RequestStateIsCompleted) {
    RequestState request = createTestRequest(1);
    request.isCompleted = true;
    
    EXPECT_FALSE(request.isPending()) << "Completed request should not be PENDING";
    EXPECT_FALSE(request.isProcessing()) << "Completed request should not be PROCESSING";
    EXPECT_TRUE(request.isCompletedState()) << "Completed request should be COMPLETED";
    EXPECT_FALSE(request.isFailedState()) << "Completed request should not be FAILED";
}

TEST_F(StateTransitionTest, RequestStateIsFailed) {
    RequestState request = createTestRequest(1);
    request.isFailed = true;
    
    EXPECT_FALSE(request.isPending()) << "Failed request should not be PENDING";
    EXPECT_FALSE(request.isProcessing()) << "Failed request should not be PROCESSING";
    EXPECT_FALSE(request.isCompletedState()) << "Failed request should not be COMPLETED";
    EXPECT_TRUE(request.isFailedState()) << "Failed request should be FAILED";
}

TEST_F(StateTransitionTest, RequestStateIsActive) {
    RequestState pendingRequest = createTestRequest(1);
    EXPECT_TRUE(pendingRequest.isActive()) << "PENDING request should be active";
    
    RequestState processingRequest = createTestRequest(2);
    processingRequest.isRunning = true;
    EXPECT_TRUE(processingRequest.isActive()) << "PROCESSING request should be active";
    
    RequestState completedRequest = createTestRequest(3);
    completedRequest.isCompleted = true;
    EXPECT_FALSE(completedRequest.isActive()) << "COMPLETED request should not be active";
    
    RequestState failedRequest = createTestRequest(4);
    failedRequest.isFailed = true;
    EXPECT_FALSE(failedRequest.isActive()) << "FAILED request should not be active";
}

TEST_F(StateTransitionTest, RequestStateIsTimeout) {
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 70 * 1000;
    
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) << "Request should be TIMEOUT";
    
    RequestState notTimeoutRequest = createTestRequest(2);
    notTimeoutRequest.startTime = getCurrentTime() - 30 * 1000;
    
    EXPECT_FALSE(notTimeoutRequest.checkTimeout(getCurrentTime(), 60.0f)) << "Request should not be TIMEOUT";
}

TEST_F(StateTransitionTest, RequestFlowFromQueueToRunning) {
    auto request = createTestRequest(1);
    scheduler_->addRequest(request);
    RequestState observed;
    bool inCompleted = false;
    ASSERT_TRUE(waitForRequestVisible(1, 2000, &observed, &inCompleted))
        << "Request should appear in running or completed list";
    if (inCompleted) {
        EXPECT_TRUE(observed.isCompleted || observed.isFailed);
    } else {
        EXPECT_TRUE(observed.isRunning);
    }
}

TEST_F(StateTransitionTest, MultipleRequestsStateTransitions) {
    const int numRequests = 5;
    
    for (int i = 0; i < numRequests; ++i) {
        auto request = createTestRequest(i + 1);
        scheduler_->addRequest(request);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    auto completedRequests = scheduler_->getCompletedRequests();
    auto runningRequests = scheduler_->getRunningRequests();
    
    for (int i = 0; i < numRequests; ++i) {
        size_t requestId = i + 1;
        
        bool inCompleted = false;
        for (const auto& req : completedRequests) {
            if (req.requestId == requestId) {
                inCompleted = true;
                break;
            }
        }
        
        bool inRunning = false;
        for (const auto& req : runningRequests) {
            if (req.requestId == requestId) {
                inRunning = true;
                break;
            }
        }
        
        EXPECT_TRUE(inCompleted || inRunning) << "Request should be in completed or running";
        
        if (inCompleted) {
            for (const auto& req : completedRequests) {
                if (req.requestId == requestId) {
                    EXPECT_TRUE(req.isCompleted || req.isFailed) << "Completed request should have final state";
                    break;
                }
            }
        }
    }
}

TEST_F(StateTransitionTest, ConcurrentStateUpdates) {
    const int numThreads = 5;
    const int requestsPerThread = 3;
    
    std::vector<std::thread> threads;
    
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
    const size_t expected = numThreads * requestsPerThread;
    size_t completed = 0;
    for (size_t requestId = 1; requestId <= expected; ++requestId) {
        if (scheduler_->waitForRequest(requestId, 2.0f)) {
            completed++;
        }
    }
    EXPECT_EQ(completed, expected) << "All requests should complete";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
