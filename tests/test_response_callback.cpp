#include <gtest/gtest.h>
#include "cllm/scheduler/scheduler.h"
#include "cllm/common/request_state.h"
#include "cllm/common/config.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

namespace cllm {

class ResponseCallbackTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("../config/test_config.yaml");
        scheduler_ = nullptr;
        callbackInvoked_ = false;
        callbackRequestId_ = 0;
        callbackState_ = RequestState();
        callbackExceptionThrown_ = false;
    }
    
    void TearDown() override {
        if (scheduler_) {
            scheduler_->stop();
            delete scheduler_;
        }
    }
    
    void createScheduler() {
        try {
            scheduler_ = new Scheduler("", "", 4, 2048);
        } catch (const std::exception& e) {
            GTEST_SKIP() << "Scheduler initialization failed: " << e.what();
        }
        scheduler_->start();
    }
    
    void setupCallback() {
        scheduler_->setResponseCallback([this](size_t requestId, const RequestState& state) {
            callbackInvoked_ = true;
            callbackRequestId_ = requestId;
            callbackState_ = state;
            
            if (callbackShouldThrow_) {
                throw std::runtime_error("Test exception in callback");
            }
        });
    }
    
    Scheduler* scheduler_;
    bool callbackInvoked_;
    size_t callbackRequestId_;
    RequestState callbackState_;
    bool callbackShouldThrow_;
    bool callbackExceptionThrown_;
};

TEST_F(ResponseCallbackTest, SetAndGetCallback) {
    createScheduler();
    
    bool callbackSet = false;
    scheduler_->setResponseCallback([&callbackSet](size_t requestId, const RequestState& state) {
        callbackSet = true;
    });
    
    EXPECT_TRUE(callbackSet);
}

TEST_F(ResponseCallbackTest, CallbackException_DoesNotCrashScheduler) {
    createScheduler();
    
    callbackShouldThrow_ = true;
    setupCallback();
    
    RequestState request;
    request.requestId = 999;
    request.maxTokens = 10;
    request.temperature = 0.7f;
    request.isCompleted = false;
    request.isFailed = false;
    request.isRunning = false;
    request.startTime = 0;
    request.completionTime = 0;
    request.tokenizedPrompt = {1, 2, 3};
    request.generatedTokens = {4, 5, 6};
    
    size_t requestId = scheduler_->addRequest(request);
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    EXPECT_TRUE(callbackInvoked_);
    EXPECT_EQ(callbackRequestId_, requestId);
}

TEST_F(ResponseCallbackTest, MultipleCallbacks_AllInvoked) {
    createScheduler();
    
    std::vector<size_t> callbackRequestIds;
    std::vector<bool> callbackStates;
    
    scheduler_->setResponseCallback([&callbackRequestIds, &callbackStates](size_t requestId, const RequestState& state) {
        callbackRequestIds.push_back(requestId);
        callbackStates.push_back(state.isCompleted);
    });
    
    std::vector<RequestState> requests;
    for (int i = 0; i < 3; ++i) {
        RequestState request;
        request.requestId = 0;
        request.maxTokens = 10;
        request.temperature = 0.7f;
        request.isCompleted = false;
        request.isFailed = false;
        request.isRunning = false;
        request.startTime = 0;
        request.completionTime = 0;
        request.tokenizedPrompt = {1, 2, 3};
        request.generatedTokens = {4, 5, 6};
        requests.push_back(request);
    }
    
    for (auto& req : requests) {
        scheduler_->addRequest(req);
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    
    EXPECT_EQ(callbackRequestIds.size(), 3);
}

TEST_F(ResponseCallbackTest, Callback_ThreadSafety) {
    createScheduler();
    
    std::atomic<int> callbackCount{0};
    
    scheduler_->setResponseCallback([&callbackCount](size_t requestId, const RequestState& state) {
        callbackCount++;
    });
    
    const int numRequests = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numRequests; ++i) {
        threads.emplace_back([this, i]() {
            RequestState request;
            request.requestId = 0;
            request.maxTokens = 10;
            request.temperature = 0.7f;
            request.isCompleted = false;
            request.isFailed = false;
            request.isRunning = false;
            request.startTime = 0;
            request.completionTime = 0;
            request.tokenizedPrompt = {1, 2, 3};
            request.generatedTokens = {4, 5, 6};
            
            scheduler_->addRequest(request);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    
    EXPECT_GT(callbackCount.load(), 0);
}

} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
