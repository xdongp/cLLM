#include <gtest/gtest.h>
#include "cllm/scheduler/scheduler.h"
#include "cllm/common/request_state.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <vector>

namespace cllm {

class ResponseCallbackSimpleTest : public ::testing::Test {
protected:
    void SetUp() override {
        callbackInvoked_ = false;
        callbackRequestId_ = 0;
        callbackState_ = RequestState();
    }
    
    void TearDown() override {
    }
    
    bool callbackInvoked_;
    size_t callbackRequestId_;
    RequestState callbackState_;
};

TEST_F(ResponseCallbackSimpleTest, CallbackFunctionType_CanBeStored) {
    ResponseCallback callback = [](size_t requestId, const RequestState& state) {
        // Do nothing
    };
    
    EXPECT_TRUE(callback != nullptr);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_CanBeInvoked) {
    bool invoked = false;
    size_t capturedRequestId = 0;
    
    ResponseCallback callback = [&invoked, &capturedRequestId](size_t requestId, const RequestState& state) {
        invoked = true;
        capturedRequestId = requestId;
    };
    
    RequestState request;
    request.requestId = 123;
    request.isCompleted = true;
    
    callback(123, request);
    
    EXPECT_TRUE(invoked);
    EXPECT_EQ(capturedRequestId, 123);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_CanCaptureState) {
    bool isCompleted = false;
    bool isFailed = false;
    
    ResponseCallback callback = [&isCompleted, &isFailed](size_t requestId, const RequestState& state) {
        isCompleted = state.isCompleted;
        isFailed = state.isFailed;
    };
    
    RequestState request;
    request.requestId = 456;
    request.isCompleted = true;
    request.isFailed = false;
    
    callback(456, request);
    
    EXPECT_TRUE(isCompleted);
    EXPECT_FALSE(isFailed);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_ExceptionHandling) {
    bool exceptionCaught = false;
    
    ResponseCallback callback = [&exceptionCaught](size_t requestId, const RequestState& state) {
        try {
            throw std::runtime_error("Test exception");
        } catch (const std::exception& e) {
            exceptionCaught = true;
        }
    };
    
    RequestState request;
    request.requestId = 789;
    
    callback(789, request);
    
    EXPECT_TRUE(exceptionCaught);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_ThreadSafety) {
    std::atomic<int> callbackCount{0};
    
    ResponseCallback callback = [&callbackCount](size_t requestId, const RequestState& state) {
        callbackCount++;
    };
    
    const int numThreads = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([callback, i]() {
            RequestState request;
            request.requestId = i;
            callback(i, request);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(callbackCount.load(), numThreads);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_MultipleCallbacks) {
    std::vector<size_t> requestIds;
    std::vector<bool> completedStates;
    
    ResponseCallback callback = [&requestIds, &completedStates](size_t requestId, const RequestState& state) {
        requestIds.push_back(requestId);
        completedStates.push_back(state.isCompleted);
    };
    
    for (int i = 0; i < 5; ++i) {
        RequestState request;
        request.requestId = i;
        request.isCompleted = (i % 2 == 0);
        callback(i, request);
    }
    
    EXPECT_EQ(requestIds.size(), 5);
    EXPECT_EQ(completedStates.size(), 5);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_LambdaCaptureByReference) {
    int counter = 0;
    
    ResponseCallback callback = [&counter](size_t requestId, const RequestState& state) {
        counter++;
    };
    
    RequestState request;
    request.requestId = 1;
    
    callback(1, request);
    callback(2, request);
    callback(3, request);
    
    EXPECT_EQ(counter, 3);
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_LambdaCaptureByValue) {
    int counter = 0;
    
    ResponseCallback callback = [counter](size_t requestId, const RequestState& state) mutable {
        counter++;
        return counter;
    };
    
    RequestState request;
    request.requestId = 1;
    
    callback(1, request);
    
    EXPECT_EQ(counter, 0);  // Original counter unchanged
}

TEST_F(ResponseCallbackSimpleTest, CallbackFunction_NullCallback) {
    ResponseCallback callback = nullptr;
    
    RequestState request;
    request.requestId = 1;
    
    if (callback) {
        callback(1, request);
    }
    
    EXPECT_TRUE(callback == nullptr);
}

} // namespace cllm

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}