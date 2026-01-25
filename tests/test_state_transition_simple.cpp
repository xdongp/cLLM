#include <gtest/gtest.h>
#include <cllm/common/request_state.h>
#include <cllm/common/logger.h>
#include <thread>
#include <chrono>

using namespace cllm;

class StateTransitionTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
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
};

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

TEST_F(StateTransitionTest, RequestStateTransitions) {
    RequestState request = createTestRequest(1);
    
    EXPECT_TRUE(request.isPending());
    EXPECT_FALSE(request.isProcessing());
    EXPECT_FALSE(request.isCompletedState());
    EXPECT_FALSE(request.isFailedState());
    
    request.isRunning = true;
    EXPECT_FALSE(request.isPending());
    EXPECT_TRUE(request.isProcessing());
    EXPECT_FALSE(request.isCompletedState());
    EXPECT_FALSE(request.isFailedState());
    
    request.isCompleted = true;
    request.isRunning = false;
    EXPECT_FALSE(request.isPending());
    EXPECT_FALSE(request.isProcessing());
    EXPECT_TRUE(request.isCompletedState());
    EXPECT_FALSE(request.isFailedState());
}

TEST_F(StateTransitionTest, RequestStateMaxTokensReached) {
    RequestState request = createTestRequest(1);
    request.maxTokens = 10;
    
    for (int i = 0; i < 10; ++i) {
        request.generatedTokens.push_back(i);
    }
    
    EXPECT_FALSE(request.isCompletedState()) << "Should not be completed before marking";
    
    request.isCompleted = true;
    EXPECT_TRUE(request.isCompletedState()) << "Should be completed after marking";
}

TEST_F(StateTransitionTest, RequestStateEosTokenReached) {
    RequestState request = createTestRequest(1);
    request.eosTokenId = 2;
    
    request.generatedTokens.push_back(1);
    EXPECT_FALSE(request.isCompletedState()) << "Should not be completed before EOS";
    
    request.generatedTokens.push_back(2);
    request.isCompleted = true;
    EXPECT_TRUE(request.isCompletedState()) << "Should be completed after EOS";
}

TEST_F(StateTransitionTest, MultipleRequestsStateTransitions) {
    const int numRequests = 5;
    std::vector<RequestState> requests;
    
    for (int i = 0; i < numRequests; ++i) {
        requests.push_back(createTestRequest(i + 1));
    }
    
    for (int i = 0; i < numRequests; ++i) {
        EXPECT_TRUE(requests[i].isPending()) << "Request " << i << " should be PENDING";
    }
    
    for (int i = 0; i < numRequests; ++i) {
        requests[i].isRunning = true;
    }
    
    for (int i = 0; i < numRequests; ++i) {
        EXPECT_TRUE(requests[i].isProcessing()) << "Request " << i << " should be PROCESSING";
    }
    
    for (int i = 0; i < numRequests; ++i) {
        requests[i].isCompleted = true;
        requests[i].isRunning = false;
    }
    
    for (int i = 0; i < numRequests; ++i) {
        EXPECT_TRUE(requests[i].isCompletedState()) << "Request " << i << " should be COMPLETED";
    }
}

TEST_F(StateTransitionTest, ConcurrentStateUpdates) {
    const int numThreads = 4;
    const int requestsPerThreadValue = 3;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<RequestState>> requestsPerThread(numThreads);
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, requestsPerThreadValue, &requestsPerThread]() {
            for (int i = 0; i < requestsPerThreadValue; ++i) {
                RequestState request = createTestRequest(t * requestsPerThreadValue + i + 1);
                
                EXPECT_TRUE(request.isPending());
                
                request.isRunning = true;
                EXPECT_TRUE(request.isProcessing());
                
                request.isCompleted = true;
                request.isRunning = false;
                EXPECT_TRUE(request.isCompletedState());
                
                requestsPerThread[t].push_back(request);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (int t = 0; t < numThreads; ++t) {
        for (int i = 0; i < requestsPerThreadValue; ++i) {
            EXPECT_TRUE(requestsPerThread[t][i].isCompletedState()) 
                << "Request " << t << "-" << i << " should be COMPLETED";
        }
    }
}

TEST_F(StateTransitionTest, RequestStateCopyAndAssignment) {
    RequestState request1 = createTestRequest(1);
    request1.isCompleted = true;
    
    RequestState request2 = request1;
    EXPECT_TRUE(request2.isCompletedState());
    EXPECT_EQ(request2.requestId, request1.requestId);
    
    RequestState request3 = createTestRequest(2);
    request3 = request1;
    EXPECT_TRUE(request3.isCompletedState());
    EXPECT_EQ(request3.requestId, request1.requestId);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
