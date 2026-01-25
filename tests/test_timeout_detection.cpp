#include <gtest/gtest.h>
#include <cllm/common/request_state.h>
#include <cllm/common/logger.h>
#include <thread>
#include <chrono>

using namespace cllm;

class TimeoutDetectionTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }

    RequestState createTestRequest(size_t requestId, float timeoutSeconds = 60.0f) {
        RequestState request;
        request.requestId = requestId;
        request.tokenizedPrompt = {1, 2, 3, 4, 5};
        request.maxTokens = 10;
        request.temperature = 0.7f;
        request.topP = 0.9f;
        request.priority = 1.0f;
        request.isCompleted = false;
        request.isRunning = true;
        request.isFailed = false;
        request.startTime = getCurrentTime();
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

    void sleepForSeconds(float seconds) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(seconds * 1000))
        );
    }
};

TEST_F(TimeoutDetectionTest, RequestStateIsTimeout) {
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 70 * 1000;
    
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) << "Request should be TIMEOUT";
    
    RequestState notTimeoutRequest = createTestRequest(2);
    notTimeoutRequest.startTime = getCurrentTime() - 30 * 1000;
    
    EXPECT_FALSE(notTimeoutRequest.checkTimeout(getCurrentTime(), 60.0f)) << "Request should not be TIMEOUT";
}

TEST_F(TimeoutDetectionTest, RequestStateProcessingTimeCalculation) {
    RequestState request = createTestRequest(1);
    size_t startTime = request.startTime;
    
    sleepForSeconds(0.1f);
    
    size_t currentTime = getCurrentTime();
    float processingTime = static_cast<float>(currentTime - startTime) / 1000.0f;
    
    EXPECT_GE(processingTime, 0.1f) << "Processing time should be at least 0.1s";
    EXPECT_LT(processingTime, 2.0f) << "Processing time should be less than 2.0s";
}

TEST_F(TimeoutDetectionTest, RequestStateTimeoutBoundary) {
    RequestState request = createTestRequest(1);
    float timeoutThreshold = 60.0f;
    
    request.startTime = getCurrentTime() - 59 * 1000;
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "Request should not be TIMEOUT before threshold";
    
    request.startTime = getCurrentTime() - 61 * 1000;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "Request should be TIMEOUT after threshold";
}

TEST_F(TimeoutDetectionTest, RequestStateTimeoutWithZeroStartTime) {
    RequestState request = createTestRequest(1);
    request.startTime = 0;
    
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "Request with zero start time should not be TIMEOUT";
}

TEST_F(TimeoutDetectionTest, RequestStateTimeoutWithVeryOldStartTime) {
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 10000 * 1000;
    
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "Request with very old start time should be TIMEOUT";
}

TEST_F(TimeoutDetectionTest, RequestStateTimeoutWithDifferentThresholds) {
    RequestState request = createTestRequest(1);
    
    request.startTime = getCurrentTime() - 30 * 1000;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 20.0f)) 
        << "Request should be TIMEOUT with 20s threshold";
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), 40.0f)) 
        << "Request should not be TIMEOUT with 40s threshold";
}

TEST_F(TimeoutDetectionTest, RequestStateTimeoutWithVeryLongThreshold) {
    RequestState request = createTestRequest(1);
    float timeoutThreshold = 3600.0f;
    
    request.startTime = getCurrentTime() - 1800 * 1000;
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "Request should not be TIMEOUT with 3600s threshold";
    
    request.startTime = getCurrentTime() - 3700 * 1000;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "Request should be TIMEOUT with 3600s threshold";
}

TEST_F(TimeoutDetectionTest, RequestStateIsActiveDoesNotAffectTimeout) {
    RequestState request = createTestRequest(1);
    request.isRunning = false;
    request.isCompleted = false;
    request.isFailed = false;
    
    request.startTime = getCurrentTime() - 70 * 1000;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "PENDING request should be TIMEOUT based on time";
    
    request.isRunning = true;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "PROCESSING request should be TIMEOUT based on time";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
