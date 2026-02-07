/**
 * @file timeout_detection_test.cpp
 * @brief 超时检测机制单元测试
 * 
 * 测试RequestState的超时检测功能
 * 迁移自：tests/test_timeout_detection.cpp
 */

#include <gtest/gtest.h>
#include "utils/test_base.h"
#include <cllm/common/request_state.h>
#include <cllm/common/logger.h>
#include <thread>
#include <chrono>

using namespace cllm;
using namespace cllm::test;

/**
 * @brief 超时检测测试类
 * 使用TestBase作为基类，提供通用的测试环境
 */
class TimeoutDetectionTest : public TestBase {
protected:
    void SetUp() override {
        TestBase::SetUp();
    }

    void TearDown() override {
        TestBase::TearDown();
    }

    /**
     * @brief 创建测试用的RequestState
     */
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

    /**
     * @brief 获取当前时间（毫秒）
     */
    size_t getCurrentTime() {
        auto now = std::chrono::high_resolution_clock::now();
        return static_cast<size_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()
            ).count()
        );
    }

    /**
     * @brief 睡眠指定秒数
     */
    void sleepForSeconds(float seconds) {
        std::this_thread::sleep_for(
            std::chrono::milliseconds(static_cast<int>(seconds * 1000))
        );
    }
};

// ========== 基础超时检测测试 ==========

TEST_F(TimeoutDetectionTest, CheckTimeout_OldRequest_ReturnsTrue) {
    // Arrange
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 70 * 1000;  // 70秒前
    
    // Act & Assert
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "70秒前的请求应该超时（阈值60秒）";
}

TEST_F(TimeoutDetectionTest, CheckTimeout_RecentRequest_ReturnsFalse) {
    // Arrange
    RequestState request = createTestRequest(2);
    request.startTime = getCurrentTime() - 30 * 1000;  // 30秒前
    
    // Act & Assert
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "30秒前的请求不应该超时（阈值60秒）";
}

// ========== 处理时间计算测试 ==========

TEST_F(TimeoutDetectionTest, ProcessingTime_AfterDelay_IsAccurate) {
    // Arrange
    RequestState request = createTestRequest(1);
    size_t startTime = request.startTime;
    
    // Act
    sleepForSeconds(0.1f);
    
    // Assert
    size_t currentTime = getCurrentTime();
    float processingTime = static_cast<float>(currentTime - startTime) / 1000.0f;
    
    EXPECT_GE(processingTime, 0.1f) << "处理时间应至少0.1秒";
    EXPECT_LT(processingTime, 2.0f) << "处理时间应小于2.0秒";
}

// ========== 边界条件测试 ==========

TEST_F(TimeoutDetectionTest, CheckTimeout_JustBeforeThreshold_ReturnsFalse) {
    // Arrange
    RequestState request = createTestRequest(1);
    float timeoutThreshold = 60.0f;
    request.startTime = getCurrentTime() - 59 * 1000;  // 59秒前
    
    // Act & Assert
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "59秒前的请求不应超时（阈值60秒）";
}

TEST_F(TimeoutDetectionTest, CheckTimeout_JustAfterThreshold_ReturnsTrue) {
    // Arrange
    RequestState request = createTestRequest(1);
    float timeoutThreshold = 60.0f;
    request.startTime = getCurrentTime() - 61 * 1000;  // 61秒前
    
    // Act & Assert
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "61秒前的请求应该超时（阈值60秒）";
}

TEST_F(TimeoutDetectionTest, CheckTimeout_ZeroStartTime_ReturnsFalse) {
    // Arrange
    RequestState request = createTestRequest(1);
    request.startTime = 0;
    
    // Act & Assert
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "startTime为0的请求不应超时";
}

TEST_F(TimeoutDetectionTest, CheckTimeout_VeryOldRequest_ReturnsTrue) {
    // Arrange
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 10000 * 1000;  // 10000秒前
    
    // Act & Assert
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "很久以前的请求应该超时";
}

// ========== 不同阈值测试 ==========

TEST_F(TimeoutDetectionTest, CheckTimeout_DifferentThresholds_WorksCorrectly) {
    // Arrange
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 30 * 1000;  // 30秒前
    
    // Act & Assert
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 20.0f)) 
        << "30秒前的请求应超时（阈值20秒）";
    
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), 40.0f)) 
        << "30秒前的请求不应超时（阈值40秒）";
}

TEST_F(TimeoutDetectionTest, CheckTimeout_VeryLongThreshold_WorksCorrectly) {
    // Arrange
    RequestState request = createTestRequest(1);
    float timeoutThreshold = 3600.0f;  // 1小时
    
    // 30分钟前
    request.startTime = getCurrentTime() - 1800 * 1000;
    EXPECT_FALSE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "30分钟前的请求不应超时（阈值1小时）";
    
    // 62分钟前
    request.startTime = getCurrentTime() - 3700 * 1000;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), timeoutThreshold)) 
        << "62分钟前的请求应该超时（阈值1小时）";
}

// ========== 状态不影响超时检测 ==========

TEST_F(TimeoutDetectionTest, CheckTimeout_RequestState_DoesNotAffectTimeout) {
    // Arrange
    RequestState request = createTestRequest(1);
    request.startTime = getCurrentTime() - 70 * 1000;  // 70秒前
    
    // 测试PENDING状态
    request.isRunning = false;
    request.isCompleted = false;
    request.isFailed = false;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "PENDING状态的请求应基于时间判断超时";
    
    // 测试PROCESSING状态
    request.isRunning = true;
    EXPECT_TRUE(request.checkTimeout(getCurrentTime(), 60.0f)) 
        << "PROCESSING状态的请求应基于时间判断超时";
}

// ========== 并发测试 ==========

TEST_F(TimeoutDetectionTest, CheckTimeout_Concurrent_ThreadSafe) {
    // Arrange
    const int numThreads = 10;
    std::vector<std::thread> threads;
    std::vector<bool> results(numThreads);
    
    // Act
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, &results]() {
            RequestState request = createTestRequest(i);
            request.startTime = getCurrentTime() - 70 * 1000;
            results[i] = request.checkTimeout(getCurrentTime(), 60.0f);
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    // Assert
    for (int i = 0; i < numThreads; ++i) {
        EXPECT_TRUE(results[i]) << "线程 " << i << " 的超时检测应该返回true";
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
