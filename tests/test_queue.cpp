#include <gtest/gtest.h>
#include "cllm/common/queue.h"
#include <thread>
#include <chrono>

using namespace cllm;

class RequestQueueTest : public ::testing::Test {
protected:
    void SetUp() override {
        queue = std::make_unique<RequestQueue>(100, 2048);
    }
    
    void TearDown() override {
        queue.reset();
    }
    
    RequestState createTestRequest(size_t requestId, size_t promptLength, int maxTokens = 100) {
        RequestState req;
        req.requestId = requestId;
        req.tokenizedPrompt = std::vector<int>(promptLength, 1);
        req.maxTokens = maxTokens;
        req.temperature = 0.7f;
        req.topK = 50;
        req.topP = 0.9f;
        req.samplingStrategy = "nucleus";
        req.isCompleted = false;
        return req;
    }
    
    std::unique_ptr<RequestQueue> queue;
};

TEST_F(RequestQueueTest, AddRequest) {
    RequestState req = createTestRequest(1, 50);
    
    bool success = queue->addRequest(req);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(queue->getQueueSize(), 1);
}

TEST_F(RequestQueueTest, AddMultipleRequests) {
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 1, 50);
        queue->addRequest(req);
    }
    
    EXPECT_EQ(queue->getQueueSize(), 5);
}

TEST_F(RequestQueueTest, AddRequestExceedsMaxSize) {
    RequestQueue smallQueue(3, 2048);
    
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, 50);
        EXPECT_TRUE(smallQueue.addRequest(req));
    }
    
    RequestState req = createTestRequest(4, 50);
    EXPECT_FALSE(smallQueue.addRequest(req));
    EXPECT_EQ(smallQueue.getQueueSize(), 3);
}

TEST_F(RequestQueueTest, RemoveRequest) {
    RequestState req1 = createTestRequest(1, 50);
    RequestState req2 = createTestRequest(2, 50);
    
    queue->addRequest(req1);
    queue->addRequest(req2);
    
    bool success = queue->removeRequest(1);
    
    EXPECT_TRUE(success);
    EXPECT_EQ(queue->getQueueSize(), 1);
}

TEST_F(RequestQueueTest, RemoveNonExistentRequest) {
    RequestState req = createTestRequest(1, 50);
    queue->addRequest(req);
    
    bool success = queue->removeRequest(999);
    
    EXPECT_FALSE(success);
    EXPECT_EQ(queue->getQueueSize(), 1);
}

TEST_F(RequestQueueTest, GetNextRequest) {
    RequestState req = createTestRequest(1, 50);
    queue->addRequest(req);
    
    RequestState nextReq = queue->getNextRequest();
    
    EXPECT_EQ(nextReq.requestId, 1);
    EXPECT_EQ(queue->getQueueSize(), 0);
}

TEST_F(RequestQueueTest, GetNextRequestBlocksWhenEmpty) {
    std::atomic<bool> requestAdded{false};
    
    std::thread consumer([&]() {
        RequestState nextReq = queue->getNextRequest();
        EXPECT_EQ(nextReq.requestId, 1);
    });
    
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    RequestState req = createTestRequest(1, 50);
    queue->addRequest(req);
    
    consumer.join();
}

TEST_F(RequestQueueTest, FormBatch) {
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 1, 100);
        queue->addRequest(req);
    }
    
    std::vector<RequestState> batch = queue->formBatch(2048);
    
    EXPECT_EQ(batch.size(), 5);
    EXPECT_EQ(queue->getQueueSize(), 0);
}

TEST_F(RequestQueueTest, FormBatchWithContextLimit) {
    for (size_t i = 0; i < 10; ++i) {
        RequestState req = createTestRequest(i + 1, 300);
        queue->addRequest(req);
    }
    
    std::vector<RequestState> batch = queue->formBatch(1024);
    
    EXPECT_LE(batch.size(), 3);
    EXPECT_GT(queue->getQueueSize(), 0);
}

TEST_F(RequestQueueTest, FormBatchWithRunningRequests) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, 200);
        running.push_back(req);
    }
    queue->updateRunningRequests(running);
    
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 10, 100);
        queue->addRequest(req);
    }
    
    std::vector<RequestState> batch = queue->formBatch(1024);
    
    EXPECT_LE(batch.size(), 4);
}

TEST_F(RequestQueueTest, FormBatchNoRequestsWhenRunningExceedsLimit) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 10; ++i) {
        RequestState req = createTestRequest(i + 1, 100);
        running.push_back(req);
    }
    queue->updateRunningRequests(running);
    
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 10, 100);
        queue->addRequest(req);
    }
    
    std::vector<RequestState> batch = queue->formBatch(1024);
    
    EXPECT_EQ(batch.size(), 0);
    EXPECT_EQ(queue->getQueueSize(), 5);
}

TEST_F(RequestQueueTest, GetQueueSize) {
    EXPECT_EQ(queue->getQueueSize(), 0);
    
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 1, 50);
        queue->addRequest(req);
    }
    
    EXPECT_EQ(queue->getQueueSize(), 5);
    
    queue->getNextRequest();
    EXPECT_EQ(queue->getQueueSize(), 4);
}

TEST_F(RequestQueueTest, GetRunningRequestsLength) {
    EXPECT_EQ(queue->getRunningRequestsLength(), 0);
    
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, 100);
        running.push_back(req);
    }
    queue->updateRunningRequests(running);
    
    EXPECT_EQ(queue->getRunningRequestsLength(), 300);
}

TEST_F(RequestQueueTest, GetAverageWaitTime) {
    EXPECT_FLOAT_EQ(queue->getAverageWaitTime(), 0.0f);
    
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, 50);
        queue->addRequest(req);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    float avgWaitTime = queue->getAverageWaitTime();
    EXPECT_GT(avgWaitTime, 0.0f);
}

TEST_F(RequestQueueTest, GetAverageRequestLength) {
    EXPECT_EQ(queue->getAverageRequestLength(), 0);
    
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, (i + 1) * 100);
        queue->addRequest(req);
    }
    
    size_t avgLength = queue->getAverageRequestLength();
    EXPECT_GT(avgLength, 0);
}

TEST_F(RequestQueueTest, UpdateRunningRequests) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 1, 100);
        running.push_back(req);
    }
    queue->updateRunningRequests(running);
    
    EXPECT_EQ(queue->getRunningRequestsLength(), 300);
    
    running.clear();
    queue->updateRunningRequests(running);
    
    EXPECT_EQ(queue->getRunningRequestsLength(), 0);
}

TEST_F(RequestQueueTest, Clear) {
    for (size_t i = 0; i < 5; ++i) {
        RequestState req = createTestRequest(i + 1, 50);
        queue->addRequest(req);
    }
    
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        RequestState req = createTestRequest(i + 10, 100);
        running.push_back(req);
    }
    queue->updateRunningRequests(running);
    
    queue->clear();
    
    EXPECT_EQ(queue->getQueueSize(), 0);
    EXPECT_EQ(queue->getRunningRequestsLength(), 0);
}

TEST_F(RequestQueueTest, PriorityBasedOrdering) {
    RequestState req1 = createTestRequest(1, 50);
    RequestState req2 = createTestRequest(2, 100);
    RequestState req3 = createTestRequest(3, 50);
    
    queue->addRequest(req1);
    queue->addRequest(req2);
    queue->addRequest(req3);
    
    RequestState nextReq = queue->getNextRequest();
    EXPECT_EQ(nextReq.requestId, 1);
    
    nextReq = queue->getNextRequest();
    EXPECT_EQ(nextReq.requestId, 3);
    
    nextReq = queue->getNextRequest();
    EXPECT_EQ(nextReq.requestId, 2);
}

TEST_F(RequestQueueTest, ThreadSafety) {
    const int numThreads = 10;
    const int requestsPerThread = 10;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back([this, i, requestsPerThread]() {
            for (int j = 0; j < requestsPerThread; ++j) {
                RequestState req = createTestRequest(i * requestsPerThread + j, 50);
                queue->addRequest(req);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_EQ(queue->getQueueSize(), numThreads * requestsPerThread);
}

TEST_F(RequestQueueTest, ConcurrentAddAndRemove) {
    const int numOperations = 100;
    std::vector<std::thread> threads;
    
    for (int i = 0; i < 5; ++i) {
        threads.emplace_back([this, numOperations]() {
            for (int j = 0; j < numOperations; ++j) {
                RequestState req = createTestRequest(j, 50);
                queue->addRequest(req);
            }
        });
    }
    
    for (int i = 0; i < 3; ++i) {
        threads.emplace_back([this, numOperations]() {
            for (int j = 0; j < numOperations; ++j) {
                queue->removeRequest(j);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    EXPECT_LE(queue->getQueueSize(), 2 * numOperations);
}

class RequestStateTest : public ::testing::Test {
protected:
    void SetUp() override {
        req.requestId = 1;
        req.tokenizedPrompt = {1, 2, 3, 4, 5};
        req.generatedTokens = {6, 7, 8};
        req.maxTokens = 100;
        req.temperature = 0.7f;
        req.topK = 50;
        req.topP = 0.9f;
        req.samplingStrategy = "nucleus";
        req.isCompleted = false;
    }
    
    RequestState req;
};

TEST_F(RequestStateTest, GetPromptLength) {
    EXPECT_EQ(req.getPromptLength(), 5);
    
    req.tokenizedPrompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(req.getPromptLength(), 10);
}

TEST_F(RequestStateTest, GetTotalLength) {
    EXPECT_EQ(req.getTotalLength(), 8);
    
    req.generatedTokens = {6, 7, 8, 9, 10};
    EXPECT_EQ(req.getTotalLength(), 10);
}

TEST_F(RequestStateTest, CalculatePriority) {
    size_t currentTime = 1000;
    req.arrivalTime = 500;
    
    float priority = req.calculatePriority(currentTime);
    
    EXPECT_GT(priority, 0.0f);
    
    req.arrivalTime = 900;
    float priority2 = req.calculatePriority(currentTime);
    
    EXPECT_GT(priority2, priority);
}

TEST_F(RequestStateTest, CalculatePriorityWithDifferentLengths) {
    size_t currentTime = 1000;
    req.arrivalTime = 500;
    
    float priority1 = req.calculatePriority(currentTime);
    
    req.tokenizedPrompt = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    float priority2 = req.calculatePriority(currentTime);
    
    EXPECT_GT(priority1, priority2);
}

TEST_F(RequestStateTest, RequestComparator) {
    RequestComparator comparator;
    
    RequestState req1;
    req1.requestId = 1;
    req1.tokenizedPrompt = {1, 2, 3};
    req1.arrivalTime = 500;
    req1.priority = 1.5f;
    
    RequestState req2;
    req2.requestId = 2;
    req2.tokenizedPrompt = {1, 2, 3, 4, 5};
    req2.arrivalTime = 600;
    req2.priority = 1.0f;
    
    EXPECT_FALSE(comparator(req1, req2)); // req1 has higher priority, so should not be placed below req2
    EXPECT_TRUE(comparator(req2, req1));  // req2 has lower priority, so should be placed below req1
}

TEST_F(RequestStateTest, RequestComparatorSamePriority) {
    RequestComparator comparator;
    
    RequestState req1;
    req1.requestId = 1;
    req1.tokenizedPrompt = {1, 2, 3};
    req1.arrivalTime = 500;
    req1.priority = 1.0f;
    
    RequestState req2;
    req2.requestId = 2;
    req2.tokenizedPrompt = {1, 2, 3, 4, 5};
    req2.arrivalTime = 600;
    req2.priority = 1.0f;
    
    EXPECT_FALSE(comparator(req1, req2)); // req1 has earlier arrival time, so should not be placed below req2
    EXPECT_TRUE(comparator(req2, req1));  // req2 has later arrival time, so should be placed below req1
}
