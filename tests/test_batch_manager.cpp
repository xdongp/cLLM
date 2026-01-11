#include <gtest/gtest.h>
#include "cllm/batch/manager.h"
#include "cllm/batch/input.h"
#include "cllm/batch/output.h"
#include "cllm/memory/float_array.h"
#include "cllm/common/config.h"

using namespace cllm;

class BatchManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        Config::instance().load("config/scheduler_config.yaml");
        batchManager = std::make_unique<BatchManager>(2048, 32);
    }
    
    void TearDown() override {
        batchManager.reset();
    }
    
    RequestState createTestRequest(size_t requestId, size_t promptLength, size_t generatedLength = 0) {
        RequestState req;
        req.requestId = requestId;
        req.tokenizedPrompt = std::vector<int>(promptLength, 1);
        req.generatedTokens = std::vector<int>(generatedLength, 2);
        req.maxTokens = 100;
        req.temperature = 0.7f;
        req.topK = 50;
        req.topP = 0.9f;
        req.samplingStrategy = "nucleus";
        req.isCompleted = false;
        return req;
    }
    
    std::unique_ptr<BatchManager> batchManager;
};

TEST_F(BatchManagerTest, FormBatchEmpty) {
    std::vector<RequestState> pending;
    std::vector<RequestState> running;
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_EQ(batch.size(), 0);
}

TEST_F(BatchManagerTest, FormBatchSingleRequest) {
    std::vector<RequestState> pending;
    pending.push_back(createTestRequest(1, 50));
    
    std::vector<RequestState> running;
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_EQ(batch.size(), 1);
    EXPECT_EQ(batch[0].requestId, 1);
}

TEST_F(BatchManagerTest, FormBatchMultipleRequests) {
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 5; ++i) {
        pending.push_back(createTestRequest(i + 1, 100));
    }
    
    std::vector<RequestState> running;
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_EQ(batch.size(), 5);
}

TEST_F(BatchManagerTest, FormBatchWithRunningRequests) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        running.push_back(createTestRequest(i + 1, 200, 50));
    }
    
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 5; ++i) {
        pending.push_back(createTestRequest(i + 10, 100));
    }
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_GT(batch.size(), 0);
    EXPECT_LE(batch.size(), 5);
}

TEST_F(BatchManagerTest, FormBatchNoRequestsWhenRunningExceedsThreshold) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 10; ++i) {
        running.push_back(createTestRequest(i + 1, 200, 50));
    }
    
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 5; ++i) {
        pending.push_back(createTestRequest(i + 10, 100));
    }
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_EQ(batch.size(), 0);
}

TEST_F(BatchManagerTest, FormBatchRespectsContextLimit) {
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 10; ++i) {
        pending.push_back(createTestRequest(i + 1, 500));
    }
    
    std::vector<RequestState> running;
    
    std::vector<RequestState> batch = batchManager->formBatch(pending, running);
    
    EXPECT_GT(batch.size(), 0);
    EXPECT_LE(batch.size(), 4);
}

TEST_F(BatchManagerTest, FormMultipleBatches) {
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 15; ++i) {
        pending.push_back(createTestRequest(i + 1, 100));
    }
    
    std::vector<RequestState> running;
    
    std::vector<RequestState> allBatches = batchManager->formMultipleBatches(pending, running);
    
    EXPECT_GT(allBatches.size(), 0);
    EXPECT_LE(allBatches.size(), 15);
}

TEST_F(BatchManagerTest, FormMultipleBatchesWithRunningRequests) {
    std::vector<RequestState> running;
    for (size_t i = 0; i < 3; ++i) {
        running.push_back(createTestRequest(i + 1, 200, 50));
    }
    
    std::vector<RequestState> pending;
    for (size_t i = 0; i < 10; ++i) {
        pending.push_back(createTestRequest(i + 10, 100));
    }
    
    std::vector<RequestState> allBatches = batchManager->formMultipleBatches(pending, running);
    
    EXPECT_GT(allBatches.size(), 0);
}

TEST_F(BatchManagerTest, PrepareBatchInput) {
    std::vector<RequestState> batch;
    batch.push_back(createTestRequest(1, 5));
    batch.push_back(createTestRequest(2, 3));
    
    BatchInput input = batchManager->prepareBatchInput(batch);
    
    EXPECT_EQ(input.batchSize, 2);
    EXPECT_EQ(input.inputIds.size(), 8);
    EXPECT_EQ(input.requestPositions.size(), 2);
    EXPECT_EQ(input.sequenceIds.size(), 2);
}

TEST_F(BatchManagerTest, PrepareBatchInputWithGeneratedTokens) {
    std::vector<RequestState> batch;
    auto req1 = createTestRequest(1, 5);
    req1.generatedTokens = {10, 20};
    batch.push_back(req1);
    
    auto req2 = createTestRequest(2, 3);
    req2.generatedTokens = {30};
    batch.push_back(req2);
    
    BatchInput input = batchManager->prepareBatchInput(batch);
    
    EXPECT_EQ(input.batchSize, 2);
    EXPECT_EQ(input.inputIds.size(), 11);
}

TEST_F(BatchManagerTest, ProcessBatchOutput) {
    std::vector<RequestState> batch;
    batch.push_back(createTestRequest(1, 5));
    batch.push_back(createTestRequest(2, 3));
    
    BatchOutput output;
    output.logits = FloatArray(1000);
    for (size_t i = 0; i < 1000; ++i) {
        output.logits[i] = static_cast<float>(i);
    }
    output.requestPositions = {{0, 1000}, {0, 1000}};
    output.sequenceIds = {1, 2};
    
    batchManager->processBatchOutput(batch, output);
    
    EXPECT_EQ(batch[0].generatedTokens.size(), 1);
    EXPECT_EQ(batch[1].generatedTokens.size(), 1);
}

TEST_F(BatchManagerTest, ProcessBatchOutputWithCompletedRequests) {
    std::vector<RequestState> batch;
    auto req1 = createTestRequest(1, 5);
    req1.isCompleted = true;
    batch.push_back(req1);
    
    auto req2 = createTestRequest(2, 3);
    batch.push_back(req2);
    
    BatchOutput output;
    output.logits = FloatArray(1000);
    for (size_t i = 0; i < 1000; ++i) {
        output.logits[i] = static_cast<float>(i);
    }
    output.requestPositions = {{0, 1000}, {0, 1000}};
    output.sequenceIds = {1, 2};
    
    batchManager->processBatchOutput(batch, output);
    
    EXPECT_TRUE(batch[0].isCompleted);
    EXPECT_EQ(batch[0].generatedTokens.size(), 0);
    EXPECT_EQ(batch[1].generatedTokens.size(), 1);
}

TEST_F(BatchManagerTest, CalculateOptimalBatchSize) {
    std::vector<RequestState> requests;
    for (size_t i = 0; i < 10; ++i) {
        requests.push_back(createTestRequest(i + 1, 100));
    }
    
    size_t optimalSize = batchManager->calculateOptimalBatchSize(requests, 100);
    
    EXPECT_GT(optimalSize, 0);
    EXPECT_LE(optimalSize, 48);
}

TEST_F(BatchManagerTest, CalculateOptimalBatchSizeLongRequests) {
    std::vector<RequestState> requests;
    for (size_t i = 0; i < 10; ++i) {
        requests.push_back(createTestRequest(i + 1, 600));
    }
    
    size_t optimalSize = batchManager->calculateOptimalBatchSize(requests, 600);
    
    EXPECT_GT(optimalSize, 0);
    EXPECT_LE(optimalSize, 16);
}

TEST_F(BatchManagerTest, CanAddToBatch) {
    std::vector<RequestState> currentBatch;
    currentBatch.push_back(createTestRequest(1, 100));
    
    RequestState newRequest = createTestRequest(2, 100);
    
    bool canAdd = batchManager->canAddToBatch(newRequest, currentBatch, 100, 32);
    
    EXPECT_TRUE(canAdd);
}

TEST_F(BatchManagerTest, CannotAddToBatchExceedsDynamicBatchSize) {
    std::vector<RequestState> currentBatch;
    for (size_t i = 0; i < 32; ++i) {
        currentBatch.push_back(createTestRequest(i + 1, 10));
    }
    
    RequestState newRequest = createTestRequest(33, 10);
    
    bool canAdd = batchManager->canAddToBatch(newRequest, currentBatch, 320, 32);
    
    EXPECT_FALSE(canAdd);
}

TEST_F(BatchManagerTest, CannotAddToBatchExceedsContextLimit) {
    std::vector<RequestState> currentBatch;
    currentBatch.push_back(createTestRequest(1, 1800));
    
    RequestState newRequest = createTestRequest(2, 500);
    
    bool canAdd = batchManager->canAddToBatch(newRequest, currentBatch, 1800, 32);
    
    EXPECT_FALSE(canAdd);
}

TEST_F(BatchManagerTest, GetStats) {
    BatchStats stats = batchManager->getStats();
    
    EXPECT_EQ(stats.totalBatches, 0);
    EXPECT_EQ(stats.totalRequests, 0);
    EXPECT_EQ(stats.totalTokens, 0);
}

TEST_F(BatchManagerTest, UpdateStats) {
    std::vector<RequestState> batch;
    for (size_t i = 0; i < 5; ++i) {
        batch.push_back(createTestRequest(i + 1, 100));
    }
    
    batchManager->formBatch(batch, {});
    
    BatchStats stats = batchManager->getStats();
    
    EXPECT_EQ(stats.totalBatches, 1);
    EXPECT_EQ(stats.totalRequests, 5);
    EXPECT_EQ(stats.totalTokens, 500);
}

TEST_F(BatchManagerTest, ResetStats) {
    std::vector<RequestState> batch;
    for (size_t i = 0; i < 5; ++i) {
        batch.push_back(createTestRequest(i + 1, 100));
    }
    
    batchManager->formBatch(batch, {});
    batchManager->resetStats();
    
    BatchStats stats = batchManager->getStats();
    
    EXPECT_EQ(stats.totalBatches, 0);
    EXPECT_EQ(stats.totalRequests, 0);
    EXPECT_EQ(stats.totalTokens, 0);
}

TEST_F(BatchManagerTest, CheckStoppingConditionsMaxTokens) {
    RequestState req = createTestRequest(1, 10);
    // Set generated tokens to maxTokens - 1 so that adding one more token will reach maxTokens
    req.generatedTokens = std::vector<int>(99, 1);
    req.maxTokens = 100;
    req.temperature = 0.0f; // Use greedy sampling to ensure deterministic behavior
    
    std::vector<RequestState> batch = {req};
    BatchOutput output;
    output.logits = FloatArray(1000);
    // Set logit at position 1 to be highest (arbitrary token)
    for (size_t i = 0; i < 1000; ++i) {
        output.logits[i] = (i == 1) ? 100.0f : 0.0f;
    }
    output.requestPositions = {{0, 1000}};
    output.sequenceIds = {1};
    
    batchManager->processBatchOutput(batch, output);
    
    EXPECT_TRUE(batch[0].isCompleted);
    EXPECT_EQ(batch[0].generatedTokens.size(), static_cast<size_t>(batch[0].maxTokens));
}

TEST_F(BatchManagerTest, CheckStoppingConditionsEOS) {
    RequestState req = createTestRequest(1, 10);
    req.generatedTokens = {1};
    req.maxTokens = 100;
    req.temperature = 0.0f; // Use greedy sampling to ensure deterministic behavior
    
    std::vector<RequestState> batch = {req};
    BatchOutput output;
    output.logits = FloatArray(1000);
    // Set logit at position 2 (EOS token) to be highest
    for (size_t i = 0; i < 1000; ++i) {
        output.logits[i] = (i == 2) ? 100.0f : 0.0f;
    }
    output.requestPositions = {{0, 1000}};
    output.sequenceIds = {1};
    
    batchManager->processBatchOutput(batch, output);
    
    EXPECT_TRUE(batch[0].isCompleted);
    EXPECT_EQ(batch[0].generatedTokens.back(), 2); // EOS token
}

// SampleToken tests removed since sampling is now handled by Sampler class

TEST_F(BatchManagerTest, BatchInputEmpty) {
    BatchInput input;
    
    EXPECT_TRUE(input.empty());
    EXPECT_EQ(input.getTotalTokens(), 0);
}

TEST_F(BatchManagerTest, BatchInputClear) {
    BatchInput input;
    input.batchSize = 5;
    input.inputIds = {1, 2, 3, 4, 5};
    input.requestPositions = {{0, 5}};
    input.sequenceIds = {1};
    
    input.clear();
    
    EXPECT_TRUE(input.empty());
    EXPECT_EQ(input.batchSize, 0);
    EXPECT_TRUE(input.inputIds.empty());
}

TEST_F(BatchManagerTest, BatchOutputGetLogitsForRequest) {
    BatchOutput output;
    // vocabSize = 32000, requestPositions = {{0, 500}, {500, 1000}}
    // 需要 1000 * 32000 的 logits
    size_t vocabSize = 32000;
    output.logits = FloatArray(1000 * vocabSize);
    for (size_t i = 0; i < output.logits.size(); ++i) {
        output.logits[i] = static_cast<float>(i % vocabSize); // 每个位置存储 vocabSize 的值
    }
    output.requestPositions = {{0, 500}, {500, 1000}};
    output.sequenceIds = {1, 2};
    
    // getLogitsForRequest 返回 vocabSize 维度的 logits
    FloatArray logits1 = output.getLogitsForRequest(0);
    EXPECT_EQ(logits1.size(), vocabSize);
    EXPECT_EQ(logits1[0], 0.0f);  // 第一个token的logits
    EXPECT_EQ(logits1[1], 1.0f);
    
    FloatArray logits2 = output.getLogitsForRequest(1);
    EXPECT_EQ(logits2.size(), vocabSize);
    EXPECT_EQ(logits2[0], 0.0f);  // 第500个token的logits
    EXPECT_EQ(logits2[1], 1.0f);
}

TEST_F(BatchManagerTest, BatchOutputGetLogitsForRequestInvalidIndex) {
    BatchOutput output;
    output.logits = FloatArray(1000);
    output.requestPositions = {{0, 1000}};
    output.sequenceIds = {1};
    
    FloatArray logits = output.getLogitsForRequest(10);
    
    EXPECT_EQ(logits.size(), 0);
}

TEST_F(BatchManagerTest, BatchOutputClear) {
    BatchOutput output;
    output.logits = FloatArray(1000);
    output.requestPositions = {{0, 1000}};
    output.sequenceIds = {1};
    
    output.clear();
    
    EXPECT_TRUE(output.empty());
    EXPECT_TRUE(output.logits.empty());
    EXPECT_TRUE(output.requestPositions.empty());
}

TEST_F(BatchManagerTest, BatchStatsUpdate) {
    BatchStats stats;
    
    stats.update(5, 500);
    
    EXPECT_EQ(stats.totalBatches, 1);
    EXPECT_EQ(stats.totalRequests, 5);
    EXPECT_EQ(stats.totalTokens, 500);
    EXPECT_FLOAT_EQ(stats.averageBatchSize, 5.0f);
    EXPECT_FLOAT_EQ(stats.averageBatchLength, 500.0f);
    EXPECT_EQ(stats.maxBatchSize, 5);
    EXPECT_EQ(stats.minBatchSize, 5);
}

TEST_F(BatchManagerTest, BatchStatsMultipleUpdates) {
    BatchStats stats;
    
    stats.update(5, 500);
    stats.update(10, 1000);
    
    EXPECT_EQ(stats.totalBatches, 2);
    EXPECT_EQ(stats.totalRequests, 15);
    EXPECT_EQ(stats.totalTokens, 1500);
    EXPECT_FLOAT_EQ(stats.averageBatchSize, 7.5f);
    EXPECT_FLOAT_EQ(stats.averageBatchLength, 750.0f);
    EXPECT_EQ(stats.maxBatchSize, 10);
    EXPECT_EQ(stats.minBatchSize, 5);
}

TEST_F(BatchManagerTest, BatchStatsReset) {
    BatchStats stats;
    stats.update(5, 500);
    
    stats.reset();
    
    EXPECT_EQ(stats.totalBatches, 0);
    EXPECT_EQ(stats.totalRequests, 0);
    EXPECT_EQ(stats.totalTokens, 0);
    EXPECT_FLOAT_EQ(stats.averageBatchSize, 0.0f);
    EXPECT_FLOAT_EQ(stats.averageBatchLength, 0.0f);
    EXPECT_EQ(stats.maxBatchSize, 0);
    EXPECT_EQ(stats.minBatchSize, 0);
}

TEST_F(BatchManagerTest, BatchStatsToString) {
    BatchStats stats;
    stats.update(5, 500);
    
    std::string str = stats.toString();
    
    EXPECT_FALSE(str.empty());
    EXPECT_NE(str.find("totalBatches"), std::string::npos);
    EXPECT_NE(str.find("totalRequests"), std::string::npos);
    EXPECT_NE(str.find("totalTokens"), std::string::npos);
}
