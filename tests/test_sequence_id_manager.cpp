#include <gtest/gtest.h>
#include <cllm/inference/llama_cpp_backend.h>
#include <cllm/common/config.h>
#include <cllm/common/logger.h>

using namespace cllm;
using namespace cllm::inference;

class SequenceIdManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        try {
            Config::instance().load("config/scheduler_config.yaml");
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to load config: " << e.what() << std::endl;
        }
        
        try {
            backend_ = std::make_unique<LlamaCppBackend>(
                ModelConfig(),
                ""
            );
            backend_->initialize();
        } catch (const std::exception& e) {
            std::cerr << "Warning: LlamaCppBackend initialization failed: " << e.what() << std::endl;
        }
        
        if (backend_) {
            CLLM_INFO("Testing sequence ID management with backend");
        } else {
            CLLM_WARN("Backend not initialized, skipping tests that require backend");
        }
    }

    void TearDown() override {
        if (backend_) {
            backend_.reset();
        }
    }

    std::unique_ptr<LlamaCppBackend> backend_;
};

TEST_F(SequenceIdManagerTest, AllocateSingleSequenceId) {
    size_t requestId = 1;
    int32_t seqId = backend_->allocateSequenceId(requestId);
    
    EXPECT_GE(seqId, 0);
    EXPECT_LT(seqId, 256);
    
    int32_t retrievedSeqId = backend_->getSequenceId(requestId);
    EXPECT_EQ(seqId, retrievedSeqId);
}

TEST_F(SequenceIdManagerTest, ReleaseSingleSequenceId) {
    size_t requestId = 1;
    int32_t seqId = backend_->allocateSequenceId(requestId);
    
    EXPECT_GE(seqId, 0);
    
    bool released = backend_->releaseSequenceId(requestId);
    EXPECT_TRUE(released);
    
    int32_t retrievedSeqId = backend_->getSequenceId(requestId);
    EXPECT_EQ(retrievedSeqId, -1);
}

TEST_F(SequenceIdManagerTest, AllocateMultipleSequenceIds) {
    const int numRequests = 10;
    std::vector<size_t> requestIds;
    std::vector<int32_t> seqIds;
    
    for (int i = 0; i < numRequests; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = backend_->allocateSequenceId(requestId);
        
        EXPECT_GE(seqId, 0);
        EXPECT_LT(seqId, 256);
        
        requestIds.push_back(requestId);
        seqIds.push_back(seqId);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        int32_t retrievedSeqId = backend_->getSequenceId(requestIds[i]);
        EXPECT_EQ(seqIds[i], retrievedSeqId);
    }
}

TEST_F(SequenceIdManagerTest, ReleaseMultipleSequenceIds) {
    const int numRequests = 10;
    std::vector<size_t> requestIds;
    
    for (int i = 0; i < numRequests; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = backend_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
        requestIds.push_back(requestId);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        bool released = backend_->releaseSequenceId(requestIds[i]);
        EXPECT_TRUE(released);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        int32_t retrievedSeqId = backend_->getSequenceId(requestIds[i]);
        EXPECT_EQ(retrievedSeqId, -1);
    }
}

TEST_F(SequenceIdManagerTest, SequenceIdReuse) {
    size_t requestId1 = 1;
    int32_t seqId1 = backend_->allocateSequenceId(requestId1);
    EXPECT_GE(seqId1, 0);
    
    bool released = backend_->releaseSequenceId(requestId1);
    EXPECT_TRUE(released);
    
    size_t requestId2 = 2;
    int32_t seqId2 = backend_->allocateSequenceId(requestId2);
    EXPECT_GE(seqId2, 0);
    
    EXPECT_EQ(seqId1, seqId2);
}

TEST_F(SequenceIdManagerTest, AllocateSameRequestIdTwice) {
    size_t requestId = 1;
    int32_t seqId1 = backend_->allocateSequenceId(requestId);
    EXPECT_GE(seqId1, 0);
    
    int32_t seqId2 = backend_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId1, seqId2);
}

TEST_F(SequenceIdManagerTest, ReleaseNonExistentRequestId) {
    size_t requestId = 999;
    bool released = backend_->releaseSequenceId(requestId);
    EXPECT_FALSE(released);
}

TEST_F(SequenceIdManagerTest, GetNonExistentRequestId) {
    size_t requestId = 999;
    int32_t seqId = backend_->getSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
}

TEST_F(SequenceIdManagerTest, ConcurrentAllocateRelease) {
    const int numThreads = 5;
    const int requestsPerThread = 10;
    
    std::vector<std::thread> threads;
    std::vector<std::vector<size_t>> requestIdsPerThread(numThreads);
    std::vector<std::vector<int32_t>> seqIdsPerThread(numThreads);
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, requestsPerThread, &requestIdsPerThread, &seqIdsPerThread]() {
            for (int i = 0; i < requestsPerThread; ++i) {
                size_t requestId = t * requestsPerThread + i + 1;
                int32_t seqId = backend_->allocateSequenceId(requestId);
                requestIdsPerThread[t].push_back(requestId);
                seqIdsPerThread[t].push_back(seqId);
            }
        });
    }
    
    for (auto& thread : threads) {
        thread.join();
    }
    
    for (int t = 0; t < numThreads; ++t) {
        for (int i = 0; i < requestsPerThread; ++i) {
            size_t requestId = requestIdsPerThread[t][i];
            int32_t expectedSeqId = seqIdsPerThread[t][i];
            int32_t retrievedSeqId = backend_->getSequenceId(requestId);
            EXPECT_EQ(expectedSeqId, retrievedSeqId);
        }
    }
    
    for (int t = 0; t < numThreads; ++t) {
        for (int i = 0; i < requestsPerThread; ++i) {
            size_t requestId = requestIdsPerThread[t][i];
            bool released = backend_->releaseSequenceId(requestId);
            EXPECT_TRUE(released);
        }
    }
}

TEST_F(SequenceIdManagerTest, SequenceIdPoolExhaustion) {
    int nSeqMax = Config::instance().backendLlamaCppNSeqMax();
    
    for (int i = 0; i < nSeqMax; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = backend_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
        EXPECT_LT(seqId, nSeqMax);
    }
    
    size_t requestId = nSeqMax + 1;
    int32_t seqId = backend_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
}

TEST_F(SequenceIdManagerTest, SequenceIdPoolAfterRelease) {
    int nSeqMax = Config::instance().backendLlamaCppNSeqMax();
    
    for (int i = 0; i < nSeqMax; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = backend_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
    }
    
    size_t requestId = nSeqMax + 1;
    int32_t seqId = backend_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
    
    bool released = backend_->releaseSequenceId(1);
    EXPECT_TRUE(released);
    
    seqId = backend_->allocateSequenceId(requestId);
    EXPECT_GE(seqId, 0);
    EXPECT_LT(seqId, nSeqMax);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
