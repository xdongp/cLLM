#include <gtest/gtest.h>
#include <cllm/inference/llama_cpp_backend.h>
#include <cllm/common/config.h>
#include <cllm/common/logger.h>

using namespace cllm;
using namespace cllm::inference;

class SimpleSequenceIdManager {
public:
    SimpleSequenceIdManager(int32_t nSeqMax = 8) : nSeqMax_(nSeqMax) {
        initializeSequenceIdPool();
    }
    
    int32_t allocateSequenceId(size_t requestId) {
        std::lock_guard<std::mutex> lock(sequenceIdMutex_);
        
        auto it = requestIdToSeqId_.find(requestId);
        if (it != requestIdToSeqId_.end()) {
            return it->second;
        }
        
        if (availableSeqIds_.empty()) {
            return -1;
        }
        
        int32_t seqId = availableSeqIds_.back();
        availableSeqIds_.pop_back();
        
        requestIdToSeqId_[requestId] = seqId;
        
        return seqId;
    }
    
    bool releaseSequenceId(size_t requestId) {
        std::lock_guard<std::mutex> lock(sequenceIdMutex_);
        
        auto it = requestIdToSeqId_.find(requestId);
        if (it == requestIdToSeqId_.end()) {
            return false;
        }
        
        int32_t seqId = it->second;
        
        requestIdToSeqId_.erase(it);
        
        availableSeqIds_.push_back(seqId);
        
        return true;
    }
    
    int32_t getSequenceId(size_t requestId) const {
        std::lock_guard<std::mutex> lock(sequenceIdMutex_);
        
        auto it = requestIdToSeqId_.find(requestId);
        if (it == requestIdToSeqId_.end()) {
            return -1;
        }
        
        return it->second;
    }
    
private:
    void initializeSequenceIdPool() {
        availableSeqIds_.clear();
        availableSeqIds_.reserve(nSeqMax_);
        for (int32_t i = 0; i < nSeqMax_; ++i) {
            availableSeqIds_.push_back(i);
        }
        
        requestIdToSeqId_.clear();
    }
    
    int32_t nSeqMax_;
    std::map<size_t, int32_t> requestIdToSeqId_;
    std::vector<int32_t> availableSeqIds_;
    mutable std::mutex sequenceIdMutex_;
};

class SequenceIdManagerTest : public ::testing::Test {
protected:
    void SetUp() override {
        manager_ = std::make_unique<SimpleSequenceIdManager>(8);
    }

    void TearDown() override {
        manager_.reset();
    }

    std::unique_ptr<SimpleSequenceIdManager> manager_;
};

TEST_F(SequenceIdManagerTest, AllocateSingleSequenceId) {
    size_t requestId = 1;
    int32_t seqId = manager_->allocateSequenceId(requestId);
    
    EXPECT_GE(seqId, 0);
    EXPECT_LT(seqId, 8);
    
    int32_t retrievedSeqId = manager_->getSequenceId(requestId);
    EXPECT_EQ(seqId, retrievedSeqId);
}

TEST_F(SequenceIdManagerTest, ReleaseSingleSequenceId) {
    size_t requestId = 1;
    int32_t seqId = manager_->allocateSequenceId(requestId);
    
    EXPECT_GE(seqId, 0);
    
    bool released = manager_->releaseSequenceId(requestId);
    EXPECT_TRUE(released);
    
    int32_t retrievedSeqId = manager_->getSequenceId(requestId);
    EXPECT_EQ(retrievedSeqId, -1);
}

TEST_F(SequenceIdManagerTest, AllocateMultipleSequenceIds) {
    const int numRequests = 5;
    std::vector<size_t> requestIds;
    std::vector<int32_t> seqIds;
    
    for (int i = 0; i < numRequests; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = manager_->allocateSequenceId(requestId);
        
        EXPECT_GE(seqId, 0);
        EXPECT_LT(seqId, 8);
        
        requestIds.push_back(requestId);
        seqIds.push_back(seqId);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        int32_t retrievedSeqId = manager_->getSequenceId(requestIds[i]);
        EXPECT_EQ(seqIds[i], retrievedSeqId);
    }
}

TEST_F(SequenceIdManagerTest, ReleaseMultipleSequenceIds) {
    const int numRequests = 5;
    std::vector<size_t> requestIds;
    
    for (int i = 0; i < numRequests; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = manager_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
        requestIds.push_back(requestId);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        bool released = manager_->releaseSequenceId(requestIds[i]);
        EXPECT_TRUE(released);
    }
    
    for (int i = 0; i < numRequests; ++i) {
        int32_t retrievedSeqId = manager_->getSequenceId(requestIds[i]);
        EXPECT_EQ(retrievedSeqId, -1);
    }
}

TEST_F(SequenceIdManagerTest, SequenceIdReuse) {
    size_t requestId1 = 1;
    int32_t seqId1 = manager_->allocateSequenceId(requestId1);
    EXPECT_GE(seqId1, 0);
    
    bool released = manager_->releaseSequenceId(requestId1);
    EXPECT_TRUE(released);
    
    size_t requestId2 = 2;
    int32_t seqId2 = manager_->allocateSequenceId(requestId2);
    EXPECT_GE(seqId2, 0);
    
    EXPECT_EQ(seqId1, seqId2);
}

TEST_F(SequenceIdManagerTest, AllocateSameRequestIdTwice) {
    size_t requestId = 1;
    int32_t seqId1 = manager_->allocateSequenceId(requestId);
    EXPECT_GE(seqId1, 0);
    
    int32_t seqId2 = manager_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId1, seqId2);
}

TEST_F(SequenceIdManagerTest, ReleaseNonExistentRequestId) {
    size_t requestId = 999;
    bool released = manager_->releaseSequenceId(requestId);
    EXPECT_FALSE(released);
}

TEST_F(SequenceIdManagerTest, GetNonExistentRequestId) {
    size_t requestId = 999;
    int32_t seqId = manager_->getSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
}

TEST_F(SequenceIdManagerTest, ConcurrentAllocateRelease) {
    const int numThreads = 4;
    const int requestsPerThread = 2;  // 减少到2，确保总数不超过8
    
    std::vector<std::thread> threads;
    std::vector<std::vector<size_t>> requestIdsPerThread(numThreads);
    std::vector<std::vector<int32_t>> seqIdsPerThread(numThreads);
    
    for (int t = 0; t < numThreads; ++t) {
        threads.emplace_back([this, t, requestsPerThread, &requestIdsPerThread, &seqIdsPerThread]() {
            for (int i = 0; i < requestsPerThread; ++i) {
                size_t requestId = t * requestsPerThread + i + 1;
                int32_t seqId = manager_->allocateSequenceId(requestId);
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
            if (expectedSeqId >= 0) {
                int32_t retrievedSeqId = manager_->getSequenceId(requestId);
                EXPECT_EQ(expectedSeqId, retrievedSeqId);
            }
        }
    }
    
    for (int t = 0; t < numThreads; ++t) {
        for (int i = 0; i < requestsPerThread; ++i) {
            size_t requestId = requestIdsPerThread[t][i];
            int32_t expectedSeqId = seqIdsPerThread[t][i];
            if (expectedSeqId >= 0) {
                bool released = manager_->releaseSequenceId(requestId);
                EXPECT_TRUE(released);
            }
        }
    }
}

TEST_F(SequenceIdManagerTest, SequenceIdPoolExhaustion) {
    int nSeqMax = 8;
    
    for (int i = 0; i < nSeqMax; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = manager_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
        EXPECT_LT(seqId, nSeqMax);
    }
    
    size_t requestId = nSeqMax + 1;
    int32_t seqId = manager_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
}

TEST_F(SequenceIdManagerTest, SequenceIdPoolAfterRelease) {
    int nSeqMax = 8;
    
    for (int i = 0; i < nSeqMax; ++i) {
        size_t requestId = i + 1;
        int32_t seqId = manager_->allocateSequenceId(requestId);
        EXPECT_GE(seqId, 0);
    }
    
    size_t requestId = nSeqMax + 1;
    int32_t seqId = manager_->allocateSequenceId(requestId);
    EXPECT_EQ(seqId, -1);
    
    bool released = manager_->releaseSequenceId(1);
    EXPECT_TRUE(released);
    
    seqId = manager_->allocateSequenceId(requestId);
    EXPECT_GE(seqId, 0);
    EXPECT_LT(seqId, nSeqMax);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
