#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <numeric>
#include <stdexcept>

namespace cllm {

BatchManager::BatchManager(size_t maxContextLength, size_t maxBatchSize)
    : maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength())
    , maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize())
    , contextUsageThreshold_(Config::instance().schedulerContextUsageThreshold())
    , executor_(nullptr) {
}

BatchManager::BatchManager(size_t maxContextLength, size_t maxBatchSize, ModelExecutor* executor)
    : maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength())
    , maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize())
    , contextUsageThreshold_(Config::instance().schedulerContextUsageThreshold())
    , executor_(executor) {
}

BatchManager::~BatchManager() {
}

std::vector<RequestState> BatchManager::formBatch(
    const std::vector<RequestState>& pendingRequests,
    const std::vector<RequestState>& runningRequests,
    size_t availableSeqIds
) {
    std::vector<RequestState> batch;
    size_t currentBatchLength = 0;
    
    size_t runningLength = calculateRunningRequestsLength(runningRequests);
    
    if (runningLength > maxContextLength_ * contextUsageThreshold_) {
        return batch;
    }
    
    size_t avgLength = calculateAverageRequestLength(pendingRequests);
    size_t dynamicBatchSize = calculateOptimalBatchSize(pendingRequests, avgLength);
    
    // 优化：考虑序列ID可用性，限制批处理大小
    if (availableSeqIds > 0) {
        dynamicBatchSize = std::min(dynamicBatchSize, availableSeqIds);
    }
    
    for (const auto& request : pendingRequests) {
        size_t requestLength = request.getTotalLength();
        size_t totalLength = runningLength + currentBatchLength + requestLength;
        
        if (totalLength <= maxContextLength_ && 
            batch.size() < dynamicBatchSize) {
            batch.push_back(request);
            currentBatchLength += requestLength;
        } else {
            break;
        }
    }
    
    updateStats(batch);
    return batch;
}

std::vector<RequestState> BatchManager::formMultipleBatches(
    const std::vector<RequestState>& pendingRequests,
    const std::vector<RequestState>& runningRequests
) {
    std::vector<RequestState> allBatches;
    std::vector<RequestState> remaining = pendingRequests;
    
    size_t runningLength = calculateRunningRequestsLength(runningRequests);
    
    while (!remaining.empty()) {
        std::vector<RequestState> batch;
        size_t currentBatchLength = 0;
        
        size_t avgLength = calculateAverageRequestLength(remaining);
        size_t dynamicBatchSize = calculateOptimalBatchSize(remaining, avgLength);
        
        std::vector<size_t> usedIndices;
        
        for (size_t i = 0; i < remaining.size(); ++i) {
            const auto& request = remaining[i];
            size_t requestLength = request.getTotalLength();
            size_t totalLength = runningLength + currentBatchLength + requestLength;
            
            if (totalLength <= maxContextLength_ && 
                batch.size() < dynamicBatchSize) {
                batch.push_back(request);
                currentBatchLength += requestLength;
                usedIndices.push_back(i);
            }
        }
        
        if (batch.empty()) {
            break;
        }
        
        allBatches.insert(allBatches.end(), batch.begin(), batch.end());
        runningLength += currentBatchLength;
        
        std::vector<RequestState> newRemaining;
        for (size_t i = 0; i < remaining.size(); ++i) {
            if (std::find(usedIndices.begin(), usedIndices.end(), i) == usedIndices.end()) {
                newRemaining.push_back(remaining[i]);
            }
        }
        remaining = newRemaining;
    }
    
    return allBatches;
}

BatchInput BatchManager::prepareBatchInput(const std::vector<RequestState>& batch) {
    BatchInput input;
    input.batchSize = batch.size();
    
    size_t currentPos = 0;
    
    for (const auto& request : batch) {
        std::vector<int> inputIds = request.tokenizedPrompt;
        inputIds.insert(inputIds.end(), 
                       request.generatedTokens.begin(), 
                       request.generatedTokens.end());
        
        input.inputIds.insert(input.inputIds.end(), 
                             inputIds.begin(), 
                             inputIds.end());
        
        input.requestPositions.push_back({currentPos, currentPos + inputIds.size()});
        input.sequenceIds.push_back(request.requestId);
        
        currentPos += inputIds.size();
    }
    
    return input;
}

void BatchManager::processBatchOutput(
    std::vector<RequestState>& batch,
    const BatchOutput& output
) {
    for (size_t i = 0; i < batch.size(); ++i) {
        if (batch[i].isCompleted) {
            continue;
        }
        
        size_t vocabSize = executor_ ? executor_->getConfig().vocabSize : 32000;
        FloatArray requestLogits = output.getLogitsForRequest(i, vocabSize);
        
        float temperature = batch[i].temperature;
        int topK = batch[i].topK;
        float topP = batch[i].topP;
        
        int nextToken = sampler_.sample(requestLogits, temperature, topK, topP);
        
        batch[i].generatedTokens.push_back(nextToken);
        
        checkStoppingConditions(batch[i], nextToken);
    }
}

size_t BatchManager::calculateOptimalBatchSize(
    const std::vector<RequestState>& requests,
    size_t avgRequestLength
) {
    if (requests.empty()) {
        return 0;
    }
    
    size_t dynamicBatchSize = maxBatchSize_;
    
    if (avgRequestLength > 500) {
        dynamicBatchSize = std::max(size_t(2), maxBatchSize_ / 2);
    } else if (avgRequestLength > 200) {
        dynamicBatchSize = std::max(size_t(3), static_cast<size_t>(maxBatchSize_ / 1.5));
    } else {
        dynamicBatchSize = static_cast<size_t>(maxBatchSize_ * 1.5);
    }
    
    return dynamicBatchSize;
}

bool BatchManager::canAddToBatch(
    const RequestState& request,
    const std::vector<RequestState>& currentBatch,
    size_t currentBatchLength,
    size_t dynamicBatchSize
) {
    if (currentBatch.size() >= dynamicBatchSize) {
        return false;
    }
    
    size_t requestLength = request.getTotalLength();
    size_t totalLength = currentBatchLength + requestLength;
    
    return totalLength <= maxContextLength_;
}

BatchStats BatchManager::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void BatchManager::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.reset();
}

size_t BatchManager::calculateRunningRequestsLength(
    const std::vector<RequestState>& runningRequests
) {
    size_t totalLength = 0;
    for (const auto& request : runningRequests) {
        totalLength += request.getTotalLength();
    }
    return totalLength;
}

size_t BatchManager::calculateAverageRequestLength(
    const std::vector<RequestState>& requests
) {
    if (requests.empty()) {
        return 0;
    }
    
    size_t totalLength = 0;
    for (const auto& request : requests) {
        totalLength += request.getTotalLength();
    }
    
    return totalLength / requests.size();
}

void BatchManager::updateStats(const std::vector<RequestState>& batch) {
    if (batch.empty()) {
        return;
    }
    
    size_t batchLength = 0;
    for (const auto& request : batch) {
        batchLength += request.getTotalLength();
    }
    
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.update(batch.size(), batchLength);
}

void BatchManager::checkStoppingConditions(RequestState& request, int nextToken) {
    if (request.generatedTokens.size() >= static_cast<size_t>(request.maxTokens)) {
        request.isCompleted = true;
        return;
    }

    // 使用请求注入的 EOS token id（避免写死 2/0 导致不同模型行为错误）
    if (request.eosTokenId >= 0 && nextToken == request.eosTokenId) {
        request.isCompleted = true;
        return;
    }
}



}
