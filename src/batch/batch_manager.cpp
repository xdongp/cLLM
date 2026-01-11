#include "cllm/batch/manager.h"
#include <algorithm>

namespace cllm {

BatchManager::BatchManager(size_t maxContextLength)
    : maxContextLength_(maxContextLength)
    , contextUsageThreshold_(0.75f) {
}

BatchManager::~BatchManager() {
}

std::vector<RequestState> BatchManager::formBatch(
    const std::vector<RequestState>& pendingRequests,
    const std::vector<RequestState>& runningRequests
) {
    std::vector<RequestState> batch;
    
    size_t runningLength = 0;
    for (const auto& req : runningRequests) {
        runningLength += req.getTotalLength();
    }
    
    if (runningLength > maxContextLength_ * contextUsageThreshold_) {
        return batch;
    }
    
    size_t availableContext = maxContextLength_ - runningLength;
    size_t currentBatchLength = 0;
    
    for (const auto& req : pendingRequests) {
        if (canAddToBatch(req, batch, currentBatchLength)) {
            batch.push_back(req);
            currentBatchLength += req.getPromptLength();
        }
    }
    
    return batch;
}

size_t BatchManager::calculateOptimalBatchSize(
    const std::vector<RequestState>& requests,
    size_t avgRequestLength
) {
    if (requests.empty() || avgRequestLength == 0) {
        return 0;
    }
    
    size_t availableContext = maxContextLength_;
    size_t optimalBatchSize = 0;
    
    if (avgRequestLength < 100) {
        optimalBatchSize = std::min(requests.size(), availableContext / avgRequestLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(16));
    } else if (avgRequestLength < 500) {
        optimalBatchSize = std::min(requests.size(), availableContext / avgRequestLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(8));
    } else {
        optimalBatchSize = std::min(requests.size(), availableContext / avgRequestLength);
        optimalBatchSize = std::min(optimalBatchSize, size_t(4));
    }
    
    return optimalBatchSize;
}

bool BatchManager::canAddToBatch(
    const RequestState& request,
    const std::vector<RequestState>& currentBatch,
    size_t currentBatchLength
) {
    size_t runningLength = 0;
    for (const auto& req : currentBatch) {
        runningLength += req.getTotalLength();
    }
    
    size_t totalLength = currentBatchLength + request.getPromptLength();
    
    return totalLength <= maxContextLength_;
}

}  // namespace cllm