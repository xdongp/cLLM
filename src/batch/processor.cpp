#include "cllm/batch/processor.h"
#include <algorithm>

namespace cllm {

BatchProcessor::BatchProcessor(BatchManager* manager)
    : manager_(manager) {
}

BatchProcessor::~BatchProcessor() {
}

void BatchProcessor::processBatch(std::vector<RequestState>& batch) {
    for (auto& request : batch) {
        if (request.isCompleted) {
            continue;
        }
        
        if (request.generatedTokens.size() >= static_cast<size_t>(request.maxTokens)) {
            request.isCompleted = true;
            continue;
        }
    }
}

bool BatchProcessor::isBatchComplete(const std::vector<RequestState>& batch) const {
    return std::all_of(batch.begin(), batch.end(), 
                      [](const RequestState& req) { return req.isCompleted; });
}

std::vector<RequestState> BatchProcessor::getActiveRequests(
    const std::vector<RequestState>& batch
) const {
    std::vector<RequestState> active;
    for (const auto& request : batch) {
        if (!request.isCompleted) {
            active.push_back(request);
        }
    }
    return active;
}

std::vector<RequestState> BatchProcessor::getCompletedRequests(
    const std::vector<RequestState>& batch
) const {
    std::vector<RequestState> completed;
    for (const auto& request : batch) {
        if (request.isCompleted) {
            completed.push_back(request);
        }
    }
    return completed;
}

void BatchProcessor::checkStoppingConditions(RequestState& request, int nextToken) {
    if (request.generatedTokens.size() >= static_cast<size_t>(request.maxTokens)) {
        request.isCompleted = true;
        return;
    }

    if (request.eosTokenId >= 0 && nextToken == request.eosTokenId) {
        request.isCompleted = true;
        return;
    }
}

}
