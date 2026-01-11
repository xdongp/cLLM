#include <iostream>
#include <cllm/batch/manager.h>
#include <cllm/batch/input.h>
#include <cllm/batch/output.h>
#include <cllm/common/request_state.h>

using namespace cllm;

int main() {
    std::cout << "cLLM Batch Processing Example" << std::endl;
    std::cout << "==============================" << std::endl;

    size_t maxContextLength = 1024;
    size_t maxBatchSize = 8;
    BatchManager batchManager(maxContextLength, maxBatchSize);

    std::cout << "Creating batch..." << std::endl;
    
    // Create sample request states
    std::vector<RequestState> pendingRequests;
    
    for (size_t i = 0; i < 3; ++i) {
        RequestState request;
        request.requestId = i + 1;
        request.tokenizedPrompt = {1, 2, static_cast<int>(3 + i * 3), static_cast<int>(4 + i * 3), static_cast<int>(5 + i * 3)};
        request.maxTokens = 10;
        request.temperature = 0.7f;
        request.isCompleted = false;
        request.isRunning = false;
        pendingRequests.push_back(request);
    }
    
    std::vector<RequestState> runningRequests;
    
    std::cout << "Forming batch..." << std::endl;
    std::vector<RequestState> batch = batchManager.formBatch(pendingRequests, runningRequests);
    
    std::cout << "Prepared batch with " << batch.size() << " requests" << std::endl;
    
    if (!batch.empty()) {
        std::cout << "Preparing batch input..." << std::endl;
        BatchInput batchInput = batchManager.prepareBatchInput(batch);
        
        std::cout << "Batch input prepared. Request positions size: " << batchInput.requestPositions.size() << std::endl;
        
        // Create a dummy batch output
        BatchOutput batchOutput;
        
        std::cout << "Processing batch output..." << std::endl;
        batchManager.processBatchOutput(batch, batchOutput);
        
        std::cout << "Batch processing completed!" << std::endl;
    } else {
        std::cout << "No requests could be batched." << std::endl;
    }
    
    return 0;
}
