#include "cllm/scheduler/stats.h"
#include "cllm/common/request_state.h"
#include <sstream>
#include <iomanip>

namespace cllm {

void SchedulerStats::update(const RequestState& request) {
    completedRequests++;
    
    float requestTime = static_cast<float>(request.completionTime - request.arrivalTime) / 1000.0f;
    float waitTime = static_cast<float>(request.startTime - request.arrivalTime) / 1000.0f;
    
    float currentAvgRequestTime = averageRequestTime.load();
    float currentAvgWaitTime = averageWaitTime.load();
    
    size_t completed = completedRequests.load();
    float newAvgRequestTime = (currentAvgRequestTime * (completed - 1) + requestTime) / completed;
    float newAvgWaitTime = (currentAvgWaitTime * (completed - 1) + waitTime) / completed;
    
    averageRequestTime.store(newAvgRequestTime);
    averageWaitTime.store(newAvgWaitTime);
}

void SchedulerStats::updateBatch(const std::vector<RequestState>& batch) {
    totalBatches++;
    
    float currentAvgBatchSize = averageBatchSize.load();
    size_t batches = totalBatches.load();
    float newAvgBatchSize = (currentAvgBatchSize * (batches - 1) + batch.size()) / batches;
    
    averageBatchSize.store(newAvgBatchSize);
}

void SchedulerStats::reset() {
    totalRequests.store(0);
    completedRequests.store(0);
    failedRequests.store(0);
    totalBatches.store(0);
    averageBatchSize.store(0.0f);
    averageRequestTime.store(0.0f);
    averageWaitTime.store(0.0f);
    peakQueueSize.store(0);
}

std::string SchedulerStats::toString() const {
    std::ostringstream oss;
    oss << "SchedulerStats:\n";
    oss << "  Total Requests: " << totalRequests.load() << "\n";
    oss << "  Completed Requests: " << completedRequests.load() << "\n";
    oss << "  Failed Requests: " << failedRequests.load() << "\n";
    oss << "  Total Batches: " << totalBatches.load() << "\n";
    oss << "  Average Batch Size: " << std::fixed << std::setprecision(2) 
        << averageBatchSize.load() << "\n";
    oss << "  Average Request Time: " << std::fixed << std::setprecision(2) 
        << averageRequestTime.load() << "s\n";
    oss << "  Average Wait Time: " << std::fixed << std::setprecision(2) 
        << averageWaitTime.load() << "s\n";
    oss << "  Peak Queue Size: " << peakQueueSize.load();
    return oss.str();
}

}
