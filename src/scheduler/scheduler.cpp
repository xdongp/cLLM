#include "cllm/scheduler/scheduler.h"
#include "cllm/common/request_state.h"
#include "cllm/common/queue.h"
#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/memory/monitor.h"
#include "cllm/common/logger.h"
#include <chrono>
#include <stdexcept>

namespace cllm {

Scheduler::Scheduler(
    ModelExecutor* modelExecutor,
    size_t maxBatchSize,
    size_t maxContextLength
) : batchManager_(
        (maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength(),
        (maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize()
    ),
    maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize()),
    maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength()),
    modelExecutor_(modelExecutor),
    ownsModelExecutor_(false) {
    
    config_.maxBatchSize = maxBatchSize_;
    config_.maxContextLength = maxContextLength_;
    config_.defaultTemperature = Config::instance().schedulerDefaultTemperature();
    config_.defaultTopP = Config::instance().schedulerDefaultTopP();
    config_.defaultTopK = Config::instance().schedulerDefaultTopK();
    config_.defaultMaxTokens = Config::instance().schedulerDefaultMaxTokens();
    config_.requestTimeout = Config::instance().schedulerRequestTimeout();
    config_.schedulerLoopInterval = Config::instance().schedulerLoopInterval();
    config_.idleLoopInterval = Config::instance().schedulerIdleLoopInterval();
    config_.contextUsageThreshold = Config::instance().schedulerContextUsageThreshold();
    
    kvCache_ = new KVCache(maxContextLength_, 10000);
    
    // 验证模型已加载
    if (!modelExecutor_->isLoaded()) {
        throw std::runtime_error("Model executor must be pre-loaded before creating Scheduler");
    }
}

Scheduler::Scheduler(
    const std::string& modelPath,
    const std::string& quantization,
    size_t maxBatchSize,
    size_t maxContextLength
) : batchManager_(
        (maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength(),
        (maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize()
    ),
    maxBatchSize_((maxBatchSize != 0) ? maxBatchSize : Config::instance().serverMaxBatchSize()),
    maxContextLength_((maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength()),
    ownsModelExecutor_(true) {
    
    config_.maxBatchSize = maxBatchSize_;
    config_.maxContextLength = maxContextLength_;
    config_.defaultTemperature = Config::instance().schedulerDefaultTemperature();
    config_.defaultTopP = Config::instance().schedulerDefaultTopP();
    config_.defaultTopK = Config::instance().schedulerDefaultTopK();
    config_.defaultMaxTokens = Config::instance().schedulerDefaultMaxTokens();
    config_.requestTimeout = Config::instance().schedulerRequestTimeout();
    config_.schedulerLoopInterval = Config::instance().schedulerLoopInterval();
    config_.idleLoopInterval = Config::instance().schedulerIdleLoopInterval();
    config_.contextUsageThreshold = Config::instance().schedulerContextUsageThreshold();
    
    modelExecutor_ = new ModelExecutor(modelPath, quantization);
    kvCache_ = new KVCache(maxContextLength_, 10000);
    
    // 加载模型
    modelExecutor_->loadModel();
}

Scheduler::~Scheduler() {
    stop();
    
    delete kvCache_;
    
    // Only delete modelExecutor_ if we own it (used by tests)
    if (ownsModelExecutor_ && modelExecutor_) {
        delete modelExecutor_;
    }
}

void Scheduler::start() {
    if (running_) {
        return;
    }
    
    running_ = true;
    schedulerThread_ = std::thread(&Scheduler::schedulerLoop, this);
}

void Scheduler::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;
    
    if (schedulerThread_.joinable()) {
        schedulerThread_.join();
    }
}

size_t Scheduler::addRequest(const RequestState& request) {
    if (!running_) {
        throw SchedulerException(
            SchedulerError::SCHEDULER_NOT_RUNNING,
            "Scheduler is not running"
        );
    }
    
    RequestState req = request;
    
    if (req.requestId == 0) {
        req.requestId = requestTracker_.addRequest(req);
    }
    
    req.arrivalTime = getCurrentTime();
    
    if (req.temperature == 0.0f) {
        req.temperature = config_.defaultTemperature;
    }
    if (req.topP == 0.0f) {
        req.topP = config_.defaultTopP;
    }
    if (req.topK == 0) {
        req.topK = config_.defaultTopK;
    }
    if (req.maxTokens == 0) {
        req.maxTokens = config_.defaultMaxTokens;
    }
    
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        requestQueue_.addRequest(req);
    }
    
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.totalRequests++;
        
        size_t queueSize = requestQueue_.getQueueSize();
        if (queueSize > stats_.peakQueueSize.load()) {
            stats_.peakQueueSize.store(queueSize);
        }
    }
    
    queueCondition_.notify_one();
    return req.requestId;
}

bool Scheduler::removeRequest(size_t requestId) {
    if (!running_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(requestsMutex_);
    
    if (runningRequests_.erase(requestId) > 0) {
        requestTracker_.removeRequest(requestId);
        return true;
    }
    
    if (completedRequests_.erase(requestId) > 0) {
        requestTracker_.removeRequest(requestId);
        return true;
    }
    
    return requestQueue_.removeRequest(requestId);
}

RequestState Scheduler::getRequestResult(size_t requestId) {
    std::lock_guard<std::mutex> lock(requestsMutex_);
    
    auto it = completedRequests_.find(requestId);
    if (it != completedRequests_.end()) {
        return it->second;
    }
    
    throw SchedulerException(
        SchedulerError::REQUEST_NOT_FOUND,
        "Request not found: " + std::to_string(requestId)
    );
}

bool Scheduler::waitForRequest(size_t requestId, float timeout) {
    auto startTime = std::chrono::steady_clock::now();
    
    while (running_) {
        {
            std::lock_guard<std::mutex> lock(requestsMutex_);
            if (completedRequests_.find(requestId) != completedRequests_.end()) {
                return true;
            }
        }
        
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<float>(currentTime - startTime).count();
        
        if (elapsed >= timeout) {
            return false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    return false;
}

std::vector<RequestState> Scheduler::getRunningRequests() const {
    std::lock_guard<std::mutex> lock(requestsMutex_);
    
    std::vector<RequestState> requests;
    requests.reserve(runningRequests_.size());
    
    for (const auto& pair : runningRequests_) {
        requests.push_back(pair.second);
    }
    
    return requests;
}

std::vector<RequestState> Scheduler::getCompletedRequests() const {
    std::lock_guard<std::mutex> lock(requestsMutex_);
    
    std::vector<RequestState> requests;
    requests.reserve(completedRequests_.size());
    
    for (const auto& pair : completedRequests_) {
        requests.push_back(pair.second);
    }
    
    return requests;
}

size_t Scheduler::getQueueSize() const {
    std::lock_guard<std::mutex> lock(queueMutex_);
    return requestQueue_.getQueueSize();
}

SchedulerStats Scheduler::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void Scheduler::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.reset();
}

void Scheduler::schedulerLoop() {
    while (running_) {
        try {
            processRequests();
            
            size_t queueSize = requestQueue_.getQueueSize();
            size_t runningCount = runningRequests_.size();
            
            if (queueSize == 0 && runningCount == 0) {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(config_.idleLoopInterval)
                );
            } else {
                std::this_thread::sleep_for(
                    std::chrono::microseconds(config_.schedulerLoopInterval)
                );
            }
            
        } catch (const std::exception& e) {
            CLLM_ERROR("Error in scheduler loop: %s", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void Scheduler::processRequests() {
    if (requestQueue_.getQueueSize() == 0 && runningRequests_.empty()) {
        return;
    }
    
    // 按照设计文档，使用BatchManager形成批处理
    std::vector<RequestState> running = getRunningRequests();
    std::vector<RequestState> batch = batchManager_.formBatch(
        requestQueue_.getPendingRequests(),
        running
    );
    
    if (batch.empty()) {
        return;
    }
    
    processBatch(batch);
}

void Scheduler::processBatch(std::vector<RequestState>& batch) {
    SchedulerBatchProcessor processor(this, modelExecutor_, kvCache_, &batchManager_);
    
    CLLM_INFO("Starting batch processing for %zu requests", batch.size());
    
    for (auto& request : batch) {
        request.startTime = getCurrentTime();
        requestTracker_.markAsRunning(request.requestId);
        
        {
            std::lock_guard<std::mutex> lock(requestsMutex_);
            runningRequests_[request.requestId] = request;
        }
    }
    
    processor.processBatch(batch);
    
    for (auto& request : batch) {
        request.completionTime = getCurrentTime();
        
        CLLM_DEBUG("Request %llu generated tokens: %zu", request.requestId, request.generatedTokens.size());
        
        if (request.isCompleted) {
            requestTracker_.markAsCompleted(request.requestId);
            stats_.update(request);
        } else if (request.isFailed) {
            requestTracker_.markAsFailed(request.requestId, request.errorMessage);
            stats_.failedRequests++;
        }
        
        {
            std::lock_guard<std::mutex> lock(requestsMutex_);
            runningRequests_.erase(request.requestId);
            completedRequests_[request.requestId] = request;
        }
        
        resultCondition_.notify_all();
    }
    
    stats_.updateBatch(batch);
}

float Scheduler::getCurrentTime() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return std::chrono::duration<float>(duration).count();
}

}
