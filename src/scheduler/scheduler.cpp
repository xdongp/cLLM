#include "cllm/scheduler/scheduler.h"
#include "cllm/common/request_state.h"
#include "cllm/common/queue.h"
#include "cllm/batch/manager.h"
#include "cllm/model/executor.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/memory/monitor.h"
#include "cllm/common/logger.h"
#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/scheduler/dynamic_batch_tuner.h"
#include <chrono>
#include <stdexcept>
#include <queue>
#include <algorithm>

namespace cllm {

namespace {
size_t resolveMaxBatchSize(size_t overrideMax) {
    const size_t schedulerMax = static_cast<size_t>(Config::instance().schedulerMaxBatchSize());
    const size_t resourceMax = static_cast<size_t>(Config::instance().serverMaxBatchSize());
    size_t desired = overrideMax != 0 ? overrideMax : schedulerMax;
    if (desired == 0) {
        desired = resourceMax;
    }
    if (resourceMax > 0) {
        desired = std::min(desired, resourceMax);
    }
    return desired > 0 ? desired : 1;
}
}

Scheduler::Scheduler(
    ModelExecutor* modelExecutor,
    size_t maxBatchSize,
    size_t maxContextLength
) : batchManager_(
        (maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength(),
        resolveMaxBatchSize(maxBatchSize)
    ),
    maxBatchSize_(resolveMaxBatchSize(maxBatchSize)),
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
    // 🔥 修复：增加最大并发请求数，支持32并发测试
    config_.maxConcurrentRequests = 64;  // 从默认8增加到64，支持高并发场景
    
    kvCache_ = new KVCache(
        static_cast<size_t>(Config::instance().serverKvCacheMaxSize()),
        static_cast<size_t>(Config::instance().serverKvCacheMaxMemoryMb())
    );
    
    // 验证模型已加载
    if (!modelExecutor_->isLoaded()) {
        throw std::runtime_error("Model executor must be pre-loaded before creating Scheduler");
    }

    enforceBackendBatchConstraints();
}

Scheduler::Scheduler(
    const std::string& modelPath,
    const std::string& quantization,
    size_t maxBatchSize,
    size_t maxContextLength
) : batchManager_(
        (maxContextLength != 0) ? maxContextLength : Config::instance().serverMaxContextLength(),
        resolveMaxBatchSize(maxBatchSize)
    ),
    maxBatchSize_(resolveMaxBatchSize(maxBatchSize)),
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
    // 🔥 修复：增加最大并发请求数，支持32并发测试
    config_.maxConcurrentRequests = 64;  // 从默认8增加到64，支持高并发场景
    
    modelExecutor_ = new ModelExecutor(modelPath, quantization);
    kvCache_ = new KVCache(
        static_cast<size_t>(Config::instance().serverKvCacheMaxSize()),
        static_cast<size_t>(Config::instance().serverKvCacheMaxMemoryMb())
    );
    
    // 加载模型
    modelExecutor_->loadModel();

    enforceBackendBatchConstraints();
}

Scheduler::~Scheduler() {
    stop();
    
    delete kvCache_;
    
    // Only delete modelExecutor_ if we own it (used by tests)
    if (ownsModelExecutor_ && modelExecutor_) {
        delete modelExecutor_;
    }
}

void Scheduler::enforceBackendBatchConstraints() {
    if (!modelExecutor_) {
        return;
    }

    const std::string backendName = modelExecutor_->getBackendName();
    if (backendName == "Kylin") {
        // Kylin 后端现在支持 per-request KV Cache，可以真正并发
        // 限制最大并发数为 16（与 KVCachePool 的 maxSlots 一致）
        static constexpr size_t kKylinMaxConcurrent = 16;
        if (maxBatchSize_ > kKylinMaxConcurrent) {
            CLLM_WARN("[Scheduler] Kylin backend limits maxBatchSize from %zu to %zu (per-request KV cache slots)",
                      maxBatchSize_, kKylinMaxConcurrent);
            maxBatchSize_ = kKylinMaxConcurrent;
            tunedMaxBatchSize_.store(kKylinMaxConcurrent, std::memory_order_relaxed);
            config_.maxBatchSize = kKylinMaxConcurrent;
            batchManager_.setMaxBatchSize(kKylinMaxConcurrent);
        } else {
            CLLM_INFO("[Scheduler] Kylin backend with per-request KV cache, maxBatchSize=%zu", maxBatchSize_);
        }
    }
}

void Scheduler::start() {
    if (running_) {
        return;
    }
    
    running_ = true;

    const auto tunerConfig = Config::instance().dynamicBatchTunerConfig();
    if (tunerConfig.enabled && tunerConfig.strategy == "static") {
        if (tunerConfig.fixedBatchSize > 0) {
            applyTunedBatchSize(static_cast<size_t>(tunerConfig.fixedBatchSize));
            CLLM_INFO("[Scheduler] 使用静态 batch_size=%d", tunerConfig.fixedBatchSize);
        }
    } else if (tunerConfig.enabled && tunerConfig.strategy != "static") {
        DynamicBatchTuner::TunerConfig config;
        config.minBatchSize = std::max<size_t>(1, static_cast<size_t>(tunerConfig.minBatchSize));
        config.maxBatchSize = std::min<size_t>(maxBatchSize_, static_cast<size_t>(tunerConfig.maxBatchSize));
        config.initialBatchSize = static_cast<size_t>(tunerConfig.initialBatchSize);
        config.probingGrowthFactor = tunerConfig.probingGrowthFactor;
        config.maxProbingAttempts = static_cast<size_t>(tunerConfig.maxProbingAttempts);
        config.timeIncreaseThreshold = tunerConfig.timeIncreaseThreshold;
        config.adjustmentFactor = tunerConfig.adjustmentFactor;
        config.validationInterval = static_cast<size_t>(tunerConfig.validationInterval);
        config.explorationInterval = static_cast<size_t>(tunerConfig.explorationInterval);
        config.probeBatchCount = static_cast<size_t>(tunerConfig.probeBatchCount);
        config.validationBatchCount = static_cast<size_t>(tunerConfig.validationBatchCount);
        config.autoAdjustEnabled = (tunerConfig.strategy == "dynamic");
        config.maxConsecutiveTimeIncreases = static_cast<size_t>(tunerConfig.maxConsecutiveTimeIncreases);

        if (config.maxBatchSize == 0) {
            config.maxBatchSize = maxBatchSize_;
        }
        if (config.minBatchSize > config.maxBatchSize) {
            config.minBatchSize = config.maxBatchSize;
        }
        if (config.initialBatchSize == 0) {
            config.initialBatchSize = config.minBatchSize;
        }
        config.initialBatchSize = std::max(config.minBatchSize, std::min(config.initialBatchSize, config.maxBatchSize));

        batchTuner_ = std::make_unique<DynamicBatchTuner>(config);
        batchTuner_->startPassive();
        applyTunedBatchSize(batchTuner_->getCurrentBatchSize());
        CLLM_INFO("[Scheduler] 动态批处理调谐器已启用: strategy=%s, batch_size=%zu",
                  tunerConfig.strategy.c_str(), batchTuner_->getCurrentBatchSize());
    }

    schedulerThread_ = std::thread(&Scheduler::schedulerLoop, this);
    cleanupThread_ = std::thread(&Scheduler::cleanupLoop, this);
}

void Scheduler::stop() {
    if (!running_) {
        return;
    }
    
    running_ = false;

    if (batchTuner_) {
        batchTuner_->stop();
    }
    
    // 通知清理线程退出
    cleanupCondition_.notify_all();
    
    if (cleanupThread_.joinable()) {
        cleanupThread_.join();
    }
    
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
            if (!requestQueue_.addRequest(req)) {
                throw SchedulerException(
                    SchedulerError::REQUEST_QUEUE_FULL,
                    "Request queue is full"
                );
            }
            // 🔥 优化步骤1: 更新原子缓存
            cachedQueueSize_.store(requestQueue_.getQueueSize(), std::memory_order_relaxed);
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
    
    std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 写操作使用独占锁
    
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
    std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 读操作使用共享锁
    
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
    auto timeoutDuration = std::chrono::duration<float>(timeout);
    
    // 🔥 优化：使用条件变量替代轮询，减少等待延迟
    std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 等待需要独占锁
    
    // 先检查是否已经完成
    if (completedRequests_.find(requestId) != completedRequests_.end()) {
        return true;
    }
    
    // 等待请求完成，使用条件变量通知
    auto deadline = startTime + std::chrono::duration_cast<std::chrono::steady_clock::duration>(timeoutDuration);
    while (running_) {
        // 使用条件变量等待，超时时间动态计算
        auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now()).count();
        if (remaining <= 0) {
            return false; // 超时
        }
        
        // 等待通知，最多等待remaining毫秒
        if (resultCondition_.wait_for(lock, std::chrono::milliseconds(remaining), [this, requestId]() {
            return completedRequests_.find(requestId) != completedRequests_.end();
        })) {
            // 请求已完成
            return true;
        }
        
        // 再次检查是否完成（可能在wait_for返回false时已经完成）
        if (completedRequests_.find(requestId) != completedRequests_.end()) {
            return true;
        }
        
        // 检查是否超时
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
    }
    
    return false;
}

std::vector<RequestState> Scheduler::getRunningRequests() const {
    std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 优化：使用共享锁，允许多个读
    
    std::vector<RequestState> requests;
    requests.reserve(runningRequests_.size());
    
    // Phase 1: 状态机核心实现 - 只返回活跃请求（PENDING或PROCESSING）
    // 过滤掉已完成的请求（COMPLETED/TIMEOUT/FAILED），避免 formBatch 计算 runningLength 时高估
    for (const auto& pair : runningRequests_) {
        const RequestState& req = pair.second;
        // 使用状态判断辅助函数：只返回活跃请求（PENDING或PROCESSING）
        if (req.isActive()) {
            requests.push_back(req);
        }
    }
    
    return requests;
}

std::vector<RequestState> Scheduler::getCompletedRequests() const {
    std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 优化：使用共享锁，允许多个读
    
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

size_t Scheduler::getRunningCount() const {
    std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 读操作使用共享锁
    return runningRequests_.size();
}

size_t Scheduler::getMaxConcurrentRequests() const {
    return config_.maxConcurrentRequests;
}

void Scheduler::setResponseCallback(ResponseCallback callback) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    responseCallback_ = callback;
}

void Scheduler::triggerResponseCallback(size_t requestId, const RequestState& state) {
    std::lock_guard<std::mutex> lock(callbackMutex_);
    if (responseCallback_) {
        try {
            responseCallback_(requestId, state);
        } catch (const std::exception& e) {
            CLLM_ERROR("Error in response callback for requestId=%zu: %s", requestId, e.what());
        } catch (...) {
            CLLM_ERROR("Unknown error in response callback for requestId=%zu", requestId);
        }
    }
}

void Scheduler::onBatchProcessed(size_t batchSize, double processingTimeMs) {
    if (!batchTuner_) {
        return;
    }

    batchTuner_->onBatchProcessed(batchSize, processingTimeMs);
    applyTunedBatchSize(batchTuner_->getCurrentBatchSize());
}

void Scheduler::applyTunedBatchSize(size_t tunedBatchSize) {
    if (tunedBatchSize == 0) {
        return;
    }

    size_t hardMax = resolveMaxBatchSize(0);
    size_t clamped = std::max<size_t>(1, std::min(tunedBatchSize, hardMax));
    size_t current = tunedMaxBatchSize_.load(std::memory_order_relaxed);
    if (current == clamped) {
        return;
    }

    tunedMaxBatchSize_.store(clamped, std::memory_order_relaxed);
    maxBatchSize_ = clamped;
    config_.maxBatchSize = clamped;
    batchManager_.setMaxBatchSize(clamped);
}

void Scheduler::schedulerLoop() {
    while (running_) {
        try {
            processRequests();
            
            checkRequestTimeout();
            
            checkKVCachEviction();
            
            std::unique_lock<std::mutex> lock(queueMutex_);
            
            // 🔥 优化步骤2: 使用原子操作快速检查（只读）
            size_t queueSize = cachedQueueSize_.load(std::memory_order_relaxed);
            size_t runningCount = cachedRunningCount_.load(std::memory_order_relaxed);
            
            // 如果需要精确值或等待条件，获取锁并同步
            if (queueSize == 0 && runningCount == 0) {
                // 双重检查：获取精确值
                queueSize = requestQueue_.getQueueSize();
                cachedQueueSize_.store(queueSize, std::memory_order_relaxed);
                
                {
                    std::shared_lock<std::shared_mutex> reqLock(requestsMutex_);  // 读操作使用共享锁
                    runningCount = runningRequests_.size();
                    cachedRunningCount_.store(runningCount, std::memory_order_relaxed);
                }
            }
            
            // 如果没有队列请求且没有运行中的请求，等待通知
            // 使用超时避免永久阻塞（用于处理运行中请求的继续处理）
            if (queueSize == 0 && runningCount == 0) {
                // 空闲时等待新请求，使用超时以允许定期检查
                queueCondition_.wait_for(
                    lock,
                    std::chrono::microseconds(config_.idleLoopInterval),
                    [this]() {
                        return requestQueue_.getQueueSize() > 0 || !running_;
                    }
                );
            } else if (runningCount > 0) {
                // 🔥 优化5: 有运行中请求，极短间隔（1μs）快速处理，最大化吞吐量
                lock.unlock();
                std::this_thread::sleep_for(
                    std::chrono::microseconds(1)  // 优化：减少到1μs，最大化吞吐量
                );
            } else {
                // 🔥 优化5: 有队列请求但未运行，短间隔（10μs）
                lock.unlock();
                std::this_thread::sleep_for(
                    std::chrono::microseconds(10)  // 优化：减少到10μs，更快响应
                );
            }
            
        } catch (const std::exception& e) {
            CLLM_ERROR("Error in scheduler loop: %s", e.what());
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
}

void Scheduler::processRequests() {
    // 🔥 优化步骤1: 使用原子操作快速检查（只读，无副作用）
    // 先快速检查缓存值，如果为0则直接返回，避免不必要的锁竞争
    size_t queueSize = cachedQueueSize_.load(std::memory_order_relaxed);
    size_t runningCount = cachedRunningCount_.load(std::memory_order_relaxed);
    
    // 如果队列为空且没有运行中的请求，直接返回
    if (queueSize == 0 && runningCount == 0) {
        return;
    }
    
    // 需要实际处理时，获取精确值（需要锁）
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        queueSize = requestQueue_.getQueueSize();
        cachedQueueSize_.store(queueSize, std::memory_order_relaxed);
    }
    
    {
        std::shared_lock<std::shared_mutex> reqLock(requestsMutex_);  // 读操作使用共享锁
        runningCount = runningRequests_.size();
        cachedRunningCount_.store(runningCount, std::memory_order_relaxed);
    }
    
    // 再次检查（获取精确值后）
    if (queueSize == 0 && runningCount == 0) {
        return;
    }
    
    // 🔥 关键优化: 批处理累积策略
    // 如果队列请求较少且没有运行中的请求，等待更多请求到达
    // 这样可以形成更大的批处理，提高吞吐量
    const size_t minBatchSize = std::min<size_t>(8, std::max<size_t>(1, maxBatchSize_));
    constexpr size_t MAX_WAIT_MS_FOR_BATCH = 50;  // 最多等待50ms
    
    if (queueSize < minBatchSize && runningCount == 0) {
        CLLM_DEBUG("[Scheduler::processRequests] Queue size (%zu) < %zu, waiting for more requests (max %dms)",
                  queueSize, minBatchSize, MAX_WAIT_MS_FOR_BATCH);
        
        // 等待更多请求到达
        std::unique_lock<std::mutex> lock(queueMutex_);
        auto waitStart = std::chrono::steady_clock::now();
        
        // 等待直到队列足够大或超时
        queueCondition_.wait_for(
            lock,
            std::chrono::milliseconds(MAX_WAIT_MS_FOR_BATCH),
            [this, minBatchSize]() {
                return requestQueue_.getQueueSize() >= minBatchSize || !running_;
            }
        );
        
        auto waitEnd = std::chrono::steady_clock::now();
        auto waitTime = std::chrono::duration_cast<std::chrono::milliseconds>(waitEnd - waitStart).count();
        CLLM_DEBUG("[Scheduler::processRequests] Waited %lldms, queue size now: %zu",
                  waitTime, requestQueue_.getQueueSize());
        
        // 更新队列大小
        queueSize = requestQueue_.getQueueSize();
        cachedQueueSize_.store(queueSize, std::memory_order_relaxed);
        
        // 如果等待后队列仍然为空，返回
        if (queueSize == 0) {
            return;
        }
    }
    
    // Phase 1: 请求流转逻辑 - RequestQueue → runningRequests_（通过formBatch间接实现）
    // 1. 从 RequestQueue 获取待处理请求（PENDING状态）
    std::vector<RequestState> running = getRunningRequests();  // 获取当前运行中的请求（PENDING或PROCESSING）
    std::vector<RequestState> pending = requestQueue_.getPendingRequests();  // 从队列获取待处理请求
    
    // 🔥 优化: 减少序列ID检查频率，避免频繁锁竞争
    // 只在队列大小较大时才检查，小队列时假设有足够ID
    size_t availableSeqIds = 0;
    if (modelExecutor_ && queueSize > 4) {
        availableSeqIds = modelExecutor_->getAvailableSequenceIdCount();
        if (availableSeqIds > 0) {
            CLLM_DEBUG("[Scheduler::processRequests] Available sequence IDs: %zu", availableSeqIds);
        }
    } else if (modelExecutor_) {
        // 小队列时，假设有足够ID（避免锁竞争）
        availableSeqIds = 64;  // 假设有足够ID
    }
    
    // 3. formBatch 形成批处理（可能包含来自 RequestQueue 的新请求和运行中的请求）
    // formBatch 会根据 maxConcurrentRequests、资源限制和可用序列ID数量决定哪些请求可以加入批处理
    // 🔥 优化步骤3: 使用批处理池，减少内存分配
    auto& batch = batchPool_.acquire();
    batch = batchManager_.formBatch(pending, running, availableSeqIds);
    
    if (!batch.empty() && availableSeqIds > 0) {
        CLLM_DEBUG("[Scheduler::processRequests] Formed batch of %zu requests (availableSeqIds: %zu)", 
                  batch.size(), availableSeqIds);
    }
    
    // 如果 formBatch 返回空，但队列中还有请求，可能是因为资源限制
    // 这种情况下，我们仍然需要继续处理，但需要通知调度器继续尝试
    if (batch.empty() && queueSize > 0) {
        // 队列中有请求但无法形成批处理，可能是资源限制
        // 返回并让调度器稍后重试
        batchPool_.release(batch);
        return;
    }
    
    if (batch.empty()) {
        batchPool_.release(batch);
        return;
    }
    
    // 🔥 优化步骤3: 批量移除请求并更新原子缓存
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        for (const auto& req : batch) {
            requestQueue_.removeRequest(req.requestId);
        }
        cachedQueueSize_.store(requestQueue_.getQueueSize(), std::memory_order_relaxed);
    }
    
    processBatch(batch);
}

void Scheduler::processBatch(std::vector<RequestState>& batch) {
    // 在开始处理前，检查并合并已存在的请求状态
    // 🔥 优化步骤2: 减少锁持有时间
    // 步骤1: 快速复制需要处理的数据（短时间持有锁）
    struct RequestInfo {
        RequestState request;
        bool existsInRunning;
        bool isCompleted;
        bool isFailed;
        std::vector<int> existingTokens;
        bool isPending;
        bool isRunning;
        size_t startTime;
    };
    
    std::vector<RequestInfo> requestsToProcess;
    requestsToProcess.reserve(batch.size());
    
    {
        std::shared_lock<std::shared_mutex> lock(requestsMutex_);  // 读操作使用共享锁
        for (const auto& request : batch) {
            RequestInfo info;
            info.request = request;
            
            // 检查请求是否已经完成
            auto completedIt = completedRequests_.find(request.requestId);
            if (completedIt != completedRequests_.end()) {
                CLLM_DEBUG("Request %llu already completed, filtering out (tokens: %zu)",
                         request.requestId, completedIt->second.generatedTokens.size());
                continue;  // 已完成的请求不处理
            }
            
            // 检查请求是否在运行中
            auto it = runningRequests_.find(request.requestId);
            if (it != runningRequests_.end()) {
                info.existsInRunning = true;
                info.isCompleted = it->second.isCompleted;
                info.isFailed = it->second.isFailed;
                info.existingTokens = it->second.generatedTokens;
                info.isPending = it->second.isPending();
                info.isRunning = it->second.isRunning;
                info.startTime = it->second.startTime;
            } else {
                info.existsInRunning = false;
            }
            
            requestsToProcess.push_back(std::move(info));
        }
    }
    
    // 步骤2: 在锁外处理数据（无锁）
    std::vector<RequestState> activeBatch;
    activeBatch.reserve(requestsToProcess.size());
    
    for (auto& info : requestsToProcess) {
        auto& request = info.request;
        
        if (info.existsInRunning) {
            // 请求已存在，合并状态
            CLLM_DEBUG("Request %llu already in runningRequests_, merging state (existing tokens: %zu, isCompleted: %d)",
                      request.requestId, info.existingTokens.size(), info.isCompleted ? 1 : 0);
            
            request.generatedTokens = std::move(info.existingTokens);
            request.isCompleted = info.isCompleted;
            request.isFailed = info.isFailed;

            // Phase 1: 状态转换 PENDING → PROCESSING
            if (info.isPending) {
                CLLM_DEBUG("Request %llu: PENDING → PROCESSING", request.requestId);
                if (info.startTime == 0) {
                    request.startTime = getCurrentTime();
                }
            }
            
            request.isRunning = true;
            request.startTime = info.startTime;
        } else {
            // 新请求
            CLLM_DEBUG("Request %llu: NEW REQUEST (PENDING), will transition to PROCESSING", request.requestId);
            request.startTime = getCurrentTime();
            request.isRunning = false;
        }
        
        // 🔥 关键修复：检查请求是否已经达到maxTokens限制
        // 如果已经达到，标记为完成，避免继续生成
        if (!request.isCompleted && !request.isFailed && request.maxTokens > 0 &&
            request.generatedTokens.size() >= static_cast<size_t>(request.maxTokens)) {
            CLLM_DEBUG("Request %llu reached max tokens limit (%zu >= %d) before batch processing, marking as completed",
                      request.requestId, request.generatedTokens.size(), request.maxTokens);
            request.isCompleted = true;
        }
        
        // Phase 1: 状态转换 PENDING → PROCESSING
        request.isRunning = true;
        requestTracker_.markAsRunning(request.requestId);
        if (modelExecutor_) {
            modelExecutor_->updateKVCacheRequestStatus(request.requestId, inference::RequestStatus::PROCESSING);
        }
        
        activeBatch.push_back(std::move(request));
    }
    
    // 步骤3: 批量更新状态（短时间持有锁）
    {
        std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 写操作使用独占锁
        for (const auto& request : activeBatch) {
            auto it = runningRequests_.find(request.requestId);
            if (it != runningRequests_.end()) {
                it->second.isRunning = request.isRunning;
                if (it->second.startTime == 0) {
                    it->second.startTime = request.startTime;
                }
            } else {
                runningRequests_[request.requestId] = request;
            }
        }
    }
    
    // 如果所有请求都已完成，直接返回，不调用 processor.processBatch
    if (activeBatch.empty()) {
        CLLM_DEBUG("All requests in batch are already completed, skipping processing");
        return;
    }
    
    CLLM_INFO("Starting batch processing for %zu requests (filtered from %zu total)",
              activeBatch.size(), batch.size());
    
    SchedulerBatchProcessor processor(this, modelExecutor_, kvCache_, &batchManager_);
    processor.processBatch(activeBatch);
    
    // 🔥 优化: 检查是否有未完成的请求，将它们重新加入队列以便重组
    std::vector<RequestState> incompleteRequests;
    for (auto& request : activeBatch) {
        if (!request.isCompleted && !request.isFailed) {
            incompleteRequests.push_back(request);
        }
    }
    
    // 如果有未完成的请求，将它们重新加入队列，以便与其他请求重组
    if (!incompleteRequests.empty() && incompleteRequests.size() < activeBatch.size() * 0.5) {
        CLLM_DEBUG("[Scheduler] %zu incomplete requests from batch of %zu, re-queuing for regrouping",
                  incompleteRequests.size(), activeBatch.size());
        for (auto& request : incompleteRequests) {
            // 重新加入队列，以便与其他请求重组
            requestQueue_.addRequest(request);
        }
    }
    
    // 更新 batch 引用，用于后续处理
    batch = std::move(activeBatch);
    
    // 🔥 优化: 立即释放已完成请求的序列ID，避免阻塞后续批处理
    for (auto& request : batch) {
        request.completionTime = getCurrentTime();
        
        // 🔥 关键优化: 如果请求已完成，立即释放序列ID和KV缓存
        if (request.isCompleted || request.isFailed) {
            if (modelExecutor_) {
                // 立即清理KV缓存和释放序列ID，而不是等到异步清理
                modelExecutor_->cleanupKVCache(request.requestId);
                modelExecutor_->releaseSequenceId(request.requestId);
                CLLM_DEBUG("[Scheduler] Immediately released seq_id and KV cache for completed request %llu", 
                          request.requestId);
            }
        }
        
        CLLM_DEBUG("Request %llu generated tokens: %zu", request.requestId, request.generatedTokens.size());
        
        {
            std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 写操作使用独占锁
            if (request.isCompleted) {
                // Phase 1: 状态转换 PROCESSING → COMPLETED
                // 请求已完成，从 runningRequests_ 移除，添加到 completedRequests_
                CLLM_DEBUG("Request %llu: PROCESSING → COMPLETED (tokens: %zu)",
                          request.requestId, request.generatedTokens.size());
                
                if (modelExecutor_) {
                    modelExecutor_->updateKVCacheRequestStatus(request.requestId, inference::RequestStatus::COMPLETED);
                    // 🔥 优化：立即同步清理资源，立即释放序列ID，避免阻塞后续批处理
                    // 之前使用异步清理导致序列ID释放延迟，限制了批处理大小
                    modelExecutor_->cleanupKVCache(request.requestId);
                    modelExecutor_->releaseSequenceId(request.requestId);
                    CLLM_DEBUG("[Scheduler] Immediately released seq_id and KV cache for completed request %llu", 
                              request.requestId);
                }
                
                requestTracker_.markAsCompleted(request.requestId);
                stats_.update(request);
                runningRequests_.erase(request.requestId);
                completedRequests_[request.requestId] = request;
                
                // 🔥 优化步骤3: 更新原子缓存
                cachedRunningCount_.store(runningRequests_.size(), std::memory_order_relaxed);
                
                // 🔥 优化：通知等待该请求的线程（使用条件变量）
                resultCondition_.notify_all();
                
                // Phase 7: 触发完成回调
                triggerResponseCallback(request.requestId, request);
            } else if (request.isFailed) {
                // Phase 1: 状态转换 PROCESSING → FAILED
                // 请求失败，从 runningRequests_ 移除，添加到 completedRequests_
                CLLM_DEBUG("Request %llu: PROCESSING → FAILED (error: %s)",
                          request.requestId, request.errorMessage.c_str());
                
                if (modelExecutor_) {
                    modelExecutor_->updateKVCacheRequestStatus(request.requestId, inference::RequestStatus::FAILED);
                    // 🔥 优化：立即同步清理资源，立即释放序列ID
                    modelExecutor_->cleanupKVCache(request.requestId);
                    modelExecutor_->releaseSequenceId(request.requestId);
                    CLLM_DEBUG("[Scheduler] Immediately released seq_id and KV cache for failed request %llu", 
                              request.requestId);
                }
                
                requestTracker_.markAsFailed(request.requestId, request.errorMessage);
                stats_.failedRequests++;
                runningRequests_.erase(request.requestId);
                completedRequests_[request.requestId] = request;
                
                // 🔥 优化步骤3: 更新原子缓存
                cachedRunningCount_.store(runningRequests_.size(), std::memory_order_relaxed);
                
                // 🔥 优化：通知等待该请求的线程（使用条件变量）
                resultCondition_.notify_all();
                
                // Phase 7: 触发失败回调
                triggerResponseCallback(request.requestId, request);
            } else {
                // Phase 1: 状态保持 PROCESSING
                // 请求还在运行（PROCESSING状态），更新 runningRequests_ 中的状态
                auto it = runningRequests_.find(request.requestId);
                if (it != runningRequests_.end()) {
                    // 更新状态，保留已有的 generatedTokens 等
                    it->second = request;
                    // 🔥 优化步骤3: 更新原子缓存（状态更新）
                    cachedRunningCount_.store(runningRequests_.size(), std::memory_order_relaxed);
                    // 确保 isRunning 标志正确
                    it->second.isRunning = true;
                    CLLM_DEBUG("Request %llu: PROCESSING (continuing, tokens: %zu)",
                              request.requestId, request.generatedTokens.size());
                }
            }
        }
        
        resultCondition_.notify_all();
    }
    
    stats_.updateBatch(batch);
    
    // 🔥 优化步骤3: 释放批处理对象回池
    batchPool_.release(batch);
    
    // Phase 1: 请求流转逻辑 - 请求完成后自动触发下一个请求的处理
    // 检查是否还有待处理的请求，如果有，通知调度器继续处理
    // 这样可以避免调度器在有空闲资源时还在等待
    {
        std::lock_guard<std::mutex> queueLock(queueMutex_);
        size_t remainingQueueSize = requestQueue_.getQueueSize();
        if (remainingQueueSize > 0) {
            // 请求完成后，如果 RequestQueue 不为空且 runningRequests_.size() < maxConcurrentRequests，
            // 自动触发下一个请求的处理（通过 queueCondition_.notify_one()）
            CLLM_DEBUG("Request completed, notifying scheduler to process next request (queue size: %zu)", remainingQueueSize);
            queueCondition_.notify_one();
        }
    }
}

void Scheduler::checkRequestTimeout() {
    std::unique_lock<std::shared_mutex> lock(requestsMutex_);  // 写操作使用独占锁
    
    size_t currentTimeMs = getCurrentTime();
    std::vector<size_t> timeoutRequests;
    
    for (auto& pair : runningRequests_) {
        size_t requestId = pair.first;
        RequestState& request = pair.second;
        
        if (request.isProcessing() && request.startTime > 0) {
            if (currentTimeMs < request.startTime) {
                continue;
            }
            float processingTimeSec = static_cast<float>(currentTimeMs - request.startTime) / 1000.0f;
            
            if (processingTimeSec > config_.requestTimeout) {
                CLLM_WARN("Request %zu: TIMEOUT (processing time: %.2fs, timeout: %.2fs)",
                          requestId, processingTimeSec, config_.requestTimeout);
                timeoutRequests.push_back(requestId);
            }
        }
    }
    
    for (size_t requestId : timeoutRequests) {
        auto it = runningRequests_.find(requestId);
        if (it != runningRequests_.end()) {
            RequestState request = it->second;
            
            CLLM_WARN("Request %zu: PROCESSING → TIMEOUT", requestId);
            
            request.isTimeout = true;
            request.isFailed = true;
            request.errorMessage = "Request timeout";
            request.completionTime = currentTimeMs;
            
            CLLM_DEBUG("Request %zu: TIMEOUT (tokens: %zu)",
                      requestId, request.generatedTokens.size());
            
            if (modelExecutor_) {
                modelExecutor_->updateKVCacheRequestStatus(requestId, inference::RequestStatus::TIMEOUT);
                // 优化：异步清理资源，避免阻塞主流程
                cleanupRequestAsync(requestId);
            }
            
            requestTracker_.markAsFailed(requestId, request.errorMessage);
            stats_.failedRequests++;
            runningRequests_.erase(requestId);
            completedRequests_[requestId] = request;
            
            // 🔥 优化：通知等待该请求的线程（使用条件变量）
            resultCondition_.notify_all();
            
            // Phase 7: 触发超时回调
            triggerResponseCallback(requestId, request);
            
            resultCondition_.notify_all();
        }
    }
}

void Scheduler::checkKVCachEviction() {
    // Phase 5: KV缓存淘汰
    if (!modelExecutor_) {
        return;
    }

    size_t evictedCount = modelExecutor_->evictKVCachesIfNeeded(config_.kvCacheEvictionThreshold);
    if (evictedCount > 0) {
        CLLM_INFO("[Scheduler] KV cache eviction completed: evicted %zu requests", evictedCount);
    }
}

size_t Scheduler::getCurrentTime() {
    auto now = std::chrono::steady_clock::now();
    auto duration = now.time_since_epoch();
    return static_cast<size_t>(
        std::chrono::duration_cast<std::chrono::milliseconds>(duration).count()
    );
}

void Scheduler::cleanupRequestAsync(size_t requestId) {
    std::lock_guard<std::mutex> lock(cleanupMutex_);
    cleanupQueue_.push(requestId);
    cleanupCondition_.notify_one();
}

void Scheduler::cleanupLoop() {
    size_t processedCount = 0;
    while (running_) {
        std::unique_lock<std::mutex> lock(cleanupMutex_);
        
        // 等待清理任务或停止信号
        cleanupCondition_.wait_for(lock, std::chrono::milliseconds(100), [this]() {
            return !cleanupQueue_.empty() || !running_;
        });
        
        // 处理所有待清理的请求
        size_t batchSize = cleanupQueue_.size();
        while (!cleanupQueue_.empty()) {
            size_t requestId = cleanupQueue_.front();
            cleanupQueue_.pop();
            
            // 释放锁，执行清理操作
            lock.unlock();
            
            auto startTime = std::chrono::high_resolution_clock::now();
            if (modelExecutor_) {
                // Phase 4: 清理KV缓存
                modelExecutor_->cleanupKVCache(requestId);
                // Phase 2: 释放序列ID
                modelExecutor_->releaseSequenceId(requestId);
            }
            auto endTime = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count();
            
            processedCount++;
            
            // 每处理100个请求记录一次统计信息
            if (processedCount % 100 == 0) {
                CLLM_DEBUG("[Scheduler::cleanupLoop] Processed %zu cleanup tasks (avg time: %.2f us)", 
                          processedCount, static_cast<double>(duration));
            }
            
            // 重新获取锁，继续处理下一个任务
            lock.lock();
        }
        
        if (batchSize > 0) {
            CLLM_DEBUG("[Scheduler::cleanupLoop] Processed batch of %zu cleanup tasks", batchSize);
        }
    }
    CLLM_INFO("[Scheduler::cleanupLoop] Cleanup thread exiting (total processed: %zu)", processedCount);
}

RequestState Scheduler::generateStreaming(const RequestState& request, TokenCallback tokenCallback) {
    RequestState result = request;
    result.isRunning = true;
    result.startTime = getCurrentTime();
    
    CLLM_DEBUG("[generateStreaming] Starting streaming generation for %d tokens", request.maxTokens);
    
    // 使用 Scheduler 的现有机制，通过轮询检查生成进度
    // 这样可以正确使用批处理和 KV cache
    
    try {
        // 添加请求到队列
        size_t reqId = addRequest(request);
        
        size_t lastTokenCount = 0;
        const float timeoutSec = std::max(60.0f, static_cast<float>(request.maxTokens) * 0.5f);
        auto startTime = std::chrono::steady_clock::now();
        
        // 轮询检查生成进度，每发现新 token 就调用回调
        while (true) {
            // 检查超时
            auto now = std::chrono::steady_clock::now();
            float elapsed = std::chrono::duration<float>(now - startTime).count();
            if (elapsed > timeoutSec) {
                CLLM_WARN("[generateStreaming] Timeout after %.1fs", elapsed);
                result.isTimeout = true;
                break;
            }
            
            // 获取当前状态
            RequestState current;
            bool found = false;
            {
                std::shared_lock<std::shared_mutex> lock(requestsMutex_);
                auto it = runningRequests_.find(reqId);
                if (it != runningRequests_.end()) {
                    current = it->second;
                    found = true;
                }
                if (!found) {
                    auto cit = completedRequests_.find(reqId);
                    if (cit != completedRequests_.end()) {
                        current = cit->second;
                        found = true;
                    }
                }
            }
            
            if (!found) {
                // 请求还在队列中，等待
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
                continue;
            }
            
            // 检查是否有新 token
            size_t currentTokenCount = current.generatedTokens.size();
            if (currentTokenCount > lastTokenCount) {
                // 有新 token，调用回调
                for (size_t i = lastTokenCount; i < currentTokenCount; ++i) {
                    int token = current.generatedTokens[i];
                    if (tokenCallback) {
                        bool shouldContinue = tokenCallback(token);
                        if (!shouldContinue) {
                            CLLM_DEBUG("[generateStreaming] Callback requested stop");
                            // TODO: 可以考虑取消请求
                            break;
                        }
                    }
                }
                lastTokenCount = currentTokenCount;
            }
            
            // 检查是否完成
            if (current.isCompleted || current.isFailed || current.isTimeout) {
                result = current;
                break;
            }
            
            // 短暂等待再检查
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        
    } catch (const std::exception& e) {
        CLLM_ERROR("[generateStreaming] Exception: %s", e.what());
        result.isFailed = true;
        result.errorMessage = e.what();
    }
    
    result.completionTime = getCurrentTime();
    CLLM_DEBUG("[generateStreaming] Completed, generated %zu tokens", result.generatedTokens.size());
    
    return result;
}

}
