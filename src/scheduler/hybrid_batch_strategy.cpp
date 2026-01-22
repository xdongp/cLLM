#include "cllm/scheduler/hybrid_batch_strategy.h"
#include "cllm/common/logger.h"
#include <algorithm>
#include <numeric>
#include <map>

namespace cllm {

HybridBatchStrategy::HybridBatchStrategy(const HybridConfig& config)
    : config_(config)
    , currentPhase_(HybridPhase::TUNING)
    , optimalBatchSize_(config.stable.batchSize)
    , tuningRequestCount_(0)
    , stableRequestCount_(0)
    , baselineThroughput_(0.0)
    , phaseStartTime_(std::chrono::steady_clock::now())
    , lastCheckTime_(std::chrono::steady_clock::now()) {
    
    if (config_.enabled && config_.tuning.enabled) {
        DynamicBatchTuner::TunerConfig tunerConfig;
        tunerConfig.searchAlgorithm = DynamicBatchTuner::SearchAlgorithm::ADAPTIVE_STEP;
        tunerConfig.minBatchSize = config_.tuning.minBatchSize;
        tunerConfig.maxBatchSize = config_.tuning.maxBatchSize;
        tunerConfig.initialBatchSize = config_.tuning.initialBatchSize;
        tunerConfig.timeIncreaseThreshold = 0.30;
        tunerConfig.timeDecreaseThreshold = 0.10;
        tunerConfig.validationInterval = 50;
        tunerConfig.maxConsecutiveTimeIncreases = 5;
        tunerConfig.autoAdjustEnabled = true;
        tunerConfig.probeBatchCount = 10;
        tunerConfig.validationBatchCount = 10;
        tunerConfig.adjustmentFactor = 0.50;
        tunerConfig.explorationInterval = 200;
        
        tuner_ = std::make_unique<DynamicBatchTuner>(tunerConfig);
        
        CLLM_INFO("[HybridBatchStrategy] 初始化混合策略，调优阶段");
        CLLM_INFO("[HybridBatchStrategy] 调优配置: min=%zu, max=%zu, initial=%zu, duration=%zu",
                config_.tuning.minBatchSize, config_.tuning.maxBatchSize,
                config_.tuning.initialBatchSize, config_.tuning.durationRequests);
    } else {
        CLLM_INFO("[HybridBatchStrategy] 初始化混合策略，稳定阶段（批处理累积策略）");
        currentPhase_ = HybridPhase::STABLE;
    }
}

HybridBatchStrategy::~HybridBatchStrategy() {
    CLLM_INFO("[HybridBatchStrategy] 销毁混合策略，最终阶段: %s, 最优batch_size=%zu",
            currentPhase_ == HybridPhase::TUNING ? "调优" : "稳定",
            optimalBatchSize_);
}

size_t HybridBatchStrategy::getBatchSize(size_t queueSize, size_t runningCount) {
    if (currentPhase_ == HybridPhase::TUNING) {
        return getTuningBatchSize(queueSize, runningCount);
    } else {
        return getStableBatchSize(queueSize, runningCount);
    }
}

void HybridBatchStrategy::onBatchProcessed(size_t batchSize, double processingTime) {
    if (currentPhase_ == HybridPhase::TUNING) {
        recordTuningMetrics(batchSize, processingTime);
        tuningRequestCount_++;
        
        if (tuningRequestCount_ >= config_.tuning.durationRequests) {
            CLLM_INFO("[HybridBatchStrategy] 调优完成，已处理 %zu 个请求", tuningRequestCount_);
            switchToStable();
        }
    } else {
        recordStableMetrics(batchSize, processingTime);
        stableRequestCount_++;
        
        if (config_.monitoring.autoRetune && 
            stableRequestCount_ >= config_.monitoring.checkIntervalRequests) {
            if (checkPerformanceDrift()) {
                CLLM_WARN("[HybridBatchStrategy] 检测到性能下降，重新启动调优");
                switchToTuning();
            }
            stableRequestCount_ = 0;
        }
    }
}

HybridPhase HybridBatchStrategy::getCurrentPhase() const {
    return currentPhase_;
}

bool HybridBatchStrategy::isTuning() const {
    return currentPhase_ == HybridPhase::TUNING;
}

bool HybridBatchStrategy::isStable() const {
    return currentPhase_ == HybridPhase::STABLE;
}

size_t HybridBatchStrategy::getOptimalBatchSize() const {
    return optimalBatchSize_;
}

void HybridBatchStrategy::setOptimalBatchSize(size_t batchSize) {
    optimalBatchSize_ = batchSize;
    CLLM_INFO("[HybridBatchStrategy] 手动设置最优batch_size=%zu", batchSize);
}

void HybridBatchStrategy::triggerRetuning() {
    CLLM_INFO("[HybridBatchStrategy] 手动触发重新调优");
    switchToTuning();
}

void HybridBatchStrategy::reset() {
    CLLM_INFO("[HybridBatchStrategy] 重置混合策略");
    
    currentPhase_ = HybridPhase::TUNING;
    tuningRequestCount_ = 0;
    stableRequestCount_ = 0;
    
    {
        std::lock_guard<std::mutex> lock(metricsMutex_);
        tuningMetrics_.clear();
        stableMetrics_.clear();
    }
    
    baselineThroughput_ = 0.0;
    phaseStartTime_ = std::chrono::steady_clock::now();
    lastCheckTime_ = std::chrono::steady_clock::now();
    
    if (tuner_) {
        tuner_->resetToSafeState();
    }
}

HybridConfig HybridBatchStrategy::getConfig() const {
    return config_;
}

void HybridBatchStrategy::updateConfig(const HybridConfig& config) {
    CLLM_INFO("[HybridBatchStrategy] 更新配置");
    config_ = config;
}

size_t HybridBatchStrategy::getTuningBatchSize(size_t queueSize, size_t runningCount) {
    if (!tuner_) {
        return config_.tuning.initialBatchSize;
    }
    
    return tuner_->getCurrentBatchSize();
}

size_t HybridBatchStrategy::getStableBatchSize(size_t queueSize, size_t runningCount) {
    size_t minBatchSize = config_.stable.minBatchSize;
    size_t targetBatchSize = optimalBatchSize_;
    
    if (!config_.stable.accumulationEnabled) {
        return std::min(queueSize, targetBatchSize);
    }
    
    if (queueSize < minBatchSize && runningCount == 0) {
        CLLM_DEBUG("[HybridBatchStrategy] 队列大小(%zu) < %zu，等待更多请求",
                 queueSize, minBatchSize);
        return minBatchSize;
    }
    
    return std::min(queueSize, targetBatchSize);
}

void HybridBatchStrategy::recordTuningMetrics(size_t batchSize, double processingTime) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    tuningMetrics_.emplace_back(batchSize, processingTime, 0.0);
    
    if (tuner_) {
        tuner_->onBatchProcessed(batchSize, processingTime);
    }
}

void HybridBatchStrategy::recordStableMetrics(size_t batchSize, double processingTime) {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    stableMetrics_.emplace_back(batchSize, processingTime, 0.0);
}

size_t HybridBatchStrategy::findOptimalBatchSize() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    if (tuningMetrics_.empty()) {
        CLLM_WARN("[HybridBatchStrategy] 没有调优数据，使用默认值");
        return config_.stable.batchSize;
    }
    
    std::map<size_t, std::vector<double>> batchTimes;
    for (const auto& metric : tuningMetrics_) {
        batchTimes[metric.batchSize].push_back(metric.processingTimeMs);
    }
    
    size_t bestBatchSize = config_.stable.batchSize;
    double bestScore = std::numeric_limits<double>::max();
    
    for (const auto& pair : batchTimes) {
        size_t batchSize = pair.first;
        const std::vector<double>& times = pair.second;
        
        if (times.empty()) continue;
        
        double avgTime = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
        double variance = 0.0;
        for (double time : times) {
            variance += (time - avgTime) * (time - avgTime);
        }
        variance /= times.size();
        double stdDev = std::sqrt(variance);
        
        double cv = stdDev / avgTime;
        double score = avgTime * (1.0 + cv);
        
        if (score < bestScore) {
            bestScore = score;
            bestBatchSize = batchSize;
        }
    }
    
    CLLM_INFO("[HybridBatchStrategy] 找到最优batch_size=%zu (平均时间=%.2fms, CV=%.2f%%%)",
            bestBatchSize, bestScore, 0.0);
    
    return bestBatchSize;
}

bool HybridBatchStrategy::checkPerformanceDrift() {
    std::lock_guard<std::mutex> lock(metricsMutex_);
    
    if (stableMetrics_.size() < 50) {
        return false;
    }
    
    size_t recentCount = std::min(size_t(100), stableMetrics_.size());
    auto startIt = stableMetrics_.end() - recentCount;
    
    double recentThroughput = 0.0;
    for (auto it = startIt; it != stableMetrics_.end(); ++it) {
        if (it->processingTimeMs > 0) {
            recentThroughput += it->batchSize / (it->processingTimeMs / 1000.0);
        }
    }
    recentThroughput /= recentCount;
    
    if (baselineThroughput_ == 0.0) {
        baselineThroughput_ = recentThroughput;
        return false;
    }
    
    double drift = (baselineThroughput_ - recentThroughput) / baselineThroughput_;
    
    CLLM_DEBUG("[HybridBatchStrategy] 性能检查: 基准=%.2f, 当前=%.2f, 漂移=%.2f%%%)",
            baselineThroughput_, recentThroughput, drift * 100.0);
    
    if (drift > config_.monitoring.driftThreshold) {
        CLLM_WARN("[HybridBatchStrategy] 性能下降 %.2f%%%，超过阈值 %.2f%%%",
                 drift * 100.0, config_.monitoring.driftThreshold * 100.0);
        return true;
    }
    
    return false;
}

void HybridBatchStrategy::switchToStable() {
    CLLM_INFO("[HybridBatchStrategy] 切换到稳定阶段");
    
    optimalBatchSize_ = findOptimalBatchSize();
    currentPhase_ = HybridPhase::STABLE;
    tuningRequestCount_ = 0;
    stableRequestCount_ = 0;
    
    {
        std::lock_guard<std::mutex> lock(metricsMutex_);
        tuningMetrics_.clear();
        stableMetrics_.clear();
    }
    
    phaseStartTime_ = std::chrono::steady_clock::now();
    
    CLLM_INFO("[HybridBatchStrategy] 稳定阶段配置: batch_size=%zu, accumulation=%s",
            optimalBatchSize_, config_.stable.accumulationEnabled ? "启用" : "禁用");
}

void HybridBatchStrategy::switchToTuning() {
    CLLM_INFO("[HybridBatchStrategy] 切换到调优阶段");
    
    currentPhase_ = HybridPhase::TUNING;
    tuningRequestCount_ = 0;
    stableRequestCount_ = 0;
    baselineThroughput_ = 0.0;
    
    {
        std::lock_guard<std::mutex> lock(metricsMutex_);
        tuningMetrics_.clear();
        stableMetrics_.clear();
    }
    
    phaseStartTime_ = std::chrono::steady_clock::now();
    
    if (tuner_) {
        tuner_->resetToSafeState();
        tuner_->forceReprobe();
    }
}

} // namespace cllm