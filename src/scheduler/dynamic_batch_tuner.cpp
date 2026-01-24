#include "cllm/scheduler/dynamic_batch_tuner.h"

#include <algorithm>
#include <cmath>
#include <thread>

#include "cllm/common/logger.h"

namespace cllm {

DynamicBatchTuner::DynamicBatchTuner(const TunerConfig& config)
    : config_(config) {
    currentBatchSize_ = config_.initialBatchSize;
    
    CLLM_INFO("[DynamicBatchTuner] 初始化完成，配置: min_batch_size=%zu, max_batch_size=%zu, 时间阈值=%.1f%%",
              config_.minBatchSize, config_.maxBatchSize, config_.timeIncreaseThreshold * 100);
}

DynamicBatchTuner::~DynamicBatchTuner() {
    stop();
}

void DynamicBatchTuner::start(std::function<double(size_t)> batchRunner) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (running_) {
        CLLM_WARN("[DynamicBatchTuner] 已在运行中");
        return;
    }
    
    batchRunner_ = std::move(batchRunner);
    running_ = true;
    phase_ = TuningPhase::INITIAL_PROBING;
    probingAttempts_ = 0;
    lowerBound_ = 0;
    upperBound_ = 0;
    lastProbeBatchSize_ = 0;
    lastProbeProcessingTime_ = 0.0;
    
    CLLM_INFO("[DynamicBatchTuner] 开始调谐，阶段: %s", getPhaseName(phase_));
    
    std::thread([this]() {
        performInitialProbing();
        
        if (running_) {
            performPrecisionSearch();
            
            if (running_) {
                phase_ = TuningPhase::STABLE_RUNNING;
                CLLM_INFO("[DynamicBatchTuner] 进入稳定运行阶段，最优 batch_size=%zu, 处理时间=%.2fms",
                          optimalBatchSize_.load(), optimalProcessingTime_.load());
            }
        }
    }).detach();
}

void DynamicBatchTuner::startPassive() {
    std::lock_guard<std::mutex> lock(mutex_);

    if (running_) {
        CLLM_WARN("[DynamicBatchTuner] 已在运行中");
        return;
    }

    batchRunner_ = nullptr;
    running_ = true;
    phase_ = TuningPhase::INITIAL_PROBING;
    probingAttempts_ = 0;
    lowerBound_ = 0;
    upperBound_ = 0;
    lastProbeBatchSize_ = 0;
    lastProbeProcessingTime_ = 0.0;

    CLLM_INFO("[DynamicBatchTuner] 启动被动调谐，阶段: %s", getPhaseName(phase_));
}

void DynamicBatchTuner::stop() {
    running_ = false;
    cv_.notify_all();
    
    CLLM_INFO("[DynamicBatchTuner] 已停止");
}

size_t DynamicBatchTuner::getCurrentBatchSize() const {
    return currentBatchSize_.load();
}

void DynamicBatchTuner::setBatchSize(size_t batchSize) {
    if (batchSize < config_.minBatchSize) {
        CLLM_WARN("[DynamicBatchTuner] batch_size=%zu 小于最小值 %zu，使用最小值",
                  batchSize, config_.minBatchSize);
        batchSize = config_.minBatchSize;
    }
    
    if (batchSize > config_.maxBatchSize) {
        CLLM_WARN("[DynamicBatchTuner] batch_size=%zu 大于最大值 %zu，使用最大值",
                  batchSize, config_.maxBatchSize);
        batchSize = config_.maxBatchSize;
    }
    
    currentBatchSize_ = batchSize;
    
    CLLM_INFO("[DynamicBatchTuner] 设置 batch_size=%zu", batchSize);
}

size_t DynamicBatchTuner::getOptimalBatchSize() const {
    return optimalBatchSize_.load();
}

double DynamicBatchTuner::getOptimalProcessingTime() const {
    return optimalProcessingTime_.load();
}

DynamicBatchTuner::TuningPhase DynamicBatchTuner::getCurrentPhase() const {
    return phase_.load();
}

bool DynamicBatchTuner::isStabilized() const {
    return phase_.load() == TuningPhase::STABLE_RUNNING;
}

void DynamicBatchTuner::onBatchProcessed(size_t batchSize, double processingTime) {
    if (!running_) {
        return;
    }
    
    batchCount_++;

    if (!batchRunner_) {
        if (phase_ == TuningPhase::INITIAL_PROBING) {
            handlePassiveProbing(batchSize, processingTime);
            return;
        }
        if (phase_ == TuningPhase::PRECISION_SEARCH) {
            handlePassivePrecisionSearch(batchSize, processingTime);
            return;
        }
    }

    if (phase_ == TuningPhase::STABLE_RUNNING && !config_.autoAdjustEnabled) {
        return;
    }
    
    if (optimalProcessingTime_ > 0) {
        double timeIncrease = (processingTime - optimalProcessingTime_) / optimalProcessingTime_;
        
        if (timeIncrease > config_.timeIncreaseThreshold) {
            handleTimeIncrease(batchSize, processingTime);
        } else {
            consecutiveTimeIncreases_ = 0;
        }
    }
    
    if (phase_ == TuningPhase::STABLE_RUNNING && 
        batchCount_ % config_.validationInterval == 0) {
        validateCurrentBatchSize();
    }
    
    if (phase_ == TuningPhase::STABLE_RUNNING && 
        explorationCount_ % config_.explorationInterval == 0) {
        performExploration();
    }
    
    explorationCount_++;
}

DynamicBatchTuner::TunerConfig DynamicBatchTuner::getConfig() const {
    std::lock_guard<std::mutex> lock(configMutex_);
    return config_;
}

void DynamicBatchTuner::updateConfig(const TunerConfig& config) {
    std::lock_guard<std::mutex> lock(configMutex_);
    config_ = config;
    
    CLLM_INFO("[DynamicBatchTuner] 配置已更新");
}

void DynamicBatchTuner::resetToSafeState() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    currentBatchSize_ = config_.minBatchSize;
    consecutiveTimeIncreases_ = 0;
    
    CLLM_WARN("[DynamicBatchTuner] 已重置到安全状态，batch_size=%zu", config_.minBatchSize);
}

void DynamicBatchTuner::forceReprobe() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    probeResults_.clear();
    
    phase_ = TuningPhase::RE_PROBING;
    currentProbeBatchSize_ = std::max(1UL, currentBatchSize_ / 2);
    
    CLLM_INFO("[DynamicBatchTuner] 强制重新探测，从 batch_size=%zu 开始", currentProbeBatchSize_);
    
    std::thread([this]() {
        performInitialProbing();
        
        if (running_) {
            performPrecisionSearch();
            
            if (running_) {
                phase_ = TuningPhase::STABLE_RUNNING;
                CLLM_INFO("[DynamicBatchTuner] 重新探测完成，最优 batch_size=%zu", optimalBatchSize_.load());
            }
        }
    }).detach();
}

double DynamicBatchTuner::runAndMeasure(size_t batchSize, size_t runCount) {
    if (!batchRunner_) {
        CLLM_ERROR("[DynamicBatchTuner] batchRunner 未设置");
        return 0.0;
    }
    
    double totalTime = 0.0;
    
    for (size_t i = 0; i < runCount; ++i) {
        double time = batchRunner_(batchSize);
        totalTime += time;
    }
    
    return totalTime / runCount;
}

void DynamicBatchTuner::performInitialProbing() {
    CLLM_INFO("[DynamicBatchTuner] 开始初始探测");
    
    probeResults_.clear();
    currentProbeBatchSize_ = config_.minBatchSize;
    size_t attempts = 0;
    const double growthFactor = std::max(1.1, config_.probingGrowthFactor);
    
    while (currentProbeBatchSize_ <= config_.maxBatchSize && running_) {
        double avgTime = runAndMeasure(currentProbeBatchSize_, config_.probeBatchCount);
        
        CLLM_DEBUG("[DynamicBatchTuner] 探测 batch_size=%zu, 处理时间=%.2fms",
                   currentProbeBatchSize_, avgTime);
        
        probeResults_[currentProbeBatchSize_] = avgTime;
        
        if (currentProbeBatchSize_ > config_.minBatchSize) {
            size_t prevBatchSize = currentProbeBatchSize_ / 2;
            double prevTime = probeResults_[prevBatchSize];
            double timeIncrease = (avgTime - prevTime) / prevTime;
            
            if (timeIncrease > config_.timeIncreaseThreshold) {
                CLLM_INFO("[DynamicBatchTuner] 探测到临界点: batch_size=%zu, 时间从 %.2fms 增加到 %.2fms (%.1f%%)",
                         prevBatchSize, prevTime, avgTime, timeIncrease * 100);
                
                lowerBound_ = prevBatchSize;
                upperBound_ = currentProbeBatchSize_;
                break;
            }
        }
        
        attempts++;
        if (attempts >= config_.maxProbingAttempts) {
            break;
        }

        size_t nextSize = static_cast<size_t>(std::ceil(currentProbeBatchSize_ * growthFactor));
        if (nextSize <= currentProbeBatchSize_) {
            nextSize = currentProbeBatchSize_ + 1;
        }
        currentProbeBatchSize_ = nextSize;
    }
    
    if (upperBound_ == 0 && running_) {
        CLLM_WARN("[DynamicBatchTuner] 未探测到临界点，使用 max_batch_size=%zu", config_.maxBatchSize);
        lowerBound_ = config_.maxBatchSize / 2;
        upperBound_ = config_.maxBatchSize;
    }
    
    CLLM_INFO("[DynamicBatchTuner] 初始探测完成，范围: [%zu, %zu]", lowerBound_, upperBound_);
}

void DynamicBatchTuner::performPrecisionSearch() {
    if (!running_ || lowerBound_ >= upperBound_) {
        return;
    }
    
    CLLM_INFO("[DynamicBatchTuner] 开始精确搜索，范围: [%zu, %zu]", lowerBound_, upperBound_);
    
    size_t left = lowerBound_;
    size_t right = upperBound_;
    
    while (left < right && running_) {
        size_t mid = left + (right - left) / 2;
        
        double midTime = runAndMeasure(mid, config_.probeBatchCount);
        double nextTime = runAndMeasure(mid + 1, config_.probeBatchCount);
        
        double timeIncrease = (nextTime - midTime) / midTime;
        
        CLLM_DEBUG("[DynamicBatchTuner] 二分查找: [%.0f, %.0f], mid=%.0f, 时间增加=%.1f%%",
                   left, right, mid, timeIncrease * 100);
        
        if (timeIncrease > config_.timeIncreaseThreshold) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    
    if (running_) {
        optimalBatchSize_.store(left);
        optimalProcessingTime_.store(probeResults_[left]);
        currentBatchSize_.store(left);
        
        CLLM_INFO("[DynamicBatchTuner] 精确搜索完成，最优 batch_size=%zu, 处理时间=%.2fms",
                  left, probeResults_[left]);
    }
}

void DynamicBatchTuner::validateCurrentBatchSize() {
    if (!running_) {
        return;
    }
    
    size_t currentSize = currentBatchSize_.load();
    double currentTime = runAndMeasure(currentSize, config_.validationBatchCount);
    
    double optimalTime = optimalProcessingTime_.load();
    double timeIncrease = (currentTime - optimalTime) / optimalTime;
    
    CLLM_DEBUG("[DynamicBatchTuner] 验证 batch_size=%zu, 当前时间=%.2fms, 最优时间=%.2fms, 差异=%.1f%%",
               currentSize, currentTime, optimalTime, timeIncrease * 100);
    
    if (timeIncrease > config_.timeIncreaseThreshold) {
        CLLM_WARN("[DynamicBatchTuner] 当前处理时间增加 %.1f%%，需要调整", timeIncrease * 100);
        
        size_t newBatchSize = std::max(config_.minBatchSize, 
                                       static_cast<size_t>(currentSize * (1 - config_.adjustmentFactor)));
        setBatchSize(newBatchSize);
    } else if (timeIncrease < -config_.timeIncreaseThreshold) {
        CLLM_INFO("[DynamicBatchTuner] 当前处理时间减少 %.1f%%，可以尝试增加", -timeIncrease * 100);
        
        if (currentSize < config_.maxBatchSize) {
            size_t newBatchSize = std::min(config_.maxBatchSize, 
                                           static_cast<size_t>(currentSize * (1 + config_.adjustmentFactor)));
            setBatchSize(newBatchSize);
        }
    }
}

void DynamicBatchTuner::handleTimeIncrease(size_t batchSize, double processingTime) {
    consecutiveTimeIncreases_.fetch_add(1);
    
    size_t count = consecutiveTimeIncreases_.load();
    
    CLLM_WARN("[DynamicBatchTuner] 处理时间增加 (连续 %zu 次): batch_size=%zu, 时间=%.2fms",
              count, batchSize, processingTime);
    
    if (count >= config_.maxConsecutiveTimeIncreases) {
        size_t newBatchSize = std::max(config_.minBatchSize, 
                                       static_cast<size_t>(batchSize * (1 - config_.adjustmentFactor)));
        setBatchSize(newBatchSize);
        
        CLLM_WARN("[DynamicBatchTuner] 连续 %zu 次处理时间增加，强制减小 batch_size 到 %zu",
                  config_.maxConsecutiveTimeIncreases, newBatchSize);
        
        consecutiveTimeIncreases_.store(0);
    }
}

void DynamicBatchTuner::performExploration() {
    if (!running_ || !config_.autoAdjustEnabled) {
        return;
    }
    
    size_t exploreBatchSize = static_cast<size_t>(optimalBatchSize_ * (0.8 + (rand() % 40) / 100.0));
    exploreBatchSize = std::max(config_.minBatchSize, 
                                std::min(config_.maxBatchSize, exploreBatchSize));
    
    CLLM_INFO("[DynamicBatchTuner] 定期探索: 尝试 batch_size=%zu", exploreBatchSize);
    
    double exploreTime = runAndMeasure(exploreBatchSize, config_.validationBatchCount);
    
    double optimalTime = optimalProcessingTime_.load();
    if (exploreTime < optimalTime * (1 - config_.timeIncreaseThreshold)) {
        CLLM_INFO("[DynamicBatchTuner] 探索发现更好的 batch_size=%zu, 处理时间=%.2fms (原最优=%.2fms)",
                  exploreBatchSize, exploreTime, optimalTime);
        
        optimalBatchSize_.store(exploreBatchSize);
        optimalProcessingTime_.store(exploreTime);
        currentBatchSize_.store(exploreBatchSize);
    }
}

void DynamicBatchTuner::adjustBatchSize() {
    size_t startBatchSize = std::max(1UL, currentBatchSize_.load() / 2);
    
    CLLM_INFO("[DynamicBatchTuner] 重新探测，从 batch_size=%zu 开始", startBatchSize);
    
    phase_ = TuningPhase::RE_PROBING;
    currentProbeBatchSize_ = startBatchSize;
    
    std::thread([this]() {
        performInitialProbing();
        
        if (running_) {
            performPrecisionSearch();
            
            if (running_) {
                phase_ = TuningPhase::STABLE_RUNNING;
                CLLM_INFO("[DynamicBatchTuner] 调整完成，最优 batch_size=%zu", optimalBatchSize_.load());
            }
        }
    }).detach();
}

void DynamicBatchTuner::handlePassiveProbing(size_t batchSize, double processingTime) {
    if (batchSize < config_.minBatchSize) {
        return;
    }

    probingAttempts_++;

    if (optimalProcessingTime_ <= 0.0 || processingTime < optimalProcessingTime_) {
        optimalProcessingTime_.store(processingTime);
        optimalBatchSize_.store(batchSize);
    }

    if (lastProbeBatchSize_ > 0 && lastProbeProcessingTime_ > 0.0) {
        double timeIncrease = (processingTime - lastProbeProcessingTime_) / lastProbeProcessingTime_;
        if (timeIncrease > config_.timeIncreaseThreshold) {
            lowerBound_ = std::max(config_.minBatchSize, optimalBatchSize_.load());
            upperBound_ = std::min(config_.maxBatchSize, std::max(batchSize, lowerBound_ + 1));

            if (lowerBound_ >= upperBound_) {
                phase_ = TuningPhase::STABLE_RUNNING;
                currentBatchSize_.store(optimalBatchSize_.load());
                CLLM_INFO("[DynamicBatchTuner] 被动探测完成，最优 batch_size=%zu", optimalBatchSize_.load());
                return;
            }

            phase_ = TuningPhase::PRECISION_SEARCH;
            currentProbeBatchSize_ = (lowerBound_ + upperBound_) / 2;
            currentBatchSize_.store(currentProbeBatchSize_);
            CLLM_INFO("[DynamicBatchTuner] 被动探测进入精确搜索，范围: [%zu, %zu]",
                      lowerBound_, upperBound_);
            return;
        }
    }

    lastProbeBatchSize_ = batchSize;
    lastProbeProcessingTime_ = processingTime;

    if (probingAttempts_ >= config_.maxProbingAttempts || batchSize >= config_.maxBatchSize) {
        phase_ = TuningPhase::STABLE_RUNNING;
        currentBatchSize_.store(optimalBatchSize_.load());
        CLLM_INFO("[DynamicBatchTuner] 被动探测结束，最优 batch_size=%zu", optimalBatchSize_.load());
        return;
    }

    const double growthFactor = std::max(1.1, config_.probingGrowthFactor);
    size_t nextSize = static_cast<size_t>(std::ceil(batchSize * growthFactor));
    if (nextSize <= batchSize) {
        nextSize = batchSize + 1;
    }
    nextSize = std::min(nextSize, config_.maxBatchSize);
    currentBatchSize_.store(nextSize);
}

void DynamicBatchTuner::handlePassivePrecisionSearch(size_t batchSize, double processingTime) {
    if (lowerBound_ == 0 || upperBound_ == 0 || lowerBound_ >= upperBound_) {
        phase_ = TuningPhase::STABLE_RUNNING;
        currentBatchSize_.store(optimalBatchSize_.load());
        CLLM_INFO("[DynamicBatchTuner] 精确搜索结束，最优 batch_size=%zu", optimalBatchSize_.load());
        return;
    }

    double bestTime = optimalProcessingTime_.load();
    if (bestTime <= 0.0) {
        bestTime = processingTime;
        optimalProcessingTime_.store(processingTime);
        optimalBatchSize_.store(batchSize);
    }

    if (processingTime < bestTime) {
        optimalProcessingTime_.store(processingTime);
        optimalBatchSize_.store(batchSize);
        lowerBound_ = std::min(config_.maxBatchSize, batchSize + 1);
    } else if (processingTime > bestTime * (1 + config_.timeIncreaseThreshold)) {
        upperBound_ = batchSize;
    } else {
        lowerBound_ = std::min(config_.maxBatchSize, batchSize + 1);
    }

    if (lowerBound_ >= upperBound_) {
        phase_ = TuningPhase::STABLE_RUNNING;
        currentBatchSize_.store(optimalBatchSize_.load());
        CLLM_INFO("[DynamicBatchTuner] 精确搜索收敛，最优 batch_size=%zu", optimalBatchSize_.load());
        return;
    }

    currentProbeBatchSize_ = (lowerBound_ + upperBound_) / 2;
    currentBatchSize_.store(currentProbeBatchSize_);
}

const char* DynamicBatchTuner::getPhaseName(TuningPhase phase) {
    switch (phase) {
        case TuningPhase::INITIAL_PROBING:
            return "INITIAL_PROBING";
        case TuningPhase::PRECISION_SEARCH:
            return "PRECISION_SEARCH";
        case TuningPhase::STABLE_RUNNING:
            return "STABLE_RUNNING";
        case TuningPhase::RE_PROBING:
            return "RE_PROBING";
        default:
            return "UNKNOWN";
    }
}

} 
