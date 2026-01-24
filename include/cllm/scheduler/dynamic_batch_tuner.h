/**
 * @file dynamic_batch_tuner.h
 * @brief 动态 Batch Size 调谐器（简化版）
 * @author cLLM Team
 * @date 2026-01-23
 */

#pragma once

#include <atomic>
#include <cstddef>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>

namespace cllm {

/**
 * @brief 动态批处理调谐器
 *
 * 通过批次耗时反馈执行指数探测与二分收敛，可选择动态调整或稳定锁定。
 */
class DynamicBatchTuner {
public:
    enum class TuningPhase {
        INITIAL_PROBING,
        PRECISION_SEARCH,
        STABLE_RUNNING,
        RE_PROBING
    };

    struct TunerConfig {
        size_t minBatchSize;
        size_t maxBatchSize;
        size_t initialBatchSize;
        double probingGrowthFactor;
        size_t maxProbingAttempts;
        double timeIncreaseThreshold;
        double adjustmentFactor;
        size_t validationInterval;
        size_t explorationInterval;
        size_t probeBatchCount;
        size_t validationBatchCount;
        bool autoAdjustEnabled;
        size_t maxConsecutiveTimeIncreases;
    };

    explicit DynamicBatchTuner(const TunerConfig& config);
    ~DynamicBatchTuner();

    void start(std::function<double(size_t)> batchRunner);
    void startPassive();
    void stop();

    void onBatchProcessed(size_t batchSize, double processingTimeMs);
    void updateConfig(const TunerConfig& config);
    TunerConfig getConfig() const;

    size_t getCurrentBatchSize() const;
    void setBatchSize(size_t batchSize);
    size_t getOptimalBatchSize() const;
    double getOptimalProcessingTime() const;
    TuningPhase getCurrentPhase() const;
    bool isStabilized() const;

    void resetToSafeState();
    void forceReprobe();

private:
    double runAndMeasure(size_t batchSize, size_t runCount);
    void performInitialProbing();
    void performPrecisionSearch();
    void validateCurrentBatchSize();
    void handleTimeIncrease(size_t batchSize, double processingTime);
    void performExploration();
    void adjustBatchSize();
    const char* getPhaseName(TuningPhase phase);

    void handlePassiveProbing(size_t batchSize, double processingTime);
    void handlePassivePrecisionSearch(size_t batchSize, double processingTime);

    TunerConfig config_;
    std::atomic<size_t> currentBatchSize_{1};
    std::atomic<size_t> optimalBatchSize_{1};
    std::atomic<double> optimalProcessingTime_{0.0};
    std::atomic<TuningPhase> phase_{TuningPhase::INITIAL_PROBING};
    std::atomic<bool> running_{false};

    std::function<double(size_t)> batchRunner_;

    std::mutex mutex_;
    mutable std::mutex configMutex_;
    std::condition_variable cv_;

    std::atomic<size_t> batchCount_{0};
    std::atomic<size_t> explorationCount_{0};
    std::atomic<size_t> consecutiveTimeIncreases_{0};

    std::map<size_t, double> probeResults_;
    size_t lowerBound_{0};
    size_t upperBound_{0};
    size_t currentProbeBatchSize_{0};
    size_t probingAttempts_{0};
    size_t lastProbeBatchSize_{0};
    double lastProbeProcessingTime_{0.0};
};

} // namespace cllm
