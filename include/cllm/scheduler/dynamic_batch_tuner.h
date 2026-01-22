#ifndef CLLM_SCHEDULER_DYNAMIC_BATCH_TUNER_H
#define CLLM_SCHEDULER_DYNAMIC_BATCH_TUNER_H

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace cllm {

class DynamicBatchTuner {
public:
    enum class SearchAlgorithm {
        STATIC,           
        ADAPTIVE_STEP,    
        EXPONENTIAL_BINARY
    };

    enum class TuningPhase {
        INITIAL_PROBING,  
        PRECISION_SEARCH, 
        STABLE_RUNNING,   
        RE_PROBING        
    };

    struct TunerConfig {
        SearchAlgorithm searchAlgorithm;
        size_t fixedBatchSize;
        size_t minBatchSize;
        size_t maxBatchSize;
        size_t initialBatchSize;
        double timeIncreaseThreshold;
        double timeDecreaseThreshold;
        size_t validationInterval;
        size_t maxConsecutiveTimeIncreases;
        bool autoAdjustEnabled;
        size_t probeBatchCount;
        size_t validationBatchCount;
        double adjustmentFactor;
        size_t explorationInterval;
    };

    struct PerformanceMetrics {
        double throughput;
        double processingTimeMs;
        size_t batchSize;
        std::chrono::steady_clock::time_point timestamp;
    };

    DynamicBatchTuner(const TunerConfig& config);
    ~DynamicBatchTuner();

    void start(std::function<double(size_t)> batchRunner);
    void stop();

    size_t getCurrentBatchSize() const;
    void setBatchSize(size_t batchSize);

    size_t getOptimalBatchSize() const;
    double getOptimalProcessingTime() const;

    TuningPhase getCurrentPhase() const;
    bool isStabilized() const;

    void onBatchProcessed(size_t batchSize, double processingTime);

    TunerConfig getConfig() const;
    void updateConfig(const TunerConfig& config);

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

    static const char* getPhaseName(TuningPhase phase);

    TunerConfig config_;
    mutable std::mutex configMutex_;

    std::atomic<TuningPhase> phase_;
    std::atomic<size_t> currentBatchSize_;
    std::atomic<size_t> optimalBatchSize_;
    std::atomic<double> optimalProcessingTime_;

    std::unordered_map<size_t, double> probeResults_;
    size_t currentProbeBatchSize_;
    size_t lowerBound_;
    size_t upperBound_;

    std::atomic<size_t> batchCount_;
    std::atomic<size_t> consecutiveTimeIncreases_;
    std::atomic<size_t> explorationCount_;

    std::function<double(size_t)> batchRunner_;

    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_;
};

} 

#endif 
