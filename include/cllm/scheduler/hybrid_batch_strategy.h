#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include "cllm/scheduler/dynamic_batch_tuner.h"

namespace cllm {

enum class HybridPhase {
    TUNING,    // 调优阶段
    STABLE      // 稳定运行阶段
};

struct HybridConfig {
    bool enabled;
    
    // 调优阶段配置
    struct {
        bool enabled;
        size_t durationRequests;
        size_t minBatchSize;
        size_t maxBatchSize;
        size_t initialBatchSize;
    } tuning;
    
    // 稳定阶段配置
    struct {
        size_t batchSize;
        bool accumulationEnabled;
        size_t minBatchSize;
        size_t maxWaitMs;
    } stable;
    
    // 性能监控配置
    struct {
        size_t checkIntervalRequests;
        double driftThreshold;
        bool autoRetune;
    } monitoring;
    
    HybridConfig()
        : enabled(true)
        , tuning{true, 100, 4, 96, 16}
        , stable{24, true, 8, 50}
        , monitoring{1000, 0.10, true} {}
};

struct PerformanceMetrics {
    size_t batchSize;
    double processingTimeMs;
    double throughput;
    std::chrono::steady_clock::time_point timestamp;
    
    PerformanceMetrics(size_t bs, double time, double tp)
        : batchSize(bs)
        , processingTimeMs(time)
        , throughput(tp)
        , timestamp(std::chrono::steady_clock::now()) {}
};

class HybridBatchStrategy {
public:
    explicit HybridBatchStrategy(const HybridConfig& config);
    ~HybridBatchStrategy();
    
    size_t getBatchSize(size_t queueSize, size_t runningCount);
    void onBatchProcessed(size_t batchSize, double processingTime);
    
    HybridPhase getCurrentPhase() const;
    bool isTuning() const;
    bool isStable() const;
    
    size_t getOptimalBatchSize() const;
    void setOptimalBatchSize(size_t batchSize);
    
    void triggerRetuning();
    void reset();
    
    HybridConfig getConfig() const;
    void updateConfig(const HybridConfig& config);
    
private:
    size_t getTuningBatchSize(size_t queueSize, size_t runningCount);
    size_t getStableBatchSize(size_t queueSize, size_t runningCount);
    
    void recordTuningMetrics(size_t batchSize, double processingTime);
    void recordStableMetrics(size_t batchSize, double processingTime);
    
    size_t findOptimalBatchSize();
    bool checkPerformanceDrift();
    void switchToStable();
    void switchToTuning();
    
    HybridConfig config_;
    HybridPhase currentPhase_;
    
    std::unique_ptr<DynamicBatchTuner> tuner_;
    
    size_t optimalBatchSize_;
    size_t tuningRequestCount_;
    size_t stableRequestCount_;
    
    std::vector<PerformanceMetrics> tuningMetrics_;
    std::vector<PerformanceMetrics> stableMetrics_;
    
    std::mutex metricsMutex_;
    
    std::chrono::steady_clock::time_point phaseStartTime_;
    std::chrono::steady_clock::time_point lastCheckTime_;
    
    double baselineThroughput_;
};

} // namespace cllm