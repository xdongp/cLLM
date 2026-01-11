#include "cllm/CTokenizer/performance_monitor.h"
#include <algorithm>
#include <numeric>

namespace cllm {

PerformanceMonitor::PerformanceMonitor()
    : startTime_(std::chrono::system_clock::now())
    , lastStatsTime_(startTime_) {
    encodeLatencies_.reserve(MAX_LATENCY_SAMPLES);
    decodeLatencies_.reserve(MAX_LATENCY_SAMPLES);
}

void PerformanceMonitor::recordEncode(double durationMs, size_t tokenCount) {
    totalEncodes_.fetch_add(1, std::memory_order_relaxed);
    totalTokensEncoded_.fetch_add(tokenCount, std::memory_order_relaxed);
    
    // 记录延迟样本（限制样本数量）
    std::lock_guard<std::mutex> lock(latencyMutex_);
    if (encodeLatencies_.size() < MAX_LATENCY_SAMPLES) {
        encodeLatencies_.push_back(durationMs);
    } else {
        // 随机替换策略（简单的蓄水池采样）
        size_t idx = totalEncodes_.load(std::memory_order_relaxed) % MAX_LATENCY_SAMPLES;
        encodeLatencies_[idx] = durationMs;
    }
}

void PerformanceMonitor::recordDecode(double durationMs, size_t tokenCount) {
    totalDecodes_.fetch_add(1, std::memory_order_relaxed);
    totalTokensDecoded_.fetch_add(tokenCount, std::memory_order_relaxed);
    
    // 记录延迟样本
    std::lock_guard<std::mutex> lock(latencyMutex_);
    if (decodeLatencies_.size() < MAX_LATENCY_SAMPLES) {
        decodeLatencies_.push_back(durationMs);
    } else {
        size_t idx = totalDecodes_.load(std::memory_order_relaxed) % MAX_LATENCY_SAMPLES;
        decodeLatencies_[idx] = durationMs;
    }
}

void PerformanceMonitor::recordCacheHit() {
    cacheHits_.fetch_add(1, std::memory_order_relaxed);
}

void PerformanceMonitor::recordCacheMiss() {
    cacheMisses_.fetch_add(1, std::memory_order_relaxed);
}

void PerformanceMonitor::updateMemoryUsage(size_t bytes) {
    currentMemoryUsage_.store(bytes, std::memory_order_relaxed);
    
    // 更新峰值内存
    size_t currentPeak = peakMemoryUsage_.load(std::memory_order_relaxed);
    while (bytes > currentPeak) {
        if (peakMemoryUsage_.compare_exchange_weak(currentPeak, bytes, std::memory_order_relaxed)) {
            break;
        }
    }
}

TokenizerPerformanceStats PerformanceMonitor::getStats() const {
    TokenizerPerformanceStats stats;
    
    // 获取原子计数器的值
    stats.totalEncodes = totalEncodes_.load(std::memory_order_relaxed);
    stats.totalDecodes = totalDecodes_.load(std::memory_order_relaxed);
    stats.totalTokensEncoded = totalTokensEncoded_.load(std::memory_order_relaxed);
    stats.totalTokensDecoded = totalTokensDecoded_.load(std::memory_order_relaxed);
    stats.cacheHits = cacheHits_.load(std::memory_order_relaxed);
    stats.cacheMisses = cacheMisses_.load(std::memory_order_relaxed);
    stats.currentMemoryUsage = currentMemoryUsage_.load(std::memory_order_relaxed);
    stats.peakMemoryUsage = peakMemoryUsage_.load(std::memory_order_relaxed);
    
    // 时间信息
    stats.startTime = startTime_;
    stats.endTime = std::chrono::system_clock::now();
    double runtimeSeconds = stats.getRuntimeSeconds();
    
    // 计算延迟统计
    std::lock_guard<std::mutex> lock(latencyMutex_);
    
    if (!encodeLatencies_.empty()) {
        // 平均延迟
        double sumEncode = std::accumulate(encodeLatencies_.begin(), encodeLatencies_.end(), 0.0);
        stats.avgEncodeLatency = sumEncode / encodeLatencies_.size();
        
        // 百分位延迟
        std::vector<double> sortedEncode = encodeLatencies_;
        std::sort(sortedEncode.begin(), sortedEncode.end());
        stats.p50EncodeLatency = calculatePercentile(sortedEncode, 0.50);
        stats.p95EncodeLatency = calculatePercentile(sortedEncode, 0.95);
        stats.p99EncodeLatency = calculatePercentile(sortedEncode, 0.99);
    }
    
    if (!decodeLatencies_.empty()) {
        // 平均延迟
        double sumDecode = std::accumulate(decodeLatencies_.begin(), decodeLatencies_.end(), 0.0);
        stats.avgDecodeLatency = sumDecode / decodeLatencies_.size();
        
        // 百分位延迟
        std::vector<double> sortedDecode = decodeLatencies_;
        std::sort(sortedDecode.begin(), sortedDecode.end());
        stats.p50DecodeLatency = calculatePercentile(sortedDecode, 0.50);
        stats.p95DecodeLatency = calculatePercentile(sortedDecode, 0.95);
        stats.p99DecodeLatency = calculatePercentile(sortedDecode, 0.99);
    }
    
    // 计算吞吐量（tokens/s）
    if (runtimeSeconds > 0) {
        stats.encodeSpeed = stats.totalTokensEncoded / runtimeSeconds;
        stats.decodeSpeed = stats.totalTokensDecoded / runtimeSeconds;
    }
    
    return stats;
}

void PerformanceMonitor::reset() {
    totalEncodes_.store(0, std::memory_order_relaxed);
    totalDecodes_.store(0, std::memory_order_relaxed);
    totalTokensEncoded_.store(0, std::memory_order_relaxed);
    totalTokensDecoded_.store(0, std::memory_order_relaxed);
    cacheHits_.store(0, std::memory_order_relaxed);
    cacheMisses_.store(0, std::memory_order_relaxed);
    currentMemoryUsage_.store(0, std::memory_order_relaxed);
    peakMemoryUsage_.store(0, std::memory_order_relaxed);
    
    std::lock_guard<std::mutex> lock(latencyMutex_);
    encodeLatencies_.clear();
    decodeLatencies_.clear();
    
    startTime_ = std::chrono::system_clock::now();
    lastStatsTime_ = startTime_;
}

double PerformanceMonitor::calculatePercentile(const std::vector<double>& latencies, double percentile) {
    if (latencies.empty()) {
        return 0.0;
    }
    
    // 假设 latencies 已经排序
    size_t idx = static_cast<size_t>(latencies.size() * percentile);
    if (idx >= latencies.size()) {
        idx = latencies.size() - 1;
    }
    
    return latencies[idx];
}

} // namespace cllm
