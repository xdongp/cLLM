#pragma once

#include <atomic>
#include <chrono>
#include <vector>
#include <mutex>
#include <algorithm>

namespace cllm {

/**
 * @brief 性能统计数据
 */
struct TokenizerPerformanceStats {
    // 基础统计
    size_t totalEncodes{0};           ///< 总编码次数
    size_t totalDecodes{0};           ///< 总解码次数
    size_t totalTokensEncoded{0};     ///< 总编码 token 数
    size_t totalTokensDecoded{0};     ///< 总解码 token 数
    
    // 延迟统计（毫秒）
    double avgEncodeLatency{0.0};     ///< 平均编码延迟
    double avgDecodeLatency{0.0};     ///< 平均解码延迟
    double p50EncodeLatency{0.0};     ///< 编码延迟 P50
    double p95EncodeLatency{0.0};     ///< 编码延迟 P95
    double p99EncodeLatency{0.0};     ///< 编码延迟 P99
    double p50DecodeLatency{0.0};     ///< 解码延迟 P50
    double p95DecodeLatency{0.0};     ///< 解码延迟 P95
    double p99DecodeLatency{0.0};     ///< 解码延迟 P99
    
    // 吞吐量统计
    double encodeSpeed{0.0};          ///< 编码速度 (tokens/s)
    double decodeSpeed{0.0};          ///< 解码速度 (tokens/s)
    
    // 缓存统计（如果实现了缓存）
    size_t cacheHits{0};              ///< 缓存命中次数
    size_t cacheMisses{0};            ///< 缓存未命中次数
    
    // 内存统计（字节）
    size_t currentMemoryUsage{0};     ///< 当前内存使用量
    size_t peakMemoryUsage{0};        ///< 峰值内存使用量
    
    // 时间范围
    std::chrono::system_clock::time_point startTime;  ///< 统计开始时间
    std::chrono::system_clock::time_point endTime;    ///< 统计结束时间
    
    /**
     * @brief 计算缓存命中率
     */
    double getCacheHitRate() const {
        size_t total = cacheHits + cacheMisses;
        return total > 0 ? static_cast<double>(cacheHits) / total : 0.0;
    }
    
    /**
     * @brief 计算运行时长（秒）
     */
    double getRuntimeSeconds() const {
        return std::chrono::duration<double>(endTime - startTime).count();
    }
};

/**
 * @brief 性能监控器接口
 * 
 * 提供编码/解码性能监控功能，支持延迟分布统计和吞吐量计算。
 */
class IPerformanceMonitor {
public:
    virtual ~IPerformanceMonitor() = default;
    
    /**
     * @brief 记录编码操作
     * @param durationMs 编码耗时（毫秒）
     * @param tokenCount 编码的 token 数量
     */
    virtual void recordEncode(double durationMs, size_t tokenCount) = 0;
    
    /**
     * @brief 记录解码操作
     * @param durationMs 解码耗时（毫秒）
     * @param tokenCount 解码的 token 数量
     */
    virtual void recordDecode(double durationMs, size_t tokenCount) = 0;
    
    /**
     * @brief 记录缓存命中
     */
    virtual void recordCacheHit() = 0;
    
    /**
     * @brief 记录缓存未命中
     */
    virtual void recordCacheMiss() = 0;
    
    /**
     * @brief 更新内存使用量
     * @param bytes 当前内存使用字节数
     */
    virtual void updateMemoryUsage(size_t bytes) = 0;
    
    /**
     * @brief 获取性能统计数据
     */
    virtual TokenizerPerformanceStats getStats() const = 0;
    
    /**
     * @brief 重置统计数据
     */
    virtual void reset() = 0;
};

/**
 * @brief 性能监控器实现
 * 
 * 线程安全的性能监控器，支持百分位延迟统计。
 */
class PerformanceMonitor : public IPerformanceMonitor {
public:
    PerformanceMonitor();
    ~PerformanceMonitor() override = default;
    
    void recordEncode(double durationMs, size_t tokenCount) override;
    void recordDecode(double durationMs, size_t tokenCount) override;
    void recordCacheHit() override;
    void recordCacheMiss() override;
    void updateMemoryUsage(size_t bytes) override;
    
    TokenizerPerformanceStats getStats() const override;
    void reset() override;
    
private:
    // 原子计数器（用于无锁操作）
    std::atomic<size_t> totalEncodes_{0};
    std::atomic<size_t> totalDecodes_{0};
    std::atomic<size_t> totalTokensEncoded_{0};
    std::atomic<size_t> totalTokensDecoded_{0};
    std::atomic<size_t> cacheHits_{0};
    std::atomic<size_t> cacheMisses_{0};
    std::atomic<size_t> currentMemoryUsage_{0};
    std::atomic<size_t> peakMemoryUsage_{0};
    
    // 延迟历史（需要互斥锁保护）
    mutable std::mutex latencyMutex_;
    std::vector<double> encodeLatencies_;
    std::vector<double> decodeLatencies_;
    
    // 时间追踪
    std::chrono::system_clock::time_point startTime_;
    mutable std::chrono::system_clock::time_point lastStatsTime_;
    
    // 配置
    static constexpr size_t MAX_LATENCY_SAMPLES = 10000;  ///< 最大延迟样本数
    
    /**
     * @brief 计算百分位延迟
     */
    static double calculatePercentile(const std::vector<double>& latencies, double percentile);
};

/**
 * @brief 性能监控 RAII 辅助类
 * 
 * 用于自动记录操作耗时的辅助类。
 * 
 * 用法：
 * ```cpp
 * {
 *     PerformanceTimer timer(monitor, PerformanceTimer::Operation::Encode, tokenCount);
 *     // 执行编码操作...
 * } // 析构时自动记录
 * ```
 */
class PerformanceTimer {
public:
    enum class Operation {
        Encode,
        Decode
    };
    
    PerformanceTimer(IPerformanceMonitor* monitor, Operation op, size_t tokenCount)
        : monitor_(monitor)
        , operation_(op)
        , tokenCount_(tokenCount)
        , startTime_(std::chrono::high_resolution_clock::now()) {}
    
    ~PerformanceTimer() {
        if (monitor_) {
            auto endTime = std::chrono::high_resolution_clock::now();
            double durationMs = std::chrono::duration<double, std::milli>(endTime - startTime_).count();
            
            if (operation_ == Operation::Encode) {
                monitor_->recordEncode(durationMs, tokenCount_);
            } else {
                monitor_->recordDecode(durationMs, tokenCount_);
            }
        }
    }
    
    // 禁止拷贝
    PerformanceTimer(const PerformanceTimer&) = delete;
    PerformanceTimer& operator=(const PerformanceTimer&) = delete;
    
private:
    IPerformanceMonitor* monitor_;
    Operation operation_;
    size_t tokenCount_;
    std::chrono::high_resolution_clock::time_point startTime_;
};

} // namespace cllm
