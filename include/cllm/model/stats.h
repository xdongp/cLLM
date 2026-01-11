/**
 * @file stats.h
 * @brief 模型统计信息结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_MODEL_STATS_H
#define CLLM_MODEL_STATS_H

#include <string>
#include <atomic>
#include <cstddef>

namespace cllm {

/**
 * @brief 模型统计信息结构
 * 
 * 记录模型推理的统计信息，所有字段都是线程安全的原子类型。
 */
struct ModelStats {
    std::atomic<size_t> totalInferences;        ///< 总推理次数
    std::atomic<size_t> totalTokens;            ///< 总tokens数
    std::atomic<float> averageInferenceTime;    ///< 平均推理时间
    std::atomic<float> averageTokensPerSecond;  ///< 平均每秒tokens数
    std::atomic<size_t> peakMemoryUsage;        ///< 峰值内存使用
    std::atomic<float> cacheHitRate;            ///< 缓存命中率
    
    /**
     * @brief 默认构造函数
     */
    ModelStats()
        : totalInferences(0)
        , totalTokens(0)
        , averageInferenceTime(0.0f)
        , averageTokensPerSecond(0.0f)
        , peakMemoryUsage(0)
        , cacheHitRate(0.0f) {}
    
    /**
     * @brief 拷贝构造函数
     * @param other 源对象
     */
    ModelStats(const ModelStats& other)
        : totalInferences(other.totalInferences.load())
        , totalTokens(other.totalTokens.load())
        , averageInferenceTime(other.averageInferenceTime.load())
        , averageTokensPerSecond(other.averageTokensPerSecond.load())
        , peakMemoryUsage(other.peakMemoryUsage.load())
        , cacheHitRate(other.cacheHitRate.load()) {}
    
    /**
     * @brief 拷贝赋值运算符
     * @param other 源对象
     * @return 当前对象引用
     */
    ModelStats& operator=(const ModelStats& other) {
        if (this != &other) {
            totalInferences.store(other.totalInferences.load());
            totalTokens.store(other.totalTokens.load());
            averageInferenceTime.store(other.averageInferenceTime.load());
            averageTokensPerSecond.store(other.averageTokensPerSecond.load());
            peakMemoryUsage.store(other.peakMemoryUsage.load());
            cacheHitRate.store(other.cacheHitRate.load());
        }
        return *this;
    }
    
    /**
     * @brief 更新统计信息
     * @param inferenceTime 推理时间
     * @param numTokens Token数量
     */
    void update(float inferenceTime, size_t numTokens) {
        size_t currentInferences = totalInferences.fetch_add(1, std::memory_order_relaxed) + 1;
        size_t currentTokens = totalTokens.fetch_add(numTokens, std::memory_order_relaxed) + numTokens;
        
        float currentAvgTime = averageInferenceTime.load(std::memory_order_relaxed);
        float newAvgTime = (currentAvgTime * (currentInferences - 1) + inferenceTime) / currentInferences;
        averageInferenceTime.store(newAvgTime, std::memory_order_relaxed);
        
        float currentAvgTPS = averageTokensPerSecond.load(std::memory_order_relaxed);
        float newAvgTPS = (currentAvgTPS * (currentInferences - 1) + (numTokens / inferenceTime)) / currentInferences;
        averageTokensPerSecond.store(newAvgTPS, std::memory_order_relaxed);
    }
    
    /**
     * @brief 重置所有统计信息
     */
    void reset() {
        totalInferences.store(0, std::memory_order_relaxed);
        totalTokens.store(0, std::memory_order_relaxed);
        averageInferenceTime.store(0.0f, std::memory_order_relaxed);
        averageTokensPerSecond.store(0.0f, std::memory_order_relaxed);
        peakMemoryUsage.store(0, std::memory_order_relaxed);
        cacheHitRate.store(0.0f, std::memory_order_relaxed);
    }
    
    /**
     * @brief 转换为字符串表示
     * @return 统计信息的字符串表示
     */
    std::string toString() const {
        return "ModelStats{totalInferences=" + std::to_string(totalInferences.load()) +
               ", totalTokens=" + std::to_string(totalTokens.load()) +
               ", averageInferenceTime=" + std::to_string(averageInferenceTime.load()) +
               ", averageTokensPerSecond=" + std::to_string(averageTokensPerSecond.load()) +
               ", peakMemoryUsage=" + std::to_string(peakMemoryUsage.load()) +
               ", cacheHitRate=" + std::to_string(cacheHitRate.load()) + "}";
    }
};

}

#endif
