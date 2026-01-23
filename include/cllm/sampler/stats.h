/**
 * @file stats.h
 * @brief 采样器统计信息类
 * @author cLLM Team
 * @date 2026-01-09
 */

#ifndef CLLM_SAMPLER_STATS_H
#define CLLM_SAMPLER_STATS_H

#include <string>
#include <mutex>

namespace cllm {

/**
 * @brief 采样器统计信息类
 * 
 * 记录采样器的详细统计信息，包括各种采样策略的使用情况。
 * 线程安全，可在多线程环境中使用。
 */
class SamplerStats {
public:
    /**
     * @brief 默认构造函数
     * 
     * 初始化所有统计计数器为0
     */
    SamplerStats();
    
    /**
     * @brief 拷贝构造函数
     * @param other 要拷贝的对象
     */
    SamplerStats(const SamplerStats& other);
    
    /**
     * @brief 赋值运算符
     * @param other 要赋值的对象
     * @return 当前对象的引用
     */
    SamplerStats& operator=(const SamplerStats& other);
    
    /**
     * @brief 析构函数
     */
    ~SamplerStats();
    
    /**
     * @brief 增加总采样次数
     */
    void incrementTotalSamples();
    
    /**
     * @brief 增加贪心采样次数
     */
    void incrementGreedySamples();
    
    /**
     * @brief 增加Top-K采样次数
     */
    void incrementTopKSamples();
    
    /**
     * @brief 增加Top-P采样次数
     */
    void incrementTopPSamples();
    
    /**
     * @brief 增加温度采样次数
     */
    void incrementTemperatureSamples();
    
    /**
     * @brief 获取总采样次数
     * @return 总采样次数
     */
    long long getTotalSamples() const;
    
    /**
     * @brief 获取贪心采样次数
     * @return 贪心采样次数
     */
    long long getGreedySamples() const;
    
    /**
     * @brief 获取Top-K采样次数
     * @return Top-K采样次数
     */
    long long getTopKSamples() const;
    
    /**
     * @brief 获取Top-P采样次数
     * @return Top-P采样次数
     */
    long long getTopPSamples() const;
    
    /**
     * @brief 获取温度采样次数
     * @return 温度采样次数
     */
    long long getTemperatureSamples() const;
    
    /**
     * @brief 获取贪心采样百分比
     * @return 贪心采样百分比（0-100）
     */
    float getGreedyPercentage() const;
    
    /**
     * @brief 获取Top-K采样百分比
     * @return Top-K采样百分比（0-100）
     */
    float getTopKPercentage() const;
    
    /**
     * @brief 获取Top-P采样百分比
     * @return Top-P采样百分比（0-100）
     */
    float getTopPPercentage() const;
    
    /**
     * @brief 获取温度采样百分比
     * @return 温度采样百分比（0-100）
     */
    float getTemperaturePercentage() const;
    
    /**
     * @brief 重置所有统计信息
     */
    void reset();
    
    /**
     * @brief 转换为字符串表示
     * @return 统计信息的字符串描述
     */
    std::string toString() const;
    
private:
    long long totalSamples_{0};       ///< 总采样次数
    long long greedySamples_{0};      ///< 贪心采样次数
    long long topKSamples_{0};        ///< Top-K采样次数
    long long topPSamples_{0};        ///< Top-P采样次数
    long long temperatureSamples_{0}; ///< 温度采样次数
    
    mutable std::mutex mutex_;        ///< 互斥锁，用于线程安全
    
    /**
     * @brief 计算百分比
     * @param count 特定类型的计数
     * @return 百分比（0-100）
     */
    float calculatePercentage(long long count) const;
};

}

#endif
