/**
 * @file config.h
 * @brief 采样器配置类
 * @author cLLM Team
 * @date 2026-01-09
 */

#ifndef CLLM_SAMPLER_CONFIG_H
#define CLLM_SAMPLER_CONFIG_H

#include <string>

namespace cllm {

/**
 * @brief 采样器配置类
 * 
 * 用于配置采样器的各种参数，包括温度、Top-K、Top-P等。
 * 提供灵活的配置管理，支持预设配置和自定义配置。
 */
class SamplerConfig {
public:
    /**
     * @brief 默认构造函数
     * 
     * 初始化为默认配置：
     * - temperature = 1.0
     * - topK = -1 (disabled)
     * - topP = -1.0 (disabled)
     * - greedyThreshold = 0.9
     */
    SamplerConfig();
    
    /**
     * @brief 析构函数
     */
    ~SamplerConfig();
    
    /**
     * @brief 设置温度参数
     * @param temperature 温度值，范围[0, ∞)，较大的值使分布更平滑
     */
    void setTemperature(float temperature);
    
    /**
     * @brief 获取温度参数
     * @return 当前温度值
     */
    float getTemperature() const;
    
    /**
     * @brief 设置Top-K参数
     * @param topK Top-K值，-1表示不使用Top-K采样
     */
    void setTopK(int topK);
    
    /**
     * @brief 获取Top-K参数
     * @return 当前Top-K值
     */
    int getTopK() const;
    
    /**
     * @brief 设置Top-P参数
     * @param topP Top-P值，范围[0, 1]，-1表示不使用Top-P采样
     */
    void setTopP(float topP);
    
    /**
     * @brief 获取Top-P参数
     * @return 当前Top-P值
     */
    float getTopP() const;
    
    /**
     * @brief 设置贪心采样阈值
     * @param threshold 阈值，当温度低于此值时使用贪心采样
     */
    void setGreedyThreshold(float threshold);
    
    /**
     * @brief 获取贪心采样阈值
     * @return 当前阈值
     */
    float getGreedyThreshold() const;
    
    /**
     * @brief 加载预设配置
     * @param presetName 预设名称，如"greedy", "creative", "balanced"
     */
    void loadPreset(const std::string& presetName);
    
    /**
     * @brief 重置为默认配置
     */
    void reset();
    
private:
    float temperature_;      ///< 温度参数
    int topK_;              ///< Top-K参数
    float topP_;            ///< Top-P参数
    float greedyThreshold_; ///< 贪心采样阈值
};

}

#endif
