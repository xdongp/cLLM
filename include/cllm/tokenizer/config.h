/**
 * @file config.h
 * @brief Tokenizer配置类
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_TOKENIZER_CONFIG_H
#define CLLM_TOKENIZER_CONFIG_H

#include <vector>
#include <string>

namespace cllm {

/**
 * @brief Tokenizer配置类
 * 
 * 包含生成相关的配置参数。
 */
class TokenizerConfig {
public:
    /**
     * @brief 默认构造函数
     */
    TokenizerConfig();
    
    /**
     * @brief 析构函数
     */
    ~TokenizerConfig();
    
    void setMaxTokens(int maxTokens);  ///< 设置最大token数
    void setTemperature(float temperature);  ///< 设置温度参数
    void setTopP(float topP);  ///< 设置Top-P参数
    void setTopK(int topK);  ///< 设置Top-K参数
    void setStopTokens(const std::vector<int>& stopTokens);  ///< 设置停止tokens
    void setRepeatPenalty(float repeatPenalty);  ///< 设置重复惩罚
    
    int getMaxTokens() const;  ///< 获取最大token数
    float getTemperature() const;  ///< 获取温度参数
    float getTopP() const;  ///< 获取Top-P参数
    int getTopK() const;  ///< 获取Top-K参数
    std::vector<int> getStopTokens() const;  ///< 获取停止tokens
    float getRepeatPenalty() const;  ///< 获取重复惩罚
    
private:
    int maxTokens_;              ///< 最大token数
    float temperature_;          ///< 温度参数
    float topP_;                 ///< Top-P参数
    int topK_;                   ///< Top-K参数
    std::vector<int> stopTokens_; ///< 停止tokens
    float repeatPenalty_;        ///< 重复惩罚
};

}

#endif
