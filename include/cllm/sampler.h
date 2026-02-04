/**
 * @file sampler.h
 * @brief 采样器，负责从tok概率分布中采样生成token
 * @author cLLM Team
 * @date 2024-01-01
 */

#ifndef CLLM_SAMPLER_H
#define CLLM_SAMPLER_H

#include "cllm/memory/float_array.h"
#include "cllm/sampler/config.h"
#include "cllm/sampler/stats.h"
#include "cllm/common/config.h"
#include <vector>
#include <random>

namespace cllm {

/**
 * @brief 采样器类
 * 
 * 负责从模型输出的logits分布中采样生成下一个token。
 * 支持贪心采样、温度采样、Top-K采样、Top-P采样等策略。
 */
class Sampler {
public:
    /**
     * @brief 默认构造函数
     */
    Sampler();
    
    /**
     * @brief 带配置的构造函数
     * @param config 采样器配置
     */
    explicit Sampler(const SamplerConfig& config);
    
    /**
     * @brief 析构函数
     */
    ~Sampler();
    
    /**
     * @brief 从log its分布中采样token
     * @param logits 模型输出的logits分布
     * @param temperature 温度参数，默认1.0
     * @param topK Top-K参数，默认-1（不使用Top-K）
     * @param topP Top-P参数，默认-1.0（不使用Top-P）
     * @return 采样得到的token ID
     */
    int sample(const FloatArray& logits, float temperature = 1.0f, int topK = -1, float topP = -1.0f);
    
    /**
     * @brief 从logits分布中采样token（带重复惩罚）
     * @param logits 模型输出的logits分布
     * @param generatedTokens 已生成的token序列（用于重复惩罚）
     * @param temperature 温度参数
     * @param topK Top-K参数
     * @param topP Top-P参数
     * @param repetitionPenalty 重复惩罚参数（>1.0 惩罚重复，1.0 不惩罚）
     * @return 采样得到的token ID
     */
    int sampleWithPenalty(const FloatArray& logits, const std::vector<int>& generatedTokens,
                          float temperature, int topK, float topP, float repetitionPenalty);
    
    /**
     * @brief 批处理采样
     * @param logits 模型输出的logits分布，形状为(batchSize, vocabSize)
     * @param temperature 温度参数，默认1.0
     * @param topK Top-K参数，默认-1（不使用Top-K）
     * @param topP Top-P参数，默认-1.0（不使用Top-P）
     * @return 采样得到的token ID列表
     */
    std::vector<int> sampleBatch(const FloatArray& logits, int batchSize, float temperature = 1.0f, int topK = -1, float topP = -1.0f);
    
    /**
     * @brief 设置配置
     * @param config 新的配置
     */
    void setConfig(const SamplerConfig& config);
    
    /**
     * @brief 获取当前配置
     * @return 当前配置
     */
    SamplerConfig getConfig() const;
    
    /**
     * @brief 获取详细统计信息
     * @return 统计信息对象
     */
    SamplerStats getStats() const;
    
    /**
     * @brief 获取采样总次数
     * @return 采样总次数
     */
    size_t getSampleCount() const;
    
    /**
     * @brief 重置采样统计信息
     */
    void resetStats();
    
private:
    /**
     * @brief 贪心采样（选择概率最高的token）
     * @param logits logits分布
     * @return token ID
     */
    int sampleGreedy(const FloatArray& logits);
    
    /**
     * @brief 温度采样
     * @param logits logits分布
     * @param temperature 温度参数
     * @return token ID
     */
    int sampleTemperature(const FloatArray& logits, float temperature);
    
    /**
     * @brief Top-K采样
     * @param logits logits分布
     * @param k Top-K参数
     * @param temperature 温度参数，默认1.0
     * @return token ID
     */
    int sampleTopK(const FloatArray& logits, int k, float temperature = 1.0f);
    
    /**
     * @brief Top-P采样
     * @param logits logits分布
     * @param p Top-P参数
     * @param temperature 温度参数，默认1.0
     * @return token ID
     */
    int sampleTopP(const FloatArray& logits, float p, float temperature = 1.0f);
    
    /**
     * @brief 从单个logits分布采样
     * @param logits logits分布
     * @param vocabSize 词汇表大小
     * @param temperature 温度参数
     * @param topK Top-K参数
     * @param topP Top-P参数
     * @return token ID
     */
    int sampleSingle(const float* logits, size_t vocabSize, float temperature, int topK, float topP);
    
    std::mt19937 gen_;        ///< 随机数生成器
    SamplerConfig config_;    ///< 采样器配置
    SamplerStats stats_;      ///< 采样统计信息
    size_t sampleCount_;      ///< 采样总次数（向后兼容）
};

}

#endif
