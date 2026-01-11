/**
 * @file config.h
 * @brief 模型配置结构
 * @author cLLM Team
 * @date 2024-01-09
 */

#ifndef CLLM_MODEL_CONFIG_H
#define CLLM_MODEL_CONFIG_H

#include <string>
#include <cstddef>

namespace cllm {

/**
 * @brief 模型配置结构
 * 
 * 包含模型的各种配置参数。
 */
struct ModelConfig {
    std::string modelType;       ///< 模型类型
    size_t vocabSize;            ///< 词表大小
    size_t hiddenSize;           ///< 隐藏层大小
    size_t numLayers;            ///< 层数
    size_t numAttentionHeads;    ///< 注意力头数 (Query heads)
    size_t numKeyValueHeads;     ///< KV 注意力头数（GQA 支持，默认等于 numAttentionHeads）
    size_t maxSequenceLength;    ///< 最大序列长度
    size_t intermediateSize;     ///< 中间层大小
    
    bool useKVCache;             ///< 是否使用KV缓存
    bool useQuantization;        ///< 是否使用量化
    bool useMemoryCompression;   ///< 是否使用内存压缩
    std::string quantizationType;///< 量化类型
    
    /**
     * @brief 默认构造函数
     */
    ModelConfig()
        : modelType("llama")
        , vocabSize(32000)
        , hiddenSize(4096)
        , numLayers(32)
        , numAttentionHeads(32)
        , numKeyValueHeads(32)  // 默认等于 numAttentionHeads（MHA）
        , maxSequenceLength(2048)
        , intermediateSize(11008)
        , useKVCache(true)
        , useQuantization(false)
        , useMemoryCompression(false)
        , quantizationType("") {}
    
    /**
     * @brief 从配置文件加载
     * @param configPath 配置文件路径
     */
    void loadFromConfigFile(const std::string& configPath);
    
    /**
     * @brief 转换为字符串表示
     * @return 配置的字符串表示
     */
    std::string toString() const;
};

}

#endif
