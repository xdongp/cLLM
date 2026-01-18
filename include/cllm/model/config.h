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
    size_t vocabSize;            ///< 词表大小（模型的 vocab_size）
    size_t tokenizerVocabSize;   ///< Tokenizer 的词表大小（用于采样时限制 logits 范围）
    size_t hiddenSize;           ///< 隐藏层大小
    size_t numLayers;            ///< 层数
    size_t numAttentionHeads;    ///< 注意力头数 (Query heads)
    size_t numKeyValueHeads;     ///< KV 注意力头数（GQA 支持，默认等于 numAttentionHeads）
    size_t maxSequenceLength;    ///< 最大序列长度
    size_t intermediateSize;     ///< 中间层大小
    float ropeTheta;              ///< RoPE theta 参数（rope.freq_base，Qwen3常见1000000）
    float rmsNormEps;             ///< RMSNorm epsilon（Qwen3常见1e-6）
    
    // P3修复：RoPE扩展参数（对齐llama.cpp的ggml_rope_ext）
    size_t ropeNctxOrig;         ///< 原始上下文长度（n_ctx_orig），默认等于maxSequenceLength
    float ropeFreqScale;         ///< 频率缩放因子（freq_scale），默认1.0
    int ropeType;                 ///< RoPE类型（rope_type），0=标准RoPE，默认0
    float ropeExtFactor;          ///< 扩展因子（ext_factor），默认1.0
    
    bool useKVCache;             ///< 是否使用KV缓存
    bool useQuantization;        ///< 是否使用量化
    bool useMemoryCompression;   ///< 是否使用内存压缩
    std::string quantizationType;///< 量化类型

    // llama.cpp 后端参数（仅 GGUF 时使用）
    size_t llamaBatchSize;       ///< llama.cpp n_batch
    int llamaNumThreads;         ///< llama.cpp n_threads
    int llamaGpuLayers;          ///< llama.cpp n_gpu_layers
    bool llamaUseMmap;           ///< llama.cpp use_mmap
    bool llamaUseMlock;          ///< llama.cpp use_mlock
    
    /**
     * @brief 默认构造函数
     */
    ModelConfig()
        : modelType("llama")
        , vocabSize(32000)
        , tokenizerVocabSize(32000)  // 默认等于 vocabSize，如果不同会在初始化时更新
        , hiddenSize(4096)
        , numLayers(32)
        , numAttentionHeads(32)
        , numKeyValueHeads(32)  // 默认等于 numAttentionHeads（MHA）
        , maxSequenceLength(2048)
        , intermediateSize(11008)
        , ropeTheta(10000.0f)    // 默认 RoPE theta（Qwen3通常为1000000）
        , rmsNormEps(1e-5f)      // 默认 RMSNorm eps（Qwen3通常为1e-6）
        , ropeNctxOrig(2048)     // P3修复：默认等于maxSequenceLength
        , ropeFreqScale(1.0f)    // P3修复：默认频率缩放因子
        , ropeType(0)            // P3修复：默认标准RoPE
        , ropeExtFactor(1.0f)    // P3修复：默认扩展因子
        , useKVCache(true)
        , useQuantization(false)
        , useMemoryCompression(false)
        , quantizationType("")
        , llamaBatchSize(512)
        , llamaNumThreads(0)
        , llamaGpuLayers(0)
        , llamaUseMmap(true)
        , llamaUseMlock(false) {}
    
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
