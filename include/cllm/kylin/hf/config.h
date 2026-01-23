/**
 * @file hf_config.h
 * @brief HuggingFace 模型配置解析
 */

#pragma once

#include <string>

namespace cllm {
namespace kylin {

/**
 * @brief HuggingFace 模型配置
 * 
 * 对应 config.json 中的字段
 */
struct HFModelConfig {
    // 基本信息
    std::string architecture;      // 如 "Qwen3ForCausalLM"
    std::string modelType;         // 如 "qwen3"
    std::string torchDtype;        // 如 "bfloat16"
    
    // 模型结构
    int hiddenSize = 0;            // 隐藏层维度
    int numHiddenLayers = 0;       // Transformer 层数
    int numAttentionHeads = 0;     // 注意力头数
    int numKeyValueHeads = 0;      // KV 头数 (GQA)
    int intermediateSize = 0;      // FFN 中间层维度
    int vocabSize = 0;             // 词表大小
    int headDim = 0;               // 每头维度
    int maxPositionEmbeddings = 0; // 最大位置
    
    // 归一化参数
    float rmsNormEps = 1e-6f;
    
    // RoPE 参数
    float ropeTheta = 10000.0f;
    
    // 特殊配置
    bool tieWordEmbeddings = false;
    bool attentionBias = false;
    std::string hiddenAct = "silu";
    
    // Token IDs
    int bosTokenId = 0;
    int eosTokenId = 0;
    int padTokenId = -1;
    
    /**
     * @brief 检查配置是否有效
     */
    bool isValid() const {
        return hiddenSize > 0 && 
               numHiddenLayers > 0 && 
               numAttentionHeads > 0 &&
               vocabSize > 0;
    }
    
    /**
     * @brief 获取每个注意力头的维度
     */
    int getHeadDim() const {
        if (headDim > 0) return headDim;
        return numAttentionHeads > 0 ? hiddenSize / numAttentionHeads : 0;
    }
    
    /**
     * @brief 获取 KV 头数（如果未指定则等于 Q 头数）
     */
    int getNumKVHeads() const {
        return numKeyValueHeads > 0 ? numKeyValueHeads : numAttentionHeads;
    }
    
    /**
     * @brief 打印配置信息
     */
    void print() const;
};

/**
 * @brief 从 config.json 加载配置
 * 
 * @param configPath config.json 文件路径
 * @return 模型配置
 */
HFModelConfig loadHFConfig(const std::string& configPath);

/**
 * @brief 从模型目录加载配置
 * 
 * @param modelDir 模型目录路径
 * @return 模型配置
 */
HFModelConfig loadHFConfigFromDir(const std::string& modelDir);

} // namespace kylin
} // namespace cllm
