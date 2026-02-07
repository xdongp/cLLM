/**
 * @file transformer_model.h
 * @brief Transformer 模型统一接口
 * 
 * 为了保持兼容性，此类扩展了 HFTransformerModel 的功能
 * 添加了 setEmbeddingWeight, setLmHeadWeight, setBlockWeights 等方法
 */

#pragma once

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/core/tensor.h"
#include "cllm/model/config.h"

namespace cllm {
namespace kylin {

/**
 * @brief Transformer 模型类
 * 
 * 继承自 HFTransformerModel，添加了权重设置接口
 * 用于支持 KylinBackend 的动态权重绑定
 */
class TransformerModel : public HFTransformerModel {
public:
    /**
     * @brief 默认构造函数
     */
    TransformerModel();
    
    /**
     * @brief 构造函数
     * @param config 模型配置
     */
    explicit TransformerModel(const ModelConfig& config);
    
    /**
     * @brief 从模型目录加载（委托给父类）
     * @param modelDir 模型目录
     * @param device 设备类型
     * @param quantType 量化类型
     */
    TransformerModel(const std::string& modelDir, 
                     DeviceType device = DeviceType::CPU,
                     QuantType quantType = QuantType::FP32);
    
    ~TransformerModel() = default;
    
    // 允许移动
    TransformerModel(TransformerModel&&) = default;
    TransformerModel& operator=(TransformerModel&&) = default;
    
    // 禁止拷贝
    TransformerModel(const TransformerModel&) = delete;
    TransformerModel& operator=(const TransformerModel&) = delete;
    
    // ========== 权重设置接口 ==========
    
    /**
     * @brief 设置 Embedding 权重
     * @param embedding 权重张量 [vocabSize, hiddenSize]
     */
    void setEmbeddingWeight(const Tensor& embedding);
    
    /**
     * @brief 设置 LM Head 权重
     * @param lmHead 权重张量 [hiddenSize, vocabSize]
     */
    void setLmHeadWeight(const Tensor& lmHead);
    
    /**
     * @brief 设置 Final Norm 权重
     * @param weight 权重张量 [hiddenSize]
     */
    void setFinalNormWeight(const Tensor& weight);
    
    /**
     * @brief 设置 Transformer Block 权重
     * @param layerIndex 层索引
     * @param wq Q 投影权重
     * @param wk K 投影权重
     * @param wv V 投影权重
     * @param wo O 投影权重
     * @param wGate Gate 投影权重
     * @param wUp Up 投影权重
     * @param wDown Down 投影权重
     * @param norm1 第一层归一化权重
     * @param norm2 第二层归一化权重
     * @param qNorm Q 归一化权重（可选）
     * @param kNorm K 归一化权重（可选）
     */
    void setBlockWeights(
        size_t layerIndex,
        const Tensor& wq,
        const Tensor& wk,
        const Tensor& wv,
        const Tensor& wo,
        const Tensor& wGate,
        const Tensor& wUp,
        const Tensor& wDown,
        const Tensor& norm1,
        const Tensor& norm2,
        const Tensor& qNorm = Tensor(),
        const Tensor& kNorm = Tensor()
    );
    
    // ========== 前向推理接口 ==========
    
    /**
     * @brief 前向推理（返回 Tensor 类型）
     * @param inputIds 输入 token IDs
     * @return 输出 logits 张量
     */
    Tensor forward(const std::vector<int>& inputIds);
    
    /**
     * @brief 前向推理（单 token）
     * @param tokenId 输入 token ID
     * @param position 位置
     * @return 输出 logits 张量
     */
    Tensor forward(int tokenId, int position);
};

} // namespace kylin
} // namespace cllm
