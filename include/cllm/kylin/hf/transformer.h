/**
 * @file hf_transformer.h
 * @brief HuggingFace 格式 Transformer 模型 - 100%后端化版本
 * 
 * 完全使用 IComputeBackend 接口，模型类只负责协调
 * CPU/GPU 计算逻辑完全分离到各自的后端实现
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include "cllm/kylin/hf/safetensors_loader.h"
#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/kylin/core/quantization.h"

#include <memory>
#include <vector>
#include <string>

namespace cllm {
namespace kylin {

/**
 * @brief HuggingFace Transformer 模型
 * 
 * 使用 IComputeBackend 接口进行计算
 * 支持 CPU 和 GPU（Metal/CUDA）后端
 */
class HFTransformerModel {
public:
    /**
     * @brief 从模型目录加载
     * 
     * @param modelDir 包含 config.json 和 model.safetensors 的目录
     * @param device 计算设备类型（默认 CPU）
     * @param quantType 权重量化类型（默认 FP32）
     */
    explicit HFTransformerModel(const std::string& modelDir, 
                                DeviceType device = DeviceType::CPU,
                                QuantType quantType = QuantType::FP32);
    ~HFTransformerModel();
    
    /**
     * @brief 获取当前量化类型
     */
    QuantType getQuantType() const { return quantType_; }
    
    /**
     * @brief 检查模型是否加载成功
     */
    bool isLoaded() const { return loaded_; }
    
    /**
     * @brief 前向推理（使用全局 KV Cache，兼容旧接口）
     * 
     * @param inputIds 输入 token IDs
     * @return logits [vocab_size]
     */
    std::vector<float> forward(const std::vector<int32_t>& inputIds);
    
    /**
     * @brief 前向推理（使用 per-request KV Cache，支持并发）
     * 
     * @param inputIds 输入 token IDs
     * @param requestId 请求 ID（用于查找 KV Cache）
     * @return logits [vocab_size]
     */
    std::vector<float> forwardWithRequestId(const std::vector<int32_t>& inputIds, size_t requestId);
    
    /**
     * @brief 批量前向推理（真正并发）
     * 
     * @param batchInputIds 每个请求的输入 token IDs
     * @param requestIds 每个请求的 ID
     * @return 每个请求的 logits
     */
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<size_t>& requestIds
    );
    
    /**
     * @brief 释放请求的 KV Cache
     * 
     * @param requestId 请求 ID
     */
    void releaseKVCache(size_t requestId);
    
    /**
     * @brief 获取模型配置
     */
    const HFModelConfig& config() const { return config_; }
    
    /**
     * @brief 获取词表大小
     */
    int vocabSize() const { return config_.vocabSize; }
    
    /**
     * @brief 获取指定请求的 KV Cache 当前长度
     * 
     * @param requestId 请求 ID
     * @return 当前 KV Cache 长度，如果请求不存在则返回 0
     */
    int getKVCacheCurrentLength(size_t requestId) const;
    
    /**
     * @brief 获取隐藏层维度
     */
    int hiddenSize() const { return config_.hiddenSize; }
    
    /**
     * @brief 获取后端名称
     */
    std::string getBackendName() const { 
        return backend_ ? backend_->getName() : "None"; 
    }
    
    /**
     * @brief 检查是否使用 GPU
     */
    bool isUsingGPU() const { return backend_ ? backend_->isGPU() : false; }

private:
    // 加载权重
    bool loadWeights();
    
    // FP32 模式加载权重（直接使用内存映射）
    bool loadWeightsFP32();

    // 量化模式加载权重（INT8/FP16）
    bool loadWeightsQuantized(QuantType qType);
    
    // 模型状态
    bool loaded_ = false;
    HFModelConfig config_;
    std::unique_ptr<SafetensorsLoader> loader_;
    DeviceType deviceType_ = DeviceType::CPU;
    QuantType quantType_ = QuantType::FP32;
    
    // 统一后端接口
    std::unique_ptr<IComputeBackend> backend_;
    
    // 模型权重（传递给后端）
    ModelWeights modelWeights_;
    
    // 量化后的权重存储（拥有权重的实际数据）
    // 仅当 quantType_ 为 INT8 或 FP16 时使用
    QuantizedWeight quantizedEmbedTokens_;
    QuantizedWeight quantizedLmHeadWeight_;
    QuantizedWeight quantizedFinalNormWeight_;
    std::vector<QuantizedWeight> quantizedLayerWeights_;  // 每层的量化权重 (7个线性层)
    std::vector<std::vector<float>> f32LayerWeights_;     // 每层的F32权重 (4个LayerNorm)
    std::vector<float> f32FinalNormWeight_;               // final norm的F32权重
};

} // namespace kylin
} // namespace cllm
