/**
 * @file cpu_backend.h
 * @brief CPU 计算后端
 * 
 * 实现 IComputeBackend 接口的 CPU 版本
 * 使用 CPU 进行 Transformer 模型的前向推理
 */

#pragma once

#include "cllm/kylin/backend/backend_interface.h"
#include <vector>
#include <memory>
#include <unordered_map>

namespace cllm {
namespace kylin {

// 前向声明内部实现
struct CPUBackendImpl;

/**
 * @brief CPU 计算后端
 * 
 * 使用 CPU 进行 Transformer 模型的前向推理
 * 支持 BF16/F32/FP16/INT8 权重
 */
class CPUBackend : public IComputeBackend {
public:
    CPUBackend();
    ~CPUBackend() override;
    
    // IComputeBackend 接口实现
    bool initialize(const HFModelConfig& config) override;
    void shutdown() override;
    bool loadWeights(const ModelWeights& weights) override;
    
    // 从 HFTransformerModel 的权重格式加载（辅助函数）
    bool loadWeightsFromHF(
        const std::vector<float>& embedTokens,
        const std::vector<float>& lmHeadWeight,
        const std::vector<float>& finalNormWeight,
        const std::vector<std::vector<float>>& layerWeights
    );
    
    std::vector<float> forward(const std::vector<int32_t>& inputIds, int requestId) override;
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) override;
    void resetKVCache(int requestId) override;
    void releaseKVCache(int requestId) override;
    std::string getName() const override { return "CPU"; }
    bool isGPU() const override { return false; }
    int getKVCacheCurrentLength(int requestId) const override;

private:
    // 模型配置
    HFModelConfig config_;
    
    // 权重数据
    ModelWeights weights_;
    bool weightsLoaded_ = false;
    
    // 初始化标志
    bool initialized_ = false;
    
    // PIMPL 模式隐藏实现细节
    std::unique_ptr<CPUBackendImpl> impl_;
};

} // namespace kylin
} // namespace cllm
