/**
 * @file gpu_backend.h
 * @brief GPU 计算后端
 * 
 * 实现 IComputeBackend 接口的 GPU 版本
 * 使用 GPU（Metal/CUDA）进行 Transformer 模型的前向推理
 */

#pragma once

#include "cllm/kylin/backend/backend_interface.h"
#include <vector>
#include <memory>

namespace cllm {
namespace kylin {

// 前向声明
class GGMLGPUBackend;

/**
 * @brief GPU 计算后端
 * 
 * 使用 GPU（Metal/CUDA）进行 Transformer 模型的前向推理
 * 封装现有的 GGMLGPUBackend
 */
class GPUBackend : public IComputeBackend {
public:
    GPUBackend();
    ~GPUBackend() override;
    
    // IComputeBackend 接口实现
    bool initialize(const HFModelConfig& config) override;
    void shutdown() override;
    bool loadWeights(const ModelWeights& weights) override;
    std::vector<float> forward(const std::vector<int32_t>& inputIds, int requestId) override;
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) override;
    void resetKVCache(int requestId) override;
    void releaseKVCache(int requestId) override;
    std::string getName() const override { return "GPU"; }
    bool isGPU() const override { return true; }
    int getKVCacheCurrentLength(int requestId) const override;

private:
    // 模型配置
    HFModelConfig config_;
    
    // 权重数据
    ModelWeights weights_;
    bool weightsLoaded_ = false;
    
    // 初始化标志
    bool initialized_ = false;
    
    // 封装的 GGML GPU 后端
    std::unique_ptr<GGMLGPUBackend> ggmlBackend_;
};

} // namespace kylin
} // namespace cllm
