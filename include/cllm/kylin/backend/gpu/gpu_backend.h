/**
 * @file gpu_backend.h
 * @brief GPU 后端接口定义
 * 
 * 提供 GPU 推理的接口，支持 Metal 等硬件加速。
 * 实际实现委托给 hf/ggml_backend.h 中的 GGMLGPUBackend。
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include <vector>
#include <string>
#include <memory>

namespace cllm {
namespace kylin {

// 前向声明
class GGMLGPUBackend;

namespace backend {

// 层权重结构 (GPU) - 用于上传权重
struct LayerWeightsGPU {
    const float* qProj = nullptr;
    const float* kProj = nullptr;
    const float* vProj = nullptr;
    const float* oProj = nullptr;
    const float* gateProj = nullptr;
    const float* upProj = nullptr;
    const float* downProj = nullptr;
    const float* inputLayernorm = nullptr;
    const float* postAttentionLayernorm = nullptr;
    const float* qNorm = nullptr;
    const float* kNorm = nullptr;
};

/**
 * @brief GPU 后端类
 * 
 * 负责管理 GPU 推理的所有操作，包括：
 * - GPU 内存管理
 * - 计算图构建和执行
 * - 权重上传和管理
 * 
 * 实际实现委托给 GGMLGPUBackend。
 */
class GPUBackend {
public:
    GPUBackend();
    ~GPUBackend();

    /**
     * @brief 初始化 GPU 后端
     * @param config 模型配置
     * @return 是否初始化成功
     */
    bool initialize(const HFModelConfig& config);

    /**
     * @brief 上传权重到 GPU
     * @param embedTokens Embedding 权重
     * @param layers 层权重数组
     * @param finalNorm 最终归一化权重
     * @param lmHead LM Head 权重
     * @return 是否上传成功
     */
    bool uploadWeights(
        const float* embedTokens,
        const std::vector<LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );

    /**
     * @brief GPU 前向传播
     * @param tokenId 输入 token ID
     * @param position 位置信息
     * @return 输出 logits
     */
    std::vector<float> forward(int tokenId, int position);

    /**
     * @brief 使用计算图的前向传播
     * @param tokenId 输入 token ID
     * @param position 位置信息
     * @return 输出 logits
     */
    std::vector<float> forwardGraph(int tokenId, int position);

    /**
     * @brief 使用 CPU 的前向传播（回退实现）
     * @param tokenId 输入 token ID
     * @param position 位置信息
     * @return 输出 logits
     */
    std::vector<float> forwardCPU(int tokenId, int position);

    /**
     * @brief 构建计算图
     * @return 是否构建成功
     */
    bool buildGraph();

    /**
     * @brief 重置 KV Cache
     */
    void resetKVCache();

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief 检查是否支持 GPU
     */
    bool isGPUSupported() const;

    /**
     * @brief 批量前向推理
     */
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<int>& tokenIds,
        const std::vector<int>& positions,
        const std::vector<size_t>& requestIds
    );

private:
    // 实际实现（Pimpl 模式）
    std::unique_ptr<GGMLGPUBackend> impl_;
    bool initialized_ = false;
};

} // namespace backend
} // namespace kylin
} // namespace cllm
