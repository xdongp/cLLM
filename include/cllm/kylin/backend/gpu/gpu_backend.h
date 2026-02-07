/**
 * @file gpu_backend.h
 * @brief GPU 后端接口定义
 * 
 * 提供 GPU 推理的接口，支持 Metal 等硬件加速。
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include <vector>
#include <string>
#include <memory>

// 前向声明 GGML 类型
struct ggml_context;
struct ggml_backend;
struct ggml_backend_buffer;
struct ggml_cgraph;
struct ggml_tensor;
struct ggml_backend_sched;

namespace cllm {
namespace kylin {
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

// 层张量结构 - 用于 GGML 张量存储
struct LayerTensors {
    struct ggml_tensor* inputLayernorm = nullptr;
    struct ggml_tensor* qProj = nullptr;
    struct ggml_tensor* kProj = nullptr;
    struct ggml_tensor* vProj = nullptr;
    struct ggml_tensor* oProj = nullptr;
    struct ggml_tensor* qNorm = nullptr;
    struct ggml_tensor* kNorm = nullptr;
    struct ggml_tensor* postAttentionLayernorm = nullptr;
    struct ggml_tensor* gateProj = nullptr;
    struct ggml_tensor* upProj = nullptr;
    struct ggml_tensor* downProj = nullptr;
};

/**
 * @brief GPU 后端类
 * 
 * 负责管理 GPU 推理的所有操作，包括：
 * - GPU 内存管理
 * - 计算图构建和执行
 * - 权重上传和管理
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
    // 模型配置
    HFModelConfig config_;
    bool initialized_ = false;
    int graphStage_ = 5;
    int kvCacheLen_ = 0;

    // GGML 上下文和后台
    struct ggml_context* weightCtx_ = nullptr;
    struct ggml_context* computeCtx_ = nullptr;
    struct ggml_backend* backend_ = nullptr;
    struct ggml_backend* backendCPU_ = nullptr;
    struct ggml_backend_buffer* weightBuffer_ = nullptr;
    struct ggml_backend_buffer* computeBuffer_ = nullptr;

    // 计算图
    struct ggml_context* graphCtx_ = nullptr;
    struct ggml_backend_buffer* graphBuffer_ = nullptr;
    struct ggml_backend_sched* graphSched_ = nullptr;
    struct ggml_cgraph* graph_ = nullptr;

    // 权重张量
    struct ggml_tensor* embedTokens_ = nullptr;
    struct ggml_tensor* embedTokensLookup_ = nullptr;
    struct ggml_tensor* finalNorm_ = nullptr;
    struct ggml_tensor* lmHead_ = nullptr;
    std::vector<LayerTensors> layers_;

    // CPU KV Cache（用于调试和数据传输）
    std::vector<std::vector<float>> kCacheCPU_;
    std::vector<std::vector<float>> vCacheCPU_;

    // RoPE 频率
    std::vector<float> ropeFreqsCos_;
    std::vector<float> ropeFreqsSin_;

    // 计算图输入/输出
    struct ggml_tensor* graphInputToken_ = nullptr;
    struct ggml_tensor* graphInputPosition_ = nullptr;
    struct ggml_tensor* graphOutput_ = nullptr;
    std::vector<struct ggml_tensor*> graphNormWeightedAllLayers_;

    // 内部方法
    bool createWeightTensors();
    void precomputeRoPE();
    void cleanup();
};

} // namespace backend
} // namespace kylin
} // namespace cllm
