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

// 层权重结构 (GPU)
struct LayerWeightsGPU {
    struct ggml_tensor* qProj = nullptr;
    struct ggml_tensor* kProj = nullptr;
    struct ggml_tensor* vProj = nullptr;
    struct ggml_tensor* oProj = nullptr;
    struct ggml_tensor* gateProj = nullptr;
    struct ggml_tensor* upProj = nullptr;
    struct ggml_tensor* downProj = nullptr;
    struct ggml_tensor* inputNorm = nullptr;
    struct ggml_tensor* postNorm = nullptr;
    struct ggml_tensor* qNorm = nullptr;
    struct ggml_tensor* kNorm = nullptr;
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
    int graphStage_ = 0;

    // 权重张量
    struct ggml_tensor* embedTokens_ = nullptr;
    struct ggml_tensor* finalNorm_ = nullptr;
    struct ggml_tensor* lmHead_ = nullptr;
    std::vector<LayerWeightsGPU> layerWeights_;

    // KV Cache 张量
    std::vector<struct ggml_tensor*> kCacheTensors_;
    std::vector<struct ggml_tensor*> vCacheTensors_;

    // RoPE 频率
    struct ggml_tensor* ropeFreqsCos_ = nullptr;
    struct ggml_tensor* ropeFreqsSin_ = nullptr;

    // 计算图输入/输出
    struct ggml_tensor* graphInputToken_ = nullptr;
    struct ggml_tensor* graphInputPosition_ = nullptr;
    struct ggml_tensor* graphOutput_ = nullptr;
    std::vector<struct ggml_tensor*> graphNormWeightedAllLayers_;

    // 内部方法
    bool initBackend();
    bool initWeights(const float* embedTokens,
                     const std::vector<LayerWeightsGPU>& layers,
                     const float* finalNorm,
                     const float* lmHead);
    bool initKVCache();
    bool initRoPE();
    void cleanup();
};

} // namespace backend
} // namespace kylin
} // namespace cllm
