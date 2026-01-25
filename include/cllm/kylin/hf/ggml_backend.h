/**
 * @file ggml_backend.h
 * @brief GGML GPU 后端封装（用于 HFTransformerModel）
 * 
 * 正确的 GPU 加速方式：
 * 1. 模型加载时：创建 GPU buffer 并上传权重
 * 2. 推理时：只更新输入，执行计算图，获取输出
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include <vector>
#include <memory>
#include <string>
#include <unordered_map>

// GGML 头文件
#include "ggml.h"
#include "ggml-backend.h"

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

namespace cllm {
namespace kylin {

/**
 * @brief GGML GPU 后端管理器
 * 
 * 管理 GPU 资源的生命周期，避免每次推理都重新分配
 */
class GGMLGPUBackend {
public:
    GGMLGPUBackend();
    ~GGMLGPUBackend();
    
    /**
     * @brief 初始化 GPU 后端
     * @param config 模型配置
     * @return 是否成功初始化
     */
    bool initialize(const HFModelConfig& config);
    
    /**
     * @brief 上传权重到 GPU
     * 
     * @param embedTokens 嵌入层权重 [vocabSize, hiddenSize]
     * @param layers 每层的权重
     * @param finalNorm 最终 norm 权重
     * @param lmHead LM Head 权重（可能与 embedTokens 共享）
     */
    bool uploadWeights(
        const float* embedTokens,
        const std::vector<struct LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );
    
    /**
     * @brief 在 GPU 上执行单层 matmul
     * 
     * @param layerIdx 层索引
     * @param projType 投影类型（q/k/v/o/gate/up/down）
     * @param input 输入向量
     * @param output 输出向量
     */
    void matmulGPU(int layerIdx, const char* projType,
                   const float* input, float* output,
                   int outFeatures, int inFeatures);
    
    /**
     * @brief 执行完整的 forward（单 token）
     * 
     * 在 GPU 上执行整个 forward pass，比单次 matmul 更高效
     */
    std::vector<float> forward(int tokenId, int position);

    /**
     * @brief 最小可用 GGML 计算图（Embedding + LM Head）
     *
     * 用于学习/验证 GGML 计算图执行流程
     */
    std::vector<float> forwardGraphMinimal(int tokenId, int position);
    
    /**
     * @brief 检查 GPU 是否可用
     */
    bool isAvailable() const { return backend_ != nullptr; }
    
    /**
     * @brief 获取后端名称
     */
    const char* getName() const;
    
private:
    // GGML 后端
    ggml_backend_t backend_ = nullptr;
    ggml_backend_t backendCPU_ = nullptr;
    ggml_backend_buffer_t weightBuffer_ = nullptr;
    ggml_backend_buffer_t computeBuffer_ = nullptr;
    
    // 权重上下文（静态，只在加载时分配）
    ggml_context* weightCtx_ = nullptr;
    
    // 计算上下文（每次 forward 重用）
    ggml_context* computeCtx_ = nullptr;

    // Stage 7: 持久化 GGML 计算图资源
    ggml_context* graphCtx_ = nullptr;
    ggml_backend_buffer_t graphBuffer_ = nullptr;
    ggml_backend_sched_t graphSched_ = nullptr;
    ggml_cgraph* graph_ = nullptr;
    ggml_tensor* graphToken_ = nullptr;
    ggml_tensor* graphPos_ = nullptr;
    ggml_tensor* graphLogits_ = nullptr;
    std::vector<ggml_tensor*> graphKCacheLayers_;
    std::vector<ggml_tensor*> graphVCacheLayers_;
    std::vector<ggml_tensor*> graphKCacheUpdLayers_;
    std::vector<ggml_tensor*> graphVCacheUpdLayers_;
    int graphBuiltPosition_ = -1;
    int graphBuiltStage_ = 0;
    
    // 权重张量（指向 GPU 内存）
    ggml_tensor* embedTokens_ = nullptr;
    ggml_tensor* finalNorm_ = nullptr;
    ggml_tensor* lmHead_ = nullptr;
    
    // 每层权重
    struct LayerTensors {
        ggml_tensor* inputLayernorm = nullptr;
        ggml_tensor* qProj = nullptr;
        ggml_tensor* kProj = nullptr;
        ggml_tensor* vProj = nullptr;
        ggml_tensor* oProj = nullptr;
        ggml_tensor* qNorm = nullptr;
        ggml_tensor* kNorm = nullptr;
        ggml_tensor* postAttentionLayernorm = nullptr;
        ggml_tensor* gateProj = nullptr;
        ggml_tensor* upProj = nullptr;
        ggml_tensor* downProj = nullptr;
    };
    std::vector<LayerTensors> layers_;
    
    // KV Cache（CPU 上，用于完整 Attention 实现）
    std::vector<std::vector<float>> kCacheCPU_;
    std::vector<std::vector<float>> vCacheCPU_;
    int kvCacheLen_ = 0;
    
    // RoPE 预计算（CPU）
    std::vector<float> ropeFreqsCos_;
    std::vector<float> ropeFreqsSin_;
    
    // CPU 权重缓存（避免每次从 GPU 获取）
    std::unordered_map<std::string, std::vector<float>> weightsCached_;

    // 学习版 GGML 计算图 KV Cache（多层）
    std::vector<std::vector<float>> kCacheGraphCPU_;
    std::vector<std::vector<float>> vCacheGraphCPU_;
    
    // 模型配置
    HFModelConfig config_;
    bool initialized_ = false;
    
    // 创建权重张量
    bool createWeightTensors();
    
    // 预计算 RoPE
    void precomputeRoPE();
    
    // 缓存权重到 CPU
    void cacheWeightsToCPU();

    // GGML 计算图阶段（0=关闭, 1=Embedding+LMHead, 2=+RMSNorm+FFN, 3=+Attention, 4=+KV Cache, 5=多层, 6=调度器, 7=持久化, 8=混合调度）
    int graphStage_ = 0;
    
public:
    /**
     * @brief 重置 KV Cache
     */
    void resetKVCache();
};

/**
 * @brief 层权重结构（用于上传）
 */
struct LayerWeightsGPU {
    const float* inputLayernorm;
    const float* qProj;
    const float* kProj;
    const float* vProj;
    const float* oProj;
    const float* qNorm;
    const float* kNorm;
    const float* postAttentionLayernorm;
    const float* gateProj;
    const float* upProj;
    const float* downProj;
};

} // namespace kylin
} // namespace cllm
