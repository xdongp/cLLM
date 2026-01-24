/**
 * @file hf_transformer.h
 * @brief HuggingFace 格式 Transformer 模型
 * 
 * 支持直接加载 safetensors 格式的 HF 模型进行推理
 * 支持 CPU 和 Metal GPU 加速
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include "cllm/kylin/hf/safetensors_loader.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/kylin/core/ggml_kernels.h"

#include <memory>
#include <vector>
#include <string>

namespace cllm {
namespace kylin {

/**
 * @brief HuggingFace Transformer 模型
 * 
 * 使用 F32 精度进行计算
 * 支持两种模式：
 * - 预转换模式（推荐）：加载时将 BF16 权重转为 F32，运行时无转换开销
 * - 实时转换模式：运行时从 BF16 转换（节省内存但较慢）
 * 
 * 设备支持：
 * - CPU：使用 BLAS/SIMD 优化
 * - Metal GPU：使用 Apple Metal 加速（macOS）
 */
class HFTransformerModel {
public:
    /**
     * @brief 从模型目录加载
     * 
     * @param modelDir 包含 config.json 和 model.safetensors 的目录
     * @param device 计算设备类型（默认 CPU）
     */
    explicit HFTransformerModel(const std::string& modelDir, DeviceType device = DeviceType::CPU);
    ~HFTransformerModel();
    
    /**
     * @brief 检查模型是否加载成功
     */
    bool isLoaded() const { return loaded_; }
    
    /**
     * @brief 前向推理
     * 
     * @param inputIds 输入 token IDs
     * @return logits [seq_len * vocab_size]
     */
    std::vector<float> forward(const std::vector<int32_t>& inputIds);
    
    /**
     * @brief 重置 KV Cache
     */
    void resetKVCache();
    
    /**
     * @brief 获取模型配置
     */
    const HFModelConfig& config() const { return config_; }
    
    /**
     * @brief 获取词表大小
     */
    int vocabSize() const { return config_.vocabSize; }
    
    /**
     * @brief 获取隐藏层维度
     */
    int hiddenSize() const { return config_.hiddenSize; }
    
private:
    // 加载权重
    bool loadWeights();
    
    // 计算组件
    void embedding(const std::vector<int32_t>& inputIds, std::vector<float>& output);
    void rmsNorm(const float* input, const float* weight, float* output, int size, float eps);
    void attention(int layerIdx, const float* input, float* output, int seqLen, int startPos);
    void ffn(int layerIdx, const float* input, float* output);
    void lmHead(const float* input, float* output);
    
    // RoPE
    void applyRoPE(float* q, float* k, int headDim, int nHeads, int nKVHeads, int seqLen, int startPos);
    
    // 矩阵乘法（BF16 权重 @ F32 输入）- 回退模式
    void matmulBF16(const uint16_t* weight, const float* input, float* output,
                    int outFeatures, int inFeatures, int batchSize = 1);
    
    // 矩阵乘法（F32 权重 @ F32 输入）- 预转换模式
    void matmulF32(const float* weight, const float* input, float* output,
                   int outFeatures, int inFeatures);
    
    // 预转换权重到 F32
    void preconvertWeights();
    
    // 模型状态
    bool loaded_ = false;
    HFModelConfig config_;
    std::unique_ptr<SafetensorsLoader> loader_;
    DeviceType deviceType_ = DeviceType::CPU;
    
    // GPU 后端（Metal）
    std::unique_ptr<GGMLGPUBackend> gpuBackend_;
    bool useGPU_ = false;
    
    // 权重指针（指向 mmap 的 BF16 数据）
    const uint16_t* embedTokens_ = nullptr;
    const uint16_t* lmHeadWeight_ = nullptr;
    const uint16_t* finalNormWeight_ = nullptr;
    
    // BF16 权重指针（指向 mmap 数据）
    struct LayerWeightsBF16 {
        const uint16_t* inputLayernorm = nullptr;
        const uint16_t* qProj = nullptr;
        const uint16_t* kProj = nullptr;
        const uint16_t* vProj = nullptr;
        const uint16_t* oProj = nullptr;
        const uint16_t* qNorm = nullptr;
        const uint16_t* kNorm = nullptr;
        const uint16_t* postAttentionLayernorm = nullptr;
        const uint16_t* gateProj = nullptr;
        const uint16_t* upProj = nullptr;
        const uint16_t* downProj = nullptr;
    };
    std::vector<LayerWeightsBF16> layers_;
    
    // F32 预转换权重（预转换模式）
    struct LayerWeightsF32 {
        std::vector<float> inputLayernorm;
        std::vector<float> qProj;
        std::vector<float> kProj;
        std::vector<float> vProj;
        std::vector<float> oProj;
        std::vector<float> qNorm;
        std::vector<float> kNorm;
        std::vector<float> postAttentionLayernorm;
        std::vector<float> gateProj;
        std::vector<float> upProj;
        std::vector<float> downProj;
    };
    std::vector<LayerWeightsF32> layersF32_;
    
    // 全局 F32 权重
    std::vector<float> embedTokensF32_;
    std::vector<float> lmHeadWeightF32_;
    std::vector<float> finalNormWeightF32_;
    
    // 是否使用预转换模式
    bool usePreconvertedWeights_ = true;
    
    // KV Cache [layer, 2, maxSeqLen, numKVHeads, headDim]
    std::vector<float> kCache_;
    std::vector<float> vCache_;
    int kvCacheLen_ = 0;
    static constexpr int kMaxSeqLen = 4096;
    
    // 预计算的 RoPE 频率
    std::vector<float> ropeFreqsCos_;
    std::vector<float> ropeFreqsSin_;
    
    // 工作缓冲区（预分配，避免运行时分配）
    std::vector<float> hiddenStates_;
    std::vector<float> residual_;
    std::vector<float> normOutput_;
    std::vector<float> attnOutput_;
    std::vector<float> ffnOutput_;
    std::vector<float> qkvBuffer_;
    
    // Attention 工作缓冲区
    std::vector<float> qBuffer_;
    std::vector<float> kBuffer_;
    std::vector<float> vBuffer_;
    std::vector<float> attnScores_;
    std::vector<float> attnOutBuffer_;
    
    // FFN 工作缓冲区
    std::vector<float> gateBuffer_;
    std::vector<float> upBuffer_;
    
    // Norm 权重缓冲区
    std::vector<float> normWeightBuffer_;
    std::vector<float> qkNormBuffer_;
    
    // Logits 缓冲区（避免每次分配）
    mutable std::vector<float> logitsBuffer_;
};

} // namespace kylin
} // namespace cllm
