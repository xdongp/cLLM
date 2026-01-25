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
#include "cllm/kylin/hf/kv_cache_pool.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/kylin/core/quantization.h"

#include <memory>
#include <vector>
#include <string>
#include <mutex>

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
     * @return logits [seq_len * vocab_size]
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
     * @brief 重置 KV Cache（全局）
     */
    void resetKVCache();
    
    /**
     * @brief 释放请求的 KV Cache
     * 
     * @param requestId 请求 ID
     */
    void releaseKVCache(size_t requestId);
    
    /**
     * @brief 获取 KV Cache 池统计信息
     */
    int getAvailableKVSlots() const;
    int getUsedKVSlots() const;
    
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
    
    // 原始 attention（使用全局 KV Cache）
    void attention(int layerIdx, const float* input, float* output, int seqLen, int startPos);
    
    // 新版 attention（使用独立 KV Cache）
    void attentionWithKVCache(int layerIdx, const float* input, float* output, 
                              int seqLen, int startPos,
                              KVCacheSlot* kvSlot, WorkBufferSlot* workBuf);
    
    void ffn(int layerIdx, const float* input, float* output);
    void ffnWithBuffer(int layerIdx, const float* input, float* output, WorkBufferSlot* workBuf);
    void lmHead(const float* input, float* output);
    
    // 单请求处理（用于 forwardBatch）
    void forwardSingle(const std::vector<int32_t>& inputIds, size_t requestId,
                       std::vector<float>& logits);
    
    // RoPE
    void applyRoPE(float* q, float* k, int headDim, int nHeads, int nKVHeads, int seqLen, int startPos);
    
    // 矩阵乘法（BF16 权重 @ F32 输入）- 回退模式
    void matmulBF16(const uint16_t* weight, const float* input, float* output,
                    int outFeatures, int inFeatures, int batchSize = 1);
    
    // 矩阵乘法（F32 权重 @ F32 输入）- 预转换模式
    void matmulF32(const float* weight, const float* input, float* output,
                   int outFeatures, int inFeatures);
    
    // 矩阵乘法（FP16 权重 @ F32 输入）- FP16 量化模式
    void matmulFP16(const uint16_t* weight, const float* input, float* output,
                    int outFeatures, int inFeatures);
    
    // 矩阵乘法（INT8 权重 @ F32 输入）- INT8 量化模式
    void matmulINT8(const int8_t* weight, const float* input, float* output,
                    int outFeatures, int inFeatures, float scale, int32_t zeroPoint);
    
    // 统一的矩阵乘法接口（根据量化类型自动选择）
    void matmulQuantized(int layerIdx, const char* weightName,
                         const float* input, float* output,
                         int outFeatures, int inFeatures);
    
    // 预转换权重到目标精度
    void preconvertWeights();
    
    // 转换权重到 FP16
    void convertWeightsToFP16();
    
    // 转换权重到 INT8
    void convertWeightsToINT8();
    
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
        std::vector<float> qkvProj;
        std::vector<float> qNorm;
        std::vector<float> kNorm;
        std::vector<float> postAttentionLayernorm;
        std::vector<float> gateProj;
        std::vector<float> upProj;
        std::vector<float> downProj;
        std::vector<float> gateUpProj;
    };
    std::vector<LayerWeightsF32> layersF32_;
    
    // 全局 F32 权重
    std::vector<float> embedTokensF32_;
    std::vector<float> lmHeadWeightF32_;
    std::vector<float> finalNormWeightF32_;
    
    // 量化类型
    QuantType quantType_ = QuantType::FP32;
    
    // 是否使用预转换模式
    bool usePreconvertedWeights_ = true;
    
    // FP16 权重（当 quantType_ == FP16 时使用）
    struct LayerWeightsFP16 {
        std::vector<uint16_t> inputLayernorm;
        std::vector<uint16_t> qProj;
        std::vector<uint16_t> kProj;
        std::vector<uint16_t> vProj;
        std::vector<uint16_t> oProj;
        std::vector<uint16_t> qkvProj;
        std::vector<uint16_t> qNorm;
        std::vector<uint16_t> kNorm;
        std::vector<uint16_t> postAttentionLayernorm;
        std::vector<uint16_t> gateProj;
        std::vector<uint16_t> upProj;
        std::vector<uint16_t> downProj;
        std::vector<uint16_t> gateUpProj;
    };
    std::vector<LayerWeightsFP16> layersFP16_;
    
    // 全局 FP16 权重
    std::vector<uint16_t> embedTokensFP16_;
    std::vector<uint16_t> lmHeadWeightFP16_;
    std::vector<uint16_t> finalNormWeightFP16_;
    
    // INT8 权重（当 quantType_ == INT8 时使用）
    struct LayerWeightsINT8 {
        std::vector<int8_t> qProj;
        std::vector<int8_t> kProj;
        std::vector<int8_t> vProj;
        std::vector<int8_t> oProj;
        std::vector<int8_t> qkvProj;      // 融合 QKV
        std::vector<int8_t> gateProj;
        std::vector<int8_t> upProj;
        std::vector<int8_t> downProj;
        std::vector<int8_t> gateUpProj;   // 融合 Gate+Up
        
        // 每个权重矩阵的量化参数（per-tensor 量化）
        float qProjScale = 1.0f, kProjScale = 1.0f, vProjScale = 1.0f;
        float oProjScale = 1.0f, qkvProjScale = 1.0f;
        float gateProjScale = 1.0f, upProjScale = 1.0f, downProjScale = 1.0f;
        float gateUpProjScale = 1.0f;
        
        int32_t qProjZP = 0, kProjZP = 0, vProjZP = 0;
        int32_t oProjZP = 0, qkvProjZP = 0;
        int32_t gateProjZP = 0, upProjZP = 0, downProjZP = 0;
        int32_t gateUpProjZP = 0;
    };
    std::vector<LayerWeightsINT8> layersINT8_;
    
    // 全局 INT8 权重
    std::vector<int8_t> embedTokensINT8_;
    std::vector<int8_t> lmHeadWeightINT8_;
    float embedTokensScale_ = 1.0f, lmHeadScale_ = 1.0f;
    int32_t embedTokensZP_ = 0, lmHeadZP_ = 0;
    
    // 全局 KV Cache（兼容旧接口）
    std::vector<float> kCache_;
    std::vector<float> vCache_;
    int kvCacheLen_ = 0;
    static constexpr int kMaxSeqLen = 4096;
    
    // Per-Request KV Cache Pool（支持并发）
    std::unique_ptr<KVCachePool> kvCachePool_;
    std::unique_ptr<WorkBufferPool> workBufferPool_;
    static constexpr int kMaxConcurrentRequests = 16;  // 最大并发请求数
    
    // 全局锁（保护全局资源）
    mutable std::mutex globalMutex_;
    
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
    std::vector<float> gateUpBuffer_;
    
    // Norm 权重缓冲区
    std::vector<float> normWeightBuffer_;
    std::vector<float> qkNormBuffer_;
    
    // Logits 缓冲区（避免每次分配）
    mutable std::vector<float> logitsBuffer_;
};

} // namespace kylin
} // namespace cllm
