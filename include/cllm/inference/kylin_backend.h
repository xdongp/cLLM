/**
 * @file kylin_backend.h
 * @brief Kylin (麒麟) 自研推理后端
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * Kylin Backend 是 cLLM 的自研高性能推理引擎，专注于 CPU 极致性能优化
 * - 纯 C++ 实现，无外部依赖
 * - 支持 Qwen3 等主流 Transformer 架构
 * - SIMD 优化（AVX2/AVX-512）
 * - 支持多种量化格式（FP32/FP16/INT8/INT4）
 */
#pragma once

#include "cllm/inference/backend_interface.h"
#include "cllm/kylin/model/transformer_model.h"
#include "cllm/kylin/gguf/transformer.h"  // GGML 原生实现
#include "cllm/kylin/hf/transformer.h"    // HuggingFace safetensors 支持
#include "cllm/kylin/gguf/operator_interface.h"
#include "cllm/model/loader_interface.h"
#include "cllm/kylin/core/tensor.h"
#include "cllm/model/config.h"

#include <memory>
#include <string>
#include <vector>

namespace cllm {
namespace inference {

/**
 * @brief Kylin (麒麟) 自研推理后端
 * 
 * 实现特点：
 * - 完全自主实现的 Transformer 模型
 * - 扁平 .bin 权重格式加载
 * - 模块化算子设计
 * - 支持占位权重（用于测试）和真实权重
 */
class KylinBackend : public IBackend {
public:
    /**
     * @brief 构造函数
     * 
     * @param config 模型配置
     * @param modelPath 模型权重路径
     *                  - .gguf: 使用 GGMLTransformerModel（直接 GGML 推理）
     *                  - .bin: 使用 TransformerModel（Native/GGML 算子）
     *                  - 如果为空：使用占位权重（测试模式）
     * @param operatorBackend 算子后端类型（默认 Auto，自动选择最优后端）
     */
    explicit KylinBackend(
        const ModelConfig &config, 
        const std::string &modelPath = std::string(),
        kylin::OperatorBackend operatorBackend = kylin::OperatorBackend::Auto
    );

    /**
     * @brief 析构函数
     */
    ~KylinBackend() override = default;

    // ========== IBackend 接口实现 ==========

    /**
     * @brief 初始化 Kylin 后端
     * 
     * 步骤：
     * 1. 验证模型配置
     * 2. 分配权重张量
     * 3. 加载权重（真实权重或占位权重）
     * 4. 绑定权重到 TransformerModel
     * 5. 执行验证推理（可选）
     * 
     * @return true 成功，false 失败
     */
    bool initialize() override;

    /**
     * @brief 单序列前向推理
     * 
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     */
    Tensor forward(const std::vector<int> &inputIds) override;

    /**
     * @brief 批处理前向推理
     * @param sequenceIds 每个请求的序列ID（requestId），用于序列ID管理（可选，默认空向量）
     * 
     * 实现策略：
     * - 逐请求调用 forward()
     * - 将结果拼接为 [total_tokens, vocab_size]
     * 
     * @param flatInputIds 展平后的所有 token id
     * @param requestPositions 每个请求在 flatInputIds 中的起止位置
     * @param batchSize 批大小
     * @return [total_tokens, vocab_size] logits 张量
     */
    Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize,
        const std::vector<size_t> &sequenceIds = {}
    ) override;

    /**
     * @brief 获取后端名称
     */
    std::string getName() const override { return "Kylin"; }

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const override { return initialized_; }

    /**
     * @brief 获取后端配置
     */
    const ModelConfig &getConfig() const override { return externalConfig_; }
    
    /**
     * @brief 从 ModelWeights 加载权重到 Kylin 后端
     * 
     * @param weights 通用权重数据结构
     * @return true 成功，false 失败
     */
    bool loadFromModelWeights(const model::ModelWeights &weights);
    
    /**
     * @brief 获取当前使用的算子后端类型
     * @return 算子后端类型
     */
    kylin::OperatorBackend getOperatorBackend() const;
    
    /**
     * @brief 获取算子后端名称
     * @return 算子后端名称（"Native" 或 "GGML"）
     */
    std::string getOperatorBackendName() const;
    
    /**
     * @brief 获取算子接口（供高级用户使用）
     * @return 算子接口指针
     */
    kylin::IOperator* getOperator() { return op_.get(); }
    
    /**
     * @brief 清理请求的 KV Cache
     * 
     * @param requestId 请求 ID
     * @return true 成功，false 失败
     */
    bool cleanupKVCache(size_t requestId);

private:
    // ========== 配置和状态 ==========
    
    /// 外部传入的完整模型配置（来自 ModelExecutor）
    ModelConfig externalConfig_;
    
    /// 内部实际使用的配置（真实权重模式下与 externalConfig_ 相同）
    ModelConfig internalConfig_;
    
    /// 初始化状态
    bool initialized_;
    
    /// 模型权重路径
    std::string modelPath_;

    // ========== 核心组件 ==========
    
    /// Transformer 模型实例
    std::unique_ptr<kylin::TransformerModel> model_;
    
    /// 模型加载器（仅在有真实权重时创建）
    std::unique_ptr<IModelLoader> loader_;
    
    /// 算子接口（Native 或 GGML）
    std::unique_ptr<kylin::IOperator> op_;
    
    /// 算子后端类型（用于日志和调试）
    kylin::OperatorBackend operatorBackendType_;
    
    /// 设备后端类型（CPU/Metal/CUDA）
    kylin::BackendType deviceBackendType_;

    // ========== 权重存储 ==========
    
    /// Embedding 权重 [vocabSize, hiddenSize]
    kylin::Tensor embedding_;
    
    /// LM Head 权重 [hiddenSize, vocabSize]
    kylin::Tensor lmHead_;
    
    /// Final Norm 权重 [hiddenSize]
    kylin::Tensor finalNormWeight_;
    
    /// 每层的 Attention 权重
    std::vector<kylin::Tensor> wq_;    // [hiddenSize, hiddenSize]
    std::vector<kylin::Tensor> wk_;
    std::vector<kylin::Tensor> wv_;
    std::vector<kylin::Tensor> wo_;
    
    /// 每层的 FFN 权重
    std::vector<kylin::Tensor> wGate_; // [hiddenSize, intermediateSize]
    std::vector<kylin::Tensor> wUp_;
    std::vector<kylin::Tensor> wDown_; // [intermediateSize, hiddenSize]
    
    /// 每层的 Norm 权重
    std::vector<kylin::Tensor> norm1_; // [hiddenSize]
    std::vector<kylin::Tensor> norm2_; // [hiddenSize]
    
    /// 每层的 Q/K 独立归一化权重（可选，Qwen3等模型需要）
    std::vector<kylin::Tensor> attnQNorm_; // [headDim] 或 [hiddenSize]，取决于模型
    std::vector<kylin::Tensor> attnKNorm_; // [headDim] 或 [hiddenSize]，取决于模型

    // ========== GGML 直接推理支持（GGUF 模型）==========
    
    /// 是否使用 GGML 直接推理（.gguf 文件）
    bool useGGMLDirect_;
    
    /// GGML Transformer 模型（用于 GGUF 文件）
    std::unique_ptr<kylin::GGMLTransformerModel> ggmlModel_;
    
    /// 当前活跃的序列ID（用于 KV cache 管理）
    /// 由于 GGMLTransformerModel 只有一个全局 KV cache，不支持多序列并发
    /// 当 sequenceId 变化时需要清除 KV cache
    size_t currentSequenceId_ = SIZE_MAX;
    
    // ========== HuggingFace safetensors 支持 ==========
    
    /// 是否使用 HuggingFace 模型（safetensors 格式）
    bool useHFModel_;
    
    /// HuggingFace Transformer 模型（用于 safetensors 文件）
    std::unique_ptr<kylin::HFTransformerModel> hfModel_;
    
    // ========== 内部方法 ==========
    
    /**
     * @brief 准备内部配置（占位权重模式）
     * 
     * 使用精简的配置以减少内存占用：
     * - hiddenSize = 128
     * - intermediateSize = 256
     * - numLayers = 2
     * - 保持 vocabSize 与外部配置一致
     */
    void prepareInternalConfig();
    
    /**
     * @brief 初始化占位权重
     * 
     * 使用简单的可重复模式填充权重，避免全零输出
     */
    void initializePlaceholderWeights();
    
    /**
     * @brief 加载真实权重
     * 
     * 使用 ModelLoader 从 .bin 文件加载权重
     * 
     * @return true 成功，false 失败
     */
    bool loadRealWeights();
    
    /**
     * @brief 绑定权重到 TransformerModel
     */
    void bindWeightsToModel();
    
    /**
     * @brief 初始化 GGML 直接推理模式（用于 GGUF 文件）
     * @return true 成功，false 失败
     */
    bool initializeGGMLDirect();
    
    /**
     * @brief 初始化 HuggingFace 模型（用于 safetensors 目录）
     * @return true 成功，false 失败
     */
    bool initializeHFModel();
};

} // namespace inference
} // namespace cllm
