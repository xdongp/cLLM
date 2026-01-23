/**
 * @file model_loader.h
 * @brief 简化版模型加载器，用于从预处理的二进制文件加载 Transformer 权重
 *
 * 当前实现对应设计文档中"Qwen3 定制版 ModelLoader（第一阶段可以借助离线转换）"的最小子集：
 * - 假设已有 Python 工具将 Qwen3 safetensors 导出为扁平二进制文件（fp32/fp16/int8）
 *   并按照约定顺序依次存放 embedding、各层权重和 lm_head；
 * - 本类根据现有 ModelConfig 中的结构参数（vocabSize/hiddenSize/numLayers 等）推导各张量形状，
 *   并按固定顺序切片填充到推理引擎内部的 Tensor 中。
 * - 支持 fp32/fp16/int8 三种数据格式，int8 需要配套 .json 元数据文件提供量化 scale
 */
#pragma once

#include <string>
#include <vector>
#include <memory>

#include "cllm/model/config.h"
#include "cllm/kylin/core/tensor.h"

// 前向声明
namespace cllm {
    class GGUFLoader;
}

namespace cllm {
namespace kylin {

enum class WeightDType {
    FP32,
    FP16,
    INT8,
    GGUF_Q4_K,  // GGUF格式，Q4_K量化
    GGUF_Q5_K,  // GGUF格式，Q5_K量化
    GGUF_Q6_K,  // GGUF格式，Q6_K量化
    GGUF_Q8_0,  // GGUF格式，Q8_0量化
    GGUF_F16,   // GGUF格式，FP16
    GGUF_F32    // GGUF格式，FP32
};

class ModelLoader {
public:
    /// 构造函数
    /// @param modelPath 预处理后的二进制权重文件路径（fp32/fp16/int8 扁平存储）
    /// @param config    模型结构配置（来自 ModelExecutor/外部配置），用于推导各张量形状
    ModelLoader(const std::string &modelPath, const ModelConfig &config);

    /// 加载整个二进制文件到内存；成功返回 true
    /// 会自动检测文件扩展名来推导 dtype (fp32/fp16/int8)，并加载对应的元数据
    bool load();

    const ModelConfig &getConfig() const { return config_; }
    const std::string &getModelPath() const { return modelPath_; }
    WeightDType getDType() const { return dtype_; }
    
    /// 获取加载的原始权重数据
    const std::vector<float>& getWeights() const { return weights_; }

    /// 将权重填充进调用方提供的张量中，张量形状由调用方根据 ModelConfig 预先分配
    ///
    /// 目前约定的权重顺序为：
    /// 1. embedding: [vocabSize, hiddenSize]
    /// 2. 对每一层 (0..numLayers-1)：
    ///    - wq:    [hiddenSize, qDim]  // qDim = numAttentionHeads * headDim
    ///    - wk:    [hiddenSize, kvDim] // kvDim = numKeyValueHeads * headDim
    ///    - wv:    [hiddenSize, kvDim]
    ///    - wo:    [qDim, hiddenSize]
    ///    - wGate: [hiddenSize, intermediateSize]
    ///    - wUp:   [hiddenSize, intermediateSize]
    ///    - wDown: [intermediateSize, hiddenSize]
    ///    - norm1: [hiddenSize]
    ///    - norm2: [hiddenSize]
    /// 3. finalNorm: [hiddenSize]
    /// 4. lmHead:    [hiddenSize, vocabSize]
    ///
    /// 支持 GQA (Grouped Query Attention)：numKeyValueHeads 可以小于 numAttentionHeads
    /// 如文件大小与上述推导不匹配，返回 false。
    bool loadInto(
        Tensor &embedding,
        std::vector<Tensor> &wq,
        std::vector<Tensor> &wk,
        std::vector<Tensor> &wv,
        std::vector<Tensor> &wo,
        std::vector<Tensor> &wGate,
        std::vector<Tensor> &wUp,
        std::vector<Tensor> &wDown,
        std::vector<Tensor> &norm1,
        std::vector<Tensor> &norm2,
        Tensor &finalNorm,
        Tensor &lmHead
    ) const;

private:
    std::string modelPath_;
    ModelConfig config_;
    std::vector<float> weights_;
    WeightDType dtype_;
    float int8Scale_;  // 用于 int8 反量化
    
    // 实际投影维度（从 .json 元数据加载，如果为 0 则使用 config_ 推导）
    size_t actualQProjDim_;
    size_t actualKVProjDim_;
    
    // GGUF支持
    std::unique_ptr<GGUFLoader> ggufLoader_;  // GGUF加载器（如果使用GGUF格式）
    bool isGGUFFormat_;  // 是否为GGUF格式

    bool loadBinaryFile();
    bool loadMetadata();  // 加载 .json 元数据（如 int8 scale 和实际投影维度）
    WeightDType detectDType() const;  // 根据文件扩展名推导 dtype
    bool detectGGUFFormat() const;  // 检测是否为GGUF格式
    bool loadGGUF();  // 加载GGUF格式模型
};

} // namespace kylin
} // namespace cllm
