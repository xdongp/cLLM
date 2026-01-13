/**
 * @file libtorch_backend.h
 * @brief LibTorch 推理后端实现（使用 PyTorch C++ API）
 * 
 * 参考文档：LibTorch后端设计.md
 */
#pragma once

#include "cllm/inference/backend_interface.h"
#include "cllm/kylin/tensor.h"
#include "cllm/model/config.h"
#include "cllm/model/weight_data.h"

#include <torch/script.h>
#include <torch/torch.h>

#include <memory>
#include <string>
#include <vector>

namespace cllm {
namespace inference {

/**
 * @brief LibTorch 推理后端
 *
 * 使用 PyTorch C++ API (LibTorch) 进行推理：
 * - 加载 TorchScript 模型（.pt 文件）
 * - 支持 LibTorch 内置量化（int8/fp16）
 * - 利用 MKL-DNN/oneDNN 优化
 * 
 * 实现 IBackend 接口，可与 Kylin 后端无缝切换
 */
class LibTorchBackend : public IBackend {
public:
    /**
     * @brief 构造函数
     * @param modelPath TorchScript 模型路径（.pt 文件）
     * @param config 模型配置
     */
    explicit LibTorchBackend(const std::string &modelPath, const ModelConfig &config);

    /**
     * @brief 析构函数
     */
    ~LibTorchBackend() override = default;

    // ========== IBackend 接口实现 ==========

    /**
     * @brief 初始化加载模型
     * @return true 成功，false 失败
     */
    bool initialize() override;

    /**
     * @brief 单序列前向推理
     * @param inputIds 输入 token id 序列
     * @return [seq_len, vocab_size] logits 张量
     */
    kylin::Tensor forward(const std::vector<int> &inputIds) override;

    /**
     * @brief 批处理前向推理
     * @param flatInputIds 展平后的所有 token id
     * @param requestPositions 每个请求的起止位置
     * @param batchSize 批大小
     * @return [total_tokens, vocab_size] logits 张量
     */
    kylin::Tensor forwardBatch(
        const std::vector<int> &flatInputIds,
        const std::vector<std::pair<size_t, size_t>> &requestPositions,
        size_t batchSize
    ) override;

    /**
     * @brief 获取后端名称
     */
    std::string getName() const override { return "LibTorch"; }

    /**
     * @brief 获取模型是否已加载
     */
    bool isInitialized() const override { return initialized_; }

    /**
     * @brief 获取模型配置
     */
    const ModelConfig &getConfig() const override { return config_; }
    
    /**
     * @brief 从 ModelWeights 加载权重到 LibTorch 后端
     * 
     * @param weights 通用权重数据结构
     * @return true 成功，false 失败
     */
    bool loadFromModelWeights(const model::ModelWeights &weights);

    /**
     * @brief 设置推理设备（必须在 initialize 之前调用）
     * 
     * @param useCuda 是否使用 CUDA
     * @param deviceId GPU 设备 ID（如果使用 CUDA）
     */
    void setDevice(bool useCuda, int deviceId = 0);

    /**
     * @brief 设置线程数（CPU 推理）
     * 
     * @param numThreads 线程数，0 表示使用默认值
     */
    void setNumThreads(int numThreads);
    
    /**
     * @brief 从权重字典加载模型权重
     * 
     * @param weightsDict 权重名称到张量的映射
     * @return true 成功，false 失败
     */
    bool loadWeightsFromDict(const std::map<std::string, torch::Tensor>& weightsDict);
    
    /**
     * @brief 从 GGUF 文件加载模型
     * 
     * @param ggufPath GGUF 文件路径
     * @return true 成功，false 失败
     */
    bool loadFromGGUF(const std::string& ggufPath);
    

    
private:
    /**
     * @brief 构建模型结构
     * 
     * @return true 成功，false 失败
     */
    bool buildModel();
    
    std::string modelPath_;           ///< TorchScript 模型路径
    ModelConfig config_;              ///< 模型配置
    torch::jit::script::Module model_; ///< LibTorch 模型
    torch::Device device_;            ///< 推理设备（CPU/GPU）
    bool initialized_;                ///< 是否已初始化

    // TorchScript trace 可能固化了输入序列长度；这里记录实际可用的 seq_len
    size_t tracedSeqLen_ = 0;

    /**
     * @brief 将 std::vector<int> 转换为 torch::Tensor
     */
    torch::Tensor vecToTensor(const std::vector<int> &vec);

    /**
     * @brief 将 torch::Tensor 转换为自定义 Tensor
     */
    kylin::Tensor torchTensorToTensor(const torch::Tensor &torchTensor);
};

} // namespace inference
} // namespace cllm
