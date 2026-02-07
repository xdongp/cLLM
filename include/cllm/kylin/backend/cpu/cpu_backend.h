/**
 * @file cpu_backend.h
 * @brief CPU 后端接口定义
 * 
 * 提供纯 CPU 推理的接口，包含完整的 Transformer CPU 计算。
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include <vector>
#include <string>
#include <unordered_map>

namespace cllm {
namespace kylin {
namespace backend {

/**
 * @brief CPU 后端类
 * 
 * 负责管理 CPU 推理的所有操作，包括：
 * - 权重缓存管理
 * - 前向传播计算
 * - KV Cache 管理
 */
class CPUBackend {
public:
    CPUBackend();
    ~CPUBackend();

    /**
     * @brief 初始化 CPU 后端
     * @param config 模型配置
     * @return 是否初始化成功
     */
    bool initialize(const HFModelConfig& config);

    /**
     * @brief 上传权重到 CPU 内存
     * @param weightsMap 权重映射表
     * @return 是否上传成功
     */
    bool uploadWeights(const std::unordered_map<std::string, std::vector<float>>& weightsMap);

    /**
     * @brief CPU 前向传播
     * @param tokenId 输入 token ID
     * @param position 位置信息
     * @return 输出 logits
     */
    std::vector<float> forward(int tokenId, int position);

    /**
     * @brief 重置 KV Cache
     */
    void resetKVCache();

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }

private:
    // 模型配置
    HFModelConfig config_;
    bool initialized_ = false;

    // 权重缓存 (CPU 内存)
    std::unordered_map<std::string, std::vector<float>> weightsCached_;

    // KV Cache
    std::vector<std::vector<float>> kCache_;  // [layer, position, head, dim]
    std::vector<std::vector<float>> vCache_;
    int cachePosition_ = 0;

    // RoPE 频率缓存
    std::vector<float> ropeFreqsCos_;
    std::vector<float> ropeFreqsSin_;

    // 初始化 RoPE
    void initRoPE();

    // 辅助函数
    void rmsNorm(const float* input, const float* weight, float* output, int size, float eps);
    void matmul(const float* weight, const float* input, float* output, int M, int K);
    void applyRoPE(float* x, int nHeads, int headDim, int position);
    void softmax(float* x, int size);
    float silu(float x);
};

} // namespace backend
} // namespace kylin
} // namespace cllm
