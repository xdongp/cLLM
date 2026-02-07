/**
 * @file backend_interface.h
 * @brief 统一计算后端接口
 * 
 * 定义 IComputeBackend 接口，用于抽象 CPU 和 GPU 计算后端
 * 模型类通过此接口与具体后端解耦
 */

#pragma once

#include "cllm/kylin/hf/config.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>

namespace cllm {
namespace kylin {

// 前向声明
enum class DeviceType;
enum class QuantType;

/**
 * @brief 模型权重数据结构
 * 
 * 包含模型所有权重的指针和元数据
 * 后端根据此结构加载权重
 */
struct ModelWeights {
    // 词嵌入权重
    const void* embedTokens = nullptr;
    
    // 输出层权重 (lm_head)
    const void* lmHeadWeight = nullptr;
    
    // 最终归一化权重
    const void* finalNormWeight = nullptr;
    
    // 每层权重
    struct LayerWeights {
        const void* inputLayernorm = nullptr;
        const void* qProj = nullptr;
        const void* kProj = nullptr;
        const void* vProj = nullptr;
        const void* oProj = nullptr;
        const void* qNorm = nullptr;
        const void* kNorm = nullptr;
        const void* postAttentionLayernorm = nullptr;
        const void* gateProj = nullptr;
        const void* upProj = nullptr;
        const void* downProj = nullptr;
    };
    std::vector<LayerWeights> layers;
    
    // 权重数据类型
    QuantType weightType;
    
    // 量化参数（如果适用）
    std::vector<float> scales;
    std::vector<int32_t> zeroPoints;
};

/**
 * @brief 计算后端接口
 * 
 * 抽象接口，定义 CPU 和 GPU 后端必须实现的方法
 * 模型类通过此接口进行前向推理，无需关心具体后端实现
 */
class IComputeBackend {
public:
    virtual ~IComputeBackend() = default;
    
    /**
     * @brief 初始化后端
     * 
     * @param config 模型配置
     * @return true 初始化成功
     * @return false 初始化失败
     */
    virtual bool initialize(const HFModelConfig& config) = 0;
    
    /**
     * @brief 关闭后端，释放资源
     */
    virtual void shutdown() = 0;
    
    /**
     * @brief 加载模型权重
     * 
     * @param weights 模型权重数据结构
     * @return true 加载成功
     * @return false 加载失败
     */
    virtual bool loadWeights(const ModelWeights& weights) = 0;
    
    /**
     * @brief 单请求前向推理
     * 
     * @param inputIds 输入 token IDs
     * @param requestId 请求 ID（用于 KV Cache 管理）
     * @return logits [vocab_size]
     */
    virtual std::vector<float> forward(
        const std::vector<int32_t>& inputIds,
        int requestId
    ) = 0;
    
    /**
     * @brief 批量前向推理
     * 
     * @param batchInputIds 每个请求的输入 token IDs
     * @param requestIds 每个请求的请求 ID
     * @return 每个请求的 logits
     */
    virtual std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) = 0;
    
    /**
     * @brief 重置指定请求的 KV Cache
     * 
     * @param requestId 请求 ID
     */
    virtual void resetKVCache(int requestId) = 0;
    
    /**
     * @brief 释放指定请求的 KV Cache
     * 
     * @param requestId 请求 ID
     */
    virtual void releaseKVCache(int requestId) = 0;
    
    /**
     * @brief 获取后端名称
     * 
     * @return 后端名称（如 "CPU", "Metal", "CUDA"）
     */
    virtual std::string getName() const = 0;
    
    /**
     * @brief 检查是否为 GPU 后端
     * 
     * @return true 是 GPU 后端
     * @return false 是 CPU 后端
     */
    virtual bool isGPU() const = 0;
    
    /**
     * @brief 获取指定请求的 KV Cache 当前长度
     * 
     * @param requestId 请求 ID
     * @return 当前 KV Cache 长度，如果请求不存在则返回 -1
     */
    virtual int getKVCacheCurrentLength(int requestId) const = 0;
};

/**
 * @brief 后端工厂
 * 
 * 根据设备类型创建对应的后端实例
 */
class BackendFactory {
public:
    /**
     * @brief 创建后端实例
     * 
     * @param device 设备类型
     * @return 后端实例指针
     */
    static std::unique_ptr<IComputeBackend> create(DeviceType device);
    
    /**
     * @brief 获取默认设备类型
     * 
     * @return 默认设备类型（根据环境变量或系统配置）
     */
    static DeviceType getDefaultDevice();
};

} // namespace kylin
} // namespace cllm
