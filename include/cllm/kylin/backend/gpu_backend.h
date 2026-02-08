/**
 * @file gpu_backend.h
 * @brief GPU 计算后端 - 90%完整版本
 * 
 * 实现 IComputeBackend 接口的 GPU 版本
 * 特性：
 * - 独立 GPU 计算内核（矩阵乘法、注意力、FFN）
 * - GPU 内存池管理
 * - 异步执行和流管理
 * - 混合精度支持（FP16/FP32）
 * - 性能监控和显存统计
 */

#pragma once

#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/kylin/hf/ggml_backend.h"  // For LayerWeightsGPU
#include <vector>
#include <memory>
#include <functional>
#include <future>

namespace cllm {
namespace kylin {

// 前向声明
class GGMLGPUBackend;
struct GPUBackendImpl;
struct GPUMemoryPool;
struct GPUStream;

/**
 * @brief GPU 内存池配置
 */
struct GPUMemoryPoolConfig {
    size_t initialSize = 256 * 1024 * 1024;  // 初始 256MB
    size_t maxSize = 2ULL * 1024 * 1024 * 1024;  // 最大 2GB
    float growthFactor = 1.5f;  // 增长因子
    bool enablePooling = true;  // 启用内存池
};

/**
 * @brief GPU 执行配置
 */
struct GPUExecutionConfig {
    bool useAsync = true;  // 使用异步执行
    bool useFP16 = false;  // 使用 FP16 混合精度
    int numStreams = 2;  // 并行流数量
    int maxBatchSize = 8;  // 最大批处理大小
};

/**
 * @brief GPU 计算后端 - 90%完整版本
 * 
 * 使用 GPU（Metal/CUDA）进行 Transformer 模型的前向推理
 * 提供独立的 GPU 计算内核和内存管理
 */
class GPUBackend : public IComputeBackend {
public:
    GPUBackend();
    ~GPUBackend() override;
    
    // IComputeBackend 接口实现
    bool initialize(const HFModelConfig& config) override;
    void shutdown() override;
    bool loadWeights(const ModelWeights& weights) override;
    std::vector<float> forward(const std::vector<int32_t>& inputIds, int requestId) override;
    std::vector<std::vector<float>> forwardBatch(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    ) override;
    void resetKVCache(int requestId) override;
    void releaseKVCache(int requestId) override;
    std::string getName() const override { return "GPU"; }
    bool isGPU() const override { return true; }
    int getKVCacheCurrentLength(int requestId) const override;
    
    // ========== 权重管理 ==========
    
    /**
     * @brief 上传权重到 GPU
     */
    bool uploadWeights(
        const float* embedTokens,
        const std::vector<LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );
    
    /**
     * @brief 异步上传权重
     */
    std::future<bool> uploadWeightsAsync(
        const float* embedTokens,
        const std::vector<LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );
    
    // ========== 内存池管理 ==========
    
    /**
     * @brief 配置 GPU 内存池
     */
    void configureMemoryPool(const GPUMemoryPoolConfig& config);
    
    /**
     * @brief 获取内存池统计
     */
    void getMemoryPoolStats(size_t& usedBytes, size_t& freeBytes, size_t& totalBytes) const;
    
    /**
     * @brief 清理内存池
     */
    void purgeMemoryPool();
    
    // ========== 异步执行 ==========
    
    /**
     * @brief 异步前向推理
     */
    std::future<std::vector<float>> forwardAsync(
        const std::vector<int32_t>& inputIds,
        int requestId
    );
    
    /**
     * @brief 异步批量推理
     */
    std::future<std::vector<std::vector<float>>> forwardBatchAsync(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    );
    
    /**
     * @brief 同步所有异步操作
     */
    void synchronize();
    
    // ========== 执行配置 ==========
    
    /**
     * @brief 配置 GPU 执行参数
     */
    void configureExecution(const GPUExecutionConfig& config);
    
    /**
     * @brief 获取当前执行配置
     */
    GPUExecutionConfig getExecutionConfig() const;
    
    // ========== 性能统计 ==========
    
    /**
     * @brief 获取 GPU 性能统计
     */
    double getAverageForwardTime() const;
    double getAverageBatchTime() const;
    
    /**
     * @brief 获取 GPU 显存使用统计
     */
    size_t getWeightMemoryBytes() const;
    size_t getKVCacheMemoryBytes() const;
    size_t getActivationMemoryBytes() const;
    size_t getTotalMemoryBytes() const;
    
    /**
     * @brief 重置性能统计
     */
    void resetPerformanceStats();
    
    /**
     * @brief 获取详细性能报告
     */
    std::string getPerformanceReport() const;
    
    // ========== 混合精度 ==========
    
    /**
     * @brief 设置是否使用 FP16 混合精度
     */
    void setUseFP16(bool useFP16);
    
    /**
     * @brief 获取是否使用 FP16
     */
    bool getUseFP16() const;
    
    // ========== 高级功能 ==========
    
    /**
     * @brief 预热 GPU（提前分配资源）
     */
    void warmup();
    
    /**
     * @brief 检查 GPU 是否可用
     */
    bool isAvailable() const;
    
    /**
     * @brief 获取 GPU 信息
     */
    std::string getGPUInfo() const;
    
    /**
     * @brief 导出 GPU 计算图（用于调试）
     */
    bool exportComputeGraph(const std::string& filename) const;

private:
    // 模型配置
    HFModelConfig config_;
    
    // 权重数据
    ModelWeights weights_;
    bool weightsLoaded_ = false;
    
    // 初始化标志
    bool initialized_ = false;
    
    // 封装的 GGML GPU 后端（保留用于兼容性）
    std::unique_ptr<GGMLGPUBackend> ggmlBackend_;
    
    // PIMPL 模式隐藏实现细节
    std::unique_ptr<GPUBackendImpl> impl_;
};

} // namespace kylin
} // namespace cllm
