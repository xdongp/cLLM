/**
 * @file gpu_backend.h
 * @brief GPU 计算后端 - 100%完整版本
 * 
 * 实现 IComputeBackend 接口的 GPU 版本
 * 特性：
 * - 多 GPU 支持（多设备管理）
 * - 动态批处理（Dynamic Batching）
 * - 计算图优化（Graph Optimization）
 * - 自动混合精度（AMP）
 * - GPU 内存池管理
 * - 异步执行和流管理
 * - 完整性能监控和显存统计
 */

#pragma once

#include "cllm/kylin/backend/backend_interface.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <map>
#include <sstream>
#include <iomanip>

namespace cllm {
namespace kylin {

// 前向声明
class GGMLGPUBackend;
struct GPUBackendImpl;
struct GPUMemoryPool;
struct GPUStream;
struct GPUComputeGraph;
struct DynamicBatchScheduler;

/**
 * @brief GPU 设备信息
 */
struct GPUDeviceInfo {
    int deviceId = 0;
    std::string name;
    size_t totalMemory = 0;
    size_t freeMemory = 0;
    int computeCapability = 0;
    bool isAvailable = false;
};

/**
 * @brief GPU 内存池配置
 */
struct GPUMemoryPoolConfig {
    size_t initialSize = 256 * 1024 * 1024;
    size_t maxSize = 2ULL * 1024 * 1024 * 1024;
    float growthFactor = 1.5f;
    bool enablePooling = true;
};

/**
 * @brief GPU 执行配置
 */
struct GPUExecutionConfig {
    bool useAsync = true;
    bool useFP16 = false;
    bool useAMP = false;  // 自动混合精度
    int numStreams = 2;
    int maxBatchSize = 8;
    int maxSequenceLength = 2048;
    bool enableGraphOptimization = true;  // 启用图优化
    bool enableDynamicBatching = true;    // 启用动态批处理
    int deviceId = 0;  // 指定GPU设备
};

/**
 * @brief 动态批处理配置
 */
struct DynamicBatchConfig {
    int maxBatchSize = 8;
    int maxWaitTimeMs = 10;  // 最大等待时间
    int minBatchSize = 1;
    bool prioritizeNewRequests = false;
    bool enableDynamicBatching = true;  // 启用动态批处理
};

/**
 * @brief 计算图优化配置
 */
struct GraphOptimizationConfig {
    bool enableFusion = true;        // 算子融合
    bool enableConstantFolding = true;  // 常量折叠
    bool enableDeadCodeElimination = true;  // 死代码消除
    bool enableMemoryPlanning = true;  // 内存规划
    int optimizationLevel = 2;  // 优化级别 0-3
};

/**
 * @brief 批处理请求
 */
struct BatchRequest {
    std::vector<int32_t> inputIds;
    int requestId = 0;
    int priority = 0;
    std::chrono::steady_clock::time_point submitTime;
    std::promise<std::vector<float>> promise;
};

/**
 * @brief GPU 计算后端 - 100%完整版本
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
    
    // ========== 多 GPU 支持 ==========
    
    /**
     * @brief 获取可用 GPU 设备列表
     */
    static std::vector<GPUDeviceInfo> getAvailableDevices();
    
    /**
     * @brief 选择指定 GPU 设备
     */
    bool selectDevice(int deviceId);
    
    /**
     * @brief 获取当前设备 ID
     */
    int getCurrentDeviceId() const;
    
    /**
     * @brief 获取当前设备信息
     */
    GPUDeviceInfo getCurrentDeviceInfo() const;
    
    // ========== 动态批处理 ==========
    
    /**
     * @brief 配置动态批处理
     */
    void configureDynamicBatching(const DynamicBatchConfig& config);
    
    /**
     * @brief 提交异步推理请求（支持动态批处理）
     */
    std::future<std::vector<float>> submitInferenceRequest(
        const std::vector<int32_t>& inputIds,
        int requestId,
        int priority = 0
    );
    
    /**
     * @brief 处理等待中的批处理请求
     */
    void processPendingBatches();
    
    /**
     * @brief 获取动态批处理统计
     */
    void getDynamicBatchStats(int& pendingRequests, int& processedBatches, double& avgBatchSize) const;
    
    // ========== 计算图优化 ==========
    
    /**
     * @brief 配置计算图优化
     */
    void configureGraphOptimization(const GraphOptimizationConfig& config);
    
    /**
     * @brief 构建优化后的计算图
     */
    bool buildOptimizedGraph(int maxSequenceLength);
    
    /**
     * @brief 使用优化图执行推理
     */
    std::vector<float> forwardWithOptimizedGraph(
        const std::vector<int32_t>& inputIds,
        int requestId
    );
    
    /**
     * @brief 导出计算图到文件
     */
    bool exportComputeGraph(const std::string& filename) const;
    
    /**
     * @brief 从文件导入计算图
     */
    bool importComputeGraph(const std::string& filename);
    
    // ========== 自动混合精度 ==========
    
    /**
     * @brief 启用/禁用自动混合精度（AMP）
     */
    void setUseAMP(bool useAMP);
    
    /**
     * @brief 获取 AMP 状态
     */
    bool getUseAMP() const;
    
    /**
     * @brief 校准 AMP 缩放因子
     */
    void calibrateAMP();
    
    // ========== 权重管理 ==========
    
    bool uploadWeights(
        const float* embedTokens,
        const std::vector<LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );
    
    std::future<bool> uploadWeightsAsync(
        const float* embedTokens,
        const std::vector<LayerWeightsGPU>& layers,
        const float* finalNorm,
        const float* lmHead
    );
    
    // ========== 内存池管理 ==========
    
    void configureMemoryPool(const GPUMemoryPoolConfig& config);
    void getMemoryPoolStats(size_t& usedBytes, size_t& freeBytes, size_t& totalBytes) const;
    void purgeMemoryPool();
    
    // ========== 异步执行 ==========
    
    std::future<std::vector<float>> forwardAsync(
        const std::vector<int32_t>& inputIds,
        int requestId
    );
    
    std::future<std::vector<std::vector<float>>> forwardBatchAsync(
        const std::vector<std::vector<int32_t>>& batchInputIds,
        const std::vector<int>& requestIds
    );
    
    void synchronize();
    
    // ========== 执行配置 ==========
    
    void configureExecution(const GPUExecutionConfig& config);
    GPUExecutionConfig getExecutionConfig() const;
    
    // ========== 性能统计 ==========
    
    double getAverageForwardTime() const;
    double getAverageBatchTime() const;
    size_t getWeightMemoryBytes() const;
    size_t getKVCacheMemoryBytes() const;
    size_t getActivationMemoryBytes() const;
    size_t getTotalMemoryBytes() const;
    void resetPerformanceStats();
    std::string getPerformanceReport() const;
    
    // ========== 混合精度 ==========
    
    void setUseFP16(bool useFP16);
    bool getUseFP16() const;
    
    // ========== 高级功能 ==========
    
    void warmup();
    bool isAvailable() const;
    std::string getGPUInfo() const;

private:
    HFModelConfig config_;
    ModelWeights weights_;
    bool weightsLoaded_ = false;
    bool initialized_ = false;
    std::unique_ptr<GGMLGPUBackend> ggmlBackend_;
    std::unique_ptr<GPUBackendImpl> impl_;
};

} // namespace kylin
} // namespace cllm
