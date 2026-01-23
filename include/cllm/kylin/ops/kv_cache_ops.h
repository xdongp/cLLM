/**
 * @file kv_cache_ops.h
 * @brief KV Cache 操作和管理接口
 * 
 * 提供 KV Cache 的分配、写入、读取、验证等操作
 */
#pragma once

#include "cllm/kylin/core/tensor_stats.h"
#include "cllm/kylin/gguf/context.h"

#include <vector>
#include <cstddef>

namespace cllm {
namespace kylin {

/**
 * @brief KV Cache 统计信息
 */
struct KVCacheStats {
    size_t layerIdx = 0;          ///< 层索引
    size_t headDim = 0;           ///< head 维度
    size_t maxSeq = 0;            ///< 最大序列长度
    size_t nKVHeads = 0;          ///< KV head 数量
    size_t currentLen = 0;        ///< 当前有效长度
    TensorStats kStats;           ///< K cache 统计
    TensorStats vStats;           ///< V cache 统计
    bool isValid = false;         ///< 是否有效（无 NaN/Inf）
};

/**
 * @brief KV Cache 比较结果
 */
struct KVCacheCompareResult {
    float kMaxDiff = 0.0f;        ///< K cache 最大差异
    float kAvgDiff = 0.0f;        ///< K cache 平均差异
    float vMaxDiff = 0.0f;        ///< V cache 最大差异
    float vAvgDiff = 0.0f;        ///< V cache 平均差异
    size_t kLargeDiffCount = 0;   ///< K cache 大差异数量（> threshold）
    size_t vLargeDiffCount = 0;   ///< V cache 大差异数量
};

/**
 * @brief KV Cache 配置
 */
struct KVCacheConfig {
    size_t nLayers = 0;           ///< 层数
    size_t nKVHeads = 0;          ///< KV head 数量
    size_t headDim = 0;           ///< head 维度
    size_t maxSeqLen = 2048;      ///< 最大序列长度
};

/**
 * @brief KV Cache 管理器
 * 
 * 负责 KV Cache 的分配、更新、验证等操作
 */
class KVCacheManager {
public:
    /**
     * @brief 构造函数
     */
    KVCacheManager() = default;
    
    /**
     * @brief 析构函数
     */
    ~KVCacheManager() = default;
    
    // 禁止拷贝
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;
    
    // ========== 初始化 ==========
    
    /**
     * @brief 分配 KV Cache
     * @param ctx GGML 上下文
     * @param config KV Cache 配置
     * @return true 成功
     */
    bool allocate(ggml_context* ctx, const KVCacheConfig& config);
    
    /**
     * @brief 清空 KV Cache
     */
    void clear();
    
    // ========== 数据操作 ==========
    
    /**
     * @brief 将新的 K/V 数据写入 cache
     * @param layerIdx 层索引
     * @param kNew 新的 K 数据张量
     * @param vNew 新的 V 数据张量
     * @param startPos 起始位置
     * @param seqLen 序列长度
     * @return true 成功
     */
    bool writeToCache(size_t layerIdx, ggml_tensor* kNew, ggml_tensor* vNew,
                      size_t startPos, size_t seqLen);
    
    /**
     * @brief 批量写入所有层的 K/V
     * @param pendingK 待写入的 K 张量列表
     * @param pendingV 待写入的 V 张量列表
     * @param startPos 起始位置
     * @param seqLen 序列长度
     * @return true 成功
     */
    bool flushPending(const std::vector<ggml_tensor*>& pendingK,
                      const std::vector<ggml_tensor*>& pendingV,
                      size_t startPos, size_t seqLen);
    
    // ========== 访问接口 ==========
    
    /**
     * @brief 获取 K cache 张量
     */
    ggml_tensor* getKCache(size_t layerIdx) const;
    
    /**
     * @brief 获取 V cache 张量
     */
    ggml_tensor* getVCache(size_t layerIdx) const;
    
    /**
     * @brief 获取当前 cache 长度
     */
    size_t getCurrentLength() const { return currentLen_; }
    
    /**
     * @brief 设置当前 cache 长度
     */
    void setCurrentLength(size_t len) { currentLen_ = len; }
    
    /**
     * @brief 获取最大 cache 长度
     */
    size_t getMaxLength() const { return config_.maxSeqLen; }
    
    // ========== 验证和统计 ==========
    
    /**
     * @brief 验证指定层的 KV Cache 数据完整性
     * @param layerIdx 层索引
     * @param expectedLen 期望的长度
     * @return true 如果验证通过
     */
    bool verifyIntegrity(size_t layerIdx, size_t expectedLen) const;
    
    /**
     * @brief 验证所有层的 KV Cache 数据完整性
     * @param expectedLen 期望的长度
     * @return true 如果所有层验证通过
     */
    bool validateAllLayers(size_t expectedLen) const;
    
    /**
     * @brief 获取指定层的 KV Cache 统计信息
     * @param layerIdx 层索引
     * @return 统计信息
     */
    KVCacheStats getStats(size_t layerIdx) const;
    
    /**
     * @brief 获取所有层的 KV Cache 统计信息
     * @return 所有层的统计信息
     */
    std::vector<KVCacheStats> getAllStats() const;
    
    /**
     * @brief 获取指定位置的 KV 数据
     * @param layerIdx 层索引
     * @param position 位置
     * @param kData 输出 K 数据
     * @param vData 输出 V 数据
     * @return true 成功
     */
    bool getDataAtPosition(size_t layerIdx, size_t position,
                           std::vector<float>& kData, std::vector<float>& vData) const;
    
private:
    KVCacheConfig config_;                ///< 配置
    std::vector<ggml_tensor*> kCaches_;   ///< K cache 张量列表
    std::vector<ggml_tensor*> vCaches_;   ///< V cache 张量列表
    size_t currentLen_ = 0;               ///< 当前有效长度
    
    /**
     * @brief 单层 K/V 写入实现
     */
    bool writeSingleLayer(size_t layerIdx, ggml_tensor* kNew, ggml_tensor* vNew,
                          size_t startPos, size_t seqLen);
};

} // namespace kylin
} // namespace cllm
