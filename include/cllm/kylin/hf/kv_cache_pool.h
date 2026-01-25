/**
 * @file kv_cache_pool.h
 * @brief Per-Request KV Cache 管理器
 * 
 * 支持多请求并发，每个请求拥有独立的 KV Cache 槽位
 */

#pragma once

#include <vector>
#include <unordered_map>
#include <mutex>
#include <cstdint>
#include <memory>

namespace cllm {
namespace kylin {

/**
 * @brief 单个请求的 KV Cache 槽位
 */
struct KVCacheSlot {
    std::vector<float> kCache;  // [numLayers, maxSeqLen, numKVHeads, headDim]
    std::vector<float> vCache;  // [numLayers, maxSeqLen, numKVHeads, headDim]
    int currentLen = 0;         // 当前序列长度
    size_t requestId = 0;       // 关联的请求 ID
    bool inUse = false;         // 是否在使用中
    
    void reset() {
        currentLen = 0;
        requestId = 0;
        inUse = false;
        // 不清零数据，只重置状态（性能优化）
    }
};

/**
 * @brief KV Cache 池管理器
 * 
 * 管理多个 KV Cache 槽位，支持分配/释放/查找
 */
class KVCachePool {
public:
    /**
     * @brief 构造函数
     * 
     * @param maxSlots 最大槽位数（最大并发请求数）
     * @param numLayers Transformer 层数
     * @param maxSeqLen 最大序列长度
     * @param numKVHeads KV Head 数量
     * @param headDim Head 维度
     */
    KVCachePool(int maxSlots, int numLayers, int maxSeqLen, int numKVHeads, int headDim);
    ~KVCachePool() = default;
    
    /**
     * @brief 为请求分配 KV Cache 槽位
     * 
     * @param requestId 请求 ID
     * @return 槽位指针，如果没有可用槽位返回 nullptr
     */
    KVCacheSlot* allocate(size_t requestId);
    
    /**
     * @brief 释放请求的 KV Cache 槽位
     * 
     * @param requestId 请求 ID
     * @return 是否成功释放
     */
    bool release(size_t requestId);
    
    /**
     * @brief 获取请求的 KV Cache 槽位
     * 
     * @param requestId 请求 ID
     * @return 槽位指针，如果不存在返回 nullptr
     */
    KVCacheSlot* get(size_t requestId);
    
    /**
     * @brief 获取或创建请求的 KV Cache 槽位
     * 
     * @param requestId 请求 ID
     * @return 槽位指针，如果没有可用槽位返回 nullptr
     */
    KVCacheSlot* getOrAllocate(size_t requestId);
    
    /**
     * @brief 获取可用槽位数量
     */
    int availableSlots() const;
    
    /**
     * @brief 获取已使用槽位数量
     */
    int usedSlots() const;
    
    /**
     * @brief 获取每层的 KV Cache 大小（每个槽位）
     */
    size_t perLayerCacheSize() const { return perLayerCacheSize_; }
    
    /**
     * @brief 获取配置参数
     */
    int numLayers() const { return numLayers_; }
    int maxSeqLen() const { return maxSeqLen_; }
    int numKVHeads() const { return numKVHeads_; }
    int headDim() const { return headDim_; }
    
private:
    int maxSlots_;
    int numLayers_;
    int maxSeqLen_;
    int numKVHeads_;
    int headDim_;
    size_t perLayerCacheSize_;  // maxSeqLen * numKVHeads * headDim
    
    std::vector<std::unique_ptr<KVCacheSlot>> slots_;
    std::unordered_map<size_t, int> requestToSlot_;  // requestId -> slotIndex
    std::vector<int> freeSlots_;  // 可用槽位索引栈
    
    mutable std::mutex mutex_;
};

/**
 * @brief 工作缓冲区槽位（用于并发计算）
 * 
 * 每个并发请求需要独立的工作缓冲区
 */
struct WorkBufferSlot {
    std::vector<float> hiddenStates;
    std::vector<float> residual;
    std::vector<float> normOutput;
    std::vector<float> attnOutput;
    std::vector<float> ffnOutput;
    std::vector<float> qkvBuffer;
    std::vector<float> qBuffer;
    std::vector<float> kBuffer;
    std::vector<float> vBuffer;
    std::vector<float> attnScores;
    std::vector<float> attnOutBuffer;
    std::vector<float> gateBuffer;
    std::vector<float> upBuffer;
    std::vector<float> gateUpBuffer;
    std::vector<float> logits;
    
    bool inUse = false;
    
    void reset() { inUse = false; }
};

/**
 * @brief 工作缓冲区池
 */
class WorkBufferPool {
public:
    WorkBufferPool(int maxSlots, int hiddenSize, int intermediateSize, 
                   int vocabSize, int numHeads, int numKVHeads, int headDim, int maxSeqLen);
    ~WorkBufferPool() = default;
    
    WorkBufferSlot* allocate();
    void release(WorkBufferSlot* slot);
    
private:
    std::vector<std::unique_ptr<WorkBufferSlot>> slots_;
    std::vector<int> freeSlots_;
    mutable std::mutex mutex_;
};

} // namespace kylin
} // namespace cllm
