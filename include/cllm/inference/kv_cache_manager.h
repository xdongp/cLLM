/**
 * @file kv_cache_manager.h
 * @brief KV缓存统计管理器（Phase 4）
 * 
 * 基于 requestId 的KV缓存统计管理，包括统计信息跟踪和清理协调。
 * 注意：KV缓存数据由 llama.cpp 内部管理，我们只需要管理统计信息和协调清理。
 * 
 * @author cLLM Team
 * @date 2026-01-18
 */

#pragma once

#include <map>
#include <mutex>
#include <cstddef>
#include <chrono>

// Forward declaration from llama.h (global namespace)
struct llama_context;

namespace cllm {
namespace inference {

/**
 * @brief KV缓存统计信息结构
 * 
 * 记录每个 requestId 对应的KV缓存统计信息，不存储实际缓存数据。
 */
struct KVCacheStats {
    size_t requestId;                    ///< 请求ID
    size_t itemCount;                    ///< 缓存条目数（序列总长度：tokenizedPrompt.size() + generatedTokens.size()）
    size_t memoryMb;                     ///< 内存占用估算（MB）
    std::chrono::steady_clock::time_point lastAccessTime;  ///< 最后访问时间（用于LRU）

    KVCacheStats()
        : requestId(0)
        , itemCount(0)
        , memoryMb(0)
        , lastAccessTime(std::chrono::steady_clock::now()) {}
    
    KVCacheStats(size_t id, size_t items, size_t memMb)
        : requestId(id)
        , itemCount(items)
        , memoryMb(memMb)
        , lastAccessTime(std::chrono::steady_clock::now()) {}
};

/**
 * @brief 请求状态枚举（用于淘汰保护）
 */
enum class RequestStatus {
    PENDING,      ///< 等待处理
    PROCESSING,   ///< 正在处理
    COMPLETED,    ///< 已完成
    TIMEOUT,      ///< 超时
    FAILED        ///< 失败
};

/**
 * @brief KV缓存统计管理器
 * 
 * 管理基于 requestId 的KV缓存统计信息，协调 llama.cpp 的KV缓存清理。
 * 
 * 职责：
 * 1. 维护 requestId 到 KV缓存统计信息的映射
 * 2. 跟踪总条目数和总内存占用
 * 3. 协调 llama.cpp 的KV缓存清理（通过 llama_memory_seq_rm）
 * 4. 维护请求状态映射（用于淘汰保护，Phase 5）
 */
class KVCacheManager {
public:
    /**
     * @brief 构造函数
     * @param maxItems 最大缓存条目数（默认：4*1024*1024）
     * @param maxMemoryMb 最大内存限制（MB，默认：1024）
     */
    explicit KVCacheManager(size_t maxItems = 4 * 1024 * 1024, size_t maxMemoryMb = 1024);

    /**
     * @brief 析构函数
     */
    ~KVCacheManager() = default;

    // 禁止拷贝和赋值
    KVCacheManager(const KVCacheManager&) = delete;
    KVCacheManager& operator=(const KVCacheManager&) = delete;

    /**
     * @brief 更新KV缓存统计信息
     * @param requestId 请求ID
     * @param sequenceLength 序列总长度（tokenizedPrompt.size() + generatedTokens.size()）
     * 
     * 注意：不存储KV缓存数据，只更新统计信息。
     * KV缓存条目数 = 序列总长度。
     * 内存占用估算 = 序列长度 × 每条目内存大小（可配置或估算）。
     */
    void updateKVCacheStats(size_t requestId, size_t sequenceLength);

    /**
     * @brief 获取KV缓存统计信息
     * @param requestId 请求ID
     * @return 统计信息，如果不存在返回默认值
     */
    KVCacheStats getKVCacheStats(size_t requestId) const;

    /**
     * @brief 检查是否存在KV缓存统计信息
     * @param requestId 请求ID
     * @return true 如果存在，false 否则
     */
    bool hasKVCacheStats(size_t requestId) const;

    /**
     * @brief 移除KV缓存（协调清理）
     * @param ctx llama.cpp 上下文（用于调用 llama_memory_seq_rm）
     * @param requestId 请求ID
     * @param seqId 序列ID（通过序列ID管理器获取）
     * @return true 如果成功清理，false 否则
     * 
     * 协调清理：
     * 1. 调用 llama_memory_seq_rm 清理 llama.cpp 中的KV缓存（需要seqId）
     * 2. 删除统计信息
     * 3. 更新总条目数和内存占用
     */
    bool removeKVCache(llama_context* ctx, size_t requestId, int32_t seqId);

    /**
     * @brief 获取总条目数
     * @return 总条目数
     */
    size_t getTotalItems() const;

    /**
     * @brief 获取总内存占用（MB）
     * @return 总内存占用（MB）
     */
    size_t getTotalMemoryMb() const;

    /**
     * @brief 获取缓存数量（requestId数量）
     * @return 缓存数量
     */
    size_t getCacheCount() const;

    /**
     * @brief 更新请求状态（用于淘汰保护，Phase 5）
     * @param requestId 请求ID
     * @param status 请求状态
     */
    void updateRequestStatus(size_t requestId, RequestStatus status);

    /**
     * @brief 获取请求状态
     * @param requestId 请求ID
     * @return 请求状态，如果不存在返回 PENDING
     */
    RequestStatus getRequestStatus(size_t requestId) const;

    /**
     * @brief 获取最大条目数限制
     */
    size_t getMaxItems() const { return maxItems_; }

    /**
     * @brief 获取最大内存限制（MB）
     */
    size_t getMaxMemoryMb() const { return maxMemoryMb_; }

    /**
     * @brief 估算单个条目内存占用（MB）
     * 
     * 这是一个粗略估算，实际内存占用取决于模型大小。
     * 默认估算：假设每个条目占用约 2MB（可配置）
     */
    static size_t estimateMemoryPerItem(size_t vocabSize = 151936, size_t hiddenSize = 4096);

    /**
     * @brief Phase 5: 检查是否需要触发淘汰
     * @param evictionThreshold 淘汰阈值（默认：0.8）
     * @return true 如果需要淘汰，false 否则
     */
    bool shouldEvict(double evictionThreshold = 0.8) const;

    /**
     * @brief Phase 5: 执行LRU淘汰
     * @param ctx llama.cpp 上下文（用于调用 llama_memory_seq_rm）
     * @param evictionThreshold 淘汰阈值（默认：0.8）
     * @param getSeqIdCallback 获取序列ID的回调函数（requestId -> seqId）
     * @return 淘汰的请求数量
     * 
     * LRU淘汰逻辑：
     * 1. 按照requestId的最后访问时间排序
     * 2. 只淘汰状态为 PENDING 或 COMPLETED 的请求对应的KV缓存
     * 3. 保护：不淘汰状态为 PROCESSING 的请求对应的KV缓存
     * 4. 持续淘汰直到条目数或内存低于阈值
     */
    size_t evictLRUCache(llama_context* ctx, double evictionThreshold = 0.8,
                        std::function<int32_t(size_t)> getSeqIdCallback = nullptr);

private:
    /**
     * @brief 计算内存占用估算
     * @param itemCount 条目数
     * @return 内存占用（MB）
     */
    size_t calculateMemoryUsage(size_t itemCount) const;

    std::map<size_t, KVCacheStats> statsMap_;           ///< requestId 到统计信息的映射
    std::map<size_t, RequestStatus> requestStatus_;     ///< requestId 到请求状态的映射（用于淘汰保护）
    mutable std::mutex mutex_;                          ///< 保护并发访问

    size_t totalItems_;                                 ///< 总条目数
    size_t totalMemoryMb_;                              ///< 总内存占用（MB）

    size_t maxItems_;                                   ///< 最大条目数限制
    size_t maxMemoryMb_;                                ///< 最大内存限制（MB）
};

} // namespace inference
} // namespace cllm