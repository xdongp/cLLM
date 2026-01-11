/**
 * @file executor_manager.h
 * @brief 模型执行器内存管理器，管理模型执行器的内存分配和释放
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <cstddef>
#include <vector>
#include <mutex>

namespace cllm {

/**
 * @brief 缓冲区信息结构
 * 
 * 记录分配的缓冲区的指针、大小和类型信息。
 */
struct BufferInfo {
    void* ptr;         ///< 缓冲区指针
    size_t size;       ///< 缓冲区大小（字节）
    bool isTemp;       ///< 是否为临时缓冲区
};

/**
 * @brief 模型执行器内存管理器类
 * 
 * 负责管理模型执行器的临时缓冲区和权重缓存的内存分配、释放和监控。
 * 提供内存限制检查和统计功能，确保内存使用不超过设定的上限。
 */
class ModelExecutorMemoryManager {
public:
    /**
     * @brief 构造函数
     * @param maxMemoryMb 最大内存限制（MB）
     */
    explicit ModelExecutorMemoryManager(size_t maxMemoryMb);
    
    /**
     * @brief 析构函数，自动释放所有分配的缓冲区
     */
    ~ModelExecutorMemoryManager();
    
    /**
     * @brief 分配临时缓冲区
     * @param size 缓冲区大小（字节）
     * @return 分配的缓冲区指针
     * @throws std::runtime_error 如果超过内存限制
     */
    void* allocateTempBuffer(size_t size);
    
    /**
     * @brief 释放临时缓冲区
     * @param ptr 要释放的缓冲区指针
     */
    void deallocateTempBuffer(void* ptr);
    
    /**
     * @brief 分配权重缓存
     * @param size 缓存大小（字节）
     * @return 分配的缓存指针
     * @throws std::runtime_error 如果超过内存限制
     */
    void* allocateWeightsCache(size_t size);
    
    /**
     * @brief 释放权重缓存
     * @param ptr 要释放的缓存指针
     */
    void deallocateWeightsCache(void* ptr);
    
    /**
     * @brief 获取临时缓冲区使用的内存大小
     * @return 已使用的临时内存（字节）
     */
    size_t getTempMemoryUsed() const;
    
    /**
     * @brief 获取权重缓存使用的内存大小
     * @return 已使用的权重缓存（字节）
     */
    size_t getWeightsMemoryUsed() const;
    
    /**
     * @brief 获取总内存使用量
     * @return 总内存使用量（字节）
     */
    size_t getTotalMemoryUsed() const;
    
    /**
     * @brief 清除所有缓冲区
     */
    void clearAll();
    
private:
    std::vector<BufferInfo> buffers_;    ///< 缓冲区信息列表
    mutable std::mutex mutex_;           ///< 互斥锁，保护并发访问
    size_t maxMemoryBytes_;              ///< 最大内存限制（字节）
    size_t tempMemoryUsed_;              ///< 已使用的临时内存（字节）
    size_t weightsMemoryUsed_;           ///< 已使用的权重缓存（字节）
};

}  // namespace cllm
