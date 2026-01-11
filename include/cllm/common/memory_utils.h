/**
 * @file memory_utils.h
 * @brief 通用内存管理工具类，包括RAII内存管理器和内存监控器
 * @author cLLM Team
 * @date 2026-01-10
 */

#pragma once

#include <cstddef>
#include <atomic>
#include <functional>

namespace cllm {

/**
 * @brief 浮点数组RAII包装器类
 * 
 * 提供对float数组的RAII管理，自动处理内存的分配和释放。
 * 支持拷贝、移动、重新分配大小等操作。
 * 注意：本类本身不是线程安全的，如需并发访问需外部同步。
 */
class FloatArray {
public:
    /**
     * @brief 构造函数
     * @param size 数组大小，默认为0
     */
    explicit FloatArray(size_t size = 0);
    
    /**
     * @brief 析构函数，自动释放内存
     */
    ~FloatArray();
    
    /**
     * @brief 拷贝构造函数
     * @param other 要拷贝的对象
     */
    FloatArray(const FloatArray& other);
    
    /**
     * @brief 拷贝赋值运算符
     * @param other 要拷贝的对象
     * @return 当前对象的引用
     */
    FloatArray& operator=(const FloatArray& other);
    
    /**
     * @brief 移动构造函数
     * @param other 要移动的对象
     */
    FloatArray(FloatArray&& other) noexcept;
    
    /**
     * @brief 移动赋值运算符
     * @param other 要移动的对象
     * @return 当前对象的引用
     */
    FloatArray& operator=(FloatArray&& other) noexcept;
    
    /**
     * @brief 重新分配数组大小
     * @param newSize 新的数组大小
     */
    void resize(size_t newSize);
    
    /**
     * @brief 获取数组指针
     * @return float数组指针
     */
    float* data();
    
    /**
     * @brief 获取常量数组指针
     * @return const float数组指针
     */
    const float* data() const;
    
    /**
     * @brief 获取数组大小
     * @return 数组大小
     */
    size_t size() const;
    
    /**
     * @brief 判断数组是否为空
     * @return true 如果数组为空，false 否则
     */
    bool empty() const;
    
    /**
     * @brief 数组下标访问运算符
     * @param index 索引
     * @return 对应位置的元素引用
     * @throws std::out_of_range 如果索引越界
     */
    float& operator[](size_t index);
    
    /**
     * @brief 常量数组下标访问运算符
     * @param index 索引
     * @return 对应位置的元素常量引用
     * @throws std::out_of_range 如果索引越界
     */
    const float& operator[](size_t index) const;
    
    /**
     * @brief 清除数组，释放内存
     */
    void clear();

private:
    float* data_;   ///< 数组数据指针
    size_t size_;   ///< 数组大小
};

/**
 * @brief 全局内存监控器类
 * 
 * 单例模式的内存监控器，用于监控和限制系统的全局内存使用。
 * 提供内存分配、释放、统计和超限回调功能。
 * 所有操作都是线程安全的，使用原子变量保证并发安全。
 */
class MemoryMonitor {
public:
    /// 内存超限回调函数类型
    typedef std::function<void(size_t used, size_t limit)> MemoryLimitCallback;
    
    /**
     * @brief 获取单例实例
     * @return MemoryMonitor的单例引用
     */
    static MemoryMonitor& instance();
    
    /**
     * @brief 设置内存限制
     * @param limitBytes 内存限制（字节），0表示无限制
     */
    void setLimit(size_t limitBytes);
    
    /**
     * @brief 获取内存限制
     * @return 内存限制（字节）
     */
    size_t getLimit() const;
    
    /**
     * @brief 记录内存分配
     * @param bytes 分配的字节数
     * @throws std::runtime_error 如果超过内存限制
     */
    void allocate(size_t bytes);
    
    /**
     * @brief 记录内存释放
     * @param bytes 释放的字节数
     */
    void deallocate(size_t bytes);
    
    /**
     * @brief 获取当前使用的内存大小
     * @return 已使用的内存（字节）
     */
    size_t getUsed() const;
    
    /**
     * @brief 获取峰值内存使用量
     * @return 峰值内存（字节）
     */
    size_t getPeak() const;
    
    /**
     * @brief 设置内存超限回调函数
     * @param callback 回调函数，当内存超限时被调用
     */
    void setLimitCallback(MemoryLimitCallback callback);
    
    /**
     * @brief 重置峰值内存统计
     */
    void resetPeak();
    
    /**
     * @brief 重置所有统计信息（包括已使用内存和峰值）
     */
    void resetAll();
    
private:
    /**
     * @brief 私有构造函数，实现单例模式
     */
    MemoryMonitor();
    
    MemoryMonitor(const MemoryMonitor&) = delete;
    MemoryMonitor& operator=(const MemoryMonitor&) = delete;
    
    std::atomic<size_t> usedMemory_;     ///< 当前使用的内存（字节）
    std::atomic<size_t> peakMemory_;     ///< 峰值内存（字节）
    std::atomic<size_t> memoryLimit_;    ///< 内存限制（字节）
    MemoryLimitCallback limitCallback_;  ///< 内存超限回调函数
};

}  // namespace cllm