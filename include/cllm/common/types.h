/**
 * @file types.h
 * @brief 通用类型定义和基础数据结构
 * @author cLLM Team
 * @date 2024-01-01
 */

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace cllm {

/// 请求ID类型
typedef size_t RequestId_t;
/// 时间戳类型
typedef size_t Timestamp_t;
/// 内存大小类型
typedef size_t MemorySize_t;

// FloatArray has been moved to memory/float_array.h

/**
 * @brief 整数数组RAII包装器
 * 
 * 提供对int数组的RAII管理，支持动态增长和容量管理。
 */
struct IntArray {
    int* data;         ///< 数组数据指针
    size_t size;       ///< 当前元素个数
    size_t capacity;   ///< 已分配的容量
    
    /**
     * @brief 构造函数
     * @param initialCapacity 初始容量，默认为0
     */
    explicit IntArray(size_t initialCapacity = 0);
    
    /**
     * @brief 析构函数，自动释放内存
     */
    ~IntArray();
    
    /**
     * @brief 拷贝构造函数
     * @param other 要拷贝的对象
     */
    IntArray(const IntArray& other);
    
    /**
     * @brief 拷贝赋值运算符
     * @param other 要拷贝的对象
     * @return 当前对象的引用
     */
    IntArray& operator=(const IntArray& other);
    
    /**
     * @brief 移动构造函数
     * @param other 要移动的对象
     */
    IntArray(IntArray&& other) noexcept;
    
    /**
     * @brief 移动赋值运算符
     * @param other 要移动的对象
     * @return 当前对象的引用
     */
    IntArray& operator=(IntArray&& other) noexcept;
    
    /**
     * @brief 重新设置数组大小
     * @param newSize 新的大小
     */
    void resize(size_t newSize);
    
    /**
     * @brief 预留容量
     * @param newCapacity 新的容量
     */
    void reserve(size_t newCapacity);
    
    /**
     * @brief 在末尾添加元素
     * @param value 要添加的值
     */
    void push_back(int value);
    
    /**
     * @brief 数组下标访问运算符
     * @param index 索引
     * @return 对应位置的元素引用
     */
    int& operator[](size_t index);
    
    /**
     * @brief 常量数组下标访问运算符
     * @param index 索引
     * @return 对应位置的元素常量引用
     */
    const int& operator[](size_t index) const;
    
    /**
     * @brief 获取数组指针
     * @return int数组指针
     */
    int* get() { return data; }
    
    /**
     * @brief 获取常量数组指针
     * @return const int数组指针
     */
    const int* get() const { return data; }
    
    /**
     * @brief 获取当前元素个数
     * @return 元素个数
     */
    size_t getSize() const { return size; }
    
    /**
     * @brief 获取已分配的容量
     * @return 容量
     */
    size_t getCapacity() const { return capacity; }
    
    /**
     * @brief 清空数组
     */
    void clear();
    
    /**
     * @brief 判断数组是否为空
     * @return true 如果为空，false 否则
     */
    bool empty() const { return size == 0; }
};

}  // namespace cllm
