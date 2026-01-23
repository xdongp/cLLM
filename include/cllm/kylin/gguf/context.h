/**
 * @file ggml_context.h
 * @brief GGML 上下文封装类
 * 
 * 参考文档：Kylin推理引擎设计.md
 * 
 * 封装 GGML 的上下文管理，提供 C++ 友好的接口
 */
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// Metal GPU 支持（仅在 macOS 上可用）
#if defined(__APPLE__) && defined(__MACH__)
#define GGML_USE_METAL
#include "ggml-metal.h"
#endif

// CUDA GPU 支持（仅在有 CUDA 时）
#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief 后端类型枚举
 */
enum class BackendType {
    CPU,      ///< CPU 后端（默认，支持 AVX2/AVX-512/NEON）
    CUDA,     ///< NVIDIA GPU（可选）
    Metal,    ///< Apple GPU（可选）
    Auto      ///< 自动选择最优后端
};

/**
 * @brief GGML 上下文封装类
 * 
 * 职责：
 * - 管理 GGML 上下文生命周期
 * - 提供张量创建的 C++ 友好接口
 * - 管理计算图构建和执行
 * - 支持多后端调度
 */
class GGMLContext {
public:
    /**
     * @brief 构造函数
     * @param memSize 上下文内存大小（字节），默认 512MB
     * @param backend 后端类型，默认 CPU
     */
    explicit GGMLContext(size_t memSize = 512 * 1024 * 1024, BackendType backend = BackendType::CPU);
    
    /**
     * @brief 析构函数
     */
    ~GGMLContext();
    
    // 禁止拷贝
    GGMLContext(const GGMLContext&) = delete;
    GGMLContext& operator=(const GGMLContext&) = delete;
    
    // 允许移动
    GGMLContext(GGMLContext&& other) noexcept;
    GGMLContext& operator=(GGMLContext&& other) noexcept;
    
    // ========== 张量创建 ==========
    
    /**
     * @brief 创建 1D 张量
     * @param type 数据类型
     * @param ne0 第一维大小
     * @return GGML 张量指针
     */
    ggml_tensor* newTensor1D(ggml_type type, int64_t ne0);
    
    /**
     * @brief 创建 2D 张量
     * @param type 数据类型
     * @param ne0 第一维大小
     * @param ne1 第二维大小
     * @return GGML 张量指针
     */
    ggml_tensor* newTensor2D(ggml_type type, int64_t ne0, int64_t ne1);
    
    /**
     * @brief 创建 3D 张量
     * @param type 数据类型
     * @param ne0 第一维大小
     * @param ne1 第二维大小
     * @param ne2 第三维大小
     * @return GGML 张量指针
     */
    ggml_tensor* newTensor3D(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2);
    
    /**
     * @brief 创建 4D 张量
     * @param type 数据类型
     * @param ne0 第一维大小
     * @param ne1 第二维大小
     * @param ne2 第三维大小
     * @param ne3 第四维大小
     * @return GGML 张量指针
     */
    ggml_tensor* newTensor4D(ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3);
    
    // ========== 计算图操作 ==========
    
    /**
     * @brief 构建计算图
     * @param output 输出张量
     * @return 计算图指针
     */
    ggml_cgraph* buildGraph(ggml_tensor* output);
    
    /**
     * @brief 执行计算图
     * @param graph 计算图
     */
    void compute(ggml_cgraph* graph);
    
    /**
     * @brief 执行计算图（使用内部后端）
     * @param graph 计算图
     * @param nThreads 线程数（0 = 自动）
     */
    void computeWithBackend(ggml_cgraph* graph, int nThreads = 0);
    
    // ========== 后端管理 ==========
    
    /**
     * @brief 设置后端类型
     * @param type 后端类型
     */
    void setBackend(BackendType type);
    
    /**
     * @brief 获取当前后端类型
     * @return 后端类型
     */
    BackendType getBackend() const { return backendType_; }
    
    /**
     * @brief 检查 GPU 是否可用
     * @return true 如果 GPU 可用
     */
    static bool isGPUAvailable();
    
    // ========== 工具方法 ==========
    
    /**
     * @brief 获取原始 GGML 上下文指针
     * @return GGML 上下文指针
     */
    ggml_context* raw() { return ctx_; }
    const ggml_context* raw() const { return ctx_; }
    
    /**
     * @brief 获取已使用内存大小
     * @return 已使用内存（字节）
     */
    size_t usedMemory() const;
    
    /**
     * @brief 获取总内存大小
     * @return 总内存（字节）
     */
    size_t totalMemory() const { return memSize_; }
    
    /**
     * @brief 重置上下文（清空所有张量）
     */
    void reset();
    
    /**
     * @brief 将数据类型转换为字符串
     * @param type GGML 数据类型
     * @return 类型名称字符串
     */
    static std::string typeToString(ggml_type type);

private:
    ggml_context* ctx_;                ///< GGML 上下文
    std::vector<uint8_t> buffer_;      ///< 内存缓冲区
    size_t memSize_;                   ///< 内存大小
    BackendType backendType_;          ///< 后端类型
    
    ggml_backend_t backend_;           ///< GGML 后端
    ggml_backend_buffer_t backendBuffer_;  ///< 后端缓冲区
    
    /**
     * @brief 初始化后端
     */
    void initBackend();
    
    /**
     * @brief 释放资源
     */
    void cleanup();
};

} // namespace kylin
} // namespace cllm
