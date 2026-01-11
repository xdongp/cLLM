/**
 * @file inference_optimizer.h
 * @brief 推理优化器，负责应用各种推理优化
 * @author cLLM Team
 * @date 2026-01-08
 */
#ifndef CLLM_MODEL_INFERENCE_OPTIMIZER_H
#define CLLM_MODEL_INFERENCE_OPTIMIZER_H

#include <string>

namespace cllm {

/**
 * @brief 优化器统计信息
 */
struct OptimizerStats {
    size_t optimizedOperations;  // 优化的操作数量
    float speedupFactor;         // 加速因子
    size_t memorySaved;          // 节省的内存（字节）
    
    /**
     * @brief 默认构造函数
     */
    OptimizerStats()
        : optimizedOperations(0), speedupFactor(1.0f), memorySaved(0) {}
};

/**
 * @brief SIMD推理优化器，负责应用SIMD优化
 */
class SIMDInferenceOptimizer {
public:
    /**
     * @brief 应用AVX2优化
     */
    void applyAVX2Optimizations();
    
    /**
     * @brief 应用AVX512优化
     */
    void applyAVX512Optimizations();
    
    /**
     * @brief 优化矩阵乘法
     * @param A 矩阵A
     * @param B 矩阵B
     * @param C 输出矩阵C
     * @param M 矩阵A的行数
     * @param N 矩阵B的列数
     * @param K 矩阵A的列数和矩阵B的行数
     */
    void optimizeMatMul(float* A, float* B, float* C, size_t M, size_t N, size_t K);
    
    /**
     * @brief 优化激活函数
     * @param input 输入张量
     * @param output 输出张量
     * @param size 张量大小
     */
    void optimizeActivation(float* input, float* output, size_t size);
    
private:
    /**
     * @brief 检查是否支持AVX2
     * @return 是否支持AVX2
     */
    bool _hasAVX2();
    
    /**
     * @brief 检查是否支持AVX512
     * @return 是否支持AVX512
     */
    bool _hasAVX512();
};

/**
 * @brief 推理优化器类，负责应用各种推理优化
 */
class InferenceOptimizer {
public:
    /**
     * @brief 构造函数
     * @param enableSIMD 是否启用SIMD优化
     */
    explicit InferenceOptimizer(bool enableSIMD = true);
    
    /**
     * @brief 析构函数
     */
    ~InferenceOptimizer();
    
    /**
     * @brief 应用优化
     */
    void applyOptimizations();
    
    /**
     * @brief 优化内存布局
     */
    void optimizeMemoryLayout();
    
    /**
     * @brief 优化计算图
     */
    void optimizeComputeGraph();
    
    /**
     * @brief 启用SIMD优化
     */
    void enableSIMDOptimizations();
    
    /**
     * @brief 启用内存优化
     */
    void enableMemoryOptimizations();
    
    /**
     * @brief 获取优化器统计信息
     * @return 优化器统计信息
     */
    OptimizerStats getStats() const;
    
private:
    /**
     * @brief 应用AVX2优化
     */
    void _applyAVX2Optimizations();
    
    /**
     * @brief 应用AVX512优化
     */
    void _applyAVX512Optimizations();
    
    bool enableSIMD_;          // 是否启用SIMD优化
    OptimizerStats stats_;     // 优化器统计信息
    SIMDInferenceOptimizer simdOptimizer_;  // SIMD优化器
};

}  // namespace cllm

#endif  // CLLM_MODEL_INFERENCE_OPTIMIZER_H