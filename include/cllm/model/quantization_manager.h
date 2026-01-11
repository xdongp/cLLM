/**
 * @file quantization_manager.h
 * @brief 模型量化管理器，负责模型权重的量化和反量化
 * @author cLLM Team
 * @date 2026-01-08
 */
#ifndef CLLM_MODEL_QUANTIZATION_MANAGER_H
#define CLLM_MODEL_QUANTIZATION_MANAGER_H

#include "cllm/memory/float_array.h"
#include <string>

namespace cllm {

/**
 * @brief 量化统计信息
 */
struct QuantizationStats {
    size_t originalSize;      // 原始大小（字节）
    size_t quantizedSize;     // 量化后大小（字节）
    float compressionRatio;   // 压缩比
    float quantizationError;  // 量化误差
    
    /**
     * @brief 默认构造函数
     */
    QuantizationStats()
        : originalSize(0), quantizedSize(0), compressionRatio(1.0f), quantizationError(0.0f) {}
};

/**
 * @brief 量化管理器类，负责模型权重的量化和反量化
 */
class QuantizationManager {
public:
    /**
     * @brief 构造函数
     * @param quantizationType 量化类型（支持int8, int4）
     */
    explicit QuantizationManager(const std::string& quantizationType);
    
    /**
     * @brief 析构函数
     */
    ~QuantizationManager();
    
    /**
     * @brief 量化模型
     * @param modelWeights 模型权重
     * @param modelSize 模型大小（字节）
     */
    void quantizeModel(void* modelWeights, size_t modelSize);
    
    /**
     * @brief 反量化模型
     * @param quantizedWeights 量化后的权重
     * @param output 输出缓冲区
     * @param size 输出大小（字节）
     */
    void dequantizeModel(void* quantizedWeights, void* output, size_t size);
    
    /**
     * @brief 量化张量
     * @param tensor 输入张量
     * @param size 张量大小
     * @return 量化后的张量
     */
    FloatArray quantizeTensor(FloatArray& tensor, size_t size);
    
    /**
     * @brief 反量化张量
     * @param quantized 量化后的张量
     * @param size 张量大小
     * @return 反量化后的张量
     */
    FloatArray dequantizeTensor(void* quantized, size_t size);
    
    /**
     * @brief 获取量化统计信息
     * @return 量化统计信息
     */
    QuantizationStats getStats() const;
    
private:
    /**
     * @brief 应用int8量化
     */
    void _applyInt8Quantization();
    
    /**
     * @brief 应用int4量化
     */
    void _applyInt4Quantization();
    
    std::string quantizationType_;  // 量化类型
    QuantizationStats stats_;       // 量化统计信息
};

}  // namespace cllm

#endif  // CLLM_MODEL_QUANTIZATION_MANAGER_H