/**
 * @file tensor_stats.h
 * @brief GGML 张量统计工具
 * 
 * 提供张量数据的统计分析功能，用于调试和验证
 */
#pragma once

#include <ggml.h>
#include <string>
#include <vector>
#include <cstddef>

namespace cllm {
namespace kylin {

/**
 * @brief 张量统计信息
 */
struct TensorStats {
    float minVal = 0.0f;           ///< 最小值
    float maxVal = 0.0f;           ///< 最大值
    double mean = 0.0;             ///< 均值
    double stddev = 0.0;           ///< 标准差
    size_t nanCount = 0;           ///< NaN 数量
    size_t infCount = 0;           ///< Inf 数量
    size_t zeroCount = 0;          ///< 零值数量
    std::vector<float> percentiles; ///< 分位数 (25%, 50%, 75%, 95%, 99%)
    std::vector<float> firstValues; ///< 前 N 个值
    
    /**
     * @brief 检查统计是否有效（无 NaN/Inf）
     */
    bool isValid() const { return nanCount == 0 && infCount == 0; }
};

// ========== 常量定义 ==========

/// 默认收集的前 N 个值数量
constexpr size_t kDefaultFirstN = 10;

/// 最大有效数据大小（防止溢出）
constexpr size_t kMaxValidDataSize = 10000000;

/// 最大有效维度值（vocab_size 可能超过 100000）
constexpr size_t kMaxValidDimension = 200000;

/// 零值判定阈值
constexpr float kZeroThreshold = 1e-9f;

/// KV Cache 差异判定阈值
constexpr float kKVDiffThreshold = 1e-3f;

// ========== 统计计算函数 ==========

/**
 * @brief 计算浮点数据的统计信息
 * @param data 数据指针
 * @param size 数据大小
 * @param firstN 收集前 N 个值的数量
 * @return 统计信息
 */
TensorStats computeTensorStats(const float* data, size_t size, size_t firstN = kDefaultFirstN);

/**
 * @brief 计算 GGML 张量的统计信息
 * @param tensor 张量指针
 * @param firstN 收集前 N 个值的数量
 * @return 统计信息
 */
TensorStats computeTensorStats(const ggml_tensor* tensor, size_t firstN = kDefaultFirstN);

/**
 * @brief 打印张量统计信息到日志
 * @param name 张量名称
 * @param tensor 张量指针
 * @param stats 统计信息
 */
void printTensorStats(const char* name, const ggml_tensor* tensor, const TensorStats& stats);

/**
 * @brief 验证张量数据有效性
 * @param tensor 张量指针
 * @return true 如果数据有效（无 NaN/Inf）
 */
bool validateTensorData(const ggml_tensor* tensor);

/**
 * @brief 计算张量有效元素数量
 * @param tensor 张量指针
 * @return 元素数量，如果无效返回 0
 */
size_t computeTensorElementCount(const ggml_tensor* tensor);

/**
 * @brief 安全计算张量统计（带边界检查）
 * @param tensor 张量指针
 * @param stats 输出统计信息
 * @return true 如果计算成功
 */
bool safeComputeTensorStats(const ggml_tensor* tensor, TensorStats& stats);

} // namespace kylin
} // namespace cllm
