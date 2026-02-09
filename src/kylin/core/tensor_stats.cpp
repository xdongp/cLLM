/**
 * @file tensor_stats.cpp
 * @brief GGML 张量统计工具实现
 */

#include "cllm/kylin/core/tensor_stats.h"
#include "cllm/common/logger.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace cllm {
namespace kylin {

TensorStats computeTensorStats(const float* data, size_t size, size_t firstN) {
    TensorStats stats;
    stats.minVal = std::numeric_limits<float>::max();
    stats.maxVal = std::numeric_limits<float>::lowest();
    stats.mean = 0.0;
    stats.nanCount = 0;
    stats.infCount = 0;
    stats.zeroCount = 0;
    
    if (size == 0 || !data) return stats;
    
    // 收集有效值
    std::vector<float> validValues;
    validValues.reserve(size);
    double sum = 0.0;
    
    for (size_t i = 0; i < size; ++i) {
        float v = data[i];
        if (std::isnan(v)) {
            stats.nanCount++;
        } else if (std::isinf(v)) {
            stats.infCount++;
        } else {
            validValues.push_back(v);
            stats.minVal = std::min(stats.minVal, v);
            stats.maxVal = std::max(stats.maxVal, v);
            sum += v;
            if (std::abs(v) < kZeroThreshold) stats.zeroCount++;
        }
    }
    
    if (validValues.empty()) {
        return stats;
    }
    
    // 计算均值
    stats.mean = sum / validValues.size();
    
    // 计算标准差
    double variance = 0.0;
    for (float v : validValues) {
        double diff = v - stats.mean;
        variance += diff * diff;
    }
    stats.stddev = std::sqrt(variance / validValues.size());
    
    // 计算分位数
    std::sort(validValues.begin(), validValues.end());
    auto percentile = [&](double p) -> float {
        size_t idx = static_cast<size_t>(p * (validValues.size() - 1));
        return validValues[idx];
    };
    stats.percentiles = {
        percentile(0.25),  // 25%
        percentile(0.50),  // 50% (median)
        percentile(0.75),  // 75%
        percentile(0.95),  // 95%
        percentile(0.99)   // 99%
    };
    
    // 保存前 N 个值
    stats.firstValues.resize(std::min(firstN, size));
    for (size_t i = 0; i < stats.firstValues.size(); ++i) {
        stats.firstValues[i] = data[i];
    }
    
    return stats;
}

TensorStats computeTensorStats(const ggml_tensor* tensor, size_t firstN) {
    TensorStats stats;
    if (!tensor || !tensor->data || tensor->type != GGML_TYPE_F32) {
        return stats;
    }
    
    size_t size = computeTensorElementCount(tensor);
    if (size == 0 || size > kMaxValidDataSize) {
        return stats;
    }
    
    const float* data = static_cast<const float*>(tensor->data);
    return computeTensorStats(data, size, firstN);
}

void printTensorStats(const char* name, const ggml_tensor* tensor, const TensorStats& stats) {
    if (!tensor || !tensor->data) return;
    
    CLLM_INFO("[Kylin Debug] %s stats:", name);
    CLLM_INFO("  Shape: [%lld, %lld, %lld, %lld]", 
               tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);
    CLLM_INFO("  Type: %s", ggml_type_name(tensor->type));
    CLLM_INFO("  Min: %.6f, Max: %.6f, Mean: %.6f, StdDev: %.6f",
               stats.minVal, stats.maxVal, stats.mean, stats.stddev);
    CLLM_INFO("  NaN: %zu, Inf: %zu, Zero: %zu",
               stats.nanCount, stats.infCount, stats.zeroCount);
    
    if (stats.percentiles.size() >= 5) {
        CLLM_INFO("  Percentiles: P25=%.6f, P50=%.6f, P75=%.6f, P95=%.6f, P99=%.6f",
                   stats.percentiles[0], stats.percentiles[1], stats.percentiles[2],
                   stats.percentiles[3], stats.percentiles[4]);
    }
    
    if (!stats.firstValues.empty()) {
        std::string valuesStr;
        for (size_t i = 0; i < stats.firstValues.size(); ++i) {
            if (i > 0) valuesStr += " ";
            valuesStr += std::to_string(stats.firstValues[i]);
        }
        CLLM_INFO("  First %zu values: %s", stats.firstValues.size(), valuesStr.c_str());
    }
}

bool validateTensorData(const ggml_tensor* tensor) {
    if (!tensor || !tensor->data) return false;
    if (tensor->type != GGML_TYPE_F32) return true;  // 非 F32 不做验证
    
    size_t size = computeTensorElementCount(tensor);
    if (size == 0 || size > kMaxValidDataSize) return false;
    
    const float* data = static_cast<const float*>(tensor->data);
    for (size_t i = 0; i < size; ++i) {
        if (std::isnan(data[i]) || std::isinf(data[i])) {
            return false;
        }
    }
    return true;
}

size_t computeTensorElementCount(const ggml_tensor* tensor) {
    if (!tensor) return 0;
    
    size_t size = 1;
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (tensor->ne[i] > 0 && tensor->ne[i] < static_cast<int64_t>(kMaxValidDimension)) {
            size *= tensor->ne[i];
        }
    }
    return size;
}

bool safeComputeTensorStats(const ggml_tensor* tensor, TensorStats& stats) {
    try {
        if (!tensor || !tensor->data || tensor->type != GGML_TYPE_F32) {
            return false;
        }
        
        // 维度边界检查
        for (int i = 0; i < GGML_MAX_DIMS; ++i) {
            if (tensor->ne[i] < 0 || tensor->ne[i] > static_cast<int64_t>(kMaxValidDimension)) {
                return false;
            }
        }
        
        size_t size = computeTensorElementCount(tensor);
        if (size == 0 || size > kMaxValidDataSize) {
            return false;
        }
        
        const float* data = static_cast<const float*>(tensor->data);
        stats = computeTensorStats(data, size, kDefaultFirstN);
        return true;
    } catch (const std::exception& e) {
        CLLM_DEBUG("[TensorStats] Exception: %s", e.what());
        return false;
    }
}

} // namespace kylin
} // namespace cllm
