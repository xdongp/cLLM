/**
 * @file test_common_types.h
 * @brief 测试框架公共类型定义
 * 
 * 包含所有测试模块共享的数据结构定义
 */

#pragma once

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>

namespace kylin_test {

// 张量信息结构 - 用于 CPU/GPU 对比测试
struct TensorInfo {
    std::string name;
    std::vector<int> shape;
    std::string dtype;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float mean = 0.0f;
    float std = 0.0f;
    size_t nan_count = 0;
    size_t inf_count = 0;
    std::vector<float> sample_values;
    
    std::string toString() const {
        std::ostringstream oss;
        oss << name << " [";
        for (size_t i = 0; i < shape.size(); ++i) {
            if (i > 0) oss << ", ";
            oss << shape[i];
        }
        oss << "] " << dtype;
        oss << " | min=" << min_val << " max=" << max_val;
        oss << " | mean=" << mean << " std=" << std;
        if (nan_count > 0) oss << " | NaN=" << nan_count;
        if (inf_count > 0) oss << " | Inf=" << inf_count;
        return oss.str();
    }
};

// 辅助函数：将 shape 转换为字符串
inline std::string shapeToString(const std::vector<int>& shape) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) oss << ", ";
        oss << shape[i];
    }
    oss << "]";
    return oss.str();
}

// ============================================================================
// 张量统计信息结构
// ============================================================================
struct TensorStats {
    float min = 0.0f;
    float max = 0.0f;
    float mean = 0.0f;
    float std = 0.0f;
};

// ============================================================================
// 张量对比结果结构
// ============================================================================
struct TensorComparison {
    std::string name;
    TensorStats cpuStats;
    TensorStats gpuStats;
    float maxDiff = 0.0f;
    float meanDiff = 0.0f;
    float rmse = 0.0f;
};

// ============================================================================
// 层输出对比结果
// ============================================================================
struct LayerComparisonResult {
    int layerIdx;
    TensorComparison inputNormComp;
    TensorComparison qkvComp;
    TensorComparison attentionComp;
    TensorComparison postNormComp;
    TensorComparison ffnComp;
    bool hasMismatch = false;
};

} // namespace kylin_test
