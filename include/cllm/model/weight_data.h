/**
 * @file weight_data.h
 * @brief 通用权重数据结构，后端无关
 * @author cLLM Team
 * @date 2026-01-13
 * 
 * 定义通用的权重数据表示结构，使 ModelLoader 接口完全后端无关。
 * 后端适配器负责将通用权重数据转换为各自的 Tensor 类型。
 */

#ifndef CLLM_MODEL_WEIGHT_DATA_H
#define CLLM_MODEL_WEIGHT_DATA_H

#include "cllm/kylin/model/model_loader.h"  // 用于 WeightDType
#include <string>
#include <vector>
#include <map>
#include <unordered_map>

namespace cllm {
namespace model {

/**
 * @brief 通用的权重数据表示，后端无关
 * 
 * 包含权重的原始数据、形状信息和元数据。
 * 所有后端（Kylin、LibTorch等）都可以从此结构转换到各自的Tensor类型。
 */
struct WeightData {
    std::vector<float> data;           ///< 权重数据（FP32格式，已反量化）
    std::vector<size_t> shape;         ///< 张量形状，例如 [vocabSize, hiddenSize]
    std::string name;                   ///< 权重名称（如 "layers.0.attention.wq.weight"）
    kylin::WeightDType dtype;           ///< 原始数据类型（FP32, FP16, INT8等）
    
    /**
     * @brief 获取数据元素数量
     */
    size_t size() const { return data.size(); }
    
    /**
     * @brief 根据形状计算元素总数
     */
    size_t elementCount() const {
        size_t count = 1;
        for (size_t dim : shape) {
            count *= dim;
        }
        return count;
    }
    
    /**
     * @brief 检查数据是否有效
     */
    bool isValid() const {
        return !data.empty() && !shape.empty() && elementCount() == data.size();
    }
};

/**
 * @brief 每层权重的结构
 * 
 * 包含 Transformer 单层的所有权重。
 */
struct LayerWeights {
    WeightData wq;      ///< Query 权重
    WeightData wk;      ///< Key 权重
    WeightData wv;      ///< Value 权重
    WeightData wo;      ///< Output 权重
    WeightData wGate;   ///< FFN Gate 权重
    WeightData wUp;     ///< FFN Up 权重
    WeightData wDown;   ///< FFN Down 权重
    WeightData norm1;   ///< 注意力层归一化权重
    WeightData norm2;   ///< FFN 层归一化权重
    WeightData attnQNorm;  ///< Q 的独立归一化权重（Qwen3等模型需要）
    WeightData attnKNorm;  ///< K 的独立归一化权重（Qwen3等模型需要）
};

/**
 * @brief 模型权重的完整集合
 * 
 * 包含整个 Transformer 模型的所有权重。
 */
struct ModelWeights {
    WeightData embedding;               ///< Token embedding 权重
    std::vector<LayerWeights> layers;  ///< 各层的权重集合
    WeightData finalNorm;               ///< 最终层归一化权重
    WeightData lmHead;                  ///< 语言模型输出头权重
    
    /**
     * @brief 构造函数
     */
    ModelWeights();
    
    /**
     * @brief 析构函数
     */
    ~ModelWeights() = default;
    
    /**
     * @brief 构造函数（禁用拷贝）
     */
    ModelWeights(const ModelWeights&) = default;
    
    /**
     * @brief 赋值操作符（禁用拷贝）
     */
    ModelWeights& operator=(const ModelWeights&) = default;
    
    /**
     * @brief 按名称查找权重（用于按名称访问的接口）
     * 
     * @param name 权重名称，例如 "layers.0.attention.wq.weight"
     * @return 指向权重数据的指针，如果未找到返回 nullptr
     */
    WeightData* findWeight(const std::string& name);
    
    /**
     * @brief 获取所有权重的名称到数据的映射
     * 
     * @return 权重名称到 WeightData 的映射。
     * @note 返回的指针指向 ModelWeights 对象内部的数据，其生命周期与 ModelWeights 对象相同。
     *       当 ModelWeights 对象被销毁时，所有返回的指针将变为无效（悬垂指针）。
     *       如果需要长期保存权重数据，请使用 getAllWeightsCopy() 方法获取副本。
     */
    std::map<std::string, WeightData*> getAllWeights() const;
    
    /**
     * @brief 获取所有权重的名称到数据副本的映射
     * 
     * @return 权重名称到 WeightData 副本的映射。
     * @note 返回的数据是内部权重数据的副本，可以安全地长期保存，不受 ModelWeights 对象生命周期的影响。
     *       注意：此方法会创建所有权重数据的副本，可能会使用大量内存，仅在必要时使用。
     */
    std::map<std::string, WeightData> getAllWeightsCopy() const;
    
    /**
     * @brief 更新权重映射（当权重结构变化时调用）
     */
    void updateWeightMap();
    
    /**
     * @brief 检查所有权重是否有效
     */
    bool isValid() const;
    
private:
    std::unordered_map<std::string, WeightData*> weightMap_;  ///< 权重名称到数据的哈希表，用于快速查找
};


} // namespace model
} // namespace cllm

#endif // CLLM_MODEL_WEIGHT_DATA_H
