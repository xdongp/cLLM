/**
 * @file weight_data.cpp
 * @brief 通用权重数据结构实现
 * @author cLLM Team
 * @date 2026-01-13
 */

#include "cllm/model/weight_data.h"
#include "cllm/common/logger.h"
#include <algorithm>

namespace cllm {
namespace model {

/**
 * @brief ModelWeights 构造函数
 */
ModelWeights::ModelWeights() {
    // 初始化权重映射
    updateWeightMap();
}

/**
 * @brief 更新权重映射
 */
void ModelWeights::updateWeightMap() {
    weightMap_.clear();
    
    // 添加embedding
    weightMap_["embedding"] = &embedding;
    
    // 添加finalNorm
    weightMap_["finalNorm"] = &finalNorm;
    weightMap_["model.norm.weight"] = &finalNorm;
    
    // 添加lmHead
    weightMap_["lmHead"] = &lmHead;
    weightMap_["lm_head.weight"] = &lmHead;
    
    // 添加各层权重
    for (size_t i = 0; i < layers.size(); ++i) {
        LayerWeights& layer = layers[i];
        std::string layerPrefix = "layers." + std::to_string(i) + ".";
        
        weightMap_[layerPrefix + "attention.wq.weight"] = &layer.wq;
        weightMap_[layerPrefix + "wq.weight"] = &layer.wq;
        
        weightMap_[layerPrefix + "attention.wk.weight"] = &layer.wk;
        weightMap_[layerPrefix + "wk.weight"] = &layer.wk;
        
        weightMap_[layerPrefix + "attention.wv.weight"] = &layer.wv;
        weightMap_[layerPrefix + "wv.weight"] = &layer.wv;
        
        weightMap_[layerPrefix + "attention.wo.weight"] = &layer.wo;
        weightMap_[layerPrefix + "wo.weight"] = &layer.wo;
        
        weightMap_[layerPrefix + "feed_forward.wGate.weight"] = &layer.wGate;
        weightMap_[layerPrefix + "wGate.weight"] = &layer.wGate;
        
        weightMap_[layerPrefix + "feed_forward.wUp.weight"] = &layer.wUp;
        weightMap_[layerPrefix + "wUp.weight"] = &layer.wUp;
        
        weightMap_[layerPrefix + "feed_forward.wDown.weight"] = &layer.wDown;
        weightMap_[layerPrefix + "wDown.weight"] = &layer.wDown;
        
        weightMap_[layerPrefix + "attention_norm.weight"] = &layer.norm1;
        weightMap_[layerPrefix + "norm1"] = &layer.norm1;
        
        weightMap_[layerPrefix + "ffn_norm.weight"] = &layer.norm2;
        weightMap_[layerPrefix + "norm2"] = &layer.norm2;
    }
}

/**
 * @brief 按名称查找权重
 * 
 * @param name 权重名称
 * @return 指向权重数据的指针，如果未找到返回 nullptr
 */
WeightData* ModelWeights::findWeight(const std::string& name) {
    auto it = weightMap_.find(name);
    if (it != weightMap_.end()) {
        return it->second;
    }
    
    CLLM_DEBUG("ModelWeights::findWeight: Weight not found: %s", name.c_str());
    return nullptr;
}

/**
 * @brief 获取所有权重的名称到数据的映射
 * 
 * @return 权重名称到 WeightData 的映射
 */
std::map<std::string, WeightData*> ModelWeights::getAllWeights() const {
    std::map<std::string, WeightData*> weightsMap;
    
    // 将哈希表转换为有序映射
    for (const auto& pair : weightMap_) {
        weightsMap[pair.first] = pair.second;
    }
    
    return weightsMap;
}

/**
 * @brief 获取所有权重的名称到数据副本的映射
 * 
 * @return 权重名称到 WeightData 副本的映射
 */
std::map<std::string, WeightData> ModelWeights::getAllWeightsCopy() const {
    std::map<std::string, WeightData> weightsMap;
    
    // 添加embedding
    weightsMap["embedding"] = embedding;
    
    // 添加finalNorm
    weightsMap["finalNorm"] = finalNorm;
    weightsMap["model.norm.weight"] = finalNorm;
    
    // 添加lmHead
    weightsMap["lmHead"] = lmHead;
    weightsMap["lm_head.weight"] = lmHead;
    
    // 添加各层权重
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerWeights& layer = layers[i];
        std::string layerPrefix = "layers." + std::to_string(i) + ".";
        
        weightsMap[layerPrefix + "attention.wq.weight"] = layer.wq;
        weightsMap[layerPrefix + "wq.weight"] = layer.wq;
        
        weightsMap[layerPrefix + "attention.wk.weight"] = layer.wk;
        weightsMap[layerPrefix + "wk.weight"] = layer.wk;
        
        weightsMap[layerPrefix + "attention.wv.weight"] = layer.wv;
        weightsMap[layerPrefix + "wv.weight"] = layer.wv;
        
        weightsMap[layerPrefix + "attention.wo.weight"] = layer.wo;
        weightsMap[layerPrefix + "wo.weight"] = layer.wo;
        
        weightsMap[layerPrefix + "feed_forward.wGate.weight"] = layer.wGate;
        weightsMap[layerPrefix + "wGate.weight"] = layer.wGate;
        
        weightsMap[layerPrefix + "feed_forward.wUp.weight"] = layer.wUp;
        weightsMap[layerPrefix + "wUp.weight"] = layer.wUp;
        
        weightsMap[layerPrefix + "feed_forward.wDown.weight"] = layer.wDown;
        weightsMap[layerPrefix + "wDown.weight"] = layer.wDown;
        
        weightsMap[layerPrefix + "attention_norm.weight"] = layer.norm1;
        weightsMap[layerPrefix + "norm1"] = layer.norm1;
        
        weightsMap[layerPrefix + "ffn_norm.weight"] = layer.norm2;
        weightsMap[layerPrefix + "norm2"] = layer.norm2;
    }
    
    return weightsMap;
}

/**
 * @brief 检查所有权重是否有效
 * 
 * @return 如果所有权重都有效返回 true，否则返回 false
 */
bool ModelWeights::isValid() const {
    // 检查embedding
    if (!embedding.isValid()) {
        CLLM_DEBUG("ModelWeights::isValid: Invalid embedding");
        return false;
    }
    
    // 检查finalNorm
    if (!finalNorm.isValid()) {
        CLLM_DEBUG("ModelWeights::isValid: Invalid finalNorm");
        return false;
    }
    
    // 检查lmHead
    if (!lmHead.isValid()) {
        CLLM_DEBUG("ModelWeights::isValid: Invalid lmHead");
        return false;
    }
    
    // 检查各层权重
    for (size_t i = 0; i < layers.size(); ++i) {
        const LayerWeights& layer = layers[i];
        
        if (!layer.wq.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wq", i);
            return false;
        }
        if (!layer.wk.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wk", i);
            return false;
        }
        if (!layer.wv.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wv", i);
            return false;
        }
        if (!layer.wo.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wo", i);
            return false;
        }
        if (!layer.wGate.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wGate", i);
            return false;
        }
        if (!layer.wUp.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wUp", i);
            return false;
        }
        if (!layer.wDown.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu wDown", i);
            return false;
        }
        if (!layer.norm1.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu norm1", i);
            return false;
        }
        if (!layer.norm2.isValid()) {
            CLLM_DEBUG("ModelWeights::isValid: Invalid layer %zu norm2", i);
            return false;
        }
    }
    
    return true;
}

} // namespace model
} // namespace cllm
