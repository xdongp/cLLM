/**
 * @file test_weight_data.cpp
 * @brief 通用权重数据结构测试
 * @author cLLM Team
 * @date 2026-01-13
 */

#include "cllm/model/weight_data.h"
#include <gtest/gtest.h>

namespace cllm {
namespace model {

TEST(WeightDataTest, BasicFunctionality) {
    // 测试WeightData基本功能
    WeightData weight;
    weight.name = "test.weight";
    weight.shape = {2, 3};
    weight.data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    weight.dtype = kylin::WeightDType::FP32;
    
    EXPECT_EQ(weight.name, "test.weight");
    EXPECT_EQ(weight.shape.size(), 2);
    EXPECT_EQ(weight.shape[0], 2);
    EXPECT_EQ(weight.shape[1], 3);
    EXPECT_EQ(weight.data.size(), 6);
    EXPECT_EQ(weight.dtype, kylin::WeightDType::FP32);
    
    EXPECT_EQ(weight.size(), 6);
    EXPECT_EQ(weight.elementCount(), 6);
    EXPECT_TRUE(weight.isValid());
}

TEST(ModelWeightsTest, FindWeight) {
    // 测试ModelWeights的findWeight方法
    ModelWeights weights;
    
    // 设置embedding
    weights.embedding.name = "embedding";
    weights.embedding.shape = {10000, 768};
    weights.embedding.data.resize(10000 * 768);
    weights.embedding.dtype = kylin::WeightDType::FP32;
    
    // 设置finalNorm
    weights.finalNorm.name = "finalNorm";
    weights.finalNorm.shape = {768};
    weights.finalNorm.data.resize(768);
    weights.finalNorm.dtype = kylin::WeightDType::FP32;
    
    // 设置lmHead
    weights.lmHead.name = "lmHead";
    weights.lmHead.shape = {768, 10000};
    weights.lmHead.data.resize(768 * 10000);
    weights.lmHead.dtype = kylin::WeightDType::FP32;
    
    // 添加一层
    weights.layers.resize(1);
    LayerWeights& layer = weights.layers[0];
    layer.wq.name = "layers.0.attention.wq.weight";
    layer.wk.name = "layers.0.attention.wk.weight";
    layer.wv.name = "layers.0.attention.wv.weight";
    layer.wo.name = "layers.0.attention.wo.weight";
    layer.wGate.name = "layers.0.feed_forward.wGate.weight";
    layer.wUp.name = "layers.0.feed_forward.wUp.weight";
    layer.wDown.name = "layers.0.feed_forward.wDown.weight";
    layer.norm1.name = "layers.0.attention_norm.weight";
    layer.norm2.name = "layers.0.ffn_norm.weight";
    
    // 为每层权重分配数据
    auto setupLayerWeight = [](WeightData& weight, const std::string& name, const std::vector<size_t>& shape) {
        weight.name = name;
        weight.shape = shape;
        
        // 计算元素总数
        size_t elementCount = 1;
        for (size_t dim : shape) {
            elementCount *= dim;
        }
        
        weight.data.resize(elementCount);
        weight.dtype = kylin::WeightDType::FP32;
    };
    
    setupLayerWeight(layer.wq, "layers.0.attention.wq.weight", {768, 768});
    setupLayerWeight(layer.wk, "layers.0.attention.wk.weight", {768, 768});
    setupLayerWeight(layer.wv, "layers.0.attention.wv.weight", {768, 768});
    setupLayerWeight(layer.wo, "layers.0.attention.wo.weight", {768, 768});
    setupLayerWeight(layer.wGate, "layers.0.feed_forward.wGate.weight", {768, 3072});
    setupLayerWeight(layer.wUp, "layers.0.feed_forward.wUp.weight", {768, 3072});
    setupLayerWeight(layer.wDown, "layers.0.feed_forward.wDown.weight", {3072, 768});
    setupLayerWeight(layer.norm1, "layers.0.attention_norm.weight", {768});
    setupLayerWeight(layer.norm2, "layers.0.ffn_norm.weight", {768});
    
    // 更新权重映射
    weights.updateWeightMap();
    
    // 测试findWeight方法
    EXPECT_EQ(weights.findWeight("embedding"), &weights.embedding);
    EXPECT_EQ(weights.findWeight("finalNorm"), &weights.finalNorm);
    EXPECT_EQ(weights.findWeight("model.norm.weight"), &weights.finalNorm);
    EXPECT_EQ(weights.findWeight("lmHead"), &weights.lmHead);
    EXPECT_EQ(weights.findWeight("lm_head.weight"), &weights.lmHead);
    EXPECT_EQ(weights.findWeight("layers.0.attention.wq.weight"), &layer.wq);
    EXPECT_EQ(weights.findWeight("layers.0.wq.weight"), &layer.wq);
    EXPECT_EQ(weights.findWeight("layers.0.attention.wk.weight"), &layer.wk);
    EXPECT_EQ(weights.findWeight("layers.0.attention.wv.weight"), &layer.wv);
    EXPECT_EQ(weights.findWeight("layers.0.attention.wo.weight"), &layer.wo);
    EXPECT_EQ(weights.findWeight("layers.0.feed_forward.wGate.weight"), &layer.wGate);
    EXPECT_EQ(weights.findWeight("layers.0.feed_forward.wUp.weight"), &layer.wUp);
    EXPECT_EQ(weights.findWeight("layers.0.feed_forward.wDown.weight"), &layer.wDown);
    EXPECT_EQ(weights.findWeight("layers.0.attention_norm.weight"), &layer.norm1);
    EXPECT_EQ(weights.findWeight("layers.0.norm1"), &layer.norm1);
    EXPECT_EQ(weights.findWeight("layers.0.ffn_norm.weight"), &layer.norm2);
    EXPECT_EQ(weights.findWeight("layers.0.norm2"), &layer.norm2);
    
    // 测试不存在的权重
    EXPECT_EQ(weights.findWeight("non_existent.weight"), nullptr);
}

TEST(ModelWeightsTest, GetAllWeights) {
    // 测试ModelWeights的getAllWeights方法
    ModelWeights weights;
    
    // 设置embedding
    weights.embedding.name = "embedding";
    weights.embedding.shape = {10000, 768};
    weights.embedding.data.resize(10000 * 768);
    weights.embedding.dtype = kylin::WeightDType::FP32;
    
    // 设置finalNorm
    weights.finalNorm.name = "finalNorm";
    weights.finalNorm.shape = {768};
    weights.finalNorm.data.resize(768);
    weights.finalNorm.dtype = kylin::WeightDType::FP32;
    
    // 设置lmHead
    weights.lmHead.name = "lmHead";
    weights.lmHead.shape = {768, 10000};
    weights.lmHead.data.resize(768 * 10000);
    weights.lmHead.dtype = kylin::WeightDType::FP32;
    
    // 添加一层
    weights.layers.resize(1);
    LayerWeights& layer = weights.layers[0];
    
    // 为每层权重分配数据
    auto setupLayerWeight = [](WeightData& weight, const std::string& name, const std::vector<size_t>& shape) {
        weight.name = name;
        weight.shape = shape;
        
        // 计算元素总数
        size_t elementCount = 1;
        for (size_t dim : shape) {
            elementCount *= dim;
        }
        
        weight.data.resize(elementCount);
        weight.dtype = kylin::WeightDType::FP32;
    };
    
    setupLayerWeight(layer.wq, "layers.0.attention.wq.weight", {768, 768});
    setupLayerWeight(layer.wk, "layers.0.attention.wk.weight", {768, 768});
    setupLayerWeight(layer.wv, "layers.0.attention.wv.weight", {768, 768});
    setupLayerWeight(layer.wo, "layers.0.attention.wo.weight", {768, 768});
    setupLayerWeight(layer.wGate, "layers.0.feed_forward.wGate.weight", {768, 3072});
    setupLayerWeight(layer.wUp, "layers.0.feed_forward.wUp.weight", {768, 3072});
    setupLayerWeight(layer.wDown, "layers.0.feed_forward.wDown.weight", {3072, 768});
    setupLayerWeight(layer.norm1, "layers.0.attention_norm.weight", {768});
    setupLayerWeight(layer.norm2, "layers.0.ffn_norm.weight", {768});
    
    // 更新权重映射
    weights.updateWeightMap();
    
    // 测试getAllWeights方法
    auto weightsMap = weights.getAllWeights();
    
    // 检查关键权重是否存在
    EXPECT_TRUE(weightsMap.find("embedding") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("finalNorm") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("model.norm.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("lmHead") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("lm_head.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.attention.wq.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.wq.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.attention.wk.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.attention.wv.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.attention.wo.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.feed_forward.wGate.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.feed_forward.wUp.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.feed_forward.wDown.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.attention_norm.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.norm1") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.ffn_norm.weight") != weightsMap.end());
    EXPECT_TRUE(weightsMap.find("layers.0.norm2") != weightsMap.end());
}

TEST(ModelWeightsTest, IsValid) {
    // 测试ModelWeights的isValid方法
    ModelWeights weights;
    
    // 设置embedding
    weights.embedding.name = "embedding";
    weights.embedding.shape = {10000, 768};
    weights.embedding.data.resize(10000 * 768);
    weights.embedding.dtype = kylin::WeightDType::FP32;
    
    // 设置finalNorm
    weights.finalNorm.name = "finalNorm";
    weights.finalNorm.shape = {768};
    weights.finalNorm.data.resize(768);
    weights.finalNorm.dtype = kylin::WeightDType::FP32;
    
    // 设置lmHead
    weights.lmHead.name = "lmHead";
    weights.lmHead.shape = {768, 10000};
    weights.lmHead.data.resize(768 * 10000);
    weights.lmHead.dtype = kylin::WeightDType::FP32;
    
    // 添加一层
    weights.layers.resize(1);
    LayerWeights& layer = weights.layers[0];
    
    // 为每层权重分配数据
    auto setupLayerWeight = [](WeightData& weight, const std::string& name, const std::vector<size_t>& shape) {
        weight.name = name;
        weight.shape = shape;
        
        // 计算元素总数
        size_t elementCount = 1;
        for (size_t dim : shape) {
            elementCount *= dim;
        }
        
        weight.data.resize(elementCount);
        weight.dtype = kylin::WeightDType::FP32;
    };
    
    setupLayerWeight(layer.wq, "layers.0.attention.wq.weight", {768, 768});
    setupLayerWeight(layer.wk, "layers.0.attention.wk.weight", {768, 768});
    setupLayerWeight(layer.wv, "layers.0.attention.wv.weight", {768, 768});
    setupLayerWeight(layer.wo, "layers.0.attention.wo.weight", {768, 768});
    setupLayerWeight(layer.wGate, "layers.0.feed_forward.wGate.weight", {768, 3072});
    setupLayerWeight(layer.wUp, "layers.0.feed_forward.wUp.weight", {768, 3072});
    setupLayerWeight(layer.wDown, "layers.0.feed_forward.wDown.weight", {3072, 768});
    setupLayerWeight(layer.norm1, "layers.0.attention_norm.weight", {768});
    setupLayerWeight(layer.norm2, "layers.0.ffn_norm.weight", {768});
    
    // 测试所有权重都有效的情况
    EXPECT_TRUE(weights.isValid());
    
    // 测试某一层权重无效的情况
    layer.wq.data.clear();
    EXPECT_FALSE(weights.isValid());
}

} // namespace model
} // namespace cllm

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
