#include <gtest/gtest.h>
#include "cllm/inference/kylin_backend.h"
#include "cllm/model/config.h"
#include "cllm/kylin/tensor.h"

using namespace cllm;
using namespace cllm::inference;

class KylinBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 设置测试用的模型配置
        config_.vocabSize = 1000;      // 较小的词汇表用于测试
        config_.hiddenSize = 128;      // 使用较小的隐藏层尺寸
        config_.numLayers = 2;         // 使用较少层数
        config_.numAttentionHeads = 4; // 较少注意力头数
        config_.intermediateSize = 256; // 中间层大小
        config_.maxSequenceLength = 512;
    }

    ModelConfig config_;
};

// 测试构造函数
TEST_F(KylinBackendTest, ConstructorTest) {
    // 测试占位权重模式（空路径）
    EXPECT_NO_THROW({
        KylinBackend backend(config_);
    });

    // 测试带路径的构造（但路径不存在，应该使用占位权重模式）
    EXPECT_NO_THROW({
        KylinBackend backend(config_, "/nonexistent/path.bin");
    });
}

// 测试初始化功能
TEST_F(KylinBackendTest, InitializationTest) {
    KylinBackend backend(config_);
    
    // 验证初始化前状态
    EXPECT_FALSE(backend.isInitialized());
    EXPECT_EQ(backend.getName(), "Kylin");
    
    // 尝试初始化
    EXPECT_TRUE(backend.initialize());
    
    // 验证初始化后状态
    EXPECT_TRUE(backend.isInitialized());
    EXPECT_EQ(backend.getConfig().vocabSize, config_.vocabSize);
}

// 测试单序列推理功能
TEST_F(KylinBackendTest, ForwardTest) {
    KylinBackend backend(config_);
    ASSERT_TRUE(backend.initialize());
    
    // 测试简单的输入序列
    std::vector<int> inputIds = {1, 2, 3, 4};
    
    EXPECT_NO_THROW({
        auto result = backend.forward(inputIds);
        
        // 验证输出张量的形状
        auto shape = result.shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], inputIds.size());  // 序列长度
        EXPECT_EQ(shape[1], config_.vocabSize); // 词汇表大小
    });
}

// 测试批处理推理功能
TEST_F(KylinBackendTest, ForwardBatchTest) {
    KylinBackend backend(config_);
    ASSERT_TRUE(backend.initialize());
    
    // 准备批处理输入：两个请求，每个请求有不同的长度
    std::vector<int> flatInputIds = {1, 2, 3, 4, 5, 6};  // 总共6个token
    std::vector<std::pair<size_t, size_t>> requestPositions = {
        {0, 3},  // 第一个请求：token [1, 2, 3]
        {3, 6}   // 第二个请求：token [4, 5, 6]
    };
    size_t batchSize = 2;
    
    EXPECT_NO_THROW({
        auto result = backend.forwardBatch(flatInputIds, requestPositions, batchSize);
        
        // 验证输出张量的形状
        auto shape = result.shape();
        EXPECT_EQ(shape.size(), 2);
        EXPECT_EQ(shape[0], flatInputIds.size());  // 总token数
        EXPECT_EQ(shape[1], config_.vocabSize);     // 词汇表大小
    });
}

// 测试错误处理
TEST_F(KylinBackendTest, ErrorHandlingTest) {
    KylinBackend backend(config_);
    
    // 测试未初始化时调用forward
    std::vector<int> inputIds = {1, 2, 3};
    EXPECT_THROW({
        backend.forward(inputIds);
    }, std::runtime_error);
    
    // 测试未初始化时调用forwardBatch
    std::vector<int> flatInputIds = {1, 2, 3};
    std::vector<std::pair<size_t, size_t>> requestPositions = {{0, 3}};
    EXPECT_THROW({
        backend.forwardBatch(flatInputIds, requestPositions, 1);
    }, std::runtime_error);
    
    // 初始化后再测试
    ASSERT_TRUE(backend.initialize());
    
    // 测试无效的批处理参数
    EXPECT_THROW({
        backend.forwardBatch({}, {}, 0);  // 空输入，batchSize=0
    }, std::invalid_argument);
    
    EXPECT_THROW({
        backend.forwardBatch({1, 2, 3}, {{0, 3}}, 2);  // 位置数量与batchSize不匹配
    }, std::invalid_argument);
    
    EXPECT_THROW({
        backend.forwardBatch({1, 2, 3}, {{0, 5}}, 1);  // 位置超出范围
    }, std::out_of_range);
}

// 测试不同配置的初始化
TEST_F(KylinBackendTest, DifferentConfigurationsTest) {
    // 测试不同的配置参数
    ModelConfig smallConfig = config_;
    smallConfig.hiddenSize = 64;
    smallConfig.intermediateSize = 128;
    smallConfig.numLayers = 1;
    
    KylinBackend backend(smallConfig);
    EXPECT_TRUE(backend.initialize());
    EXPECT_TRUE(backend.isInitialized());
}

// 测试边界情况
TEST_F(KylinBackendTest, EdgeCasesTest) {
    KylinBackend backend(config_);
    ASSERT_TRUE(backend.initialize());
    
    // 测试单个token的输入
    std::vector<int> singleInput = {42};
    EXPECT_NO_THROW({
        auto result = backend.forward(singleInput);
        auto shape = result.shape();
        EXPECT_EQ(shape[0], 1);
        EXPECT_EQ(shape[1], config_.vocabSize);
    });
    
    // 测试空批处理（虽然在实际应用中不太可能出现）
    std::vector<int> emptyFlatInput;
    std::vector<std::pair<size_t, size_t>> emptyPositions;
    EXPECT_THROW({
        backend.forwardBatch(emptyFlatInput, emptyPositions, 0);
    }, std::invalid_argument);
    
    // 测试包含空请求的批处理
    std::vector<int> mixedInput = {1, 2};
    std::vector<std::pair<size_t, size_t>> mixedPositions = {
        {0, 2},  // 有效请求
        {2, 2}   // 空请求（start == end）
    };
    EXPECT_NO_THROW({
        auto result = backend.forwardBatch(mixedInput, mixedPositions, 2);
        auto shape = result.shape();
        EXPECT_EQ(shape[0], mixedInput.size());  // 应该等于非空请求的token总数
        EXPECT_EQ(shape[1], config_.vocabSize);
    });
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}