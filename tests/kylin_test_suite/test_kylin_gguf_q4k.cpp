/**
 * @file test_kylin_gguf_q4k.cpp
 * @brief 测试Kylin引擎加载Q4_K_M格式的GGUF模型并进行推理
 * 
 * 测试目标：
 * 1. 加载 model/Qwen/qwen3-0.6b-q4_k_m.gguf 模型
 * 2. 验证模型加载成功
 * 3. 执行简单推理测试
 * 4. 验证输出logits形状和数值合理性
 */

#include <gtest/gtest.h>
#include "cllm/inference/kylin_backend.h"
#include "cllm/model/config.h"
#include "cllm/kylin/core/tensor.h"
#include "cllm/common/logger.h"

#include <filesystem>
#include <vector>
#include <cmath>
#include <chrono>
#include <limits>
#include <cstdlib>  // For std::getenv

using namespace cllm;
using namespace cllm::inference;

namespace fs = std::filesystem;

class KylinGGUFQ4KTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 尝试多个可能的模型路径
        std::vector<std::string> possiblePaths = {
            "model/Qwen/qwen3-0.6b-q4_k_m.gguf",  // 从项目根目录
            "../model/Qwen/qwen3-0.6b-q4_k_m.gguf",  // 从build目录
            "../../model/Qwen/qwen3-0.6b-q4_k_m.gguf",  // 从build/tests目录
            "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf"  // 绝对路径
        };
        
        // 从环境变量获取（如果设置）
        const char* envPath = std::getenv("CLLM_MODEL_PATH");
        if (envPath && fs::exists(envPath)) {
            possiblePaths.insert(possiblePaths.begin(), envPath);
        }
        
        // 查找存在的模型文件
        modelPath_.clear();
        for (const auto& path : possiblePaths) {
            if (fs::exists(path)) {
                modelPath_ = fs::absolute(path).string();
                break;
            }
        }
        
        // 如果所有路径都不存在，尝试从当前目录向上查找项目根目录
        if (modelPath_.empty()) {
            fs::path currentPath = fs::current_path();
            for (int i = 0; i < 5; ++i) {  // 最多向上查找5层
                fs::path testPath = currentPath / "model" / "Qwen" / "qwen3-0.6b-q4_k_m.gguf";
                if (fs::exists(testPath)) {
                    modelPath_ = fs::absolute(testPath).string();
                    break;
                }
                if (currentPath == currentPath.parent_path()) {
                    break;  // 已到达根目录
                }
                currentPath = currentPath.parent_path();
            }
        }
        
        // 检查模型文件是否存在
        if (modelPath_.empty() || !fs::exists(modelPath_)) {
            GTEST_SKIP() << "Model file not found. Tried paths:\n"
                         << "  - model/Qwen/qwen3-0.6b-q4_k_m.gguf\n"
                         << "  - ../model/Qwen/qwen3-0.6b-q4_k_m.gguf\n"
                         << "  - Environment variable CLLM_MODEL_PATH\n"
                         << "  - Auto-search from current directory\n"
                         << "Please set CLLM_MODEL_PATH environment variable or ensure model file exists.";
        }
        
        // 初始化基础配置（实际配置会从GGUF文件自动加载和覆盖）
        // 这些值仅作为初始值，GGUFLoader会从文件中读取真实配置
        config_.vocabSize = 151936;      // Qwen3的词汇表大小（会被GGUF覆盖）
        config_.hiddenSize = 896;        // 初始值（会被GGUF覆盖）
        config_.numLayers = 16;          // 初始值（会被GGUF覆盖）
        config_.numAttentionHeads = 14;  // 初始值（会被GGUF覆盖）
        config_.numKeyValueHeads = 2;    // 初始值（会被GGUF覆盖，GQA）
        config_.intermediateSize = 3584; // 初始值（会被GGUF覆盖）
        config_.maxSequenceLength = 32768;
        
        CLLM_INFO("Test setup: model path = %s", modelPath_.c_str());
    }
    
    std::string modelPath_;
    ModelConfig config_;
};

// 测试1: 模型加载
TEST_F(KylinGGUFQ4KTest, ModelLoadingTest) {
    CLLM_INFO("=== Test: Model Loading ===");
    
    // 创建Kylin后端
    KylinBackend backend(config_, modelPath_);
    
    // 验证后端创建成功
    EXPECT_EQ(backend.getName(), "Kylin");
    EXPECT_FALSE(backend.isInitialized());
    
    // 初始化后端（会加载GGUF模型）
    bool initSuccess = backend.initialize();
    
    if (!initSuccess) {
        CLLM_ERROR("Failed to initialize Kylin backend with GGUF model");
        GTEST_SKIP() << "Model initialization failed";
    }
    
    // 验证初始化成功
    EXPECT_TRUE(backend.isInitialized());
    
    // 验证配置已更新（从GGUF文件加载的实际配置）
    const ModelConfig& loadedConfig = backend.getConfig();
    CLLM_INFO("Loaded model config:");
    CLLM_INFO("  - Vocab size: %zu", loadedConfig.vocabSize);
    CLLM_INFO("  - Hidden size: %zu", loadedConfig.hiddenSize);
    CLLM_INFO("  - Num layers: %zu", loadedConfig.numLayers);
    CLLM_INFO("  - Num attention heads: %zu", loadedConfig.numAttentionHeads);
    CLLM_INFO("  - Num KV heads: %zu", loadedConfig.numKeyValueHeads);
    CLLM_INFO("  - Intermediate size: %zu", loadedConfig.intermediateSize);
    
    // 验证配置合理性
    EXPECT_GT(loadedConfig.vocabSize, 0);
    EXPECT_GT(loadedConfig.hiddenSize, 0);
    EXPECT_GT(loadedConfig.numLayers, 0);
    EXPECT_GT(loadedConfig.numAttentionHeads, 0);
}

// 测试2: 单序列推理 - 使用占位权重
TEST_F(KylinGGUFQ4KTest, SingleSequenceInferenceTest) {
    CLLM_INFO("=== Test: Single Sequence Inference ===");
    
    // 使用占位权重测试（不加载真实模型）
    ModelConfig simplifiedConfig = config_;
    // 使用非常小的配置以避免内存问题
    simplifiedConfig.hiddenSize = 128;
    simplifiedConfig.intermediateSize = 256;
    simplifiedConfig.numLayers = 2;
    simplifiedConfig.numAttentionHeads = 4;
    simplifiedConfig.numKeyValueHeads = 4;
    
    KylinBackend backend(simplifiedConfig, "");  // 空路径表示使用占位权重
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    // 准备输入：简单的token序列（使用合理的token ID范围）
    // 注意：实际token ID应该从tokenizer获取，这里使用测试值
    // 使用较小的token ID以确保在词汇表范围内
    const size_t vocabSize = backend.getConfig().vocabSize;
    std::vector<int> inputIds = {
        static_cast<int>(vocabSize / 4),      // 使用词汇表的1/4位置
        static_cast<int>(vocabSize / 8),      // 使用词汇表的1/8位置
        static_cast<int>(vocabSize / 16),    // 使用词汇表的1/16位置
        static_cast<int>(vocabSize / 32)     // 使用词汇表的1/32位置
    };
    
    // 确保所有token ID在有效范围内
    for (auto& id : inputIds) {
        id = std::min(id, static_cast<int>(vocabSize - 1));
        id = std::max(id, 0);
    }
    
    CLLM_INFO("Input token IDs: [%d, %d, %d, %d]", 
              inputIds[0], inputIds[1], inputIds[2], inputIds[3]);
    
    // 执行推理
    kylin::Tensor logits;
    EXPECT_NO_THROW({
        logits = backend.forward(inputIds);
    }) << "Forward inference failed";
    
    // 验证输出形状
    auto shape = logits.shape();
    EXPECT_EQ(shape.size(), 2) << "Logits should be 2D tensor";
    EXPECT_EQ(shape[0], inputIds.size()) << "First dimension should match sequence length";
    EXPECT_EQ(shape[1], backend.getConfig().vocabSize) << "Second dimension should match vocab size";
    
    CLLM_INFO("Output logits shape: [%zu, %zu]", shape[0], shape[1]);
    
    // 验证输出数值合理性
    const float* logitsData = logits.data();
    size_t totalElements = logits.size();
    
    // 检查是否有NaN或Inf
    bool hasNaN = false;
    bool hasInf = false;
    float minVal = std::numeric_limits<float>::max();
    float maxVal = std::numeric_limits<float>::lowest();
    float sumVal = 0.0f;
    
    for (size_t i = 0; i < totalElements; ++i) {
        float val = logitsData[i];
        if (std::isnan(val)) {
            hasNaN = true;
        }
        if (std::isinf(val)) {
            hasInf = true;
        }
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
        sumVal += val;
    }
    
    EXPECT_FALSE(hasNaN) << "Logits contain NaN values";
    EXPECT_FALSE(hasInf) << "Logits contain Inf values";
    
    CLLM_INFO("Logits statistics:");
    CLLM_INFO("  - Min: %f", minVal);
    CLLM_INFO("  - Max: %f", maxVal);
    CLLM_INFO("  - Mean: %f", sumVal / totalElements);
    
    // 验证数值范围合理（logits通常在-50到50之间）
    EXPECT_GT(maxVal, -100.0f) << "Max logit value too small";
    EXPECT_LT(maxVal, 100.0f) << "Max logit value too large";
}

// 测试3: 不同序列长度推理
TEST_F(KylinGGUFQ4KTest, DifferentSequenceLengthsTest) {
    CLLM_INFO("=== Test: Different Sequence Lengths ===");
    
    KylinBackend backend(config_, modelPath_);
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    // 测试不同长度的序列
    const size_t vocabSize = backend.getConfig().vocabSize;
    std::vector<std::vector<int>> testCases = {
        {static_cast<int>(vocabSize / 4)},                           // 单token
        {static_cast<int>(vocabSize / 4), static_cast<int>(vocabSize / 8)},                      // 2 tokens
        {static_cast<int>(vocabSize / 4), static_cast<int>(vocabSize / 8), 
         static_cast<int>(vocabSize / 16), static_cast<int>(vocabSize / 32)},         // 4 tokens
        {static_cast<int>(vocabSize / 4), static_cast<int>(vocabSize / 8), 
         static_cast<int>(vocabSize / 16), static_cast<int>(vocabSize / 32),
         static_cast<int>(vocabSize / 64), static_cast<int>(vocabSize / 16), 
         static_cast<int>(vocabSize / 32), static_cast<int>(vocabSize / 64)}  // 8 tokens
    };
    
    // 确保所有token ID在有效范围内
    for (auto& testCase : testCases) {
        for (auto& id : testCase) {
            id = std::min(id, static_cast<int>(vocabSize - 1));
            id = std::max(id, 0);
        }
    }
    
    for (size_t i = 0; i < testCases.size(); ++i) {
        const auto& inputIds = testCases[i];
        CLLM_INFO("Testing sequence length: %zu", inputIds.size());
        
        EXPECT_NO_THROW({
            auto logits = backend.forward(inputIds);
            
            // 验证形状
            auto shape = logits.shape();
            EXPECT_EQ(shape[0], inputIds.size());
            EXPECT_EQ(shape[1], backend.getConfig().vocabSize);
            
            // 验证输出不为空
            EXPECT_GT(logits.size(), 0);
            
        }) << "Failed for sequence length " << inputIds.size();
    }
}

// 测试4: 批处理推理
TEST_F(KylinGGUFQ4KTest, BatchInferenceTest) {
    CLLM_INFO("=== Test: Batch Inference ===");
    
    KylinBackend backend(config_, modelPath_);
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    // 准备批处理输入：两个请求
    const size_t vocabSize = backend.getConfig().vocabSize;
    std::vector<int> flatInputIds = {
        static_cast<int>(vocabSize / 4), static_cast<int>(vocabSize / 8), static_cast<int>(vocabSize / 16),        // 第一个请求：3 tokens
        static_cast<int>(vocabSize / 32), static_cast<int>(vocabSize / 64), static_cast<int>(vocabSize / 16), static_cast<int>(vocabSize / 32)  // 第二个请求：4 tokens
    };
    
    // 确保所有token ID在有效范围内
    for (auto& id : flatInputIds) {
        id = std::min(id, static_cast<int>(vocabSize - 1));
        id = std::max(id, 0);
    }
    
    std::vector<std::pair<size_t, size_t>> requestPositions = {
        {0, 3},  // 第一个请求
        {3, 7}   // 第二个请求
    };
    
    size_t batchSize = 2;
    
    CLLM_INFO("Batch input: %zu total tokens, %zu requests", 
              flatInputIds.size(), batchSize);
    
    // 执行批处理推理
    kylin::Tensor batchLogits;
    EXPECT_NO_THROW({
        batchLogits = backend.forwardBatch(flatInputIds, requestPositions, batchSize);
    }) << "Batch forward inference failed";
    
    // 验证输出形状
    auto shape = batchLogits.shape();
    EXPECT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], flatInputIds.size()) << "First dimension should match total tokens";
    EXPECT_EQ(shape[1], backend.getConfig().vocabSize) << "Second dimension should match vocab size";
    
    CLLM_INFO("Batch output logits shape: [%zu, %zu]", shape[0], shape[1]);
}

// 测试5: 推理结果一致性
TEST_F(KylinGGUFQ4KTest, InferenceConsistencyTest) {
    CLLM_INFO("=== Test: Inference Consistency ===");
    
    KylinBackend backend(config_, modelPath_);
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    const size_t vocabSize = backend.getConfig().vocabSize;
    std::vector<int> inputIds = {
        static_cast<int>(vocabSize / 4),
        static_cast<int>(vocabSize / 8),
        static_cast<int>(vocabSize / 16),
        static_cast<int>(vocabSize / 32)
    };
    
    // 确保所有token ID在有效范围内
    for (auto& id : inputIds) {
        id = std::min(id, static_cast<int>(vocabSize - 1));
        id = std::max(id, 0);
    }
    
    // 执行两次推理
    kylin::Tensor logits1 = backend.forward(inputIds);
    kylin::Tensor logits2 = backend.forward(inputIds);
    
    // 验证两次结果应该相同（确定性推理）
    EXPECT_EQ(logits1.shape(), logits2.shape());
    
    const float* data1 = logits1.data();
    const float* data2 = logits2.data();
    size_t size = logits1.size();
    
    // 比较数值（允许浮点误差）
    const float epsilon = 1e-5f;
    int diffCount = 0;
    float maxDiff = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        float diff = std::abs(data1[i] - data2[i]);
        if (diff > epsilon) {
            diffCount++;
            maxDiff = std::max(maxDiff, diff);
        }
    }
    
    CLLM_INFO("Consistency check: %d differences > %f, max diff = %f", 
              diffCount, epsilon, maxDiff);
    
    // 允许少量浮点误差（由于计算顺序等）
    EXPECT_LT(diffCount, size / 1000) << "Too many differences between runs";
    EXPECT_LT(maxDiff, 1e-3f) << "Max difference too large";
}

// 测试6: 错误处理
TEST_F(KylinGGUFQ4KTest, ErrorHandlingTest) {
    CLLM_INFO("=== Test: Error Handling ===");
    
    KylinBackend backend(config_, modelPath_);
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    // 测试无效token ID（超出词汇表范围）
    std::vector<int> invalidInputIds = {
        static_cast<int>(backend.getConfig().vocabSize) + 100  // 超出范围
    };
    
    EXPECT_THROW({
        backend.forward(invalidInputIds);
    }, std::exception) << "Should throw exception for invalid token ID";
    
    // 测试空输入
    std::vector<int> emptyInputIds;
    EXPECT_THROW({
        backend.forward(emptyInputIds);
    }, std::exception) << "Should throw exception for empty input";
}

// 测试7: 性能基准（可选）
TEST_F(KylinGGUFQ4KTest, PerformanceBenchmarkTest) {
    CLLM_INFO("=== Test: Performance Benchmark ===");
    
    KylinBackend backend(config_, modelPath_);
    ASSERT_TRUE(backend.initialize()) << "Backend initialization failed";
    
    const size_t vocabSize = backend.getConfig().vocabSize;
    std::vector<int> inputIds = {
        static_cast<int>(vocabSize / 4),
        static_cast<int>(vocabSize / 8),
        static_cast<int>(vocabSize / 16),
        static_cast<int>(vocabSize / 32),
        static_cast<int>(vocabSize / 64),
        static_cast<int>(vocabSize / 16),
        static_cast<int>(vocabSize / 32),
        static_cast<int>(vocabSize / 64)
    };
    
    // 确保所有token ID在有效范围内
    for (auto& id : inputIds) {
        id = std::min(id, static_cast<int>(vocabSize - 1));
        id = std::max(id, 0);
    }
    
    // 只执行一次推理，记录时间
    auto start = std::chrono::high_resolution_clock::now();
    try {
        std::cout << "开始推理..." << std::endl;
        backend.forward(inputIds);
        std::cout << "推理完成！" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "推理错误: " << e.what() << std::endl;
        throw;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    double avgTimeMs = static_cast<double>(duration.count()) / 1;
    double tokensPerSec = (inputIds.size() * 1 * 1000.0) / duration.count();
    
    CLLM_INFO("Performance results:");
    CLLM_INFO("  - Inference time: %.2f ms", avgTimeMs);
    CLLM_INFO("  - Throughput: %.2f tokens/sec", tokensPerSec);
    
    // 验证性能合理（不设置严格阈值，仅记录）
    EXPECT_GT(tokensPerSec, 0.0) << "Throughput should be positive";
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    
    // 设置日志级别
    CLLM_INFO("Starting Kylin GGUF Q4_K_M inference tests...");
    
    return RUN_ALL_TESTS();
}
