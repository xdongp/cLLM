/**
 * @file test_gguf_loader_complete.cpp
 * @brief GGUF格式加载器完整单元测试
 * @author cLLM Team
 * @date 2026-01-13
 */

#include "cllm/model/gguf_loader_new.h"
#include "cllm/model/weight_data.h"
#include "cllm/common/logger.h"
#include <gtest/gtest.h>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>

using json = nlohmann::json;

namespace cllm {
namespace model {

// 测试配置结构体
struct TestConfig {
    std::string modelPath;         // 模型文件路径
    std::string modelJsonPath;     // 模型描述文件路径
    bool expectSuccess;            // 预期加载结果
    std::string expectedModelType; // 预期模型类型
};

// GGUF加载器完整测试类
class GGUFLoaderCompleteTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 配置测试参数
        config_.modelPath = "../model/Qwen/qwen3-0.6b-q4_k_m.gguf";
        config_.modelJsonPath = "../model/Qwen/qwen3-0.6b-q4_k_m.json";
        config_.expectSuccess = true;
        config_.expectedModelType = "qwen3";
        
        // 初始化日志
        cllm::Logger::instance().setLevel(spdlog::level::info);
        
        // 检查模型文件是否存在
        modelExists_ = std::filesystem::exists(config_.modelPath);
        jsonExists_ = std::filesystem::exists(config_.modelJsonPath);
        
        // 如果模型文件不存在，跳过测试
        if (!modelExists_) {
            GTEST_SKIP() << "模型文件不存在: " << config_.modelPath;
        }
        if (!jsonExists_) {
            GTEST_SKIP() << "模型描述文件不存在: " << config_.modelJsonPath;
        }
        
        // 读取模型描述文件
        if (jsonExists_) {
            std::ifstream jsonFile(config_.modelJsonPath);
            if (jsonFile.is_open()) {
                try {
                    modelJson_ = json::parse(jsonFile);
                    jsonFile.close();
                } catch (const json::exception& e) {
                    jsonFile.close();
                    GTEST_SKIP() << "解析模型描述文件失败: " << e.what();
                }
            }
        }
        
        // 创建加载器实例
        loader_ = std::make_unique<GGUFLoader>(config_.modelPath, false, false);
        EXPECT_NE(loader_, nullptr) << "创建GGUF加载器失败";
    }
    
    void TearDown() override {
        // 清理资源
        loader_.reset();
    }
    
    // 计算文件大小的辅助函数
    size_t getFileSize(const std::string& filePath) {
        std::filesystem::path path(filePath);
        if (std::filesystem::exists(path)) {
            return std::filesystem::file_size(path);
        }
        return 0;
    }
    
    // 格式化文件大小的辅助函数
    std::string formatFileSize(size_t size) {
        if (size >= 1024 * 1024 * 1024) {
            return std::to_string(size / (1024 * 1024 * 1024)) + " GB";
        } else if (size >= 1024 * 1024) {
            return std::to_string(size / (1024 * 1024)) + " MB";
        } else if (size >= 1024) {
            return std::to_string(size / 1024) + " KB";
        }
        return std::to_string(size) + " B";
    }
    
    // 测试配置
    TestConfig config_;
    
    // 模型文件状态
    bool modelExists_ = false;
    bool jsonExists_ = false;
    
    // 模型描述JSON
    json modelJson_;
    
    // 加载器实例
    std::unique_ptr<GGUFLoader> loader_;
};

// 测试边界情况的类
class GGUFLoaderEdgeCaseTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 初始化日志
        cllm::Logger::instance().setLevel(spdlog::level::info);
    }
};

// 测试1: 加载器构造函数和文件打开
TEST_F(GGUFLoaderCompleteTest, Constructor) {
    std::cout << "=== 测试加载器构造函数 ===" << std::endl;
    
    // 检查文件大小
    size_t fileSize = getFileSize(config_.modelPath);
    std::cout << "模型文件大小: " << formatFileSize(fileSize) << std::endl;
    EXPECT_GT(fileSize, 0) << "模型文件大小为0";
    
    // 验证加载器是否成功创建
    EXPECT_NE(loader_, nullptr) << "GGUF加载器创建失败";
    
    // 验证文件路径
    EXPECT_EQ(loader_->getModelPath(), config_.modelPath) << "模型文件路径不正确";
}

// 测试2: 文件头解析
TEST_F(GGUFLoaderCompleteTest, ParseHeader) {
    std::cout << "=== 测试文件头解析 ===" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 解析文件头
    GGUFHeader header = loader_->parseHeader();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "解析文件头耗时: " << duration << " ms" << std::endl;
    
    // 验证文件头信息
    EXPECT_EQ(header.magic, 0x46554747) << "魔数不正确";
    EXPECT_GT(header.version, 0) << "版本号不正确";
    EXPECT_GT(header.tensorCount, 0) << "张量数量不正确";
    EXPECT_GT(header.metadataCount, 0) << "元数据数量不正确";
    
    // 输出文件头信息
    std::cout << "文件头信息: " << std::endl;
    std::cout << "  魔数: 0x" << std::hex << header.magic << std::dec << std::endl;
    std::cout << "  版本: " << header.version << std::endl;
    std::cout << "  张量数量: " << header.tensorCount << std::endl;
    std::cout << "  元数据数量: " << header.metadataCount << std::endl;
}

// 测试3: 元数据解析
TEST_F(GGUFLoaderCompleteTest, ParseMetadata) {
    std::cout << "=== 测试元数据解析 ===" << std::endl;
    
    // 首先解析文件头
    GGUFHeader header = loader_->parseHeader();
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 解析元数据
    std::unordered_map<std::string, GGUFMetadata> metadata = loader_->parseMetadata(header.metadataCount);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "解析元数据耗时: " << duration << " ms" << std::endl;
    
    // 验证元数据数量
    EXPECT_EQ(metadata.size(), header.metadataCount) << "元数据数量不匹配";
    
    // 检查关键元数据
    EXPECT_NE(metadata.find("general.architecture"), metadata.end()) << "未找到架构信息";
    EXPECT_NE(metadata.find("general.name"), metadata.end()) << "未找到模型名称";
    EXPECT_NE(metadata.find("llama.context_length"), metadata.end()) << "未找到上下文长度";
    
    // 输出部分元数据
    std::cout << "元数据信息: " << std::endl;
    for (const auto& [key, value] : metadata) {
        if (key.find("general") == 0 || key.find("llama") == 0) {
            std::cout << "  " << key << ": ";
            if (value.type == GGUFMetadata::ValueType::STRING) {
                std::cout << value.string_val << std::endl;
            } else if (value.type == GGUFMetadata::ValueType::INT64) {
                std::cout << value.value.i64_val << std::endl;
            }
        }
    }
}

// 测试4: 张量数据解析
TEST_F(GGUFLoaderCompleteTest, ParseTensorData) {
    std::cout << "=== 测试张量数据解析 ===" << std::endl;
    
    // 首先解析文件头
    GGUFHeader header = loader_->parseHeader();
    
    // 解析元数据
    loader_->parseMetadata(header.metadataCount);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 解析张量数据
    std::unordered_map<std::string, uint64_t> tensorOffsets = loader_->parseTensorData(header.tensorCount);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "解析张量数据耗时: " << duration << " ms" << std::endl;
    
    // 验证张量数据数量
    EXPECT_EQ(tensorOffsets.size(), header.tensorCount) << "张量数据数量不匹配";
    
    // 检查关键张量
    EXPECT_NE(tensorOffsets.find("embedding.weight"), tensorOffsets.end()) << "未找到嵌入层权重";
    EXPECT_NE(tensorOffsets.find("final_norm.weight"), tensorOffsets.end()) << "未找到最终归一化层权重";
    EXPECT_NE(tensorOffsets.find("lm_head.weight"), tensorOffsets.end()) << "未找到语言模型头权重";
    
    std::cout << "成功解析张量数据，共 " << tensorOffsets.size() << " 个张量" << std::endl;
}

// 测试5: 完整加载流程
TEST_F(GGUFLoaderCompleteTest, LoadComplete) {
    std::cout << "=== 测试完整加载流程 ===" << std::endl;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 执行完整加载
    bool success = loader_->load();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "完整加载流程耗时: " << duration << " ms" << std::endl;
    std::cout << "加载结果: " << (success ? "成功" : "失败") << std::endl;
    
    // 验证加载结果
    EXPECT_EQ(success, config_.expectSuccess) << "加载结果与预期不符";
    
    if (success) {
        // 验证模型配置
        const ModelConfig& modelConfig = loader_->getConfig();
        EXPECT_EQ(modelConfig.modelType, config_.expectedModelType) << "模型类型与预期不符";
        EXPECT_GT(modelConfig.numLayers, 0) << "层数不正确";
        EXPECT_GT(modelConfig.hiddenSize, 0) << "隐藏层大小不正确";
        EXPECT_GT(modelConfig.vocabSize, 0) << "词汇表大小不正确";
        EXPECT_GT(modelConfig.maxSequenceLength, 0) << "最大序列长度不正确";
        
        // 输出模型配置
        std::cout << "模型配置: " << std::endl;
        std::cout << "  模型类型: " << modelConfig.modelType << std::endl;
        std::cout << "  层数: " << modelConfig.numLayers << std::endl;
        std::cout << "  隐藏层大小: " << modelConfig.hiddenSize << std::endl;
        std::cout << "  词汇表大小: " << modelConfig.vocabSize << std::endl;
        std::cout << "  最大序列长度: " << modelConfig.maxSequenceLength << std::endl;
    }
}

// 测试6: 权重加载
TEST_F(GGUFLoaderCompleteTest, LoadWeights) {
    std::cout << "=== 测试权重加载 ===" << std::endl;
    
    // 首先执行完整加载
    bool loadSuccess = loader_->load();
    EXPECT_TRUE(loadSuccess) << "模型加载失败，无法继续测试权重加载";
    
    // 创建权重容器
    ModelWeights weights;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 加载权重
    bool weightSuccess = loader_->loadWeights(weights);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "权重加载耗时: " << duration << " ms" << std::endl;
    std::cout << "权重加载结果: " << (weightSuccess ? "成功" : "失败") << std::endl;
    
    // 验证权重加载结果
    EXPECT_TRUE(weightSuccess) << "权重加载失败";
    EXPECT_EQ(weights.layers.size(), loader_->getConfig().numLayers) << "层数与配置不符";
    std::cout << "创建的层数: " << weights.layers.size() << std::endl;
    
    // 检查权重映射是否正确
    EXPECT_NE(weights.findWeight("embedding.weight"), nullptr) << "未找到embedding.weight在权重映射中";
    EXPECT_NE(weights.findWeight("final_norm.weight"), nullptr) << "未找到final_norm.weight在权重映射中";
    EXPECT_NE(weights.findWeight("lm_head.weight"), nullptr) << "未找到lm_head.weight在权重映射中";
    
    // 检查第一层的权重是否存在
    EXPECT_NE(weights.findWeight("layers.0.attention.wq.weight"), nullptr) << "未找到layers.0.attention.wq.weight在权重映射中";
}

// 测试7: 特定权重加载
TEST_F(GGUFLoaderCompleteTest, LoadSpecificWeight) {
    std::cout << "=== 测试特定权重加载 ===" << std::endl;
    
    // 首先执行完整加载
    bool loadSuccess = loader_->load();
    EXPECT_TRUE(loadSuccess) << "模型加载失败，无法继续测试特定权重加载";
    
    // 测试加载嵌入层权重
    WeightData embeddingWeight;
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    bool embeddingSuccess = loader_->loadWeightByName("embedding.weight", embeddingWeight);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "加载embedding.weight耗时: " << duration << " ms" << std::endl;
    std::cout << "embedding.weight加载结果: " << (embeddingSuccess ? "成功" : "失败") << std::endl;
    
    // 验证权重加载结果
    EXPECT_TRUE(embeddingSuccess) << "embedding.weight加载失败";
    EXPECT_EQ(embeddingWeight.name, "embedding.weight") << "权重名称不正确";
    EXPECT_GT(embeddingWeight.shape.size(), 0) << "权重形状为空";
    EXPECT_GT(embeddingWeight.data.size(), 0) << "权重数据为空";
    
    // 输出权重信息
    std::cout << "embedding.weight形状: ";
    for (const auto& dim : embeddingWeight.shape) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "embedding.weight数据大小: " << embeddingWeight.data.size() << " 字节" << std::endl;
}

// 测试8: 内存映射模式加载
TEST_F(GGUFLoaderCompleteTest, LoadWithMemoryMap) {
    std::cout << "=== 测试内存映射模式加载 ===" << std::endl;
    
    // 创建使用内存映射的加载器
    GGUFLoader memoryMapLoader(config_.modelPath, true, false);
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    // 执行完整加载
    bool success = memoryMapLoader.load();
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
    
    std::cout << "内存映射加载耗时: " << duration << " ms" << std::endl;
    std::cout << "内存映射加载结果: " << (success ? "成功" : "失败") << std::endl;
    
    // 验证加载结果
    EXPECT_TRUE(success) << "内存映射加载失败";
    
    if (success) {
        // 验证模型配置
        const ModelConfig& modelConfig = memoryMapLoader.getConfig();
        EXPECT_EQ(modelConfig.modelType, config_.expectedModelType) << "模型类型与预期不符";
        EXPECT_GT(modelConfig.numLayers, 0) << "层数不正确";
    }
}

// 测试边界情况: 加载不存在的模型
TEST_F(GGUFLoaderEdgeCaseTest, LoadNonExistentModel) {
    std::cout << "=== 测试边界情况：加载不存在的模型文件 ===" << std::endl;
    
    std::string nonExistentPath = "./non_existent_model.gguf";
    
    // 验证文件不存在
    EXPECT_FALSE(std::filesystem::exists(nonExistentPath)) << "文件应该不存在";
    
    // 尝试创建加载器并加载
    try {
        GGUFLoader loader(nonExistentPath);
        bool success = loader.load();
        EXPECT_FALSE(success) << "加载不存在的模型文件应该失败";
    } catch (const std::exception& e) {
        std::cout << "捕获到预期异常: " << e.what() << std::endl;
    }
}

// 测试边界情况: 加载空文件
TEST_F(GGUFLoaderEdgeCaseTest, LoadEmptyFile) {
    std::cout << "=== 测试边界情况：加载空文件 ===" << std::endl;
    
    std::string emptyFilePath = "./empty_model.gguf";
    
    // 创建空文件
    std::ofstream emptyFile(emptyFilePath, std::ios::binary);
    emptyFile.close();
    
    // 验证文件存在且为空
    EXPECT_TRUE(std::filesystem::exists(emptyFilePath)) << "空文件应该存在";
    EXPECT_EQ(std::filesystem::file_size(emptyFilePath), 0) << "文件应该为空";
    
    // 尝试创建加载器并解析文件头
    try {
        GGUFLoader loader(emptyFilePath);
        GGUFHeader header = loader.parseHeader();
        FAIL() << "解析空文件的文件头应该失败";
    } catch (const std::exception& e) {
        std::cout << "捕获到预期异常: " << e.what() << std::endl;
    }
    
    // 清理空文件
    std::filesystem::remove(emptyFilePath);
}

// 测试边界情况: 加载损坏的GGUF文件
TEST_F(GGUFLoaderEdgeCaseTest, LoadCorruptedFile) {
    std::cout << "=== 测试边界情况：加载损坏的GGUF文件 ===" << std::endl;
    
    std::string corruptedFilePath = "./corrupted_model.gguf";
    
    // 创建损坏的GGUF文件（只写入部分头信息）
    std::ofstream corruptedFile(corruptedFilePath, std::ios::binary);
    if (corruptedFile.is_open()) {
        // 写入部分GGUF文件标识
        const char magic[2] = { 'G', 'G' };
        corruptedFile.write(magic, 2);
        corruptedFile.close();
    }
    
    // 验证文件存在但损坏
    EXPECT_TRUE(std::filesystem::exists(corruptedFilePath)) << "损坏文件应该存在";
    EXPECT_GT(std::filesystem::file_size(corruptedFilePath), 0) << "文件应该不为空";
    
    // 尝试创建加载器并解析文件头
    try {
        GGUFLoader loader(corruptedFilePath);
        GGUFHeader header = loader.parseHeader();
        FAIL() << "解析损坏文件的文件头应该失败";
    } catch (const std::exception& e) {
        std::cout << "捕获到预期异常: " << e.what() << std::endl;
    }
    
    // 清理损坏文件
    std::filesystem::remove(corruptedFilePath);
}

} // namespace model
} // namespace cllm

// 主函数
int main(int argc, char **argv) {
    std::cout << "==============================================================" << std::endl;
    std::cout << "          GGUF Loader 完整单元测试开始执行                 " << std::endl;
    std::cout << "==============================================================" << std::endl;
    
    // 初始化Google Test
    ::testing::InitGoogleTest(&argc, argv);
    
    // 运行所有测试
    int result = RUN_ALL_TESTS();
    
    std::cout << "==============================================================" << std::endl;
    std::cout << "          GGUF Loader 完整单元测试执行结束                 " << std::endl;
    std::cout << "==============================================================" << std::endl;
    
    return result;
}