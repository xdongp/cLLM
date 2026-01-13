/**
 * @file test_loader_factory.cpp
 * @brief 模型加载器工厂测试
 * @author cLLM Team
 * @date 2026-01-13
 */

#include "cllm/model/loader_interface.h"
#include "cllm/model/config.h"
#include <fstream>
#include <gtest/gtest.h>

namespace cllm {

TEST(ModelLoaderFactoryTest, DetectFormat) {
    // 测试ModelLoaderFactory的detectFormat方法
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.bin"), ModelFormat::BINARY);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.fp16.bin"), ModelFormat::BINARY);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.int8.bin"), ModelFormat::BINARY);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.gguf"), ModelFormat::GGUF);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.f16.gguf"), ModelFormat::GGUF);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.q4_k_m.gguf"), ModelFormat::GGUF);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.safetensors"), ModelFormat::SAFETENSORS);
    EXPECT_EQ(ModelLoaderFactory::detectFormat("model.unknown"), ModelFormat::UNKNOWN);
    EXPECT_EQ(ModelLoaderFactory::detectFormat(""), ModelFormat::UNKNOWN);
}

TEST(ModelLoaderFactoryTest, IsFormatSupported) {
    // 测试ModelLoaderFactory的isFormatSupported方法
    EXPECT_TRUE(ModelLoaderFactory::isFormatSupported(ModelFormat::BINARY));
    EXPECT_FALSE(ModelLoaderFactory::isFormatSupported(ModelFormat::GGUF));
    EXPECT_FALSE(ModelLoaderFactory::isFormatSupported(ModelFormat::SAFETENSORS));
    EXPECT_FALSE(ModelLoaderFactory::isFormatSupported(ModelFormat::UNKNOWN));
}

TEST(ModelLoaderFactoryTest, FormatToString) {
    // 测试ModelLoaderFactory的formatToString方法
    EXPECT_EQ(ModelLoaderFactory::formatToString(ModelFormat::BINARY), "BINARY");
    EXPECT_EQ(ModelLoaderFactory::formatToString(ModelFormat::GGUF), "GGUF");
    EXPECT_EQ(ModelLoaderFactory::formatToString(ModelFormat::SAFETENSORS), "SAFETENSORS");
    EXPECT_EQ(ModelLoaderFactory::formatToString(ModelFormat::UNKNOWN), "UNKNOWN");
}

TEST(ModelLoaderFactoryTest, CreateBinaryLoader) {
    // 测试ModelLoaderFactory创建BinaryModelLoader
    ModelConfig config;
    config.vocabSize = 10000;
    config.hiddenSize = 768;
    config.numLayers = 12;
    config.numAttentionHeads = 12;
    config.numKeyValueHeads = 12;
    config.intermediateSize = 3072;
    
    // 使用 Google Test 的临时目录
    std::string tempDir = testing::TempDir();
    std::string testPath = tempDir + "/test_model.bin";
    
    // 创建一个测试用的空二进制文件
    std::ofstream testFile(testPath);
    testFile.close();
    
    // 预期不会抛出异常，因为我们的代码应该能处理空文件
    EXPECT_NO_THROW(
        ModelLoaderFactory::createLoader(testPath, config)
    );
    
    // 清理测试文件
    std::remove(testPath.c_str());
}

} // namespace cllm

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
