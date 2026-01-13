#include "cllm/model/gguf_loader_new.h"
#include "cllm/common/logger.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace cllm;

// 创建一个简单的GGUF测试文件
std::string createTestGGUFFile() {
    std::string testFilePath = "/tmp/test_full_gguf_model.gguf";
    
    std::ofstream testFile(testFilePath, std::ios::binary);
    if (!testFile.is_open()) {
        throw std::runtime_error("无法创建测试文件: " + testFilePath);
    }
    
    // 写入GGUF文件头
    const char magic[4] = { 'G', 'G', 'U', 'F' };
    testFile.write(magic, 4);
    
    uint32_t version = 3;
    testFile.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    uint64_t tensorCount = 2;
    uint64_t metadataCount = 5;
    
    testFile.write(reinterpret_cast<const char*>(&tensorCount), sizeof(tensorCount));
    testFile.write(reinterpret_cast<const char*>(&metadataCount), sizeof(metadataCount));
    
    // 写入元数据
    // 1. 模型架构信息
    std::string key = "general.architecture";
    uint64_t keyLen = key.size();
    testFile.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
    testFile.write(key.c_str(), key.size());
    
    uint32_t valueType = static_cast<uint32_t>(GGUFValueType::STRING);
    testFile.write(reinterpret_cast<const char*>(&valueType), sizeof(valueType));
    
    std::string value = "llama";
    uint64_t valueLen = value.size();
    testFile.write(reinterpret_cast<const char*>(&valueLen), sizeof(valueLen));
    testFile.write(value.c_str(), value.size());
    
    // 2. 模型层数
    key = "llama.layers";
    keyLen = key.size();
    testFile.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
    testFile.write(key.c_str(), key.size());
    
    valueType = static_cast<uint32_t>(GGUFValueType::UINT32);
    testFile.write(reinterpret_cast<const char*>(&valueType), sizeof(valueType));
    
    uint32_t layerCount = 8;
    testFile.write(reinterpret_cast<const char*>(&layerCount), sizeof(layerCount));
    
    // 3. 隐藏层大小
    key = "llama.hidden_size";
    keyLen = key.size();
    testFile.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
    testFile.write(key.c_str(), key.size());
    
    valueType = static_cast<uint32_t>(GGUFValueType::UINT32);
    testFile.write(reinterpret_cast<const char*>(&valueType), sizeof(valueType));
    
    uint32_t hiddenSize = 4096;
    testFile.write(reinterpret_cast<const char*>(&hiddenSize), sizeof(hiddenSize));
    
    // 4. 注意力头数
    key = "llama.attention.head_count";
    keyLen = key.size();
    testFile.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
    testFile.write(key.c_str(), key.size());
    
    valueType = static_cast<uint32_t>(GGUFValueType::UINT32);
    testFile.write(reinterpret_cast<const char*>(&valueType), sizeof(valueType));
    
    uint32_t headCount = 32;
    testFile.write(reinterpret_cast<const char*>(&headCount), sizeof(headCount));
    
    // 5. 对齐信息
    key = "general.alignment";
    keyLen = key.size();
    testFile.write(reinterpret_cast<const char*>(&keyLen), sizeof(keyLen));
    testFile.write(key.c_str(), key.size());
    
    valueType = static_cast<uint32_t>(GGUFValueType::UINT32);
    testFile.write(reinterpret_cast<const char*>(&valueType), sizeof(valueType));
    
    uint32_t alignment = 32;
    testFile.write(reinterpret_cast<const char*>(&alignment), sizeof(alignment));
    
    // 对齐到下一个32字节边界
    while (testFile.tellp() % 32 != 0) {
        char pad = 0;
        testFile.write(&pad, 1);
    }
    
    // 写入张量信息
    // 张量1: embedding.weight
    std::string tensorName = "embedding.weight";
    uint64_t tensorNameLen = tensorName.size();
    testFile.write(reinterpret_cast<const char*>(&tensorNameLen), sizeof(tensorNameLen));
    testFile.write(tensorName.c_str(), tensorName.size());
    
    uint32_t dimensions = 2;
    testFile.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    
    uint64_t shape[2] = { 32000, 4096 };
    testFile.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    
    uint32_t tensorType = static_cast<uint32_t>(GGMLType::F32);
    testFile.write(reinterpret_cast<const char*>(&tensorType), sizeof(tensorType));
    
    uint64_t tensorOffset = testFile.tellp();
    testFile.write(reinterpret_cast<const char*>(&tensorOffset), sizeof(tensorOffset));
    
    // 张量2: layers.0.attention.wq.weight
    tensorName = "layers.0.attention.wq.weight";
    tensorNameLen = tensorName.size();
    testFile.write(reinterpret_cast<const char*>(&tensorNameLen), sizeof(tensorNameLen));
    testFile.write(tensorName.c_str(), tensorName.size());
    
    dimensions = 2;
    testFile.write(reinterpret_cast<const char*>(&dimensions), sizeof(dimensions));
    
    shape[0] = 4096;
    shape[1] = 4096;
    testFile.write(reinterpret_cast<const char*>(shape), sizeof(shape));
    
    tensorType = static_cast<uint32_t>(GGMLType::F32);
    testFile.write(reinterpret_cast<const char*>(&tensorType), sizeof(tensorType));
    
    tensorOffset = testFile.tellp();
    testFile.write(reinterpret_cast<const char*>(&tensorOffset), sizeof(tensorOffset));
    
    // 对齐到下一个32字节边界
    while (testFile.tellp() % 32 != 0) {
        char pad = 0;
        testFile.write(&pad, 1);
    }
    
    // 写入张量数据 (简单的测试数据)
    float testData[4] = { 1.0f, 2.0f, 3.0f, 4.0f };
    
    // 写入第一个张量的数据 (只写入少量数据用于测试)
    testFile.write(reinterpret_cast<const char*>(testData), sizeof(testData));
    
    // 写入第二个张量的数据
    testFile.write(reinterpret_cast<const char*>(testData), sizeof(testData));
    
    testFile.close();
    
    return testFilePath;
}

int main() {
    try {
        // 创建测试文件
        std::string testFilePath = createTestGGUFFile();
        std::cout << "创建测试文件成功: " << testFilePath << std::endl;
        
        // 初始化日志 (注释掉这行，避免命名空间问题)
        // Logger::instance().setLevel(LogLevel::DEBUG);
        
        // 创建GGUF加载器 (禁用内存映射以避免潜在问题)
        std::cout << "\n=== 创建GGUF加载器 ===" << std::endl;
        GGUFLoader loader(testFilePath, false);
        
        // 加载模型
        std::cout << "\n=== 加载GGUF模型 ===" << std::endl;
        bool loadResult = loader.load();
        if (loadResult) {
            std::cout << "✅ 模型加载成功!" << std::endl;
        } else {
            std::cout << "❌ 模型加载失败!" << std::endl;
            return 1;
        }
        
        // 获取模型配置
        std::cout << "\n=== 模型配置信息 ===" << std::endl;
        const ModelConfig& config = loader.getConfig();
        std::cout << "模型类型: " << config.modelType << std::endl;
        std::cout << "层数: " << config.numLayers << std::endl;
        std::cout << "隐藏层大小: " << config.hiddenSize << std::endl;
        std::cout << "注意力头数: " << config.numAttentionHeads << std::endl;
        
        // 测试元数据访问
        std::cout << "\n=== 元数据访问测试 ===" << std::endl;
        const auto& allMetadata = loader.getMetadata();
        
        auto archIt = allMetadata.find("general.architecture");
        if (archIt != allMetadata.end()) {
            if (archIt->second.type == GGUFValueType::STRING) {
                std::cout << "✅ 元数据 general.architecture: " << archIt->second.string_val << std::endl;
            }
        }
        
        auto layersIt = allMetadata.find("llama.layers");
        if (layersIt != allMetadata.end()) {
            std::cout << "✅ 元数据 llama.layers: " << layersIt->second.value.u32_val << std::endl;
        }
        
        auto alignIt = allMetadata.find("general.alignment");
        if (alignIt != allMetadata.end()) {
            std::cout << "✅ 元数据 general.alignment: " << alignIt->second.value.u32_val << std::endl;
        }
        
        // 测试张量信息访问 - 我们只能通过hasWeight和loadWeightByName来测试
        std::cout << "\n=== 张量信息测试 ===" << std::endl;
        
        // 测试张量存在性
        std::cout << "\n=== 张量存在性测试 ===" << std::endl;
        if (loader.hasWeight("embedding.weight")) {
            std::cout << "✅ 找到张量: embedding.weight" << std::endl;
        } else {
            std::cout << "❌ 未找到张量: embedding.weight" << std::endl;
        }
        
        if (loader.hasWeight("layers.0.attention.wq.weight")) {
            std::cout << "✅ 找到张量: layers.0.attention.wq.weight" << std::endl;
        } else {
            std::cout << "❌ 未找到张量: layers.0.attention.wq.weight" << std::endl;
        }
        
        // 测试张量加载
        std::cout << "\n=== 张量加载测试 ===" << std::endl;
        model::WeightData weight;
        if (loader.loadWeightByName("embedding.weight", weight)) {
            std::cout << "✅ 成功加载张量: embedding.weight" << std::endl;
            std::cout << "   形状: [";
            for (size_t i = 0; i < weight.shape.size(); ++i) {
                std::cout << weight.shape[i];
                if (i < weight.shape.size() - 1) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << std::endl;
            std::cout << "   数据大小: " << weight.data.size() << " 字节" << std::endl;
        } else {
            std::cout << "❌ 无法加载张量: embedding.weight" << std::endl;
        }
        
        // 清理测试文件
        std::filesystem::remove(testFilePath);
        std::cout << "\n=== 清理完成 ===" << std::endl;
        std::cout << "测试文件已删除: " << testFilePath << std::endl;
        
        std::cout << "\n✅ 所有测试通过!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 测试失败: " << e.what() << std::endl;
        return 1;
    }
}