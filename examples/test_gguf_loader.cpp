/**
 * @file test_gguf_loader.cpp
 * @brief GGUF格式加载器测试示例
 * @author cLLM Team
 * @date 2026-01-13
 */

#include "cllm/model/gguf_loader_new.h"
#include "cllm/model/loader_interface.h"
#include "cllm/common/logger.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    // 检查命令行参数
    if (argc < 2) {
        std::cerr << "用法: " << argv[0] << " <GGUF模型文件路径>" << std::endl;
        return 1;
    }
    
    std::string modelPath = argv[1];
    
    try {
        // 创建GGUF加载器
        auto loader = std::make_unique<cllm::GGUFLoader>(modelPath);
        
        if (!loader) {
            CLLM_ERROR("无法创建GGUF加载器");
            return 1;
        }
        
        // 加载模型
        if (!loader->load()) {
            CLLM_ERROR("加载GGUF模型失败");
            return 1;
        }
        
        CLLM_INFO("成功加载GGUF模型: {}", modelPath);
        
        // 获取模型配置
        cllm::ModelConfig config = loader->getConfig();
        
        // 输出模型配置
        CLLM_INFO("模型配置:");
        CLLM_INFO("  模型类型: {}", config.modelType);
        CLLM_INFO("  词表大小: {}", config.vocabSize);
        CLLM_INFO("  隐藏层大小: {}", config.hiddenSize);
        CLLM_INFO("  层数: {}", config.numLayers);
        CLLM_INFO("  注意力头数: {}", config.numAttentionHeads);
        CLLM_INFO("  KV头数: {}", config.numKeyValueHeads);
        CLLM_INFO("  最大序列长度: {}", config.maxSequenceLength);
        CLLM_INFO("  中间层大小: {}", config.intermediateSize);
        CLLM_INFO("  是否使用KV缓存: {}", config.useKVCache ? "是" : "否");
        CLLM_INFO("  是否使用量化: {}", config.useQuantization ? "是" : "否");
        
        // 测试加载权重数据
        cllm::model::ModelWeights weights;
        if (loader->loadWeights(weights)) {
            CLLM_INFO("成功加载权重数据");
            
            // 检查一些关键权重是否存在
            if (weights.findWeight("embedding") != nullptr) {
                CLLM_INFO("  已找到embedding权重");
            }
            
            if (weights.findWeight("finalNorm") != nullptr) {
                CLLM_INFO("  已找到finalNorm权重");
            }
            
            if (weights.findWeight("lmHead") != nullptr) {
                CLLM_INFO("  已找到lmHead权重");
            }
            
            if (!weights.layers.empty()) {
                CLLM_INFO("  已找到{}层权重", weights.layers.size());
            }
        } else {
            CLLM_ERROR("加载权重数据失败");
        }
        
        // 测试加载Tokenizer元数据
        try {
            auto ggufLoader = dynamic_cast<cllm::GGUFLoader*>(loader.get());
            if (ggufLoader) {
                ggufLoader->loadTokenizerMetadata();
                CLLM_INFO("成功加载Tokenizer元数据");
            }
        } catch (const std::exception& e) {
            CLLM_WARN("加载Tokenizer元数据失败: {}", e.what());
        }
        
    } catch (const std::exception& e) {
        CLLM_ERROR("发生错误: {}", e.what());
        return 1;
    }
    
    return 0;
}
