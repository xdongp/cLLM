#include "cllm/model/gguf_loader_new.h"
#include <iostream>
#include <string>

using namespace cllm;

int main() {
    try {
        // 指定GGUF模型路径
        std::string modelPath = "/Users/dannypan/PycharmProjects/xllm/cpp/cLLM/model/Qwen/qwen3-0.6b-q4_k_m.gguf";
        
        std::cout << "=== 加载GGUF模型 ===" << std::endl;
        std::cout << "模型路径: " << modelPath << std::endl;
        
        // 创建GGUF加载器
        GGUFLoader loader(modelPath, false); // 禁用内存映射
        
        // 加载模型
        std::cout << "正在加载模型..." << std::endl;
        bool loadResult = loader.load();
        if (!loadResult) {
            std::cerr << "❌ 模型加载失败!" << std::endl;
            return 1;
        }
        
        std::cout << "✅ 模型加载成功!" << std::endl;
        
        // 获取并显示模型配置
        std::cout << "\n=== 模型配置信息 ===" << std::endl;
        const ModelConfig& config = loader.getConfig();
        std::cout << "模型类型: " << config.modelType << std::endl;
        std::cout << "层数: " << config.numLayers << std::endl;
        std::cout << "隐藏层大小: " << config.hiddenSize << std::endl;
        std::cout << "注意力头数: " << config.numAttentionHeads << std::endl;
        std::cout << "KV头数: " << config.numKeyValueHeads << std::endl;
        
        // 显示部分元数据
        std::cout << "\n=== 元数据信息（前10项） ===" << std::endl;
        const auto& metadata = loader.getMetadata();
        int count = 0;
        for (const auto& [key, value] : metadata) {
            std::cout << "  " << key << " = ";
            switch (value.type) {
                case GGUFValueType::STRING:
                    std::cout << value.string_val << std::endl;
                    break;
                case GGUFValueType::UINT8:
                    std::cout << static_cast<int>(value.value.u8_val) << std::endl;
                    break;
                case GGUFValueType::INT32:
                    std::cout << value.value.i32_val << std::endl;
                    break;
                case GGUFValueType::UINT32:
                    std::cout << value.value.u32_val << std::endl;
                    break;
                case GGUFValueType::INT64:
                    std::cout << value.value.i64_val << std::endl;
                    break;
                case GGUFValueType::UINT64:
                    std::cout << value.value.u64_val << std::endl;
                    break;
                case GGUFValueType::FLOAT32:
                    std::cout << value.value.f32_val << std::endl;
                    break;
                case GGUFValueType::FLOAT64:
                    std::cout << value.value.f64_val << std::endl;
                    break;
                case GGUFValueType::BOOL:
                    std::cout << (value.value.bool_val ? "true" : "false") << std::endl;
                    break;
                case GGUFValueType::ARRAY:
                    std::cout << "[Array: type=" << static_cast<int>(value.array_val.elementType) << ", count=" << value.array_val.elementCount << "]" << std::endl;
                    break;
                default:
                    std::cout << "[Unknown type: " << static_cast<int>(value.type) << "]" << std::endl;
                    break;
            }
            
            if (++count >= 10) {
                break;
            }
        }
        
        std::cout << "... 共 " << metadata.size() << " 项元数据" << std::endl;
        
        // 检查几个重要的元数据
        std::cout << "\n=== 关键元数据 ===" << std::endl;
        auto it = metadata.find("general.architecture");
        if (it != metadata.end() && it->second.type == GGUFValueType::STRING) {
            std::cout << "模型架构: " << it->second.string_val << std::endl;
        }
        
        it = metadata.find("general.alignment");
        if (it != metadata.end()) {
            std::cout << "对齐值: " << it->second.value.u32_val << std::endl;
        }
        
        it = metadata.find("general.file_type");
        if (it != metadata.end()) {
            std::cout << "文件类型: " << it->second.value.u32_val << std::endl;
        }
        
        // 显示张量统计信息
        std::cout << "\n=== 张量信息 ===" << std::endl;
        std::cout << "注意: 由于接口限制，无法直接获取张量列表" << std::endl;
        
        // 尝试加载几个关键张量
        std::cout << "\n=== 尝试加载关键张量 ===" << std::endl;
        
        // 检查embedding.weight张量
        if (loader.hasWeight("embedding.weight")) {
            std::cout << "✅ 找到embedding.weight张量" << std::endl;
        } else {
            std::cout << "❌ 未找到embedding.weight张量" << std::endl;
        }
        
        // 检查第一个注意力层的权重
        if (loader.hasWeight("layers.0.attention.wq.weight")) {
            std::cout << "✅ 找到layers.0.attention.wq.weight张量" << std::endl;
        } else {
            std::cout << "❌ 未找到layers.0.attention.wq.weight张量" << std::endl;
        }
        
        std::cout << "\n=== 加载完成 ===" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ 加载失败: " << e.what() << std::endl;
        return 1;
    }
}