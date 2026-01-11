#include "cllm/CTokenizer/model_detector.h"
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace cllm {

ModelType ModelDetector::detectModelType(const std::string& configPath) {
    try {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            return ModelType::SPM; // 如果无法打开文件，返回默认类型
        }
        
        // 读取配置文件内容
        std::string content;
        std::string line;
        while (std::getline(file, line)) {
            content += line + "\n";
        }
        
        // 解析JSON配置
        nlohmann::json config;
        try {
            std::ifstream jsonFile(configPath);
            jsonFile >> config;
        } catch (...) {
            // 如果JSON解析失败，使用字符串匹配
            if (content.find("Qwen") != std::string::npos || content.find("qwen") != std::string::npos) {
                return ModelType::QWEN;
            } else if (content.find("DeepSeek") != std::string::npos || content.find("deepseek") != std::string::npos) {
                if (content.find("DeepSeek3") != std::string::npos || content.find("deepseek3") != std::string::npos) {
                    return ModelType::DEEPSEEK3_LLM;
                } else if (content.find("Coder") != std::string::npos || content.find("coder") != std::string::npos) {
                    return ModelType::DEEPSEEK_CODER;
                } else {
                    return ModelType::DEEPSEEK_LLM;
                }
            } else if (content.find("Llama") != std::string::npos || content.find("llama") != std::string::npos) {
                return ModelType::LLAMA;
            } else if (content.find("Bert") != std::string::npos || content.find("bert") != std::string::npos) {
                return ModelType::BERT;
            } else if (content.find("GPT2") != std::string::npos || content.find("gpt2") != std::string::npos) {
                return ModelType::GPT2;
            } else {
                return ModelType::SPM;
            }
        }
        
        // 检查tokenizer_class字段
        if (config.contains("tokenizer_class")) {
            std::string tokenizerClass = config["tokenizer_class"].get<std::string>();
            
            if (tokenizerClass.find("Qwen") != std::string::npos) {
                return ModelType::QWEN;
            } else if (tokenizerClass.find("DeepSeek") != std::string::npos) {
                if (tokenizerClass.find("DeepSeek3") != std::string::npos) {
                    return ModelType::DEEPSEEK3_LLM;
                } else if (tokenizerClass.find("Coder") != std::string::npos) {
                    return ModelType::DEEPSEEK_CODER;
                } else {
                    return ModelType::DEEPSEEK_LLM;
                }
            } else if (tokenizerClass.find("Llama") != std::string::npos) {
                return ModelType::LLAMA;
            } else if (tokenizerClass.find("Bert") != std::string::npos) {
                return ModelType::BERT;
            } else if (tokenizerClass.find("GPT2") != std::string::npos) {
                return ModelType::GPT2;
            }
        }
        
        // 检查chat_template字段
        if (config.contains("chat_template")) {
            std::string chatTemplate = config["chat_template"].get<std::string>();
            
            if (chatTemplate.find("qwen") != std::string::npos) {
                return ModelType::QWEN;
            } else if (chatTemplate.find("deepseek") != std::string::npos) {
                if (chatTemplate.find("deepseek3") != std::string::npos) {
                    return ModelType::DEEPSEEK3_LLM;
                } else if (chatTemplate.find("coder") != std::string::npos) {
                    return ModelType::DEEPSEEK_CODER;
                } else {
                    return ModelType::DEEPSEEK_LLM;
                }
            }
        }
        
        // 检查model_type字段
        if (config.contains("model_type")) {
            std::string modelType = config["model_type"].get<std::string>();
            
            if (modelType.find("qwen") != std::string::npos) {
                return ModelType::QWEN;
            } else if (modelType.find("deepseek") != std::string::npos) {
                if (modelType.find("deepseek3") != std::string::npos) {
                    return ModelType::DEEPSEEK3_LLM;
                } else if (modelType.find("coder") != std::string::npos) {
                    return ModelType::DEEPSEEK_CODER;
                } else {
                    return ModelType::DEEPSEEK_LLM;
                }
            }
        }
        
        // 检查特殊token名称模式
        if (config.contains("added_tokens_decoder")) {
            auto tokens = config["added_tokens_decoder"];
            for (auto& item : tokens.items()) {
                if (item.value().contains("content")) {
                    std::string content = item.value()["content"];
                    if (content == "<|fim_begin|>" || content == "<|fim_end|>" || content == "<|fim_pad|>" || 
                        content == "<|fim_suf|>" || content == "<|fim_pre|>") {
                        return ModelType::QWEN;  // Qwen特有的FIM tokens
                    } else if (content.find("deepseek") != std::string::npos) {
                        return ModelType::DEEPSEEK_LLM;
                    }
                }
            }
        }
        
        // 默认返回SPM
        return ModelType::SPM;
    } catch (const std::exception& e) {
        // 如果解析失败，返回默认类型
        return ModelType::SPM;
    }
}

} // namespace cllm