#include "cllm/tokenizer/unified_tokenizer.h"
#include "cllm/tokenizer/qwen2_tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>
#include <regex>
#include <llama.h>

namespace cllm {

// 内部实现结构
struct UnifiedTokenizerImpl {
    llama_model* model;
    llama_vocab* vocab;
    UnifiedTokenizer::ModelType modelType;
    
    UnifiedTokenizerImpl() : model(nullptr), vocab(nullptr), modelType(UnifiedTokenizer::AUTO) {}
    ~UnifiedTokenizerImpl() {
        if (model) {
            llama_model_free(model);
            model = nullptr;
        }
    }
};

// 从llama.cpp源码中提取的配置检测逻辑
UnifiedTokenizer::ModelType UnifiedTokenizer::detectModelType(const std::string& configPath) {
    try {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open config file: " + configPath);
        }
        
        nlohmann::json config;
        file >> config;
        
        // 检查tokenizer_class字段
        if (config.contains("tokenizer_class")) {
            std::string tokenizerClass = config["tokenizer_class"].get<std::string>();
            
            if (tokenizerClass.find("Qwen") != std::string::npos) {
                return QWEN;
            } else if (tokenizerClass.find("DeepSeek") != std::string::npos) {
                if (tokenizerClass.find("DeepSeek3") != std::string::npos) {
                    return DEEPSEEK3_LLM;
                } else if (tokenizerClass.find("Coder") != std::string::npos) {
                    return DEEPSEEK_CODER;
                } else {
                    return DEEPSEEK_LLM;
                }
            }
        }
        
        // 检查chat_template字段
        if (config.contains("chat_template")) {
            std::string chatTemplate = config["chat_template"].get<std::string>();
            
            if (chatTemplate.find("qwen") != std::string::npos) {
                return QWEN;
            } else if (chatTemplate.find("deepseek") != std::string::npos) {
                if (chatTemplate.find("deepseek3") != std::string::npos) {
                    return DEEPSEEK3_LLM;
                } else if (chatTemplate.find("coder") != std::string::npos) {
                    return DEEPSEEK_CODER;
                } else {
                    return DEEPSEEK_LLM;
                }
            }
        }
        
        // 检查model_type字段
        if (config.contains("model_type")) {
            std::string modelType = config["model_type"].get<std::string>();
            
            if (modelType.find("qwen") != std::string::npos) {
                return QWEN;
            } else if (modelType.find("deepseek") != std::string::npos) {
                if (modelType.find("deepseek3") != std::string::npos) {
                    return DEEPSEEK3_LLM;
                } else if (modelType.find("coder") != std::string::npos) {
                    return DEEPSEEK_CODER;
                } else {
                    return DEEPSEEK_LLM;
                }
            }
        }
        
        // 检查特殊token名称
        if (config.contains("added_tokens_decoder")) {
            auto tokens = config["added_tokens_decoder"];
            for (auto& item : tokens.items()) {
                if (item.value().contains("content")) {
                    std::string content = item.value()["content"];
                    if (content == "" || content == "" || content == "" || 
                        content == "" || content == "") {
                        return QWEN;  // Qwen特有的FIM tokens
                    }
                }
            }
        }
        
        // 默认返回AUTO
        return AUTO;
    } catch (const std::exception& e) {
        // 如果解析失败，返回AUTO
        return AUTO;
    }
}

UnifiedTokenizer::UnifiedTokenizer(const std::string& modelPath, ModelType modelType)
    : modelPath_(modelPath), modelType_(modelType) {
    
    // 如果模型类型为AUTO，尝试自动检测
    if (modelType_ == AUTO) {
        std::string configPath = modelPath + "/config.json";
        if (modelPath.find_last_of('/') != std::string::npos) {
            configPath = modelPath + "/config.json";
        } else if (modelPath.find_last_of('.') != std::string::npos) {
            // 如果modelPath是文件路径，则取目录
            size_t lastSlash = modelPath.find_last_of('/');
            if (lastSlash != std::string::npos) {
                configPath = modelPath.substr(0, lastSlash) + "/config.json";
            }
        }
        
        modelType_ = detectModelType(configPath);
    }
    
    // 初始化分词器
    initializeTokenizer();
}

UnifiedTokenizer::~UnifiedTokenizer() {
    // 清理内部实现
    if (tokenizerImpl_) {
        // llama_vocab的清理
        // 这里需要调用适当的清理函数
    }
}

void UnifiedTokenizer::initializeTokenizer() {
    // 由于我们直接使用llama.cpp的实现，需要加载模型词汇表
    // 这里我们简化实现，实际上需要从模型文件或配置中加载词汇表
    
    // 创建内部实现结构
    auto deleter = [](void* p) { /* 适当的清理函数 */ };
    tokenizerImpl_ = std::unique_ptr<void, decltype(deleter)>(
        new UnifiedTokenizerImpl(), deleter
    );
    
    // 为了演示目的，我们暂时使用Qwen2Tokenizer作为后备实现
    // 实际实现中，这里应该使用llama.cpp的分词器
    vocabSize_ = 151643; // Qwen2的典型词汇表大小
    
    // 加载特殊tokens
    std::string configPath = modelPath_;
    if (configPath.find("config.json") == std::string::npos) {
        if (configPath.back() != '/') {
            configPath += "/";
        }
        configPath += "config.json";
    }
    loadSpecialTokens(configPath);
}

void UnifiedTokenizer::loadModelConfig(const std::string& configPath) {
    // 加载模型配置
    // 这里简化实现
}

void UnifiedTokenizer::loadSpecialTokens(const std::string& configPath) {
    try {
        std::ifstream file(configPath);
        if (!file.is_open()) {
            return; // 配置文件不存在，使用默认值
        }
        
        nlohmann::json config;
        file >> config;
        
        // 提取特殊token信息
        if (config.contains("bos_token_id")) {
            bosTokenId_ = config["bos_token_id"].get<int>();
        }
        if (config.contains("eos_token_id")) {
            eosTokenId_ = config["eos_token_id"].get<int>();
        }
        if (config.contains("pad_token_id")) {
            padTokenId_ = config["pad_token_id"].get<int>();
        }
        if (config.contains("unk_token_id")) {
            unkTokenId_ = config["unk_token_id"].get<int>();
        }
        
        // 从added_tokens_decoder获取特殊token
        if (config.contains("added_tokens_decoder")) {
            auto tokens = config["added_tokens_decoder"];
            for (auto& item : tokens.items()) {
                int tokenId = std::stoi(item.key());
                if (item.value().contains("content")) {
                    std::string content = item.value()["content"];
                    specialTokenToId_[content] = tokenId;
                    idToSpecialToken_[tokenId] = content;
                }
            }
        }
        
        // 根据模型类型设置默认特殊token
        if (modelType_ == QWEN) {
            // Qwen模型的特殊token
            if (specialTokenToId_.find("<|endoftext|>") != specialTokenToId_.end()) {
                eosTokenId_ = specialTokenToId_["<|endoftext|>"];
            }
            if (specialTokenToId_.find("<|im_start|>") != specialTokenToId_.end()) {
                // 这可能是Qwen的特殊token
            }
        } else if (modelType_ == DEEPSEEK_LLM || modelType_ == DEEPSEEK_CODER || modelType_ == DEEPSEEK3_LLM) {
            // DeepSeek模型的特殊token
            if (specialTokenToId_.find("<|eos_token|>") != specialTokenToId_.end()) {
                eosTokenId_ = specialTokenToId_["<|eos_token|>"];
            }
            if (specialTokenToId_.find("<|endoftext|>") != specialTokenToId_.end()) {
                eosTokenId_ = specialTokenToId_["<|endoftext|>"];
            }
        }
    } catch (const std::exception& e) {
        // 配置加载失败，使用默认值
    }
}

std::vector<int> UnifiedTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    // 实际实现中，这里应该调用llama_vocab的tokenize函数
    // 为了演示，我们使用Qwen2Tokenizer作为后备
    
    if (modelType_ == QWEN) {
        // 对于Qwen模型，使用专门的Qwen分词器
        Qwen2Tokenizer qwenTokenizer(modelPath_);
        return qwenTokenizer.encode(text, addSpecialTokens);
    } else {
        // 对于其他模型，这里应该使用llama.cpp的分词器实现
        // 为了演示目的，我们返回一个简化的实现
        std::vector<int> result;
        
        // 实际上需要调用llama.cpp的tokenize函数
        // llama_tokenize(vocab, text.c_str(), text.length(), tokens, max_tokens, addSpecialTokens, true);
        
        // 模拟分词结果
        if (!text.empty()) {
            // 这里应该使用实际的分词逻辑
            // 为了演示，我们返回一个简单的模拟结果
            result.push_back(1); // 假设1是BOS token
            // 实际分词逻辑应该在这里实现
            result.push_back(2); // 假设2是EOS token
        }
        
        return result;
    }
}

std::string UnifiedTokenizer::decode(const std::vector<int>& tokenIds, bool skipSpecialTokens) {
    // 实际实现中，这里应该调用llama_vocab的detokenize函数
    
    if (modelType_ == QWEN) {
        // 对于Qwen模型，使用专门的Qwen分词器
        Qwen2Tokenizer qwenTokenizer(modelPath_);
        return qwenTokenizer.decode(tokenIds, skipSpecialTokens);
    } else {
        // 对于其他模型，这里应该使用llama.cpp的分词器实现
        std::string result;
        
        // 实际上需要调用llama.cpp的detokenize函数
        // llama_detokenize(vocab, tokenIds.data(), tokenIds.size(), text_buffer, buffer_size, skipSpecialTokens, true);
        
        // 模拟解码结果
        for (int tokenId : tokenIds) {
            auto it = idToSpecialToken_.find(tokenId);
            if (it != idToSpecialToken_.end()) {
                if (!skipSpecialTokens) {
                    result += it->second;
                }
            } else {
                // 这里应该查找普通token的文本表示
                result += "[TOKEN_" + std::to_string(tokenId) + "]";
            }
        }
        
        return result;
    }
}

int UnifiedTokenizer::getVocabSize() const {
    // 实际实现中，应该返回llama_vocab的实际词汇表大小
    return vocabSize_;
}

std::string UnifiedTokenizer::getTokenText(int tokenId) const {
    // 实际实现中，应该调用llama_vocab_get_text
    auto it = idToSpecialToken_.find(tokenId);
    if (it != idToSpecialToken_.end()) {
        return it->second;
    }
    
    // 这里应该调用llama_vocab_get_text
    return "[TOKEN_" + std::to_string(tokenId) + "]";
}

bool UnifiedTokenizer::isSpecialToken(int tokenId) const {
    // 实际实现中，应该调用llama_vocab的相应函数
    return idToSpecialToken_.find(tokenId) != idToSpecialToken_.end();
}

} // namespace cllm