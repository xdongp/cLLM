#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/common/utils.h"
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cctype>

namespace cllm {

GGUFTokenizer::GGUFTokenizer() 
    : bosTokenId_(-1),
      eosTokenId_(-1),
      padTokenId_(-1),
      unkTokenId_(-1),
      vocabSize_(0),
      modelType_(ModelType::LLAMA),
      loaded_(false) {
}

GGUFTokenizer::~GGUFTokenizer() {
}

bool GGUFTokenizer::load(const std::string& modelPath) {
    try {
        modelPath_ = modelPath;
        
        // 创建GGUFLoader实例
        GGUFLoader loader(modelPath, true);
        
        // 加载GGUF文件
        if (!loader.load()) {
            CLLM_ERROR("GGUFTokenizer::load: Failed to load GGUF file: %s", modelPath.c_str());
            return false;
        }
        
        // 从loader中加载tokenizer数据
        return loadFromGGUFLoader(loader);
        
    } catch (const std::exception& e) {
        CLLM_ERROR("GGUFTokenizer::load: Exception: %s", e.what());
        return false;
    }
}

bool GGUFTokenizer::loadFromGGUFLoader(const GGUFLoader& loader) {
    try {
        // 加载词汇表
        loadVocabulary(loader);
        
        // 加载特殊tokens
        loadSpecialTokens(loader);
        
        // 加载合并规则
        loadMergeRules(loader);
        
        // 初始化编码逻辑
        initializeEncoding();
        
        loaded_ = true;
        CLLM_INFO("GGUFTokenizer::loadFromGGUFLoader: Successfully loaded tokenizer");
        CLLM_INFO("GGUFTokenizer::loadFromGGUFLoader: Vocab size: %d", vocabSize_);
        CLLM_INFO("GGUFTokenizer::loadFromGGUFLoader: Model type: %d", static_cast<int>(modelType_));
        
        return true;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("GGUFTokenizer::loadFromGGUFLoader: Exception: %s", e.what());
        return false;
    }
}

void GGUFTokenizer::loadVocabulary(const GGUFLoader& loader) {
    // 获取loader的元数据
    const auto& metadata = loader.getMetadata();
    
    // 提取词汇表大小
    if (metadata.count("tokenizer.ggml.vocab_size") > 0) {
        vocabSize_ = metadata.at("tokenizer.ggml.vocab_size").value.i32_val;
    } else {
        CLLM_WARN("GGUFTokenizer::loadVocabulary: vocab_size not found, using default: 32000");
        vocabSize_ = 32000;
    }
    
    // 尝试从多种可能的词汇表字段中加载
    std::vector<std::string> vocabFields = {
        "tokenizer.ggml.vocab",
        "tokenizer.tokens",
        "tokens"
    };
    
    for (const auto& field : vocabFields) {
        if (metadata.count(field) > 0) {
            const auto& vocabData = metadata.at(field);
            
            if (vocabData.type == GGUFMetadata::ValueType::ARRAY) {
                // 处理数组类型的词汇表
                const auto& vocabArray = vocabData.array_val;
                for (size_t i = 0; i < vocabArray.size(); ++i) {
                    idToTokenMap_[i] = vocabArray[i];
                    tokenToIdMap_[vocabArray[i]] = i;
                }
                CLLM_INFO("GGUFTokenizer::loadVocabulary: Loaded %zu tokens from field: %s", 
                          vocabArray.size(), field.c_str());
                return;
            }
        }
    }
    
    // 如果词汇表不是作为元数据存储，可能是作为张量存储
    // 这里需要特殊处理，因为张量数据的格式取决于模型
    CLLM_WARN("GGUFTokenizer::loadVocabulary: Vocabulary not found in metadata fields");
}

void GGUFTokenizer::loadSpecialTokens(const GGUFLoader& loader) {
    const auto& metadata = loader.getMetadata();
    
    // 提取特殊token ID
    if (metadata.count("tokenizer.ggml.bos_token_id") > 0) {
        bosTokenId_ = metadata.at("tokenizer.ggml.bos_token_id").value.i32_val;
    }
    
    if (metadata.count("tokenizer.ggml.eos_token_id") > 0) {
        eosTokenId_ = metadata.at("tokenizer.ggml.eos_token_id").value.i32_val;
    }
    
    if (metadata.count("tokenizer.ggml.pad_token_id") > 0) {
        padTokenId_ = metadata.at("tokenizer.ggml.pad_token_id").value.i32_val;
    }
    
    if (metadata.count("tokenizer.ggml.unk_token_id") > 0) {
        unkTokenId_ = metadata.at("tokenizer.ggml.unk_token_id").value.i32_val;
    }
    
    // 输出特殊token信息
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: BOS token ID: %d", bosTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: EOS token ID: %d", eosTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: PAD token ID: %d", padTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: UNK token ID: %d", unkTokenId_);
}

void GGUFTokenizer::loadMergeRules(const GGUFLoader& loader) {
    const auto& metadata = loader.getMetadata();
    
    // 尝试从多种可能的合并规则字段中加载
    std::vector<std::string> mergeFields = {
        "tokenizer.ggml.merges",
        "tokenizer.merges",
        "merges"
    };
    
    for (const auto& field : mergeFields) {
        if (metadata.count(field) > 0) {
            const auto& mergesData = metadata.at(field);
            
            if (mergesData.type == GGUFMetadata::ValueType::ARRAY) {
                // 处理数组类型的合并规则
                const auto& mergeArray = mergesData.array_val;
                for (const auto& mergeStr : mergeArray) {
                    // 合并规则格式通常是 "a b"，表示将a和b合并
                    size_t spacePos = mergeStr.find(' ');
                    if (spacePos != std::string::npos) {
                        std::string first = mergeStr.substr(0, spacePos);
                        std::string second = mergeStr.substr(spacePos + 1);
                        mergeRules_.emplace_back(first, second);
                    }
                }
                CLLM_INFO("GGUFTokenizer::loadMergeRules: Loaded %zu merge rules from field: %s", 
                          mergeRules_.size(), field.c_str());
                return;
            }
        }
    }
    
    // 如果没有找到合并规则，可能是SentencePiece格式
    CLLM_WARN("GGUFTokenizer::loadMergeRules: Merge rules not found in metadata fields");
}

void GGUFTokenizer::initializeEncoding() {
    // 初始化编码/解码逻辑
    // 这里简化实现，实际需要根据模型类型选择合适的编码算法
    CLLM_INFO("GGUFTokenizer::initializeEncoding: Using default encoding implementation");
}

std::vector<int> GGUFTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    if (!loaded_) {
        throw std::runtime_error("GGUFTokenizer::encode: Tokenizer not loaded");
    }
    
    // 这里实现简化的编码逻辑
    std::vector<int> tokenIds;
    
    // 如果需要，添加BOS token
    if (addSpecialTokens && bosTokenId_ != -1) {
        tokenIds.push_back(bosTokenId_);
    }
    
    // 简单的空格分割实现（实际需要使用BPE或其他算法）
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // 查找token
        auto it = tokenToIdMap_.find(word);
        if (it != tokenToIdMap_.end()) {
            tokenIds.push_back(it->second);
        } else {
            // 如果找不到，使用UNK token
            if (unkTokenId_ != -1) {
                tokenIds.push_back(unkTokenId_);
            }
        }
    }
    
    // 如果需要，添加EOS token
    if (addSpecialTokens && eosTokenId_ != -1) {
        tokenIds.push_back(eosTokenId_);
    }
    
    return tokenIds;
}

std::string GGUFTokenizer::decode(const std::vector<int>& ids, bool skipSpecialTokens) {
    if (!loaded_) {
        throw std::runtime_error("GGUFTokenizer::decode: Tokenizer not loaded");
    }
    
    // 这里实现简化的解码逻辑
    std::string text;
    
    for (int id : ids) {
        // 如果需要跳过特殊token
        if (skipSpecialTokens) {
            if (id == bosTokenId_ || id == eosTokenId_ || id == padTokenId_) {
                continue;
            }
        }
        
        // 查找token
        auto it = idToTokenMap_.find(id);
        if (it != idToTokenMap_.end()) {
            text += it->second + " ";
        } else {
            // 如果找不到，使用UNK token
            text += "[UNK] ";
        }
    }
    
    // 移除末尾的空格
    if (!text.empty() && text.back() == ' ') {
        text.pop_back();
    }
    
    return text;
}

int GGUFTokenizer::getVocabSize() const {
    return vocabSize_;
}

std::string GGUFTokenizer::idToToken(int id) const {
    if (!loaded_) {
        throw std::runtime_error("GGUFTokenizer::idToToken: Tokenizer not loaded");
    }
    
    auto it = idToTokenMap_.find(id);
    if (it != idToTokenMap_.end()) {
        return it->second;
    }
    
    return "[UNK]";
}

int GGUFTokenizer::tokenToId(const std::string& token) const {
    if (!loaded_) {
        throw std::runtime_error("GGUFTokenizer::tokenToId: Tokenizer not loaded");
    }
    
    auto it = tokenToIdMap_.find(token);
    if (it != tokenToIdMap_.end()) {
        return it->second;
    }
    
    return unkTokenId_;
}

int GGUFTokenizer::getBosId() const {
    return bosTokenId_;
}

int GGUFTokenizer::getEosId() const {
    return eosTokenId_;
}

int GGUFTokenizer::getPadId() const {
    return padTokenId_;
}

int GGUFTokenizer::getUnkId() const {
    return unkTokenId_;
}

ModelType GGUFTokenizer::getModelType() const {
    return modelType_;
}

} // namespace cllm