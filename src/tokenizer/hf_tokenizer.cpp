#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/common/logger.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace cllm {

HFTokenizer::HFTokenizer(ModelType modelType) 
    : modelType_(modelType) {
}

HFTokenizer::~HFTokenizer() = default;

bool HFTokenizer::load(const std::string& modelPath) {
#ifdef USE_TOKENIZERS_CPP
    namespace fs = std::filesystem;
    
    // Step 1: 检测tokenizer.json
    std::string tokenizerJsonPath = modelPath;
    if (fs::is_directory(modelPath)) {
        tokenizerJsonPath = (fs::path(modelPath) / "tokenizer.json").string();
    }
    
    if (!fs::exists(tokenizerJsonPath)) {
        CLLM_ERROR("tokenizer.json not found: %s", tokenizerJsonPath.c_str());
        return false;
    }
    
    try {
        // Step 2: 加载tokenizer
        std::ifstream f(tokenizerJsonPath);
        if (!f.is_open()) {
            CLLM_ERROR("Failed to open tokenizer.json: %s", tokenizerJsonPath.c_str());
            return false;
        }
        std::string json_blob((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        
        tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_blob);
        
        if (!tokenizer_) {
            CLLM_ERROR("Failed to create tokenizer from blob: %s", tokenizerJsonPath.c_str());
            return false;
        }
        
        // Step 3: 加载配置 (获取特殊Token IDs)
        loadConfig(modelPath);
        
        CLLM_INFO("✅ HFTokenizer loaded successfully from: %s", tokenizerJsonPath.c_str());
        CLLM_INFO("   Vocab size: %d, BOS: %d, EOS: %d", getVocabSize(), bosId_, eosId_);
        
        return true;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Exception loading HFTokenizer: %s", e.what());
        return false;
    }
#else
    CLLM_ERROR("HFTokenizer requires USE_TOKENIZERS_CPP to be enabled");
    CLLM_ERROR("Please rebuild with: cmake .. -DUSE_TOKENIZERS_CPP=ON");
    return false;
#endif
}

std::vector<int> HFTokenizer::encode(const std::string& text, bool addSpecialTokens) {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) {
        CLLM_ERROR("Tokenizer not loaded");
        return {};
    }
    
    try {
        // tokenizers-cpp API: Encode(text)
        auto encoding = tokenizer_->Encode(text);
        
        // 转换为std::vector<int>
        std::vector<int> ids;
        ids.reserve(encoding.size());
        for (auto id : encoding) {
            ids.push_back(static_cast<int>(id));
        }
        
        return ids;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Encode failed: %s", e.what());
        return {};
    }
#else
    CLLM_ERROR("HFTokenizer::encode requires USE_TOKENIZERS_CPP");
    return {};
#endif
}

std::string HFTokenizer::decode(const std::vector<int>& ids, bool skipSpecialTokens) {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) {
        CLLM_ERROR("Tokenizer not loaded");
        return "";
    }
    
    try {
        // 转换为tokenizers-cpp需要的类型（同时按需跳过特殊 tokens）
        std::vector<int32_t> tokenIds;
        tokenIds.reserve(ids.size());
        for (int id : ids) {
            if (skipSpecialTokens && isSpecialToken(id)) {
                continue;
            }
            tokenIds.push_back(static_cast<int32_t>(id));
        }

        // Decode
        std::string text = tokenizer_->Decode(tokenIds);
        return text;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Decode failed: %s", e.what());
        return "";
    }
#else
    CLLM_ERROR("HFTokenizer::decode requires USE_TOKENIZERS_CPP");
    return "";
#endif
}

void HFTokenizer::loadConfig(const std::string& modelPath) {
    namespace fs = std::filesystem;
    
    // 尝试多个配置文件
    std::vector<std::string> configFiles = {
        "tokenizer_config.json",
        "config.json"
    };
    
    for (const auto& configFile : configFiles) {
        std::string configPath = (fs::path(modelPath) / configFile).string();
        
        if (!fs::exists(configPath)) continue;
        
        std::ifstream f(configPath);
        if (!f.is_open()) continue;
        
        try {
            auto config = nlohmann::json::parse(f);
            
            // 读取特殊Token IDs
            if (config.contains("bos_token_id")) {
                bosId_ = config["bos_token_id"].get<int>();
            }
            if (config.contains("eos_token_id")) {
                eosId_ = config["eos_token_id"].get<int>();
            }
            if (config.contains("pad_token_id")) {
                if (!config["pad_token_id"].is_null()) {
                    padId_ = config["pad_token_id"].get<int>();
                }
            }
            if (config.contains("unk_token_id")) {
                unkId_ = config["unk_token_id"].get<int>();
            }
            
            // 读取added_tokens_decoder (完整的特殊Token列表)
            if (config.contains("added_tokens_decoder")) {
                auto tokens = config["added_tokens_decoder"];
                for (auto& [key, value] : tokens.items()) {
                    int tokenId = std::stoi(key);
                    specialTokenIds_.insert(tokenId);
                }
            }
            
            CLLM_INFO("Loaded config from: %s", configPath.c_str());
            break;
            
        } catch (const std::exception& e) {
            CLLM_WARN("Failed to parse %s: %s", configPath.c_str(), e.what());
        }
    }
}

int HFTokenizer::getVocabSize() const {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) return 0;
    return tokenizer_->GetVocabSize();
#else
    return 0;
#endif
}

std::string HFTokenizer::idToToken(int id) const {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) return "[UNK]";
    
    try {
        return tokenizer_->IdToToken(static_cast<uint32_t>(id));
    } catch (...) {
        return "[UNK]";
    }
#else
    return "[UNK]";
#endif
}

int HFTokenizer::tokenToId(const std::string& token) const {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) return unkId_;
    
    try {
        return static_cast<int>(tokenizer_->TokenToId(token));
    } catch (...) {
        return unkId_;
    }
#else
    return unkId_;
#endif
}

bool HFTokenizer::isSpecialToken(int tokenId) const {
    return specialTokenIds_.count(tokenId) > 0;
}

std::vector<std::string> HFTokenizer::tokenize(const std::string& text) {
#ifdef USE_TOKENIZERS_CPP
    if (!tokenizer_) return {};
    
    auto encoding = tokenizer_->Encode(text);
    std::vector<std::string> tokens;
    for (auto id : encoding) {
        tokens.push_back(tokenizer_->IdToToken(id));
    }
    return tokens;
#else
    return {};
#endif
}

// Getter实现
int HFTokenizer::getBosId() const { return bosId_; }
int HFTokenizer::getEosId() const { return eosId_; }
int HFTokenizer::getPadId() const { return padId_; }
int HFTokenizer::getUnkId() const { return unkId_; }
ModelType HFTokenizer::getModelType() const { return modelType_; }

void HFTokenizer::loadSpecialTokens(const std::string& configPath) {
    // 已合并到loadConfig中
}

} // namespace cllm