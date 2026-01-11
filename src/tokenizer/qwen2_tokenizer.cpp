#include "cllm/tokenizer/qwen2_tokenizer.h"
#include "cllm/common/json.h"
#include "cllm/common/logger.h"
#include <fstream>
#include <stdexcept>

namespace cllm {

Qwen2Tokenizer::Qwen2Tokenizer(const std::string& modelPath) {
    // 加载JSON格式的分词器
    std::ifstream vocabFile(modelPath + "/vocab.json");
    nlohmann::json vocab = nlohmann::json::parse(vocabFile);
    
    // 初始化词汇表映射
    for (auto& [token, id] : vocab.items()) {
        tokenToId_[token] = id;
        idToToken_[id] = token;
    }
    
    // 加载特殊token配置
    loadSpecialTokens(modelPath + "/tokenizer_config.json");
    
    CLLM_INFO("Qwen2Tokenizer loaded successfully. Vocab size: %zu", tokenToId_.size());
}

Qwen2Tokenizer::~Qwen2Tokenizer() = default;

void Qwen2Tokenizer::loadSpecialTokens(const std::string& configPath) {
    try {
        std::ifstream configFile(configPath);
        nlohmann::json config = nlohmann::json::parse(configFile);
        
        // 加载特殊token
        auto addedTokens = config["added_tokens_decoder"];
        for (auto& [idStr, tokenInfo] : addedTokens.items()) {
            int id = std::stoi(idStr);
            std::string content = tokenInfo["content"];
            bool isSpecial = tokenInfo["special"];
            
            specialTokenToId_[content] = id;
            idToSpecialToken_[id] = content;
            
            if (content == "<|endoftext|>") {
                padTokenId_ = id;
            } else if (content == "<|im_end|>") {
                eosTokenId_ = id;
            }
        }
        
        // 设置BOS和UNK token
        if (config.contains("bos_token_id")) {
            bosTokenId_ = config["bos_token_id"].get<int>();
        }
        if (config.contains("unk_token_id")) {
            unkTokenId_ = config["unk_token_id"].get<int>();
        }
        
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to load special tokens: %s", e.what());
        throw;
    }
}

std::vector<int> Qwen2Tokenizer::encode(const std::string& text, bool addSpecialTokens) {
    std::vector<int> tokenIds;
    
    // 先处理特殊token
    std::string remaining = text;
    while (!remaining.empty()) {
        bool foundSpecial = false;
        for (const auto& [token, id] : specialTokenToId_) {
            if (remaining.compare(0, token.size(), token) == 0) {
                tokenIds.push_back(id);
                remaining = remaining.substr(token.size());
                foundSpecial = true;
                break;
            }
        }
        
        if (!foundSpecial) {
            // 1. 处理数字和符号组合
            if (std::isdigit(remaining[0]) || std::ispunct(remaining[0])) {
                for (char c : remaining) {
                    if (std::isdigit(c) || std::ispunct(c)) {
                        std::string charStr(1, c);
                        auto it = tokenToId_.find(charStr);
                        if (it != tokenToId_.end()) {
                            tokenIds.push_back(it->second);
                        } else if (unkTokenId_ >= 0) {
                            tokenIds.push_back(unkTokenId_);
                        }
                    } else {
                        break;
                    }
                }
                continue;
            }
            
            // 2. 处理子词分词
            size_t maxLen = std::min(remaining.size(), size_t(10)); // 最大子词长度
            for (size_t len = maxLen; len >= 1; --len) {
                std::string subword = remaining.substr(0, len);
                auto it = tokenToId_.find(subword);
                if (it != tokenToId_.end()) {
                    tokenIds.push_back(it->second);
                    remaining = remaining.substr(len);
                    foundSpecial = true;
                    break;
                }
            }
            
            // 3. 回退到按空格分词
            if (!foundSpecial) {
                size_t spacePos = remaining.find(' ');
                std::string word = (spacePos == std::string::npos) ? remaining : remaining.substr(0, spacePos);
                
                if (unkTokenId_ >= 0) {
                    tokenIds.push_back(unkTokenId_);
                }
                remaining = (spacePos == std::string::npos) ? "" : remaining.substr(spacePos + 1);
            }
        }
    }
    
    if (addSpecialTokens) {
        if (bosTokenId_ >= 0) {
            tokenIds.insert(tokenIds.begin(), bosTokenId_);
        }
        if (eosTokenId_ >= 0) {
            tokenIds.push_back(eosTokenId_);
        }
    }
    
    return tokenIds;
}

std::string Qwen2Tokenizer::decode(const std::vector<int>& tokenIds, bool skipSpecialTokens) {
    std::string text;
    for (int id : tokenIds) {
        if (skipSpecialTokens && isSpecialToken(id)) {
            continue;
        }
        
        auto it = idToToken_.find(id);
        if (it != idToToken_.end()) {
            // 添加空格分隔（除了标点符号）
            if (!text.empty() && 
                !std::ispunct(it->second.front()) && 
                !std::ispunct(text.back())) {
                text += " ";
            }
            text += it->second;
        } else {
            text += "[UNK]";
        }
    }
    return text;
}

int Qwen2Tokenizer::getVocabSize() const {
    return static_cast<int>(tokenToId_.size());
}

std::string Qwen2Tokenizer::getTokenText(int tokenId) const {
    auto it = idToToken_.find(tokenId);
    return it != idToToken_.end() ? it->second : "[UNK]";
}

bool Qwen2Tokenizer::isSpecialToken(int tokenId) const {
    return idToSpecialToken_.find(tokenId) != idToSpecialToken_.end() ||
           tokenId == padTokenId_ || 
           tokenId == eosTokenId_ || 
           tokenId == bosTokenId_;
}

} // namespace cllm