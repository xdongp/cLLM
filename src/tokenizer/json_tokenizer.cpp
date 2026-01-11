#include "cllm/tokenizer/json_tokenizer.h"
#include "cllm/common/json.h"
#include "cllm/common/logger.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace cllm {

JsonTokenizer::JsonTokenizer(const std::string& jsonPath) {
    loadFromJson(jsonPath);
}

JsonTokenizer::~JsonTokenizer() = default;

void JsonTokenizer::loadFromJson(const std::string& jsonPath) {
    try {
        nlohmann::json tokenizerJson;
        std::ifstream jsonFile(jsonPath);
        jsonFile >> tokenizerJson;

        // 加载词汇表
        auto vocab = tokenizerJson["model"]["vocab"];
        for (auto it = vocab.begin(); it != vocab.end(); ++it) {
            int id = it.value().get<int>();
            std::string token = it.key();
            tokenToId_[token] = id;
            idToToken_[id] = token;
        }

        // 加载特殊token
        auto specialTokens = tokenizerJson["added_tokens"];
        for (const auto& token : specialTokens) {
            std::string text = token["content"];
            int id = token["id"];
            if (text == "<|endoftext|>") {
                eosTokenId_ = id;
            } else if (text == "<|startoftext|>") {
                bosTokenId_ = id;
            } else if (text == "<|pad|>") {
                padTokenId_ = id;
            } else if (text == "<|unk|>") {
                unkTokenId_ = id;
            }
        }

        CLLM_INFO("JSON tokenizer loaded successfully: vocab_size=%zu", tokenToId_.size());
    } catch (const std::exception& e) {
        CLLM_ERROR("Failed to load JSON tokenizer: %s", e.what());
        throw std::runtime_error("JSON tokenizer loading failed: " + std::string(e.what()));
    }
}

std::vector<int> JsonTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    std::vector<int> tokenIds;
    
    // 简单的基于空格的分词（实际应该使用更复杂的分词算法）
    std::istringstream iss(text);
    std::string word;
    
    while (iss >> word) {
        // 查找单词对应的token ID
        auto it = tokenToId_.find(word);
        if (it != tokenToId_.end()) {
            tokenIds.push_back(it->second);
        } else {
            // 如果单词不在词汇表中，使用UNK token
            if (unkTokenId_ >= 0) {
                tokenIds.push_back(unkTokenId_);
            } else {
                // 如果没有UNK token，尝试将单词拆分为字符
                for (char c : word) {
                    std::string charStr(1, c);
                    auto charIt = tokenToId_.find(charStr);
                    if (charIt != tokenToId_.end()) {
                        tokenIds.push_back(charIt->second);
                    } else {
                        // 如果字符也不在词汇表中，跳过
                        tokenIds.push_back(-1);
                    }
                }
            }
        }
    }
    
    if (addSpecialTokens) {
        if (bosTokenId_ >= 0 && !tokenIds.empty()) {
            tokenIds.insert(tokenIds.begin(), bosTokenId_);
        }
        if (eosTokenId_ >= 0 && !tokenIds.empty()) {
            tokenIds.push_back(eosTokenId_);
        }
    }
    
    return tokenIds;
}

std::string JsonTokenizer::decode(const std::vector<int>& tokenIds, bool skipSpecialTokens) {
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

int JsonTokenizer::getVocabSize() const {
    return tokenToId_.size();
}

std::string JsonTokenizer::getTokenText(int tokenId) const {
    auto it = idToToken_.find(tokenId);
    return it != idToToken_.end() ? it->second : "[UNK]";
}

bool JsonTokenizer::isSpecialToken(int tokenId) const {
    return tokenId == padTokenId_ || 
           tokenId == eosTokenId_ || 
           tokenId == bosTokenId_ ||
           tokenId == unkTokenId_;
}

} // namespace cllm