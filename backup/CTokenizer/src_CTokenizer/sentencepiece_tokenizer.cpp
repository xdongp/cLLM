#include "cllm/CTokenizer/sentencepiece_tokenizer.h"
#include <sentencepiece_processor.h>
#include <fstream>
#include <sstream>
#include <nlohmann/json.hpp>

namespace cllm {

SentencePieceTokenizer::SentencePieceTokenizer(ModelType modelType) 
    : modelType_(modelType), processor_(nullptr) {
    processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
}

SentencePieceTokenizer::~SentencePieceTokenizer() = default;

bool SentencePieceTokenizer::load(const std::string& modelPath) {
    // 加载SentencePiece模型
    const auto status = processor_->Load(modelPath);
    if (!status.ok()) {
        return false;
    }

    // 加载模型配置和特殊token信息
    std::string configPath = modelPath.substr(0, modelPath.rfind('.')) + ".json";
    loadModelConfig(configPath);
    
    return true;
}

std::vector<llama_token> SentencePieceTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    if (!processor_) {
        return {};
    }

    std::vector<int> ids;
    if (addSpecialTokens) {
        processor_->Encode(text, &ids);
    } else {
        processor_->Encode(text, &ids);
    }

    // 转换为llama_token类型
    std::vector<llama_token> result;
    result.reserve(ids.size());
    for (int id : ids) {
        result.push_back(static_cast<llama_token>(id));
    }

    return result;
}

std::string SentencePieceTokenizer::decode(const std::vector<llama_token>& ids, bool skipSpecialTokens) {
    if (!processor_) {
        return "";
    }

    // 转换为int类型
    std::vector<int> int_ids;
    int_ids.reserve(ids.size());
    for (llama_token id : ids) {
        int_ids.push_back(static_cast<int>(id));
    }

    std::string result;
    processor_->Decode(int_ids, &result);

    return result;
}

int SentencePieceTokenizer::getVocabSize() const {
    if (!processor_) {
        return 0;
    }
    return processor_->GetPieceSize();
}

std::string SentencePieceTokenizer::idToToken(llama_token id) const {
    if (!processor_) {
        return "";
    }
    return processor_->IdToPiece(static_cast<int>(id));
}

llama_token SentencePieceTokenizer::tokenToId(const std::string& token) const {
    if (!processor_) {
        return -1;
    }
    return static_cast<llama_token>(processor_->PieceToId(token));
}

void SentencePieceTokenizer::loadModelConfig(const std::string& configPath) {
    std::ifstream file(configPath);
    if (!file.is_open()) {
        return;
    }

    nlohmann::json config;
    try {
        file >> config;
        
        // 读取特殊token ID
        if (config.contains("bos_token_id")) {
            bosId_ = static_cast<llama_token>(config["bos_token_id"]);
        }
        if (config.contains("eos_token_id")) {
            eosId_ = static_cast<llama_token>(config["eos_token_id"]);
        }
        if (config.contains("pad_token_id")) {
            padId_ = static_cast<llama_token>(config["pad_token_id"]);
        }
        if (config.contains("unk_token_id")) {
            unkId_ = static_cast<llama_token>(config["unk_token_id"]);
        }
        
        // 读取特殊token映射
        if (config.contains("added_tokens_decoder")) {
            auto tokens = config["added_tokens_decoder"];
            for (auto& item : tokens.items()) {
                if (item.value().contains("content")) {
                    std::string content = item.value()["content"];
                    int id = item.key().empty() ? 0 : std::stoi(item.key());
                    specialTokens_[content] = id;
                    idToTokenMap_[id] = content;
                }
            }
        }
    } catch (const std::exception& e) {
        // 忽略配置文件解析错误
    }
}

void SentencePieceTokenizer::loadSpecialTokens(const std::string& configPath) {
    // 加载特殊token
    loadModelConfig(configPath);
}

void SentencePieceTokenizer::initializeRegexPatterns() {
    // 初始化正则表达式模式（如果需要）
}

} // namespace cllm