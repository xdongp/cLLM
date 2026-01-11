#include "cllm/tokenizer/native_tokenizer.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <sstream>

namespace cllm {

NativeTokenizer::NativeTokenizer(ModelType modelType) 
    : modelType_(modelType) {}

NativeTokenizer::~NativeTokenizer() = default;

bool NativeTokenizer::load(const std::string& modelPath) {
    modelPath_ = modelPath;
    processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    
    // 加载SentencePiece模型
    std::string spModelPath = modelPath;
    if (spModelPath.back() != '/') spModelPath += '/';
    spModelPath += "tokenizer.model";
    
    auto status = processor_->Load(spModelPath);
    if (!status.ok()) {
        return false;
    }
    
    // 加载特殊Token配置
    loadSpecialTokens(modelPath + "/config.json");
    return true;
}

std::vector<int> NativeTokenizer::encode(const std::string& text, bool addSpecialTokens) {
    std::string processed = preprocessText(text);
    
    std::vector<int> ids;
    auto status = processor_->Encode(processed, &ids);
    if (!status.ok()) return {};
    
    if (addSpecialTokens) {
        if (bosId_ >= 0) ids.insert(ids.begin(), bosId_);
        if (eosId_ >= 0) ids.push_back(eosId_);
    }
    return ids;
}

std::string NativeTokenizer::decode(const std::vector<int>& ids, bool skipSpecialTokens) {
    std::vector<int> filteredIds;
    for (int id : ids) {
        if (!skipSpecialTokens || (id != bosId_ && id != eosId_ && id != padId_)) {
            filteredIds.push_back(id);
        }
    }
    
    std::string text;
    auto status = processor_->Decode(filteredIds, &text);
    return status.ok() ? text : "";
}

int NativeTokenizer::getVocabSize() const {
    return processor_ ? processor_->GetPieceSize() : 0;
}

std::string NativeTokenizer::idToToken(int id) const {
    return processor_ ? processor_->IdToPiece(id) : "[UNK]";
}

int NativeTokenizer::tokenToId(const std::string& token) const {
    return processor_ ? processor_->PieceToId(token) : unkId_;
}

void NativeTokenizer::loadSpecialTokens(const std::string& configPath) {
    std::ifstream f(configPath);
    if (!f.is_open()) return;
    
    auto config = nlohmann::json::parse(f);
    
    if (config.contains("bos_token_id")) bosId_ = config["bos_token_id"];
    if (config.contains("eos_token_id")) eosId_ = config["eos_token_id"];
    if (config.contains("pad_token_id")) padId_ = config["pad_token_id"];
    if (config.contains("unk_token_id")) unkId_ = config["unk_token_id"];
}

std::string NativeTokenizer::preprocessText(const std::string& text) {
    // Unicode NFC 规范化：确保相同视觉字符使用相同的编码
    // 例如 "café" 的不同编码形式（组合字符 vs 预组合字符）会被统一
    std::string result = UnicodeUtils::normalizeNFC(text);
    
    // 模型特定的预处理可以在这里添加
    
    return result;
}

// 获取特殊Token IDs
int NativeTokenizer::getBosId() const { return bosId_; }
int NativeTokenizer::getEosId() const { return eosId_; }
int NativeTokenizer::getPadId() const { return padId_; }
int NativeTokenizer::getUnkId() const { return unkId_; }
ModelType NativeTokenizer::getModelType() const { return modelType_; }

} // namespace cllm