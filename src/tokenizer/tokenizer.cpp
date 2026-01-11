#include "cllm/tokenizer/tokenizer.h"
#include "cllm/common/utils.h"
#include "cllm/common/json.h"
#include "cllm/common/logger.h"
#include <stdexcept>
#include <cstring>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <unordered_map>
#include <sentencepiece_processor.h>

namespace cllm {

Tokenizer::Tokenizer(const std::string& modelPath)
    : modelPath_(modelPath),
      padTokenId_(-1),
      eosTokenId_(-1),
      bosTokenId_(-1),
      unkTokenId_(-1),
      modelType_(ModelType::SPM),
      loaded_(false) {
    loadModel(modelPath);
}

Tokenizer::~Tokenizer() {
    unloadModel();
}

// ITokenizer接口实现
bool Tokenizer::load(const std::string& modelPath) {
    try {
        loadModel(modelPath);
        return loaded_;
    } catch (...) {
        return false;
    }
}

std::vector<int> Tokenizer::encode(const std::string& text, bool addSpecialTokens) {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    
    std::vector<int> tokenIds;
    auto status = processor_->Encode(text, &tokenIds);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece encoding failed: " + status.ToString());
    }
    
    // 添加特殊token
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

std::string Tokenizer::decode(const std::vector<int>& ids, bool skipSpecialTokens) {
    if (!loaded_) {
        throw std::runtime_error("Tokenizer is not loaded");
    }
    
    // 过滤特殊token
    std::vector<int> filteredIds;
    for (int tokenId : ids) {
        if (!skipSpecialTokens || !isSpecialToken(tokenId)) {
            filteredIds.push_back(tokenId);
        }
    }
    
    std::string text;
    auto status = processor_->Decode(filteredIds, &text);
    if (!status.ok()) {
        throw std::runtime_error("SentencePiece decoding failed: " + status.ToString());
    }
    
    return text;
}

int Tokenizer::getVocabSize() const {
    return loaded_ ? processor_->GetPieceSize() : 0;
}

std::string Tokenizer::idToToken(int id) const {
    if (!loaded_) {
        return "[UNK]";
    }
    return processor_->IdToPiece(id);
}

int Tokenizer::tokenToId(const std::string& token) const {
    if (!loaded_) {
        return -1;
    }
    return processor_->PieceToId(token);
}

int Tokenizer::getBosId() const {
    return bosTokenId_;
}

int Tokenizer::getEosId() const {
    return eosTokenId_;
}

int Tokenizer::getPadId() const {
    return padTokenId_;
}

int Tokenizer::getUnkId() const {
    return unkTokenId_;
}

ModelType Tokenizer::getModelType() const {
    return modelType_;
}

// 保留Tokenizer特有的方法
std::string Tokenizer::getTokenText(int tokenId) const {
    if (!loaded_) {
        return "[UNK]";
    }
    return processor_->IdToPiece(tokenId);
}

bool Tokenizer::isSpecialToken(int tokenId) const {
    if (!loaded_) {
        return false;
    }
    return tokenId == padTokenId_ || 
           tokenId == eosTokenId_ || 
           tokenId == bosTokenId_ ||
           processor_->IsControl(tokenId);
}

void Tokenizer::setPadToken(int tokenId) {
    padTokenId_ = tokenId;
}

void Tokenizer::setEosToken(int tokenId) {
    eosTokenId_ = tokenId;
}

void Tokenizer::setBosToken(int tokenId) {
    bosTokenId_ = tokenId;
}

int Tokenizer::getPadToken() const {
    return padTokenId_;
}

int Tokenizer::getEosToken() const {
    return eosTokenId_;
}

int Tokenizer::getBosToken() const {
    return bosTokenId_;
}

void Tokenizer::loadModel(const std::string& modelPath) {
    if (loaded_) {
        unloadModel();
    }
    
    modelPath_ = modelPath;
    
    // 初始化SentencePiece处理器
    processor_ = std::make_unique<sentencepiece::SentencePieceProcessor>();
    
    // 尝试加载SentencePiece模型
    std::string spModelPath = modelPath;
    if (spModelPath.back() != '/') {
        spModelPath += '/';
    }
    spModelPath += "tokenizer.model";
    
    auto status = processor_->Load(spModelPath);
    if (!status.ok()) {
        CLLM_ERROR("Failed to load SentencePiece model from %s: %s", 
                   spModelPath.c_str(), status.ToString().c_str());
        throw std::runtime_error("SentencePiece model loading failed: " + status.ToString());
    }
    
    // 从SentencePiece获取特殊token ID
    padTokenId_ = processor_->pad_id();
    eosTokenId_ = processor_->eos_id();
    bosTokenId_ = processor_->bos_id();
    unkTokenId_ = processor_->unk_id();
    
    CLLM_INFO("SentencePiece model loaded successfully: vocab_size=%d, pad=%d, eos=%d, bos=%d",
              processor_->GetPieceSize(), padTokenId_, eosTokenId_, bosTokenId_);
    
    loaded_ = true;
}

void Tokenizer::unloadModel() {
    processor_.reset();
    loaded_ = false;
}

bool Tokenizer::isLoaded() const {
    return loaded_;
}

}