#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <optional>

// Forward declaration since we can't include llama.h directly here due to path issues
using llama_token = int32_t;

namespace cllm {

enum class ModelType {
    AUTO,           // 自动检测
    QWEN,           // Qwen系列模型
    QWEN2,          // Qwen2系列模型
    DEEPSEEK_LLM,   // DeepSeek LLM模型
    DEEPSEEK_CODER, // DeepSeek Coder模型
    DEEPSEEK3_LLM,  // DeepSeek3 LLM模型
    LLAMA,          // Llama系列模型
    BERT,           // BERT系列模型
    GPT2,           // GPT2系列模型
    SPM,            // SentencePiece模型
    BPE,            // BPE模型
    WPM             // WordPiece模型
};

class CTokenizer {
public:
    virtual ~CTokenizer() = default;
    
    // 核心功能
    virtual std::vector<llama_token> encode(
        const std::string& text, 
        bool addSpecialTokens = true
    ) = 0;
    
    virtual std::string decode(
        const std::vector<llama_token>& ids,
        bool skipSpecialTokens = true
    ) = 0;
    
    // 词汇表操作
    virtual int getVocabSize() const = 0;
    virtual std::string idToToken(llama_token id) const = 0;
    virtual llama_token tokenToId(const std::string& token) const = 0;
    
    // 特殊Token
    virtual llama_token getBosId() const = 0;
    virtual llama_token getEosId() const = 0;
    virtual llama_token getPadId() const = 0;
    virtual llama_token getUnkId() const = 0;
    
    // 模型类型
    virtual ModelType getModelType() const = 0;
    
    // 加载模型
    virtual bool load(const std::string& modelPath) = 0;
};

} // namespace cllm