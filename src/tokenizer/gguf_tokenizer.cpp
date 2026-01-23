#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/common/logger.h"
#include "cllm/common/utils.h"
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <cctype>
#include <limits>

namespace cllm {

namespace {

static bool isAsciiWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

static bool isAsciiAlnum(char c) {
    return (c >= '0' && c <= '9') ||
           (c >= 'A' && c <= 'Z') ||
           (c >= 'a' && c <= 'z');
}

static bool isAsciiPunct(char c) {
    return std::ispunct(static_cast<unsigned char>(c)) != 0;
}

static std::vector<std::string> splitUtf8(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t len = 1;
        if (c < 0x80) {
            len = 1;
        } else if ((c >> 5) == 0x6) {
            len = 2;
        } else if ((c >> 4) == 0xE) {
            len = 3;
        } else if ((c >> 3) == 0x1E) {
            len = 4;
        }
        if (i + len > s.size()) {
            len = 1;
        }
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

static std::string codepointToUtf8(uint32_t cp) {
    std::string out;
    if (cp <= 0x7F) {
        out.push_back(static_cast<char>(cp));
    } else if (cp <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((cp >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else if (cp <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((cp >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((cp >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
    }
    return out;
}

static std::string byteEncodeWithTable(
    const std::string& s,
    const std::unordered_map<unsigned char, std::string>& encoder
) {
    std::string out;
    for (unsigned char ch : s) {
        auto it = encoder.find(ch);
        if (it != encoder.end()) {
            out += it->second;
        } else {
            out.push_back(static_cast<char>(ch));
        }
    }
    return out;
}

static std::string byteDecodeWithTable(
    const std::string& s,
    const std::unordered_map<std::string, unsigned char>& decoder
) {
    if (decoder.empty()) {
        return s;
    }
    std::string out;
    auto chars = splitUtf8(s);
    for (const auto& ch : chars) {
        auto it = decoder.find(ch);
        if (it != decoder.end()) {
            out.push_back(static_cast<char>(it->second));
        } else {
            out += ch;
        }
    }
    return out;
}

static bool isSpecialTokenString(const std::string& token) {
    if (token.size() >= 4 && token.rfind("<|", 0) == 0 && token.rfind("|>") == token.size() - 2) {
        return true;
    }
    if (token.size() >= 2 && token.front() == '<' && token.back() == '>') {
        return true;
    }
    return false;
}

} // namespace

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
        CLLM_INFO("GGUFTokenizer::load: Starting load from: %s", modelPath.c_str());
        modelPath_ = modelPath;
        
        // 创建GGUFLoader实例
        // 禁用内存映射以避免与 llama.cpp 的 mmap 冲突
        CLLM_INFO("GGUFTokenizer::load: Creating GGUFLoader (mmap disabled)...");
        GGUFLoader loader(modelPath, false);
        CLLM_INFO("GGUFTokenizer::load: GGUFLoader created");
        
        // 加载GGUF文件
        CLLM_INFO("GGUFTokenizer::load: Calling loader.load()...");
        if (!loader.load()) {
            CLLM_ERROR("GGUFTokenizer::load: Failed to load GGUF file: %s", modelPath.c_str());
            return false;
        }
        CLLM_INFO("GGUFTokenizer::load: loader.load() succeeded");
        
        // 从loader中加载tokenizer数据
        CLLM_INFO("GGUFTokenizer::load: Calling loadFromGGUFLoader...");
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
        const auto& md = metadata.at("tokenizer.ggml.vocab_size");
        switch (md.type) {
            case GGUFValueType::UINT32:
                vocabSize_ = static_cast<int>(md.value.u32_val);
                break;
            case GGUFValueType::INT32:
                vocabSize_ = static_cast<int>(md.value.i32_val);
                break;
            case GGUFValueType::UINT64:
                vocabSize_ = static_cast<int>(md.value.u64_val);
                break;
            case GGUFValueType::INT64:
                vocabSize_ = static_cast<int>(md.value.i64_val);
                break;
            default:
                CLLM_WARN("GGUFTokenizer::loadVocabulary: vocab_size type not supported: %u", static_cast<uint32_t>(md.type));
                break;
        }
    } else {
        CLLM_WARN("GGUFTokenizer::loadVocabulary: vocab_size not found, using default: 32000");
        vocabSize_ = 32000;
    }
    
    // 尝试从多种可能的词汇表字段中加载
    std::vector<std::string> vocabFields = {
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.vocab",
        "tokenizer.tokens",
        "tokens"
    };
    
    for (const auto& field : vocabFields) {
        if (metadata.count(field) > 0) {
            const auto& vocabData = metadata.at(field);
            
            if (vocabData.type == GGUFValueType::ARRAY) {
                // 处理数组类型的词汇表
                const auto& vocabArray = vocabData.array_val;
                if (vocabArray.elementType != GGUFValueType::STRING) {
                    CLLM_WARN("GGUFTokenizer::loadVocabulary: vocab array element type not STRING: %u", static_cast<uint32_t>(vocabArray.elementType));
                }
                size_t count = vocabArray.elements.size();
                for (size_t i = 0; i < count; ++i) {
                    const auto& elem = vocabArray.elements[i];
                    if (elem.type != GGUFValueType::STRING) {
                        continue;
                    }
                    idToTokenMap_[i] = elem.string_val;
                    tokenToIdMap_[elem.string_val] = i;
                }
                vocabSize_ = static_cast<int>(count);
                CLLM_INFO("GGUFTokenizer::loadVocabulary: Loaded %zu tokens from field: %s", 
                          count, field.c_str());
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
    
    auto readTokenId = [&](const std::string& key, int& outId) {
        if (metadata.count(key) == 0) {
            return;
        }
        const auto& md = metadata.at(key);
        switch (md.type) {
            case GGUFValueType::UINT32:
                outId = static_cast<int>(md.value.u32_val);
                break;
            case GGUFValueType::INT32:
                outId = static_cast<int>(md.value.i32_val);
                break;
            case GGUFValueType::UINT64:
                outId = static_cast<int>(md.value.u64_val);
                break;
            case GGUFValueType::INT64:
                outId = static_cast<int>(md.value.i64_val);
                break;
            default:
                CLLM_WARN("GGUFTokenizer::loadSpecialTokens: token id type not supported: %u (key=%s)", static_cast<uint32_t>(md.type), key.c_str());
                return;
        }
        if (outId >= 0) {
            specialTokenIds_.insert(outId);
        }
    };

    // 提取特殊token ID（兼容不同字段名）
    readTokenId("tokenizer.ggml.bos_token_id", bosTokenId_);
    readTokenId("tokenizer.ggml.eos_token_id", eosTokenId_);
    readTokenId("tokenizer.ggml.pad_token_id", padTokenId_);
    readTokenId("tokenizer.ggml.unk_token_id", unkTokenId_);
    readTokenId("tokenizer.ggml.unknown_token_id", unkTokenId_);
    
    // 从词汇表中提取特殊 token 字符串（格式如 <|...|>）
    for (const auto& [id, token] : idToTokenMap_) {
        if (isSpecialTokenString(token)) {
            specialTokenIds_.insert(id);
            specialTokenStrings_.push_back(token);
        }
    }
    
    // 如果未从元数据获取到 UNK，尝试从词表中推断
    if (unkTokenId_ == -1) {
        const std::vector<std::string> unkCandidates = {
            "<|unk|>", "<unk>", "[UNK]", "<|unknown|>", "<|unk>"
        };
        for (const auto& cand : unkCandidates) {
            auto it = tokenToIdMap_.find(cand);
            if (it != tokenToIdMap_.end()) {
                unkTokenId_ = it->second;
                specialTokenIds_.insert(unkTokenId_);
                break;
            }
        }
    }
    
    // 输出特殊token信息
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: BOS token ID: %d", bosTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: EOS token ID: %d", eosTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: PAD token ID: %d", padTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: UNK token ID: %d", unkTokenId_);
    CLLM_INFO("GGUFTokenizer::loadSpecialTokens: Found %zu special token strings", specialTokenStrings_.size());
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
            
            if (mergesData.type == GGUFValueType::ARRAY) {
                // 处理数组类型的合并规则
                const auto& mergeArray = mergesData.array_val;
                if (mergeArray.elementType != GGUFValueType::STRING) {
                    CLLM_WARN("GGUFTokenizer::loadMergeRules: merge array element type not STRING: %u", static_cast<uint32_t>(mergeArray.elementType));
                }
                for (const auto& elem : mergeArray.elements) {
                    if (elem.type != GGUFValueType::STRING) {
                        continue;
                    }
                    const std::string& mergeStr = elem.string_val;
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
    // 构建 BPE ranks（用于快速查找 merge 规则的优先级）
    bpeRanks_.clear();
    for (size_t i = 0; i < mergeRules_.size(); ++i) {
        const auto& rule = mergeRules_[i];
        std::string key = rule.first + " " + rule.second;  // 合并规则作为 key
        bpeRanks_[key] = static_cast<int>(i);
    }
    
    // 构建字节编码器（用于 byte-level BPE）
    buildByteEncoder();
    
    CLLM_INFO("GGUFTokenizer::initializeEncoding: Initialized BPE encoding");
    CLLM_INFO("  - Merge rules: %zu", mergeRules_.size());
    CLLM_INFO("  - Vocab size: %zu", idToTokenMap_.size());
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
    
    // 完整的 BPE 编码流程
    // 1. 预处理：检查特殊 token
    std::string remaining = text;
    std::vector<std::string> preTokens;
    
    // 先提取特殊 tokens（如果存在）
    while (!remaining.empty()) {
        bool foundSpecial = false;
        for (const auto& specialToken : specialTokenStrings_) {
            size_t pos = remaining.find(specialToken);
            if (pos == 0) {
                preTokens.push_back(specialToken);
                remaining = remaining.substr(specialToken.size());
                foundSpecial = true;
                break;
            }
        }
        if (!foundSpecial) {
            break;
        }
    }
    
    // 2. 预分词：将文本分割为单词/片段
    std::vector<std::string> words = preTokenize(remaining);
    
    // 3. 对每个单词应用 BPE
    for (const auto& word : words) {
        if (word.empty()) continue;
        
        // 检查是否是特殊 token
        auto specialIt = std::find(specialTokenStrings_.begin(), specialTokenStrings_.end(), word);
        if (specialIt != specialTokenStrings_.end()) {
            // 直接查找特殊 token ID
            auto it = tokenToIdMap_.find(word);
            if (it != tokenToIdMap_.end()) {
                tokenIds.push_back(it->second);
            }
            continue;
        }
        
        // byte-level 预编码（确保与词表编码一致）
        std::string byteEncoded = byteEncodeWithTable(word, byteEncoder_);

        // 应用 BPE 合并
        std::vector<std::string> bpeTokens = bpe(byteEncoded);
        
        // 4. 将 BPE tokens 转换为 token IDs
        auto appendByteFallback = [&](const std::string& raw) -> bool {
            bool appended = false;
            for (unsigned char ch : raw) {
                auto beIt = byteEncoder_.find(ch);
                if (beIt == byteEncoder_.end()) {
                    continue;
                }
                auto tokIt = tokenToIdMap_.find(beIt->second);
                if (tokIt != tokenToIdMap_.end()) {
                    tokenIds.push_back(tokIt->second);
                    appended = true;
                }
            }
            return appended;
        };

        for (const auto& token : bpeTokens) {
            auto it = tokenToIdMap_.find(token);
            if (it != tokenToIdMap_.end()) {
                tokenIds.push_back(it->second);
                continue;
            }

            // byte-level fallback（避免中文等字符导致空编码）
            if (appendByteFallback(token)) {
                continue;
            }

            // 如果仍找不到，使用 UNK token
            if (unkTokenId_ != -1) {
                tokenIds.push_back(unkTokenId_);
            } else {
                CLLM_WARN("GGUFTokenizer::encode: Token not found in vocab: %s", token.c_str());
            }
        }
    }
    
    // 确保非空编码（避免 total tokens=0）
    if (tokenIds.empty() && !text.empty()) {
        int fallbackId = -1;
        if (unkTokenId_ != -1) {
            fallbackId = unkTokenId_;
        } else if (bosTokenId_ != -1) {
            fallbackId = bosTokenId_;
        } else if (eosTokenId_ != -1) {
            fallbackId = eosTokenId_;
        } else if (vocabSize_ > 0) {
            fallbackId = 0;
        }
        if (fallbackId != -1) {
            tokenIds.push_back(fallbackId);
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
    
    // 解码逻辑：将 token IDs 转换为文本
    std::string text;
    
    for (int id : ids) {
        // 如果需要跳过特殊token
        if (skipSpecialTokens) {
            if (id == bosTokenId_ || id == eosTokenId_ || id == padTokenId_) {
                continue;
            }
            // 检查是否是特殊 token ID
            if (specialTokenIds_.count(id) > 0) {
                continue;
            }
        }
        
        // 查找token
        auto it = idToTokenMap_.find(id);
        if (it != idToTokenMap_.end()) {
            const std::string& token = it->second;
            
            // 检查是否是特殊 token（格式如 <|...|>）
            if (isSpecialTokenString(token)) {
                // 特殊 token 不添加空格
                text += token;
            } else {
                // 普通 token：直接拼接（BPE tokens 已经是正确的格式）
                text += token;
            }
        } else {
            // 如果找不到，使用UNK token
            if (!skipSpecialTokens || unkTokenId_ == -1 || id != unkTokenId_) {
                text += "[UNK]";
            }
        }
    }

    // 将 byte-encoded 文本还原为 UTF-8
    text = byteDecodeWithTable(text, byteDecoder_);
    
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

void GGUFTokenizer::buildByteEncoder() {
    // GPT-2/llama-style byte encoder
    byteEncoder_.clear();
    byteDecoder_.clear();

    std::vector<int> bytes;
    for (int b = 33; b <= 126; ++b) bytes.push_back(b);
    for (int b = 161; b <= 172; ++b) bytes.push_back(b);
    for (int b = 174; b <= 255; ++b) bytes.push_back(b);

    std::vector<int> unicode = bytes;
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        if (std::find(bytes.begin(), bytes.end(), b) == bytes.end()) {
            bytes.push_back(b);
            unicode.push_back(256 + n);
            ++n;
        }
    }

    for (size_t i = 0; i < bytes.size(); ++i) {
        unsigned char byte = static_cast<unsigned char>(bytes[i]);
        std::string mapped = codepointToUtf8(static_cast<uint32_t>(unicode[i]));
        byteEncoder_[byte] = mapped;
        byteDecoder_[mapped] = byte;
    }

    CLLM_DEBUG("GGUFTokenizer::buildByteEncoder: Built byte encoder/decoder (256 entries)");
}

std::vector<std::string> GGUFTokenizer::preTokenize(const std::string& text) const {
    // 预分词：将文本分割为单词/片段
    // 简化实现：使用空格和标点符号分割
    // 更复杂的实现可以使用正则表达式（如 GPT-2 的 regex）
    // 参考 llama.cpp 的 unicode_regex_split
    
    std::vector<std::string> tokens;
    if (text.empty()) {
        return tokens;
    }
    
    // 简化版本：使用空白符分割
    // TODO: 可以改进为支持 GPT-2 风格的 regex 预分词
    std::string current;
    for (size_t i = 0; i < text.size(); ++i) {
        char c = text[i];
        
        if (isAsciiWhitespace(c)) {
            if (!current.empty()) {
                tokens.push_back(current);
                current.clear();
            }
            // 空白符作为一个单独的 token（简化处理）
            tokens.push_back(std::string(1, c));
        } else {
            current.push_back(c);
        }
    }
    
    if (!current.empty()) {
        tokens.push_back(current);
    }
    
    return tokens;
}

std::vector<std::string> GGUFTokenizer::bpe(const std::string& token) const {
    // BPE 合并算法
    // 参考 llama.cpp 的 llm_tokenizer_bpe_session::tokenize
    // 简化版本：贪心算法应用 merge rules
    
    if (mergeRules_.empty()) {
        // 如果没有 merge rules，直接返回字符序列
        std::vector<std::string> chars;
        for (char c : token) {
            chars.push_back(std::string(1, c));
        }
        return chars;
    }
    
    // 1. 将 token 转换为字符序列（或字节序列，取决于 tokenizer 类型）
    // 简化：使用 UTF-8 字符
    std::vector<std::string> word;
    auto utf8Chars = splitUtf8(token);
    for (const auto& ch : utf8Chars) {
        word.push_back(ch);
    }
    
    if (word.empty()) {
        return word;
    }
    
    // 如果只有一个字符，直接返回
    if (word.size() == 1) {
        return word;
    }
    
    // 2. 应用 BPE merge rules（贪心算法）
    // 每次选择优先级最高（rank 最小）的可以合并的 pair
    while (word.size() > 1) {
        // 找到所有可能的 pairs 及其 rank
        std::vector<std::pair<int, size_t>> pairs;  // (rank, index)
        
        for (size_t i = 0; i < word.size() - 1; ++i) {
            std::string pair = word[i] + " " + word[i + 1];
            auto it = bpeRanks_.find(pair);
            if (it != bpeRanks_.end()) {
                pairs.push_back({it->second, i});
            }
        }
        
        if (pairs.empty()) {
            break;  // 没有更多可以合并的 pairs
        }
        
        // 选择 rank 最小的 pair（优先级最高）
        auto minPair = std::min_element(pairs.begin(), pairs.end(),
            [](const std::pair<int, size_t>& a, const std::pair<int, size_t>& b) {
                return a.first < b.first;
            });
        
        size_t mergeIndex = minPair->second;
        
        // 合并 pair
        std::string merged = word[mergeIndex] + word[mergeIndex + 1];
        word[mergeIndex] = merged;
        word.erase(word.begin() + mergeIndex + 1);
    }
    
    return word;
}

} // namespace cllm