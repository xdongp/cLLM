#include "cllm/tokenizer/manager.h"
#include "cllm/model/executor.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/tokenizer/generator.h"
#include "cllm/tokenizer/hf_tokenizer.h"
#include "cllm/tokenizer/native_tokenizer.h"
#include "cllm/tokenizer/gguf_tokenizer.h"
#include "cllm/common/logger.h"
#include <chrono>
#include <stdexcept>
#include <nlohmann/json.hpp>
#include <fstream>
#include <filesystem>

namespace cllm {

// 检测tokenizer格式辅助函数
namespace {
    bool hasTokenizerJson(const std::string& modelPath) {
        namespace fs = std::filesystem;
        if (modelPath.empty()) {
            return false;
        }
        if (fs::is_directory(modelPath)) {
            return fs::exists(fs::path(modelPath) / "tokenizer.json");
        }
        return false;
    }
    
    bool hasTokenizerModel(const std::string& modelPath) {
        namespace fs = std::filesystem;
        if (modelPath.empty()) {
            return false;
        }
        if (fs::is_directory(modelPath)) {
            return fs::exists(fs::path(modelPath) / "tokenizer.model");
        }
        return false;
    }
    
    bool isGgufFile(const std::string& modelPath) {
        namespace fs = std::filesystem;
        if (modelPath.empty()) {
            return false;
        }
        // 检查是否为 .gguf 文件
        if (fs::is_regular_file(modelPath)) {
            std::string ext = fs::path(modelPath).extension().string();
            return (ext == ".gguf");
        }
        // 检查目录中是否有 .gguf 文件
        if (fs::is_directory(modelPath)) {
            for (const auto& entry : fs::directory_iterator(modelPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".gguf") {
                        return true;
                    }
                }
            }
        }
        return false;
    }
}

// 检测模型类型辅助函数
ModelType detectModelType(const std::string& modelPath) {
    namespace fs = std::filesystem;
    
    if (modelPath.empty()) {
        // 如果模型路径为空，返回默认模型类型
        return ModelType::SPM;
    }
    
    // 读取config.json
    fs::path configPath = fs::path(modelPath) / "config.json";
    if (!fs::exists(configPath)) {
        // 如果没有config.json，根据路径判断
        if (modelPath.find("qwen") != std::string::npos || modelPath.find("Qwen") != std::string::npos) {
            return ModelType::QWEN;
        } else if (modelPath.find("deepseek") != std::string::npos || modelPath.find("DeepSeek") != std::string::npos) {
            return ModelType::DEEPSEEK_LLM;
        }
        return ModelType::SPM;
    }
    
    std::ifstream f(configPath);
    if (!f.is_open()) return ModelType::SPM;
    
    try {
        auto config = nlohmann::json::parse(f);
        
        // 检测model_type字段
        if (config.contains("model_type")) {
            std::string modelTypeStr = config["model_type"];
            if (modelTypeStr.find("qwen2") != std::string::npos) return ModelType::QWEN2;
            if (modelTypeStr.find("qwen") != std::string::npos) return ModelType::QWEN;
            if (modelTypeStr.find("deepseek") != std::string::npos) return ModelType::DEEPSEEK_LLM;
        }
        
        // 检测tokenizer_class字段
        if (config.contains("tokenizer_class")) {
            std::string tokenizerClass = config["tokenizer_class"];
            if (tokenizerClass.find("Qwen2") != std::string::npos) return ModelType::QWEN2;
            if (tokenizerClass.find("Qwen") != std::string::npos) return ModelType::QWEN;
            if (tokenizerClass.find("DeepSeek") != std::string::npos) return ModelType::DEEPSEEK_LLM;
        }
        
    } catch (const std::exception& e) {
        CLLM_WARN("Failed to detect model type: %s", e.what());
    }
    
    return ModelType::SPM;
}

TokenizerManager::TokenizerManager(
    const std::string& modelPath,
    ModelExecutor* modelExecutor,
    TokenizerImpl impl
) : modelExecutor_(modelExecutor), kvCache_(nullptr) {
    ModelType modelType = detectModelType(modelPath);
    
    // 根据实现类型选择tokenizer
    switch(impl) {
        case TokenizerImpl::HF:
            // 强制使用HF
            CLLM_INFO("🔸 Force using HFTokenizer");
            tokenizer_ = new HFTokenizer(modelType);
            break;
            
        case TokenizerImpl::NATIVE:
            // 强制使用Native
            CLLM_INFO("🔸 Force using NativeTokenizer");
            tokenizer_ = new NativeTokenizer(modelType);
            break;
            
        case TokenizerImpl::AUTO:
        default:
            // ✅ 自动检测: 优先检测 HuggingFace tokenizer.json
            // 重要：对于 HuggingFace 模型目录（包含 tokenizer.json），应该优先使用 HFTokenizer
            // 即使目录中存在 .gguf 文件，也应该使用 tokenizer.json
            if (hasTokenizerJson(modelPath)) {
                CLLM_INFO("✅ Detected HuggingFace format (tokenizer.json), using HFTokenizer");
                tokenizer_ = new HFTokenizer(modelType);
                
            } else if (isGgufFile(modelPath)) {
                // 只有在没有 tokenizer.json 的情况下才使用 GGUFTokenizer
                CLLM_INFO("✅ Detected GGUF format, using GGUFTokenizer");
                tokenizer_ = new GGUFTokenizer();
                
            } else if (hasTokenizerModel(modelPath)) {
                CLLM_INFO("✅ Detected SentencePiece format (tokenizer.model), using NativeTokenizer");
                tokenizer_ = new NativeTokenizer(modelType);
                
            } else {
                // 回退到Native实现 (可能使用其他格式)
                CLLM_WARN("⚠️  No standard tokenizer format found, trying NativeTokenizer");
                tokenizer_ = new NativeTokenizer(modelType);
            }
            break;
    }
    
    // 加载tokenizer
    if (!tokenizer_->load(modelPath)) {
        delete tokenizer_;
        throw std::runtime_error("Failed to load tokenizer from: " + modelPath);
    }
    
    CLLM_INFO("✅ TokenizerManager initialized successfully");
    
    // 加载停止tokens
    namespace fs = std::filesystem;
    std::string configPath = (fs::path(modelPath) / "config.json").string();
    if (fs::exists(configPath)) {
        loadStopTokens(configPath);
    }
}

TokenizerManager::~TokenizerManager() {
    if (tokenizer_ != nullptr) {
        delete tokenizer_;
        tokenizer_ = nullptr;
    }
}

std::vector<int> TokenizerManager::encode(const std::string& text) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<int> tokenIds = tokenizer_->encode(text, true);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float>(endTime - startTime).count();
    
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.incrementEncodeCount();
        stats_.addEncodeTime(duration);
    }
    
    return tokenIds;
}

std::string TokenizerManager::decode(const std::vector<int>& tokenIds) {
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::string text = tokenizer_->decode(tokenIds, true);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float>(endTime - startTime).count();
    
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.incrementDecodeCount();
        stats_.addDecodeTime(duration);
    }
    
    return text;
}

std::string TokenizerManager::generate(
    const std::string& requestId,
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP
) {
    if (modelExecutor_ == nullptr) {
        throw std::runtime_error("ModelExecutor is not set");
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<int> inputIds = encodePrompt(prompt);
    
    // 使用ModelExecutor的generate方法生成所有token
    std::vector<int> generatedTokens = modelExecutor_->generate(inputIds, maxTokens, temperature);
    
    // 检查是否遇到停止token并截断
    for (size_t i = 0; i < generatedTokens.size(); ++i) {
        if (isStopToken(generatedTokens[i])) {
            generatedTokens.resize(i);
            break;
        }
    }
    
    std::string generatedText = decodeTokens(generatedTokens);
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float>(endTime - startTime).count();
    
    updateStats(requestId, static_cast<int>(generatedTokens.size()), duration);
    
    return generatedText;
}

std::vector<GenerationResponse> TokenizerManager::generateStream(
    const std::string& requestId,
    const std::string& prompt,
    int maxTokens,
    float temperature,
    float topP
) {
    if (modelExecutor_ == nullptr) {
        throw std::runtime_error("ModelExecutor is not set");
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    std::vector<int> inputIds = encodePrompt(prompt);
    
    StreamGenerator generator(
        requestId,
        inputIds,
        maxTokens,
        temperature,
        modelExecutor_,
        tokenizer_
    );
    
    std::vector<GenerationResponse> responses;
    
    while (generator.hasNext()) {
        GenerationResponse response = generator.next();
        responses.push_back(response);
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    float duration = std::chrono::duration<float>(endTime - startTime).count();
    
    {
        std::lock_guard<std::mutex> lock(statsMutex_);
        stats_.incrementStreamGenerateCount();
        stats_.addStreamGenerateTime(duration);
        stats_.addGeneratedTokens(generator.getGeneratedTokenCount());
    }
    
    return responses;
}

void TokenizerManager::setModelExecutor(ModelExecutor* modelExecutor) {
    modelExecutor_ = modelExecutor;
}

void TokenizerManager::setKVCache(KVCache* kvCache) {
    kvCache_ = kvCache;
}

ITokenizer* TokenizerManager::getTokenizer() const {
    return tokenizer_;
}

ModelExecutor* TokenizerManager::getModelExecutor() const {
    return modelExecutor_;
}

KVCache* TokenizerManager::getKVCache() const {
    return kvCache_;
}

TokenizerStats TokenizerManager::getStats() const {
    std::lock_guard<std::mutex> lock(statsMutex_);
    return stats_;
}

void TokenizerManager::resetStats() {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.reset();
}

std::vector<int> TokenizerManager::encodePrompt(const std::string& prompt) {
    return tokenizer_->encode(prompt, true);
}

std::string TokenizerManager::decodeTokens(const std::vector<int>& tokens) {
    return tokenizer_->decode(tokens, true);
}

bool TokenizerManager::isStopToken(int tokenId) {
    for (int stopToken : stopTokens_) {
        if (tokenId == stopToken) {
            return true;
        }
    }
    return tokenId == tokenizer_->getEosId();
}

void TokenizerManager::updateStats(const std::string& requestId, int tokenCount, float time) {
    std::lock_guard<std::mutex> lock(statsMutex_);
    stats_.incrementGenerateCount();
    stats_.addGenerateTime(time);
    stats_.addGeneratedTokens(tokenCount);
}

void TokenizerManager::loadStopTokens(const std::string& configPath) {
    // 从配置文件加载停止token
    std::ifstream f(configPath);
    if (!f.is_open()) return;
    
    try {
        auto config = nlohmann::json::parse(f);
        if (config.contains("stop_token_ids")) {
            stopTokens_ = config["stop_token_ids"].get<std::vector<int>>();
        }
        // 同时添加eos_token_id作为停止token
        if (config.contains("eos_token_id")) {
            int eosId = config["eos_token_id"];
            stopTokens_.push_back(eosId);
        }
    } catch (const std::exception& e) {
        CLLM_WARN("Failed to load stop tokens from %s: %s", configPath.c_str(), e.what());
    }
}

} // namespace cllm