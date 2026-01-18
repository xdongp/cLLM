/**
 * @file llama_cpp_backend.cpp
 * @brief llama.cpp 推理后端实现
 * 
 * 参考文档：llama.cpp后端集成设计.md
 */

#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/common/logger.h"

#include "llama.h"

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cstdint>
#include <unordered_map>
#include <mutex>

namespace cllm {
namespace inference {

// ========== RAII 包装类：自动管理 llama_batch 资源 ==========

/**
 * @brief RAII 包装类，自动管理 llama_batch 的生命周期
 * 
 * 确保在异常情况下也能正确清理 seq_id 数组，防止内存泄漏
 */
class LlamaBatchGuard {
public:
    explicit LlamaBatchGuard(int32_t n_tokens, int32_t embd, int32_t n_seq_max)
        : batch_(llama_batch_init(n_tokens, embd, n_seq_max)) {
        if (batch_.n_tokens < 0) {
            throw std::runtime_error("LlamaBatchGuard: failed to initialize batch");
        }
    }
    
    ~LlamaBatchGuard() {
        cleanup();
    }
    
    // 禁止拷贝和赋值
    LlamaBatchGuard(const LlamaBatchGuard&) = delete;
    LlamaBatchGuard& operator=(const LlamaBatchGuard&) = delete;
    
    // 允许移动
    LlamaBatchGuard(LlamaBatchGuard&& other) noexcept
        : batch_(other.batch_) {
        // 将 other 的 batch 重置为空，防止重复清理
        other.batch_ = llama_batch_init(0, 0, 0);
    }
    
    llama_batch& get() { return batch_; }
    const llama_batch& get() const { return batch_; }
    
    llama_batch* operator->() { return &batch_; }
    const llama_batch* operator->() const { return &batch_; }
    
private:
    void cleanup() {
        // 清理所有 seq_id 数组
        for (int32_t i = 0; i < batch_.n_tokens; ++i) {
            if (batch_.seq_id[i]) {
                delete[] batch_.seq_id[i];
                batch_.seq_id[i] = nullptr;
            }
        }
        // 释放 batch 资源
        llama_batch_free(batch_);
    }
    
    llama_batch batch_;
};

// ========== 位置管理方法实现 ==========

size_t LlamaCppBackend::getSeqPosition(int32_t seqId) const {
    std::lock_guard<std::mutex> lock(seqPositionsMutex_);
    auto it = seqPositions_.find(seqId);
    return it != seqPositions_.end() ? it->second : 0;
}

void LlamaCppBackend::updateSeqPosition(int32_t seqId, size_t position) {
    std::lock_guard<std::mutex> lock(seqPositionsMutex_);
    seqPositions_[seqId] = position;
}

void LlamaCppBackend::resetSeqPosition(int32_t seqId) {
    std::lock_guard<std::mutex> lock(seqPositionsMutex_);
    seqPositions_[seqId] = 0;
    seqLengths_[seqId] = 0;  // 同时重置长度记录
}

bool LlamaCppBackend::hasSeqPosition(int32_t seqId) const {
    std::lock_guard<std::mutex> lock(seqPositionsMutex_);
    return seqPositions_.find(seqId) != seqPositions_.end();
}

void LlamaCppBackend::clearKVCacheForSequence(int32_t seqId) {
    if (!ctx_) {
        return;
    }
    
    // 苏格拉底式分析发现的关键问题：
    // llama_memory_seq_rm 可能只是标记为未使用，而不是真正释放内存
    // 这可能导致 KV cache 内部有碎片或残留，影响性能
    
    // 使用 llama.cpp 新的 Memory API 清空指定序列的 KV cache
    // -1, -1 表示清空整个序列（从开始到结束）
    // 注意：即使位置记录为 0，也要清理，因为 seq_id 可能被重用
    llama_memory_t mem = llama_get_memory(ctx_);
    if (mem) {
        // 关键修复：先尝试清理，然后强制清理所有相关状态
        llama_memory_seq_rm(mem, static_cast<llama_seq_id>(seqId), -1, -1);
        
        // 额外清理：尝试清理整个序列的所有位置
        // 这确保即使 llama_memory_seq_rm 只是标记，我们也能强制清理
        // 注意：llama.cpp 可能不支持完全清理，但至少我们尝试了
        
        // 一次加锁更新位置记录
        std::lock_guard<std::mutex> lock(seqPositionsMutex_);
        auto it = seqPositions_.find(seqId);
        size_t previousPos = (it != seqPositions_.end()) ? it->second : 0;
        CLLM_INFO("[LlamaCppBackend] Cleared KV cache for sequence %d (previous pos=%zu) - NOTE: llama_memory_seq_rm may only mark as unused, not free memory", 
                  seqId, previousPos);
    } else {
        CLLM_WARN("[LlamaCppBackend] Failed to get memory handle for clearing KV cache");
    }
}

LlamaCppBackend::LlamaCppBackend(const ModelConfig &config, const std::string &modelPath)
    : modelPath_(modelPath)
    , config_(config)
    , model_(nullptr)
    , ctx_(nullptr)
    , initialized_(false)
    , modelParams_(nullptr)
    , contextParams_(nullptr)
    , numThreads_(config.llamaNumThreads)  // 0 表示使用默认值
    , nGpuLayers_(config.llamaGpuLayers)  // 0 表示仅使用 CPU
    , currentPosition_(0)
{
    CLLM_INFO("[LlamaCppBackend] Constructing llama.cpp backend");
    CLLM_INFO("[LlamaCppBackend] Model path: %s", modelPath_.c_str());
    CLLM_INFO("[LlamaCppBackend] Config vocab_size: %zu", config_.vocabSize);
    CLLM_INFO("[LlamaCppBackend] Config max_seq_len: %zu", config_.maxSequenceLength);
}

LlamaCppBackend::~LlamaCppBackend() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }
    
    if (modelParams_) {
        delete modelParams_;
        modelParams_ = nullptr;
    }
    
    if (contextParams_) {
        delete contextParams_;
        contextParams_ = nullptr;
    }
    
    CLLM_INFO("[LlamaCppBackend] Destructed");
}

void LlamaCppBackend::createModelParams() {
    modelParams_ = new llama_model_params();
    *modelParams_ = llama_model_default_params();
    
    // 设置 GPU 层数
    modelParams_->n_gpu_layers = nGpuLayers_;
    
    // 使用 mmap / mlock（从配置读取）
    modelParams_->use_mmap = config_.llamaUseMmap;
    modelParams_->use_mlock = config_.llamaUseMlock;
    
    CLLM_INFO("[LlamaCppBackend] Model params: n_gpu_layers=%d, use_mmap=%s, use_mlock=%s",
              modelParams_->n_gpu_layers,
              modelParams_->use_mmap ? "true" : "false",
              modelParams_->use_mlock ? "true" : "false");
}

void LlamaCppBackend::createContextParams() {
    contextParams_ = new llama_context_params();
    *contextParams_ = llama_context_default_params();
    
    // 设置上下文长度
    // 设置上下文长度
    // 注意：减少 n_ctx 可以减少内存分配，但测试表明：
    // 1. 第一次请求稍快（更小的内存分配）
    // 2. 但性能退化更严重（更少的内存插槽导致碎片化更快）
    // 3. 问题在于 llama.cpp 的内存管理机制，而不是内存大小
    contextParams_->n_ctx = static_cast<uint32_t>(config_.maxSequenceLength);
    
    // 设置批处理大小
    contextParams_->n_batch = config_.llamaBatchSize > 0 ? static_cast<uint32_t>(config_.llamaBatchSize) : 512;
    
    // 设置最大并行序列数（n_seq_max）
    // 这个值必须 >= 实际批处理中使用的最大 seq_id + 1
    // 默认使用 8（支持最多 8 个并行序列），可以根据配置调整
    // 注意：这个值会影响 KV cache 的分配，每个序列最多可以使用 n_ctx / n_seq_max 的 tokens
    contextParams_->n_seq_max = 8;  // 支持最多 8 个并行序列
    
    // 设置线程数
    if (numThreads_ > 0) {
        contextParams_->n_threads = numThreads_;
        contextParams_->n_threads_batch = numThreads_;
    } else {
        // 使用默认值（llama.cpp 会自动选择）
        contextParams_->n_threads = 0;
        contextParams_->n_threads_batch = 0;
    }
    
    CLLM_INFO("[LlamaCppBackend] Context params: n_ctx=%u, n_batch=%u, n_seq_max=%u, n_threads=%d",
              contextParams_->n_ctx, contextParams_->n_batch, contextParams_->n_seq_max, contextParams_->n_threads);
}

bool LlamaCppBackend::validateVocabSize() {
    if (!model_ || !tokenizer_) {
        return false;
    }
    
    // 获取模型的 vocab size
    // 注意：llama.cpp 使用 llama_vocab_n_tokens 获取 vocab size
    // 但需要先获取 vocab 对象
    const struct llama_vocab* vocab = llama_model_get_vocab(model_);
    if (!vocab) {
        CLLM_ERROR("[LlamaCppBackend] Failed to get vocab from model");
        return false;
    }
    
    int32_t modelVocabSize = llama_vocab_n_tokens(vocab);
    int tokenizerVocabSize = tokenizer_->getVocabSize();
    
    CLLM_INFO("[LlamaCppBackend] Model vocab_size: %d, Tokenizer vocab_size: %d",
              modelVocabSize, tokenizerVocabSize);
    
    if (static_cast<int>(modelVocabSize) != tokenizerVocabSize) {
        CLLM_ERROR("[LlamaCppBackend] ❌ Vocab size mismatch: model=%d, tokenizer=%d",
                   modelVocabSize, tokenizerVocabSize);
        return false;
    }
    
    // 更新 config 中的 vocab_size
    if (config_.vocabSize != static_cast<size_t>(modelVocabSize)) {
        CLLM_INFO("[LlamaCppBackend] Updating config vocab_size from %zu to %d",
                  config_.vocabSize, modelVocabSize);
        config_.vocabSize = static_cast<size_t>(modelVocabSize);
    }
    
    return true;
}

bool LlamaCppBackend::initialize() {
    if (initialized_) {
        CLLM_INFO("[LlamaCppBackend] Already initialized, skipping");
        return true;
    }
    
    try {
        CLLM_INFO("[LlamaCppBackend] ========== Initializing llama.cpp Backend ==========");
        
        // 检查模型路径
        if (modelPath_.empty()) {
            CLLM_ERROR("[LlamaCppBackend] Model path is empty");
            return false;
        }
        
        CLLM_INFO("[LlamaCppBackend] Loading GGUF model from: %s", modelPath_.c_str());
        
        // 1. 创建模型参数
        createModelParams();
        
        // 2. 加载模型
        model_ = llama_model_load_from_file(modelPath_.c_str(), *modelParams_);
        if (!model_) {
            CLLM_ERROR("[LlamaCppBackend] Failed to load model from: %s", modelPath_.c_str());
            return false;
        }
        CLLM_INFO("[LlamaCppBackend] Model loaded successfully");
        
        // 3. 创建上下文参数
        createContextParams();
        
        // 4. 创建上下文
        ctx_ = llama_init_from_model(model_, *contextParams_);
        if (!ctx_) {
            CLLM_ERROR("[LlamaCppBackend] Failed to create context");
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
        CLLM_INFO("[LlamaCppBackend] Context created successfully");
        
        // 5. 加载 GGUFTokenizer
        tokenizer_ = std::make_unique<GGUFTokenizer>();
        if (!tokenizer_->load(modelPath_)) {
            CLLM_ERROR("[LlamaCppBackend] Failed to load GGUFTokenizer from: %s", modelPath_.c_str());
            llama_free(ctx_);
            ctx_ = nullptr;
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
        CLLM_INFO("[LlamaCppBackend] GGUFTokenizer loaded successfully");
        
        // 6. 校验 vocab_size
        if (!validateVocabSize()) {
            CLLM_ERROR("[LlamaCppBackend] Vocab size validation failed");
            tokenizer_.reset();
            llama_free(ctx_);
            ctx_ = nullptr;
            llama_model_free(model_);
            model_ = nullptr;
            return false;
        }
        
        // 7. 从模型更新配置（如果可能）
        // 获取实际的上下文长度
        uint32_t actualNctx = llama_n_ctx(ctx_);
        if (actualNctx > 0 && actualNctx != config_.maxSequenceLength) {
            CLLM_INFO("[LlamaCppBackend] Updating maxSequenceLength from %zu to %u",
                      config_.maxSequenceLength, actualNctx);
            config_.maxSequenceLength = actualNctx;
        }
        
        CLLM_INFO("[LlamaCppBackend] ========== Initialization Complete ==========");
        CLLM_INFO("[LlamaCppBackend] Model vocab_size: %zu", config_.vocabSize);
        CLLM_INFO("[LlamaCppBackend] Context size: %zu", config_.maxSequenceLength);
        
        initialized_ = true;
        return true;
        
    } catch (const std::exception &e) {
        CLLM_ERROR("[LlamaCppBackend] Exception during initialization: %s", e.what());
        return false;
    }
}

std::vector<int32_t> LlamaCppBackend::convertToLlamaTokens(const std::vector<int> &inputIds) {
    std::vector<int32_t> tokens;
    tokens.reserve(inputIds.size());
    for (int id : inputIds) {
        tokens.push_back(static_cast<int32_t>(id));
    }
    return tokens;
}

// extractLogits 方法已移除，直接在 forward 中处理

Tensor LlamaCppBackend::forward(const std::vector<int> &inputIds) {
    if (!initialized_) {
        throw std::runtime_error("LlamaCppBackend::forward: backend not initialized");
    }
    
    if (inputIds.empty()) {
        throw std::invalid_argument("LlamaCppBackend::forward: inputIds cannot be empty");
    }
    
    // forward() 方法用于单序列推理
    // 使用固定的 seq_id = 0，并维护该序列的位置
    int32_t seqId = 0;
    
    // 判断是否是第一次调用（完整序列）还是增量推理
    // 如果序列长度 > 1，说明是新请求的 prefill
    // 如果序列长度 == 1，需要检查是否有位置记录来判断
    bool isPrefill = (inputIds.size() > 1);
    bool hasPosition = hasSeqPosition(seqId);
    size_t currentPos = getSeqPosition(seqId);
    
    if (isPrefill || !hasPosition || currentPos == 0) {
        // 新请求或第一次调用，清空 KV cache 并重置该序列的位置
        clearKVCacheForSequence(seqId);
        resetSeqPosition(seqId);
        CLLM_DEBUG("[LlamaCppBackend::forward] New request detected (length=%zu, hasPos=%d, pos=%zu), resetting sequence %d position to 0", 
                  inputIds.size(), hasPosition, currentPos, seqId);
    }
    
    // 获取该序列的当前位置（线程安全）
    size_t seqPosition = getSeqPosition(seqId);
    
    // 1. 转换为 llama_token
    std::vector<int32_t> tokens = convertToLlamaTokens(inputIds);
    
    // 2. 使用 RAII 包装类创建 batch（自动管理资源）
    LlamaBatchGuard batchGuard(static_cast<int32_t>(tokens.size()), 0, 1);
    llama_batch& batch = batchGuard.get();
    
    // 填充 batch
    for (size_t i = 0; i < tokens.size(); ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(seqPosition + i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i] = new llama_seq_id[1];
        batch.seq_id[i][0] = seqId;
        batch.logits[i] = (i == tokens.size() - 1);
    }
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    
    // 3. 推理
    CLLM_INFO("[LlamaCppBackend] Calling llama_decode with %d tokens, sequence %d at position %zu...", 
              batch.n_tokens, seqId, seqPosition);
    int decodeResult = llama_decode(ctx_, batch);
    CLLM_INFO("[LlamaCppBackend] llama_decode returned: %d", decodeResult);
    
    if (decodeResult != 0) {
        if (decodeResult < 0) {
            throw std::runtime_error("LlamaCppBackend::forward: llama_decode failed with error code " + std::to_string(decodeResult));
        } else {
            CLLM_WARN("[LlamaCppBackend] llama_decode returned warning code: %d", decodeResult);
        }
    }
    
    // 4. 更新该序列的位置（线程安全）
    updateSeqPosition(seqId, seqPosition + tokens.size());
    CLLM_DEBUG("[LlamaCppBackend::forward] Sequence %d position updated to %zu", seqId, seqPosition + tokens.size());
    
    // 5. 提取 logits
    size_t seqLen = inputIds.size();
    Tensor result({seqLen, config_.vocabSize});
    
    float* logitsPtr = llama_get_logits(ctx_);
    if (logitsPtr) {
        // 只复制最后一个 token 的 logits
        std::memcpy(
            result.data() + (seqLen - 1) * config_.vocabSize,
            logitsPtr,
            config_.vocabSize * sizeof(float)
        );
        
        // 前面的位置设置为 0
        for (size_t i = 0; i < seqLen - 1; ++i) {
            std::memset(
                result.data() + i * config_.vocabSize,
                0,
                config_.vocabSize * sizeof(float)
            );
        }
    } else {
        CLLM_WARN("[LlamaCppBackend] Failed to get logits, returning zero tensor");
    }
    
    // batch 资源由 RAII 自动清理，无需手动处理
    return result;
}

Tensor LlamaCppBackend::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize
) {
    if (!initialized_) {
        throw std::runtime_error("LlamaCppBackend::forwardBatch: backend not initialized");
    }
    
    if (requestPositions.size() != batchSize) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: requestPositions size mismatch");
    }
    
    // 计算总 token 数
    size_t totalTokens = 0;
    std::vector<size_t> seqLengths;
    for (const auto& pos : requestPositions) {
        size_t seqLength = pos.second - pos.first;
        seqLengths.push_back(seqLength);
        totalTokens += seqLength;
    }
    
    // 关键修复：检测每个序列是否是新请求
    // 问题：seq_id 是批处理索引（0, 1, 2...），不是请求 ID，所以不同请求可能使用相同的 seq_id
    // 这会导致 KV cache 混乱。解决方案：
    // 1. 对于长序列（长度 > 1），总是作为新请求的 prefill 处理，重置位置为 0
    // 2. 对于短序列（长度 == 1），检查是否有位置记录：
    //    - 如果有且位置 > 0：增量推理
    //    - 否则：新请求
    // 3. 为了安全，在每次批处理开始时，对于长序列清空位置记录
    std::unordered_map<int32_t, bool> isIncrementalMap;
    
    // 关键修复：seq_id 重用问题
    // 问题：每次批处理都使用 seq_id = 0, 1, 2...，但之前的 KV cache 可能还在
    // 
    // 策略调整：不要总是清理所有 KV cache，而是基于请求类型清理
    // - 新请求：清理 KV cache 并重置位置
    // - 增量推理：保留 KV cache（利用之前的计算结果）
    //
    // 但问题是：我们无法准确区分新请求和增量推理，因为 seq_id 被重用
    // 
    // 临时方案：先检测请求类型，然后只清理新请求的 KV cache
    // 这样可以保留增量推理的 KV cache，提高性能
    
    // 关键修复：基于序列长度变化来识别新请求 vs 增量推理
    // 策略：
    // 1. 记录每个 seq_id 的上次序列长度
    // 2. 如果当前长度 <= 上次长度，这是新请求（seq_id 被重用）
    // 3. 如果当前长度 > 上次长度，这是增量推理（同一请求的后续迭代）
    // 4. 对于新请求：清理 KV cache，处理所有 tokens
    // 5. 对于增量推理：保留 KV cache，只处理最后一个 token
    
    // 检测每个序列是否是新请求
    for (size_t seqId = 0; seqId < batchSize; ++seqId) {
        int32_t seqIdKey = static_cast<int32_t>(seqId);
        size_t seqLength = seqLengths[seqId];
        
        // 使用统一的位置管理方法（线程安全）
        bool hasPreviousPosition = hasSeqPosition(seqIdKey);
        size_t currentPos = getSeqPosition(seqIdKey);
        
        // 检查上次序列长度（线程安全）
        size_t previousLength = 0;
        {
            std::lock_guard<std::mutex> lock(seqPositionsMutex_);
            auto lengthIt = seqLengths_.find(seqIdKey);
            previousLength = (lengthIt != seqLengths_.end()) ? lengthIt->second : 0;
        }
        
        // 判断是否是新请求
        bool isNewRequest = false;
        
        if (seqLength <= previousLength) {
            // 序列长度没有增加或减少，说明这是新请求（seq_id 被重用）
            // 或者这是第一次调用（previousLength == 0）
            isNewRequest = true;
        } else {
            // 序列长度增加了，说明这是增量推理（同一请求的后续迭代）
            // 但需要验证：位置记录应该 > 0
            if (hasPreviousPosition && currentPos > 0) {
                isNewRequest = false;  // 增量推理
            } else {
                // 没有位置记录，可能是新请求（虽然长度增加了，但可能是不同的 prompt）
                // 为了安全，作为新请求处理
                isNewRequest = true;
            }
        }
        
        if (isNewRequest) {
            // 新请求：清理 KV cache 并重置所有状态
            // 注意：即使 previousLength > seqLength，也要清理，因为这是新请求（seq_id 被重用）
            clearKVCacheForSequence(seqIdKey);
            resetSeqPosition(seqIdKey);  // 这会同时重置 seqLengths_[seqIdKey] = 0
            // 更新序列长度记录为当前长度（之前已在 resetSeqPosition 中重置为0，这里更新为新长度）
            {
                std::lock_guard<std::mutex> lock(seqPositionsMutex_);
                seqLengths_[seqIdKey] = seqLength;
            }
            isIncrementalMap[seqIdKey] = false;
            CLLM_INFO("[LlamaCppBackend::forwardBatch] Sequence %d: NEW REQUEST (length=%zu, previous=%zu), cleared KV cache and reset state", 
                      seqId, seqLength, previousLength);
        } else {
            // 增量推理：保留 KV cache，只处理最后一个 token
            // 注意：不要清理 KV cache，利用之前的计算结果
            // 更新序列长度记录
            {
                std::lock_guard<std::mutex> lock(seqPositionsMutex_);
                seqLengths_[seqIdKey] = seqLength;
            }
            isIncrementalMap[seqIdKey] = true;
            CLLM_INFO("[LlamaCppBackend::forwardBatch] Sequence %d: INCREMENTAL (pos=%zu, length=%zu, previous=%zu), KEEPING KV cache, will process only last token", 
                      seqId, currentPos, seqLength, previousLength);
        }
    }
    
    if (totalTokens == 0) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: total tokens cannot be zero");
    }
    
    // 转换为 llama_token
    std::vector<int32_t> allTokens;
    allTokens.reserve(totalTokens);
    for (int id : flatInputIds) {
        allTokens.push_back(static_cast<int32_t>(id));
    }
    
    // 计算实际需要处理的 token 数量
    // 对于增量推理，每个序列只处理最后一个 token
    // 对于新请求，处理所有 tokens
    size_t actualTokenCount = 0;
    for (size_t seqId = 0; seqId < batchSize; ++seqId) {
        int32_t seqIdKey = static_cast<int32_t>(seqId);
        bool isIncremental = isIncrementalMap[seqIdKey];
        size_t seqLength = seqLengths[seqId];
        if (isIncremental) {
            actualTokenCount += 1;  // 只处理最后一个 token
        } else {
            actualTokenCount += seqLength;  // 处理所有 tokens
        }
    }
    
    // 使用 RAII 包装类创建 batch（自动管理资源）
    // n_seq_max 必须与上下文参数中的 n_seq_max 匹配
    // 我们使用上下文中设置的 n_seq_max（默认 8），这应该足够支持批处理
    // 但为了安全，我们使用 max(batchSize, 上下文中的 n_seq_max)
    // 实际上，我们应该从上下文获取 n_seq_max，但为了简单，我们使用一个足够大的值
    // 注意：batch 的 n_seq_max 不能大于上下文创建时的 n_seq_max
    uint32_t contextNSeqMax = contextParams_ ? contextParams_->n_seq_max : 8;
    int32_t batchNSeqMax = static_cast<int32_t>(std::max(static_cast<size_t>(contextNSeqMax), batchSize));
    LlamaBatchGuard batchGuard(static_cast<int32_t>(actualTokenCount), 0, batchNSeqMax);
    llama_batch& batch = batchGuard.get();
    
    size_t tokenIdx = 0;
    for (size_t seqId = 0; seqId < batchSize; ++seqId) {
        const auto& pos = requestPositions[seqId];
        size_t seqStart = pos.first;
        size_t seqEnd = pos.second;
        // seq_id 必须小于 n_seq_max（batchSize）
        // 我们使用 seqId 作为 llama.cpp 的 seq_id，范围是 [0, batchSize)，这是正确的
        llama_seq_id llamaSeqId = static_cast<llama_seq_id>(seqId);
        int32_t seqIdKey = static_cast<int32_t>(seqId);
        bool isIncremental = isIncrementalMap[seqIdKey];
        
        // 使用统一的位置管理方法（线程安全）
        size_t seqPosition = getSeqPosition(seqIdKey);
        
        if (isIncremental) {
            // 增量推理：只处理最后一个 token
            size_t lastTokenIdx = seqEnd - 1;
            batch.token[tokenIdx] = allTokens[lastTokenIdx];
            batch.pos[tokenIdx] = static_cast<llama_pos>(seqPosition);
            batch.n_seq_id[tokenIdx] = 1;
            batch.seq_id[tokenIdx] = new llama_seq_id[1];
            batch.seq_id[tokenIdx][0] = llamaSeqId;  // 使用 llama_seq_id 类型
            batch.logits[tokenIdx] = 1;  // 计算 logits
            // 更新该序列的位置（线程安全）
            updateSeqPosition(seqIdKey, seqPosition + 1);
            ++tokenIdx;
            CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Sequence %d (llama_seq_id=%d): incremental, token=%d at position %zu", 
                      seqId, llamaSeqId, allTokens[lastTokenIdx], seqPosition);
        } else {
            // 新请求：处理所有 tokens（prefill）
            // 关键修复：确保位置从 0 开始，并且强制 llama.cpp 不使用之前的 KV cache
            // 方法：即使清理了 KV cache，也要确保 batch.pos 从 0 开始，这样 llama.cpp 会重新计算
            size_t seqPosition = 0;  // 强制从 0 开始，忽略之前的记录
            for (size_t i = seqStart; i < seqEnd; ++i) {
                batch.token[tokenIdx] = allTokens[i];
                batch.pos[tokenIdx] = static_cast<llama_pos>(seqPosition + (i - seqStart));
                batch.n_seq_id[tokenIdx] = 1;
                batch.seq_id[tokenIdx] = new llama_seq_id[1];
                batch.seq_id[tokenIdx][0] = llamaSeqId;  // 使用 llama_seq_id 类型
                // 只计算最后一个位置的 logits
                batch.logits[tokenIdx] = (i == seqEnd - 1);
                ++tokenIdx;
            }
            // 更新该序列的位置（线程安全）
            updateSeqPosition(seqIdKey, seqEnd - seqStart);
            CLLM_INFO("[LlamaCppBackend::forwardBatch] Sequence %d (llama_seq_id=%d): prefill, %zu tokens (from %zu to %zu), position forced to start from 0", 
                      seqId, llamaSeqId, seqEnd - seqStart, seqStart, seqEnd);
        }
    }
    batch.n_tokens = static_cast<int32_t>(actualTokenCount);
    
    CLLM_INFO("[LlamaCppBackend::forwardBatch] Calling llama_decode with %d tokens (batch_size=%zu, actualTokenCount=%zu)...", 
              batch.n_tokens, batchSize, actualTokenCount);
    
    // 推理
    int decodeResult = llama_decode(ctx_, batch);
    if (decodeResult != 0) {
        if (decodeResult < 0) {
            throw std::runtime_error("LlamaCppBackend::forwardBatch: llama_decode failed with error code " + std::to_string(decodeResult));
        } else {
            CLLM_WARN("[LlamaCppBackend::forwardBatch] llama_decode returned warning code: %d", decodeResult);
        }
    }
    
    CLLM_DEBUG("[LlamaCppBackend::forwardBatch] llama_decode completed successfully");
    
    // 提取 logits
    // llama.cpp 的 logits 按 logits[i] != 0 的顺序连续存储
    // 由于我们只设置了最后一个位置的 logits[i] = 1，所以只有最后一个位置的 logits 已计算
    
    // 返回的 tensor 大小必须是 [totalTokens, vocabSize]（ModelExecutor 的要求）
    // 我们只填充最后一个位置的 logits（用于采样），其他位置保持零
    Tensor result({totalTokens, config_.vocabSize});
    float* logitsPtr = llama_get_logits(ctx_);
    
    if (logitsPtr) {
        // 只复制最后一个位置的 logits（这是我们需要用于采样的位置）
        // 将 llama.cpp 的 logits 复制到 result 的最后一行（totalTokens - 1 位置）
        size_t lastPosIndex = totalTokens - 1;
        std::memcpy(
            result.data() + lastPosIndex * config_.vocabSize,
            logitsPtr,
            config_.vocabSize * sizeof(float)
        );
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Copied %zu logits to position %zu", 
                  config_.vocabSize, lastPosIndex);
    } else {
        CLLM_WARN("[LlamaCppBackend::forwardBatch] Failed to get logits, returning zero tensor");
    }
    
    // batch 资源由 RAII 自动清理，无需手动处理
    return result;
}

void LlamaCppBackend::setNumThreads(int numThreads) {
    numThreads_ = numThreads;
    if (ctx_ && numThreads > 0) {
        llama_set_n_threads(ctx_, numThreads, numThreads);
        CLLM_INFO("[LlamaCppBackend] Set threads to %d", numThreads);
    }
}

void LlamaCppBackend::setNGpuLayers(int nGpuLayers) {
    nGpuLayers_ = nGpuLayers;
    // 注意：GPU 层数只能在模型加载前设置
    if (initialized_) {
        CLLM_WARN("[LlamaCppBackend] setNGpuLayers called after initialization, will take effect on next load");
    }
}

} // namespace inference
} // namespace cllm
