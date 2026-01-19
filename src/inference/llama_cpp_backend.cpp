/**
 * @file llama_cpp_backend.cpp
 * @brief llama.cpp 推理后端实现
 * 
 * 参考文档：llama.cpp后端集成设计.md
 */

#include "cllm/inference/llama_cpp_backend.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"

#include "llama.h"

#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cstdint>

namespace cllm {
namespace inference {

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
    , nSeqMax_(0)  // Phase 2: 将在 initialize() 中从配置读取
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
    contextParams_->n_ctx = static_cast<uint32_t>(config_.maxSequenceLength);
    
    // 设置批处理大小
    contextParams_->n_batch = config_.llamaBatchSize > 0 ? static_cast<uint32_t>(config_.llamaBatchSize) : 512;
    
    // 设置线程数
    if (numThreads_ > 0) {
        contextParams_->n_threads = numThreads_;
        contextParams_->n_threads_batch = numThreads_;
    } else {
        // 使用默认值（llama.cpp 会自动选择）
        contextParams_->n_threads = 0;
        contextParams_->n_threads_batch = 0;
    }
    
    // Phase 2: 从配置读取并设置 n_seq_max（必须在创建context之前设置）
    nSeqMax_ = Config::instance().backendLlamaCppNSeqMax();
    contextParams_->n_seq_max = nSeqMax_;
    
    CLLM_INFO("[LlamaCppBackend] Context params: n_ctx=%u, n_batch=%u, n_threads=%d, n_seq_max=%d",
              contextParams_->n_ctx, contextParams_->n_batch, contextParams_->n_threads, contextParams_->n_seq_max);
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
        
        // 8. Phase 2: 初始化序列ID池
        initializeSequenceIdPool();
        
        // 9. Phase 4: 初始化KV缓存管理器
        // 从配置读取参数（如果配置中有），否则使用默认值
        size_t maxItems = 4 * 1024 * 1024;  // 默认值：4M条目
        size_t maxMemoryMb = 1024;  // 默认值：1024MB
        
        // TODO: 从配置读取 maxKVCachesItems 和 kvCacheMaxMemoryMb（如果配置中已添加）
        // 当前先使用默认值，后续可以从 Config 读取
        kvCacheManager_ = std::make_unique<KVCacheManager>(maxItems, maxMemoryMb);
        CLLM_INFO("[LlamaCppBackend] KV cache manager initialized: maxItems=%zu, maxMemoryMb=%zu", 
                  maxItems, maxMemoryMb);
        
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
    
    // 判断是否是第一次调用（完整序列）还是增量推理
    bool isFirstCall = (inputIds.size() > 1);
    if (isFirstCall) {
        currentPosition_ = 0;
    }
    
    // 1. 转换为 llama_token
    std::vector<int32_t> tokens = convertToLlamaTokens(inputIds);
    
    // 2. 创建 batch
    struct llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    
    try {
        // 填充 batch
        for (size_t i = 0; i < tokens.size(); ++i) {
            batch.token[i] = tokens[i];
            batch.pos[i] = static_cast<llama_pos>(currentPosition_ + i);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i] = new llama_seq_id[1];
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (i == tokens.size() - 1);
        }
        batch.n_tokens = static_cast<int32_t>(tokens.size());
        
        // 3. 推理
        CLLM_INFO("[LlamaCppBackend] Calling llama_decode with %d tokens, starting at position %zu...", 
                  batch.n_tokens, currentPosition_);
        int decodeResult = llama_decode(ctx_, batch);
        CLLM_INFO("[LlamaCppBackend] llama_decode returned: %d", decodeResult);
        
        if (decodeResult != 0) {
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (batch.seq_id[i]) {
                    delete[] batch.seq_id[i];
                    batch.seq_id[i] = nullptr;
                }
            }
            llama_batch_free(batch);
            
            if (decodeResult < 0) {
                throw std::runtime_error("LlamaCppBackend::forward: llama_decode failed with error code " + std::to_string(decodeResult));
            } else {
                CLLM_WARN("[LlamaCppBackend] llama_decode returned warning code: %d", decodeResult);
            }
        }
        
        // 4. 更新当前位置
        currentPosition_ += tokens.size();
        
        // 5. 提取 logits
        size_t seqLen = inputIds.size();
        Tensor result({seqLen, config_.vocabSize});
        
        float* logitsPtr = llama_get_logits(ctx_);
        if (logitsPtr) {
            // 只复制最后一个 token 的 logits
            // 前面的位置可以设置为 0 或重复最后一个
            std::memcpy(
                result.data() + (seqLen - 1) * config_.vocabSize,
                logitsPtr,
                config_.vocabSize * sizeof(float)
            );
            
            // 前面的位置设置为 0（或复制最后一个）
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
        
        // 5. 清理 batch（包括 seq_id 数组）
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.seq_id[i]) {
                delete[] batch.seq_id[i];
                batch.seq_id[i] = nullptr;
            }
        }
        llama_batch_free(batch);
        
        return result;
        
    } catch (...) {
        // 确保清理 batch（包括 seq_id 数组）
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.seq_id[i]) {
                delete[] batch.seq_id[i];
                batch.seq_id[i] = nullptr;
            }
        }
        llama_batch_free(batch);
        throw;
    }
}

Tensor LlamaCppBackend::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize,
    const std::vector<size_t> &sequenceIds
) {
    if (!initialized_) {
        throw std::runtime_error("LlamaCppBackend::forwardBatch: backend not initialized");
    }
    
    if (requestPositions.size() != batchSize) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: requestPositions size mismatch");
    }
    
    // Phase 2: 验证 sequenceIds 参数（必须提供 requestId 列表）
    if (sequenceIds.empty()) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: sequenceIds is required (requestId list)");
    }
    if (sequenceIds.size() != batchSize) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: sequenceIds size mismatch");
    }
    
    // 统计总 token 数（用于输出张量大小）
    size_t totalTokens = 0;
    std::vector<size_t> seqLengths(batchSize, 0);
    for (size_t i = 0; i < batchSize; ++i) {
        size_t seqLength = requestPositions[i].second - requestPositions[i].first;
        seqLengths[i] = seqLength;
        totalTokens += seqLength;
    }
    
    if (totalTokens == 0) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: total tokens cannot be zero");
    }
    
    // 计算实际需要送入 llama_decode 的 token 数：
    // 新请求送全量序列；已有请求仅送最后一个 token
    std::vector<bool> isNewRequest(batchSize, false);
    std::vector<int32_t> cachedSeqIds(batchSize, -1);
    size_t actualTokenCount = 0;
    for (size_t i = 0; i < batchSize; ++i) {
        size_t requestId = sequenceIds[i];
        int32_t seqId = getSequenceId(requestId);
        cachedSeqIds[i] = seqId;
        if (seqId == -1) {
            isNewRequest[i] = true;
            actualTokenCount += seqLengths[i];
        } else {
            actualTokenCount += 1;
        }
    }
    
    if (actualTokenCount == 0) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: actual tokens cannot be zero");
    }
    
    // 转换为 llama_token
    std::vector<int32_t> allTokens;
    allTokens.reserve(totalTokens);
    for (int id : flatInputIds) {
        allTokens.push_back(static_cast<int32_t>(id));
    }
    
    // 创建 batch
    struct llama_batch batch = llama_batch_init(static_cast<int32_t>(actualTokenCount), 0, static_cast<int32_t>(batchSize));
    
    try {
        size_t tokenIdx = 0;
        for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            const auto& pos = requestPositions[batchIdx];
            size_t seqStart = pos.first;
            size_t seqEnd = pos.second;
            
            // Phase 2: 获取 requestId 并分配/查询 seq_id
            size_t requestId = sequenceIds[batchIdx];
            int32_t seqId = cachedSeqIds[batchIdx];
            
            // 如果是新请求（首次分配），分配 seq_id
            if (seqId == -1) {
                seqId = allocateSequenceId(requestId);
                if (seqId == -1) {
                    throw std::runtime_error(
                        "LlamaCppBackend::forwardBatch: failed to allocate seq_id for request " + std::to_string(requestId) + " (pool exhausted)"
                    );
                }
                cachedSeqIds[batchIdx] = seqId;
            }
            
            size_t actualSeqStart = seqStart;
            size_t actualSeqEnd = seqEnd;
            if (!isNewRequest[batchIdx]) {
                if (seqEnd == seqStart) {
                    continue;
                }
                actualSeqStart = seqEnd - 1;
            }
            
            for (size_t i = actualSeqStart; i < actualSeqEnd; ++i) {
                batch.token[tokenIdx] = allTokens[i];
                size_t posInSeq = isNewRequest[batchIdx] ? (i - seqStart) : (seqEnd - seqStart - 1);
                batch.pos[tokenIdx] = static_cast<llama_pos>(posInSeq);
                batch.n_seq_id[tokenIdx] = 1;
                // seq_id 必须指向有效的数组，不能为 nullptr
                batch.seq_id[tokenIdx] = new llama_seq_id[1];
                // Phase 2: 使用基于 requestId 的 seq_id（而不是批处理索引）
                batch.seq_id[tokenIdx][0] = static_cast<llama_seq_id>(seqId);
                // 只计算最后一个位置的 logits
                batch.logits[tokenIdx] = (i == actualSeqEnd - 1);
                ++tokenIdx;
            }
        }
        batch.n_tokens = static_cast<int32_t>(actualTokenCount);
        
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Calling llama_decode with %d tokens (totalTokens=%zu)...", 
                  batch.n_tokens, totalTokens);
        
        // 推理
        int decodeResult = llama_decode(ctx_, batch);
        if (decodeResult != 0) {
            // 清理 batch（包括 seq_id 数组）
            for (int32_t i = 0; i < batch.n_tokens; ++i) {
                if (batch.seq_id[i]) {
                    delete[] batch.seq_id[i];
                    batch.seq_id[i] = nullptr;
                }
            }
            llama_batch_free(batch);
            
            if (decodeResult < 0) {
                throw std::runtime_error("LlamaCppBackend::forwardBatch: llama_decode failed with error code " + std::to_string(decodeResult));
            } else {
                CLLM_WARN("[LlamaCppBackend::forwardBatch] llama_decode returned warning code: %d", decodeResult);
            }
        }
        
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] llama_decode completed successfully");
        
        // Phase 4: 更新KV缓存统计信息
        // requestPositions 对应每个请求的完整输入序列（prompt + 已生成 tokens）
        if (kvCacheManager_) {
            for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
                size_t requestId = sequenceIds[batchIdx];
                const auto& pos = requestPositions[batchIdx];
                size_t sequenceLength = pos.second - pos.first;  // 当前序列长度（tokens）
                
                // 更新统计信息（序列长度 = token数量）
                kvCacheManager_->updateKVCacheStats(requestId, sequenceLength);
            }
        }
        
        // 提取 logits
        // llama.cpp 的 logits 按 logits[i] != 0 的顺序连续存储
        // 由于我们只设置了最后一个位置的 logits[i] = 1，所以只有最后一个位置的 logits 已计算
        
        // 返回的 tensor 大小必须是 [totalTokens, vocabSize]（ModelExecutor 的要求）
        // 我们只填充最后一个位置的 logits（用于采样），其他位置保持零
        Tensor result({totalTokens, config_.vocabSize});
        float* logitsPtr = llama_get_logits(ctx_);
        
        if (logitsPtr) {
            // 跟踪哪些 token 位置需要 logits，以及它们在 batch 中的索引
            // llama_get_logits(ctx_) 返回的 logits 是按 batch.logits[i] = 1 的顺序连续存储的
            size_t logitsOffset = 0;
            size_t tokenIdx = 0;
            
            for (size_t batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
                const auto& pos = requestPositions[batchIdx];
                size_t seqStart = pos.first;
                size_t seqEnd = pos.second;
                
                size_t actualSeqStart = seqStart;
                size_t actualSeqEnd = seqEnd;
                if (!isNewRequest[batchIdx]) {
                    if (seqEnd == seqStart) {
                        continue;
                    }
                    actualSeqStart = seqEnd - 1;
                }
                
                // 遍历该请求的所有 token
                for (size_t i = actualSeqStart; i < actualSeqEnd; ++i) {
                    if (batch.logits[tokenIdx]) {
                        // 这个 token 需要 logits，从 logitsPtr 中提取
                        size_t resultPos = i;  // 在 result 中的位置
                        std::memcpy(
                            result.data() + resultPos * config_.vocabSize,
                            logitsPtr + logitsOffset * config_.vocabSize,
                            config_.vocabSize * sizeof(float)
                        );
                        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Copied logits for request %zu, token position %zu (batch tokenIdx=%zu, logitsOffset=%zu)", 
                                  batchIdx, resultPos, tokenIdx, logitsOffset);
                        logitsOffset++;
                    }
                    tokenIdx++;
                }
            }
            
            CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Extracted logits for %zu positions", logitsOffset);
        } else {
            CLLM_WARN("[LlamaCppBackend::forwardBatch] Failed to get logits, returning zero tensor");
        }
        
        // 清理 batch（包括 seq_id 数组）
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.seq_id[i]) {
                delete[] batch.seq_id[i];
                batch.seq_id[i] = nullptr;
            }
        }
        llama_batch_free(batch);
        
        return result;
        
    } catch (...) {
        // 确保清理 batch（包括 seq_id 数组）
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.seq_id[i]) {
                delete[] batch.seq_id[i];
                batch.seq_id[i] = nullptr;
            }
        }
        llama_batch_free(batch);
        throw;
    }
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

void LlamaCppBackend::initializeSequenceIdPool() {
    // Phase 2: n_seq_max 已在 createContextParams() 中从配置读取并设置
    // 这里只需要初始化序列ID池
    CLLM_INFO("[LlamaCppBackend] Initializing sequence ID pool with n_seq_max=%d", nSeqMax_);
    
    // 初始化可用序列ID池：0 到 n_seq_max-1
    availableSeqIds_.clear();
    availableSeqIds_.reserve(nSeqMax_);
    for (int32_t i = 0; i < nSeqMax_; ++i) {
        availableSeqIds_.push_back(i);
    }
    
    // 清空映射
    requestIdToSeqId_.clear();
    
    CLLM_INFO("[LlamaCppBackend] Sequence ID pool initialized: %d available IDs", nSeqMax_);
}

int32_t LlamaCppBackend::allocateSequenceId(size_t requestId) {
    std::lock_guard<std::mutex> lock(sequenceIdMutex_);
    
    // 检查是否已经分配过
    auto it = requestIdToSeqId_.find(requestId);
    if (it != requestIdToSeqId_.end()) {
        CLLM_DEBUG("[LlamaCppBackend] Request %zu already has seq_id %d", requestId, it->second);
        return it->second;
    }
    
    // 检查可用池是否为空
    if (availableSeqIds_.empty()) {
        CLLM_WARN("[LlamaCppBackend] No available sequence IDs for request %zu (pool exhausted)", requestId);
        return -1;
    }
    
    // 从可用池中分配一个序列ID
    int32_t seqId = availableSeqIds_.back();
    availableSeqIds_.pop_back();
    
    // 建立映射
    requestIdToSeqId_[requestId] = seqId;
    
    CLLM_DEBUG("[LlamaCppBackend] Allocated seq_id %d for request %zu (remaining: %zu)",
              seqId, requestId, availableSeqIds_.size());
    
    return seqId;
}

bool LlamaCppBackend::releaseSequenceId(size_t requestId) {
    std::lock_guard<std::mutex> lock(sequenceIdMutex_);
    
    // 查找 requestId 对应的 seqId
    auto it = requestIdToSeqId_.find(requestId);
    if (it == requestIdToSeqId_.end()) {
        CLLM_WARN("[LlamaCppBackend] Request %zu not found in sequence ID mapping", requestId);
        return false;
    }
    
    int32_t seqId = it->second;
    
    // 删除映射
    requestIdToSeqId_.erase(it);
    
    // 将 seqId 返回可用池
    availableSeqIds_.push_back(seqId);
    
    CLLM_DEBUG("[LlamaCppBackend] Released seq_id %d for request %zu (available: %zu)",
              seqId, requestId, availableSeqIds_.size());
    
    return true;
}

int32_t LlamaCppBackend::getSequenceId(size_t requestId) const {
    std::lock_guard<std::mutex> lock(sequenceIdMutex_);
    
    auto it = requestIdToSeqId_.find(requestId);
    if (it == requestIdToSeqId_.end()) {
        return -1;
    }
    
    return it->second;
}

bool LlamaCppBackend::cleanupKVCache(size_t requestId) {
    if (!kvCacheManager_ || !ctx_) {
        CLLM_WARN("[LlamaCppBackend] Cannot clean KV cache: kvCacheManager_ or ctx_ is nullptr");
        return false;
    }
    
    // 获取 requestId 对应的 seqId
    int32_t seqId = getSequenceId(requestId);
    if (seqId < 0) {
        CLLM_WARN("[LlamaCppBackend] Cannot clean KV cache: seqId not found for requestId=%zu", requestId);
        return false;
    }
    
    // 调用 KVCacheManager 清理KV缓存
    bool success = kvCacheManager_->removeKVCache(ctx_, requestId, seqId);
    
    if (success) {
        CLLM_DEBUG("[LlamaCppBackend] Cleaned KV cache for requestId=%zu, seqId=%d", requestId, seqId);
    } else {
        CLLM_WARN("[LlamaCppBackend] Failed to clean KV cache for requestId=%zu", requestId);
    }
    
    return success;
}

} // namespace inference
} // namespace cllm
