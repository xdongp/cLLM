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
    
    CLLM_INFO("[LlamaCppBackend] Context params: n_ctx=%u, n_batch=%u, n_threads=%d",
              contextParams_->n_ctx, contextParams_->n_batch, contextParams_->n_threads);
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
    size_t batchSize
) {
    if (!initialized_) {
        throw std::runtime_error("LlamaCppBackend::forwardBatch: backend not initialized");
    }
    
    if (requestPositions.size() != batchSize) {
        throw std::invalid_argument("LlamaCppBackend::forwardBatch: requestPositions size mismatch");
    }
    
    // 检测是否是新请求
    // 如果第一个请求的起始位置是 0，说明这是一个新请求
    bool isNewRequest = (requestPositions[0].first == 0);
    
    // 如果是新请求，重置 currentPosition_
    if (isNewRequest) {
        currentPosition_ = 0;
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] New request detected, resetting currentPosition_ to 0");
    }
    
    // 对于 llama.cpp，我们需要检测是否是增量推理
    // 如果是增量推理，我们只处理最后一个 token，并从 currentPosition 开始
    bool isIncremental = false;
    size_t totalTokens = 0;
    for (const auto& pos : requestPositions) {
        size_t seqLength = pos.second - pos.first;
        totalTokens += seqLength;
    }
    
    // 检测是否是增量推理：
    // - 如果 currentPosition_ > 0，说明已经进行过推理
    // - 如果 totalTokens > 1，说明传递了多个 tokens（包括 prompt 和生成的 tokens）
    // - 在这种情况下，我们只处理最后一个 token（新生成的 token）
    if (currentPosition_ > 0 && totalTokens > 1) {
        isIncremental = true;
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
    
    // 对于增量推理，我们只处理最后一个 token
    size_t actualTokenCount = isIncremental ? 1 : totalTokens;
    size_t tokenStartIndex = isIncremental ? (totalTokens - 1) : 0;
    
    // 创建 batch
    struct llama_batch batch = llama_batch_init(static_cast<int32_t>(actualTokenCount), 0, static_cast<int32_t>(batchSize));
    
    try {
        size_t tokenIdx = 0;
        for (size_t seqId = 0; seqId < batchSize; ++seqId) {
            const auto& pos = requestPositions[seqId];
            size_t seqStart = pos.first;
            size_t seqEnd = pos.second;
            
            // 对于增量推理，只处理最后一个 token
            size_t actualSeqStart = isIncremental ? (seqEnd - 1) : seqStart;
            size_t actualSeqEnd = seqEnd;
            
            for (size_t i = actualSeqStart; i < actualSeqEnd; ++i) {
                batch.token[tokenIdx] = allTokens[i];
                // 对于增量推理，位置从 currentPosition 开始
                // 对于第一次推理，位置从 0 开始
                if (isIncremental) {
                    batch.pos[tokenIdx] = static_cast<llama_pos>(currentPosition_);
                } else {
                    batch.pos[tokenIdx] = static_cast<llama_pos>(i - seqStart);
                }
                batch.n_seq_id[tokenIdx] = 1;
                // seq_id 必须指向有效的数组，不能为 nullptr
                batch.seq_id[tokenIdx] = new llama_seq_id[1];
                batch.seq_id[tokenIdx][0] = static_cast<llama_seq_id>(seqId);
                // 只计算最后一个位置的 logits
                batch.logits[tokenIdx] = (i == actualSeqEnd - 1);
                ++tokenIdx;
            }
        }
        batch.n_tokens = static_cast<int32_t>(actualTokenCount);
        
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] Calling llama_decode with %d tokens (incremental=%d, currentPosition=%zu)...", 
                  batch.n_tokens, isIncremental, currentPosition_);
        
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
        
        // 更新当前位置
        currentPosition_ += actualTokenCount;
        
        CLLM_DEBUG("[LlamaCppBackend::forwardBatch] llama_decode completed successfully, currentPosition now: %zu", currentPosition_);
        
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

} // namespace inference
} // namespace cllm
