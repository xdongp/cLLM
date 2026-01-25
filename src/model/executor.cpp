/**
 * @file executor.cpp
 * @brief 模型执行器模块实现
 * @author cLLM Team
 * @date 2026-01-08
 */
#include "cllm/model/executor.h"
#include "cllm/sampler.h"
#include "cllm/kv_cache/cache.h"
#include "cllm/common/logger.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <errno.h>
#include <fstream>
#include <stdexcept>

namespace cllm {

ModelExecutor::ModelExecutor(
    const std::string& modelPath,
    const std::string& quantization,
    bool enableSIMD,
    bool useLibTorch,
    const std::string& backendType,
    const ModelConfig* initialConfig
)
    : modelPath_(modelPath)
    , quantization_(quantization)
    , enableSIMD_(enableSIMD)
    , useLibTorch_(useLibTorch)
    , backendType_(backendType)
    , modelHandle_(nullptr)
    , modelWeights_(nullptr)
    , modelSize_(0)
    , sampler_(nullptr)
    , kvCache_(nullptr)
    , quantizationManager_(nullptr)
    , inferenceOptimizer_(nullptr)
    , batchProcessor_(nullptr)
    , isModelLoaded_(false)
    , inferenceBuffer_()
    , inputBuffer_()
    , inferenceEngine_(nullptr) {
    
    if (!backendType_.empty()) {
        CLLM_INFO("[ModelExecutor] Initializing with backend: %s", backendType_.c_str());
    } else {
        CLLM_INFO("[ModelExecutor] Initializing with %s backend", 
                  (useLibTorch_ ? "LibTorch" : "Kylin"));
    }
    
    sampler_ = std::make_unique<Sampler>();

    if (initialConfig) {
        config_ = *initialConfig;
    }
    
    // Note: config_ is default-constructed, check useKVCache only if explicitly set
    // For now, skip KV cache initialization in constructor
    // kvCache_ will be initialized in loadModel() if needed
    
    if (!quantization.empty()) {
        quantizationManager_ = std::make_unique<QuantizationManager>(quantization);
    }
    
    inferenceOptimizer_ = std::make_unique<InferenceOptimizer>(enableSIMD);
    batchProcessor_ = std::make_unique<BatchProcessor>(this);
    
    // 初始化 InferenceEngine
    if (!backendType_.empty()) {
        inferenceEngine_ = std::make_unique<inference::InferenceEngine>(config_, modelPath_, backendType_);
    } else {
        inferenceEngine_ = std::make_unique<inference::InferenceEngine>(config_, modelPath_, useLibTorch_);
    }
    if (!inferenceEngine_->initialize()) {
        throw std::runtime_error("ModelExecutor: failed to initialize inference engine");
    }
    
    // 同步backend更新后的config(例如LibTorch自动检测vocab_size)
    config_ = inferenceEngine_->getConfig();
    CLLM_INFO("[ModelExecutor] Config synchronized from InferenceEngine");
    CLLM_INFO("[ModelExecutor]   vocab_size: %u", config_.vocabSize);
    
    _optimizeMemoryUsage();
}

ModelExecutor::~ModelExecutor() {
    unloadModel();
}

void ModelExecutor::loadModel() {
    if (isModelLoaded_) {
        CLLM_WARN("Model already loaded");
        return;
    }

    // ✅ InferenceEngine 会在构造函数里完成 initialize()，并由后端负责权重加载：
    // - KylinBackend: 通过 ModelLoader 读取 .bin
    // - LibTorchBackend: 加载 TorchScript .pt
    // 这里不再重复把整个权重文件读入内存，避免双份加载导致 OOM。
    if (inferenceEngine_ && inferenceEngine_->isInitialized()) {
        CLLM_INFO("[ModelExecutor] InferenceEngine already initialized, skipping raw weight loading");
        _applyInferenceOptimizations();
        _warmupModel();
        isModelLoaded_ = true;
        return;
    }

    // 回退路径：理论上不应发生
    throw std::runtime_error("ModelExecutor: inference engine not initialized");
}

void ModelExecutor::unloadModel() {
    if (modelWeights_) {
        delete[] static_cast<float*>(modelWeights_);
        modelWeights_ = nullptr;
        modelHandle_ = nullptr;  // 确保handle也被清空
    } else if (modelHandle_) {
        // 对于量化模型，modelWeights_可能为空但modelHandle_不为空
        modelHandle_ = nullptr;
    }
    
    modelSize_ = 0;
    isModelLoaded_ = false;
}

void ModelExecutor::_loadFullPrecisionModel() {
    CLLM_INFO("Attempting to load full precision model from: %s", modelPath_.c_str());
    std::ifstream file(modelPath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        CLLM_ERROR("Failed to open model file: %s, error: %s", modelPath_.c_str(), strerror(errno));
        throw std::runtime_error("Failed to open model file: " + modelPath_ + ", error: " + strerror(errno));
    }
    
    modelSize_ = file.tellg();
    CLLM_INFO("Model file size: %zu bytes", modelSize_);
    
    if (modelSize_ == 0) {
        file.close();
        CLLM_ERROR("Model file is empty: %s", modelPath_.c_str());
        throw std::runtime_error("Model file is empty: " + modelPath_);
    }
    
    size_t numFloats = modelSize_ / sizeof(float);
    CLLM_DEBUG("Number of floats: %zu", numFloats);
    
    modelWeights_ = new float[numFloats];
    if (!modelWeights_) {
        file.close();
        CLLM_ERROR("Failed to allocate memory for model weights");
        throw std::runtime_error("Failed to allocate memory for model weights");
    }
    
    file.seekg(0, std::ios::beg);
    file.read(static_cast<char*>(modelWeights_), modelSize_);
    
    if (!file) {
        CLLM_ERROR("Failed to read model file: %s, bytes read: %zu", modelPath_.c_str(), file.gcount());
        delete[] static_cast<float*>(modelWeights_);
        modelWeights_ = nullptr;
        file.close();
        throw std::runtime_error("Failed to read model file: " + modelPath_);
    }
    
    file.close();
    modelHandle_ = modelWeights_;
    
    CLLM_INFO("Model weights loaded successfully");
}

void ModelExecutor::_loadInt8QuantizedModel() {
    CLLM_INFO("Loading int8 quantized model from: %s", modelPath_.c_str());
    std::ifstream file(modelPath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        CLLM_ERROR("Failed to open int8 quantized model file: %s, error: %s", 
                  modelPath_.c_str(), strerror(errno));
        throw std::runtime_error("Failed to open int8 quantized model file: " + modelPath_ + ", error: " + strerror(errno));
    }
    
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (fileSize == 0) {
        file.close();
        CLLM_ERROR("Int8 quantized model file is empty: %s", modelPath_.c_str());
        throw std::runtime_error("Int8 quantized model file is empty: " + modelPath_);
    }
    
    int8_t* quantizedWeights = new (std::nothrow) int8_t[fileSize];
    if (!quantizedWeights) {
        file.close();
        CLLM_ERROR("Failed to allocate memory for int8 quantized weights: %zu bytes", fileSize);
        throw std::runtime_error("Failed to allocate memory for int8 quantized weights");
    }
    
    file.read(reinterpret_cast<char*>(quantizedWeights), fileSize);
    if (!file) {
        delete[] quantizedWeights;
        file.close();
        CLLM_ERROR("Failed to read int8 quantized model file: %s, bytes read: %zu", 
                  modelPath_.c_str(), file.gcount());
        throw std::runtime_error("Failed to read int8 quantized model file: " + modelPath_);
    }
    file.close();
    
    // 使用量化管理器进行处理
    if (quantizationManager_) {
        quantizationManager_->quantizeModel(quantizedWeights, fileSize);
    }
    
    // 转换为float用于推理
    modelSize_ = fileSize * sizeof(float);
    modelWeights_ = new (std::nothrow) float[fileSize];
    if (!modelWeights_) {
        delete[] quantizedWeights;
        CLLM_ERROR("Failed to allocate memory for float weights: %zu bytes", modelSize_);
        throw std::runtime_error("Failed to allocate memory for float weights");
    }
    
    for (size_t i = 0; i < fileSize; ++i) {
        static_cast<float*>(modelWeights_)[i] = static_cast<float>(quantizedWeights[i]) / 127.0f;
    }
    
    delete[] quantizedWeights;
    modelHandle_ = modelWeights_;
    CLLM_INFO("Int8 quantized model loaded successfully, size: %zu bytes", modelSize_);
}

void ModelExecutor::_loadInt4QuantizedModel() {
    CLLM_INFO("Loading int4 quantized model from: %s", modelPath_.c_str());
    std::ifstream file(modelPath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        CLLM_ERROR("Failed to open int4 quantized model file: %s, error: %s", 
                  modelPath_.c_str(), strerror(errno));
        throw std::runtime_error("Failed to open int4 quantized model file: " + modelPath_ + ", error: " + strerror(errno));
    }
    
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (fileSize == 0) {
        file.close();
        CLLM_ERROR("Int4 quantized model file is empty: %s", modelPath_.c_str());
        throw std::runtime_error("Int4 quantized model file is empty: " + modelPath_);
    }
    
    uint8_t* quantizedWeights = new (std::nothrow) uint8_t[fileSize];
    if (!quantizedWeights) {
        file.close();
        CLLM_ERROR("Failed to allocate memory for int4 quantized weights: %zu bytes", fileSize);
        throw std::runtime_error("Failed to allocate memory for int4 quantized weights");
    }
    
    file.read(reinterpret_cast<char*>(quantizedWeights), fileSize);
    if (!file) {
        delete[] quantizedWeights;
        file.close();
        CLLM_ERROR("Failed to read int4 quantized model file: %s, bytes read: %zu", 
                  modelPath_.c_str(), file.gcount());
        throw std::runtime_error("Failed to read int4 quantized model file: " + modelPath_);
    }
    file.close();
    
    // 使用量化管理器进行处理
    if (quantizationManager_) {
        quantizationManager_->quantizeModel(quantizedWeights, fileSize);
    }
    
    // 转换为float用于推理
    modelSize_ = fileSize * 2 * sizeof(float);
    modelWeights_ = new (std::nothrow) float[fileSize * 2];
    if (!modelWeights_) {
        delete[] quantizedWeights;
        CLLM_ERROR("Failed to allocate memory for float weights: %zu bytes", modelSize_);
        throw std::runtime_error("Failed to allocate memory for float weights");
    }
    
    for (size_t i = 0; i < fileSize; ++i) {
        uint8_t packed = quantizedWeights[i];
        static_cast<float*>(modelWeights_)[i * 2] = (static_cast<float>(packed & 0x0F) - 7.5f) / 7.5f;
        static_cast<float*>(modelWeights_)[i * 2 + 1] = (static_cast<float>((packed >> 4) & 0x0F) - 7.5f) / 7.5f;
    }
    
    delete[] quantizedWeights;
    modelHandle_ = modelWeights_;
    CLLM_INFO("Int4 quantized model loaded successfully, size: %zu bytes", modelSize_);
}

void ModelExecutor::_applyInferenceOptimizations() {
    if (inferenceOptimizer_) {
        inferenceOptimizer_->applyOptimizations();
    }
}

void ModelExecutor::_warmupModel() {
    std::vector<int> dummyInput(config_.vocabSize > 10 ? 10 : config_.vocabSize, 1);
    
    try {
        generate(dummyInput, 1, 1.0f);
    } catch (...) {
        
    }
}

FloatArray ModelExecutor::_prepareInput(const std::vector<int>& inputIds) {
    size_t inputSize = inputIds.size();
    FloatArray inputTensor(inputSize);
    
    for (size_t i = 0; i < inputSize; ++i) {
        inputTensor[i] = static_cast<float>(inputIds[i]);
    }
    
    return inputTensor;
}

void ModelExecutor::_processOutput(FloatArray& logits, size_t batchSize, size_t vocabSize) {
    if (kvCache_ && config_.useKVCache) {
        
    }
}

BatchOutput ModelExecutor::forward(const BatchInput& input) {
    // 🔥 优化：移除不必要的计时开销（在性能测试中可以跳过）
    // auto startTime = std::chrono::high_resolution_clock::now();
    
    if (!isModelLoaded_) {
        throw std::runtime_error("Model is not loaded");
    }
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("ModelExecutor::forward() called");
    CLLM_DEBUG("  Input batch size: %zu", input.batchSize);
    CLLM_DEBUG("  Input IDs size: %zu", input.inputIds.size());
    CLLM_DEBUG("  Request positions count: %zu", input.requestPositions.size());
    #endif
    
    // 🔥 优化：移除modelMutex_锁，因为：
    // 1. InferenceEngine（LlamaCppBackend）内部已经有锁保护llama_decode
    // 2. 外层调用者（如Scheduler）会提供必要的同步
    // 3. 减少锁竞争，提升并发性能
    // std::lock_guard<std::mutex> lock(modelMutex_);  // 已移除
    
    // 🔥 优化：移除_prepareInput()调用，因为InferenceEngine::forwardBatch()直接接受std::vector<int>
    // 这样可以避免不必要的int到float转换和FloatArray创建
    // FloatArray inputTensor = _prepareInput(input.inputIds);  // 已移除
    
    size_t totalTokens = input.getTotalTokens();
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("  Total tokens: %zu", totalTokens);
    #endif
    
    // 🔥 优化：直接获取Tensor，避免数据拷贝
    if (!inferenceEngine_) {
        throw std::runtime_error("ModelExecutor::forward: inference engine not initialized");
    }
    
    inference::Tensor logitsTensor = inferenceEngine_->forwardBatch(
        input.inputIds,
        input.requestPositions,
        input.batchSize,
        input.sequenceIds
    );
    
    const auto& logitsShape = logitsTensor.shape();
    if (logitsShape.size() != 2 || logitsShape[0] != totalTokens || logitsShape[1] != config_.vocabSize) {
        throw std::runtime_error("ModelExecutor::forward: logits shape mismatch");
    }
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("  Output tensor size: %zu", logitsTensor.size());
    CLLM_DEBUG("  Expected size: %zu * %zu = %zu", totalTokens, config_.vocabSize, totalTokens * config_.vocabSize);
    #endif
    
    // 🔥 优化：_processOutput()目前是空实现，可以跳过
    // _processOutput(outputTensor, input.batchSize, config_.vocabSize);
    
    BatchOutput output;
    // 🔥 优化：直接使用Tensor，完全避免数据拷贝
    // inference::Tensor就是kylin::Tensor（通过using Tensor = kylin::Tensor），可以直接移动
    output.logitsTensor = std::make_unique<kylin::Tensor>(std::move(logitsTensor));
    // 为了兼容性，仍然创建空的FloatArray（getLogitsForRequest会优先使用logitsTensor）
    output.logits = FloatArray();  // 空FloatArray，getLogitsForRequest会使用logitsTensor
    output.requestPositions = input.requestPositions;
    output.sequenceIds = input.sequenceIds;
    
    #ifdef CLLM_DEBUG_MODE
    CLLM_DEBUG("  BatchOutput created with logits size: %zu", output.logits.size());
    #endif
    
    // 🔥 优化：减少不必要的统计更新开销（在性能测试中可以跳过）
    // auto endTime = std::chrono::high_resolution_clock::now();
    // float inferenceTime = std::chrono::duration<float>(endTime - startTime).count();
    // stats_.update(inferenceTime, totalTokens);  // 暂时跳过，减少开销
    
    return output;
}

std::vector<int> ModelExecutor::generate(
    const std::vector<int>& inputIds,
    size_t maxNewTokens,
    float temperature
) {
    std::vector<int> generatedTokens;
    std::vector<int> currentInput = inputIds;
    
    for (size_t i = 0; i < maxNewTokens; ++i) {
        BatchInput batchInput;
        batchInput.inputIds = currentInput;
        batchInput.batchSize = 1;
        batchInput.requestPositions = {{0, currentInput.size()}};
        batchInput.sequenceIds = {0};
        
        BatchOutput output = forward(batchInput);
        
        // logits 存储布局是按 model vocab 展开的，但采样应限制在 tokenizer vocab 内
        // （两者可能不同，例如 Qwen3: model vocab=151936, tokenizer vocab=151669）
        FloatArray fullLogits = output.getLogitsForRequest(0, config_.vocabSize);
        size_t sampleVocabSize = std::min(config_.tokenizerVocabSize, config_.vocabSize);
        FloatArray logits(sampleVocabSize);
        for (size_t j = 0; j < sampleVocabSize; ++j) {
            logits[j] = fullLogits[j];
        }

        int nextToken = sampler_->sample(logits, temperature);
        
        generatedTokens.push_back(nextToken);
        currentInput.push_back(nextToken);
        
        if (nextToken == 0) {
            break;
        }
    }
    
    return generatedTokens;
}

ModelStats ModelExecutor::getStats() const {
    return stats_;
}

void ModelExecutor::resetStats() {
    stats_.reset();
}

Sampler* ModelExecutor::getSampler() const {
    return sampler_.get();
}

KVCache* ModelExecutor::getKVCache() const {
    return kvCache_.get();
}

const ModelConfig& ModelExecutor::getConfig() const {
    return config_;
}

std::string ModelExecutor::getBackendName() const {
    if (!inferenceEngine_) {
        return "None";
    }
    return inferenceEngine_->getBackendType();
}

void ModelExecutor::setConfig(const ModelConfig& config) {
    config_ = config;
    // 如果 InferenceEngine 已初始化，也更新它的配置
    if (inferenceEngine_) {
        // InferenceEngine 的配置是在构造时传入的，需要重新创建
        if (!backendType_.empty()) {
            inferenceEngine_ = std::make_unique<inference::InferenceEngine>(config_, modelPath_, backendType_);
        } else {
            inferenceEngine_ = std::make_unique<inference::InferenceEngine>(config_, modelPath_, useLibTorch_);
        }
        if (!inferenceEngine_->initialize()) {
            throw std::runtime_error("ModelExecutor::setConfig: failed to reinitialize inference engine");
        }
    }
}

void ModelExecutor::setTokenizerVocabSize(size_t tokenizerVocabSize) {
    config_.tokenizerVocabSize = tokenizerVocabSize;
    // tokenizerVocabSize 不影响 InferenceEngine，所以不需要重新初始化
}

bool ModelExecutor::releaseSequenceId(size_t requestId) const {
    if (!inferenceEngine_) {
        return false;
    }
    
    // Phase 2: 委托给 InferenceEngine
    return inferenceEngine_->releaseSequenceId(requestId);
}

bool ModelExecutor::cleanupKVCache(size_t requestId) const {
    if (!inferenceEngine_) {
        return false;
    }
    
    // Phase 4: 委托给 InferenceEngine
    return inferenceEngine_->cleanupKVCache(requestId);
}

bool ModelExecutor::updateKVCacheRequestStatus(size_t requestId, inference::RequestStatus status) const {
    if (!inferenceEngine_) {
        return false;
    }

    return inferenceEngine_->updateKVCacheRequestStatus(requestId, status);
}

size_t ModelExecutor::evictKVCachesIfNeeded(double evictionThreshold) const {
    if (!inferenceEngine_) {
        return 0;
    }

    return inferenceEngine_->evictKVCachesIfNeeded(evictionThreshold);
}

size_t ModelExecutor::getAvailableSequenceIdCount() const {
    if (!inferenceEngine_) {
        return 0;
    }

    return inferenceEngine_->getAvailableSequenceIdCount();
}

int ModelExecutor::sampleToken(const std::vector<int>& inputIds, float temperature) {
    if (!isModelLoaded_) {
        throw std::runtime_error("Model is not loaded");
    }
    
    // 🔥 优化：移除modelMutex_锁（原因同forward()）
    // std::lock_guard<std::mutex> lock(modelMutex_);  // 已移除
    
    BatchInput batchInput;
    batchInput.inputIds = inputIds;
    batchInput.batchSize = 1;
    batchInput.requestPositions = {{0, inputIds.size()}};
    batchInput.sequenceIds = {0};
    
    BatchOutput output = forward(batchInput);
    
    // logits 存储布局是按 model vocab 展开的，但采样应限制在 tokenizer vocab 内
    FloatArray fullLogits = output.getLogitsForRequest(0, config_.vocabSize);
    size_t sampleVocabSize = std::min(config_.tokenizerVocabSize, config_.vocabSize);
    FloatArray logits(sampleVocabSize);
    for (size_t j = 0; j < sampleVocabSize; ++j) {
        logits[j] = fullLogits[j];
    }

    return sampler_->sample(logits, temperature);
}

FloatArray ModelExecutor::_executeModelInference(const BatchInput& input) {
    size_t totalTokens = input.getTotalTokens();
    size_t outputSize = totalTokens * config_.vocabSize;
    
    CLLM_DEBUG("[ModelExecutor::_executeModelInference] Total tokens: %zu", totalTokens);
    CLLM_DEBUG("[ModelExecutor::_executeModelInference] Vocab size: %zu", config_.vocabSize);
    CLLM_DEBUG("[ModelExecutor::_executeModelInference] Output size: %zu (%zu * %zu)", outputSize, totalTokens, config_.vocabSize);
    
    if (!inferenceEngine_) {
        throw std::runtime_error("ModelExecutor::_executeModelInference: inference engine not initialized");
    }
    
    // 调用自研推理引擎，获得 [total_tokens, vocab_size] logits
    // Phase 2: 传递 sequenceIds（requestId）用于序列ID管理
    inference::Tensor logitsTensor = inferenceEngine_->forwardBatch(
        input.inputIds,
        input.requestPositions,
        input.batchSize,
        input.sequenceIds  // 传递 requestId 列表
    );
    
    const auto& logitsShape = logitsTensor.shape();
    if (logitsShape.size() != 2 || logitsShape[0] != totalTokens || logitsShape[1] != config_.vocabSize) {
        throw std::runtime_error("ModelExecutor::_executeModelInference: logits shape mismatch");
    }
    
    // 🔥 优化：直接使用Tensor，避免数据拷贝
    // 注意：由于BatchOutput现在支持直接使用Tensor，我们可以避免从Tensor到FloatArray的拷贝
    // 但是，由于forward()返回BatchOutput，而BatchOutput需要FloatArray或Tensor，
    // 我们仍然需要创建一个FloatArray（为了兼容性），或者直接使用Tensor
    // 
    // 为了最大化性能，我们直接使用Tensor，避免拷贝
    // 但是，由于forward()的返回类型是BatchOutput，而BatchOutput.logits是FloatArray，
    // 我们需要在forward()中设置logitsTensor而不是logits
    
    // 暂时仍然使用FloatArray（为了兼容性），但可以考虑后续优化为直接使用Tensor
    FloatArray outputTensor(outputSize);
    const float* src = logitsTensor.data();
    std::memcpy(outputTensor.data(), src, outputSize * sizeof(float));
    
    CLLM_DEBUG("[ModelExecutor::_executeModelInference] Generated logits for %zu token positions", totalTokens);
    return outputTensor;
}

void ModelExecutor::_optimizeMemoryUsage() {
    // 优化内存使用
    _enableMemoryCompression();
    
    // 根据模型大小分配推理缓冲区
    size_t bufferSize = config_.vocabSize * config_.maxSequenceLength * sizeof(float);
    inferenceBuffer_ = FloatArray(bufferSize);
    inputBuffer_ = FloatArray(bufferSize);
}

void ModelExecutor::_enableMemoryCompression() {
    // 启用内存压缩
    if (config_.useMemoryCompression) {
        // 实现内存压缩逻辑
    }
}



}  // namespace cllm
