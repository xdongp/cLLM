/**
 * @file inference_engine.cpp
 * @brief 推理引擎统一接口实现
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/inference/kylin_backend.h"
#include "cllm/inference/libtorch_backend.h"
#ifdef CLLM_USE_LLAMA_CPP
#include "cllm/inference/llama_cpp_backend.h"
#endif
// BackendFactory is defined in backend_interface.h
#include "cllm/common/logger.h"
#include <stdexcept>
#include <filesystem>

namespace cllm {
namespace inference {

InferenceEngine::InferenceEngine(
    const ModelConfig &config,
    const std::string &modelPath,
    bool useLibTorch
)
    : config_(config)
    , useLibTorch_(useLibTorch)
    , backend_(nullptr) {
    
    CLLM_INFO("[InferenceEngine] Initializing inference engine...");
    
    // 自动检测后端类型（优先检查 .gguf 文件）
    std::string backendType;
    if (!modelPath.empty()) {
        namespace fs = std::filesystem;
        std::string ext = fs::path(modelPath).extension().string();
        if (ext == ".gguf") {
#ifdef CLLM_USE_LLAMA_CPP
            backendType = "llama_cpp";
            CLLM_INFO("[InferenceEngine] Detected GGUF format, using llama.cpp backend");
#else
            throw std::runtime_error("GGUF model detected but llama.cpp backend is not available in this build");
#endif
        } else if (useLibTorch) {
            backendType = "libtorch";
            CLLM_INFO("[InferenceEngine] Using LibTorch backend");
        } else {
            backendType = "kylin";
            CLLM_INFO("[InferenceEngine] Using Kylin (麒麟) backend");
        }
    } else {
        // 空路径，使用 Kylin（占位权重模式）
        backendType = "kylin";
        CLLM_INFO("[InferenceEngine] Empty model path, using Kylin backend (placeholder weights)");
    }
    
    CLLM_INFO("[InferenceEngine] Model path: %s", (modelPath.empty() ? "(placeholder weights)" : modelPath.c_str()));
    
    // 使用 BackendFactory 创建后端实例
    try {
        backend_ = BackendFactory::createBackend(backendType, config_, modelPath);
        CLLM_INFO("[InferenceEngine] Backend created: %s", backendType.c_str());
    } catch (const std::exception& e) {
        CLLM_ERROR("[InferenceEngine] Failed to create backend %s: %s", backendType.c_str(), e.what());
        throw;
    }
}

InferenceEngine::InferenceEngine(
    const ModelConfig &config,
    const std::string &modelPath,
    const std::string &backendType
)
    : config_(config)
    , useLibTorch_(backendType == "libtorch" || backendType == "LibTorch")
    , backend_(nullptr) {
    
    CLLM_INFO("[InferenceEngine] Initializing inference engine with backend: %s", backendType.c_str());
    CLLM_INFO("[InferenceEngine] Model path: %s", (modelPath.empty() ? "(placeholder weights)" : modelPath.c_str()));
    
    // 使用 BackendFactory 创建后端实例
    try {
        backend_ = BackendFactory::createBackend(backendType, config_, modelPath);
        CLLM_INFO("[InferenceEngine] Backend created: %s", backendType.c_str());
    } catch (const std::exception& e) {
        CLLM_ERROR("[InferenceEngine] Failed to create backend %s: %s", backendType.c_str(), e.what());
        throw;
    }
}

bool InferenceEngine::initialize() {
    if (!backend_) {
        CLLM_ERROR("[InferenceEngine] Error: Backend not created");
        return false;
    }
    
    CLLM_INFO("[InferenceEngine] Initializing backend...");
    
    if (!backend_->initialize()) {
        CLLM_ERROR("[InferenceEngine] Error: Backend initialization failed");
        return false;
    }
    
    // 同步backend更新后的config(例如vocab_size自动检测)
    config_ = backend_->getConfig();
    
    CLLM_INFO("[InferenceEngine] Backend initialized successfully");
    CLLM_INFO("[InferenceEngine] Backend type: %s", backend_->getName().c_str());
    CLLM_INFO("[InferenceEngine] Config vocab_size: %u", config_.vocabSize);
    return true;
}

Tensor InferenceEngine::forward(const std::vector<int> &inputIds) const {
    if (!backend_) {
        throw std::runtime_error("InferenceEngine::forward: backend not created");
    }
    
    if (!backend_->isInitialized()) {
        throw std::runtime_error("InferenceEngine::forward: backend not initialized");
    }
    
    return backend_->forward(inputIds);
}

Tensor InferenceEngine::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize,
    const std::vector<size_t> &sequenceIds
) const {
    if (!backend_) {
        throw std::runtime_error("InferenceEngine::forwardBatch: backend not created");
    }
    
    if (!backend_->isInitialized()) {
        throw std::runtime_error("InferenceEngine::forwardBatch: backend not initialized");
    }
    
    return backend_->forwardBatch(flatInputIds, requestPositions, batchSize, sequenceIds);
}

const ModelConfig &InferenceEngine::getConfig() const {
    if (!backend_) {
        throw std::runtime_error("InferenceEngine::getConfig: backend not created");
    }
    
    return backend_->getConfig();
}

std::string InferenceEngine::getBackendType() const {
    if (!backend_) {
        return "None";
    }
    
    return backend_->getName();
}

bool InferenceEngine::isInitialized() const {
    if (!backend_) {
        return false;
    }
    
    return backend_->isInitialized();
}

bool InferenceEngine::releaseSequenceId(size_t requestId) const {
    if (!backend_) {
        return false;
    }
    
    // Phase 2: 只对 LlamaCppBackend 有效
    std::string backendName = backend_->getName();
    if (backendName == "llama.cpp") {
        #ifdef CLLM_USE_LLAMA_CPP
        // 动态转换为 LlamaCppBackend 并调用 releaseSequenceId
        auto* llamaBackend = dynamic_cast<LlamaCppBackend*>(backend_.get());
        if (llamaBackend) {
            return llamaBackend->releaseSequenceId(requestId);
        }
        #endif
    }
    
    // 其他后端不需要序列ID管理
    return false;
}

bool InferenceEngine::cleanupKVCache(size_t requestId) const {
    if (!backend_) {
        return false;
    }
    
    // Phase 4: 只对 LlamaCppBackend 有效
    std::string backendName = backend_->getName();
    if (backendName == "llama.cpp") {
        #ifdef CLLM_USE_LLAMA_CPP
        // 动态转换为 LlamaCppBackend 并调用 cleanupKVCache
        auto* llamaBackend = dynamic_cast<LlamaCppBackend*>(backend_.get());
        if (llamaBackend) {
            return llamaBackend->cleanupKVCache(requestId);
        }
        #endif
    }
    
    // 其他后端不需要KV缓存清理
    return false;
}

bool InferenceEngine::updateKVCacheRequestStatus(size_t requestId, RequestStatus status) const {
    if (!backend_) {
        return false;
    }

    std::string backendName = backend_->getName();
    if (backendName == "llama.cpp") {
        #ifdef CLLM_USE_LLAMA_CPP
        auto* llamaBackend = dynamic_cast<LlamaCppBackend*>(backend_.get());
        if (llamaBackend) {
            auto* kvCacheManager = llamaBackend->getKVCacheManager();
            if (kvCacheManager) {
                kvCacheManager->updateRequestStatus(requestId, status);
                return true;
            }
        }
        #endif
    }

    return false;
}

size_t InferenceEngine::evictKVCachesIfNeeded(double evictionThreshold) const {
    if (!backend_) {
        return 0;
    }

    std::string backendName = backend_->getName();
    if (backendName == "llama.cpp") {
        #ifdef CLLM_USE_LLAMA_CPP
        auto* llamaBackend = dynamic_cast<LlamaCppBackend*>(backend_.get());
        if (llamaBackend) {
            auto* kvCacheManager = llamaBackend->getKVCacheManager();
            if (!kvCacheManager) {
                return 0;
            }
            if (!kvCacheManager->shouldEvict(evictionThreshold)) {
                return 0;
            }
            auto getSeqIdCallback = [llamaBackend](size_t requestId) -> int32_t {
                return llamaBackend->getSequenceId(requestId);
            };
            return kvCacheManager->evictLRUCache(
                llamaBackend->getContext(),
                evictionThreshold,
                getSeqIdCallback
            );
        }
        #endif
    }

    return 0;
}

} // namespace inference
} // namespace cllm
