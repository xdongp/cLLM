/**
 * @file inference_engine.cpp
 * @brief 推理引擎统一接口实现
 */

#include "cllm/inference/inference_engine.h"
#include "cllm/inference/kylin_backend.h"
#include "cllm/inference/libtorch_backend.h"
#include "cllm/common/logger.h"
#include <stdexcept>

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
    CLLM_INFO("[InferenceEngine] Backend: %s", (useLibTorch_ ? "LibTorch" : "Kylin (麒麟)"));
    CLLM_INFO("[InferenceEngine] Model path: %s", (modelPath.empty() ? "(placeholder weights)" : modelPath.c_str()));
    
    // 创建后端实例
    if (useLibTorch_) {
        // LibTorch 后端
        backend_ = std::make_unique<LibTorchBackend>(modelPath, config_);
    } else {
        // Kylin 后端
        backend_ = std::make_unique<KylinBackend>(config_, modelPath);
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
    size_t batchSize
) const {
    if (!backend_) {
        throw std::runtime_error("InferenceEngine::forwardBatch: backend not created");
    }
    
    if (!backend_->isInitialized()) {
        throw std::runtime_error("InferenceEngine::forwardBatch: backend not initialized");
    }
    
    return backend_->forwardBatch(flatInputIds, requestPositions, batchSize);
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

} // namespace inference
} // namespace cllm
