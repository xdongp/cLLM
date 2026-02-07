/**
 * @file gpu_backend.cpp
 * @brief GPU 计算后端实现
 */

#include "cllm/kylin/backend/gpu_backend.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/common/logger.h"

#include <unordered_map>

namespace cllm {
namespace kylin {

// 内部实现结构
struct GPUBackendImpl {
    // 模型配置
    HFModelConfig config;
    
    // GGML GPU 后端
    std::unique_ptr<GGMLGPUBackend> ggmlBackend;
    
    // KV Cache 管理（请求 ID -> 序列长度）
    std::unordered_map<int, int> kvCacheLengths;
    
    // 初始化标志
    bool initialized = false;
};

GPUBackend::GPUBackend() = default;

GPUBackend::~GPUBackend() {
    shutdown();
}

bool GPUBackend::initialize(const HFModelConfig& config) {
    config_ = config;
    
    // 分配实现
    impl_ = std::make_unique<GPUBackendImpl>();
    impl_->config = config;
    
#ifdef GGML_USE_METAL
    // 初始化 GGML GPU 后端
    impl_->ggmlBackend = std::make_unique<GGMLGPUBackend>();
    if (!impl_->ggmlBackend) {
        CLLM_ERROR("[GPUBackend] Failed to create GGML GPU backend");
        return false;
    }
    
    // TODO: 初始化 GGML 后端（需要权重）
    
    impl_->initialized = true;
    initialized_ = true;
    CLLM_INFO("[GPUBackend] Initialized GGML GPU backend");
    return true;
#else
    CLLM_ERROR("[GPUBackend] GPU not compiled, cannot initialize");
    impl_->initialized = false;
    initialized_ = false;
    return false;
#endif
}

void GPUBackend::shutdown() {
    if (impl_) {
        impl_->ggmlBackend.reset();
        impl_->initialized = false;
    }
    initialized_ = false;
    weightsLoaded_ = false;
    CLLM_INFO("[GPUBackend] Shutdown");
}

bool GPUBackend::loadWeights(const ModelWeights& weights) {
    weights_ = weights;
    
    if (!impl_) {
        CLLM_ERROR("[GPUBackend] Not initialized");
        return false;
    }
    
#ifdef GGML_USE_METAL
    if (impl_->ggmlBackend) {
        // TODO: 加载权重到 GPU
        // 需要转换 ModelWeights 到 GGMLGPUBackend 需要的格式
        weightsLoaded_ = true;
        CLLM_INFO("[GPUBackend] Weights loaded to GPU");
        return true;
    }
#endif
    
    return false;
}

std::vector<float> GPUBackend::forward(
    const std::vector<int32_t>& inputIds,
    int requestId
) {
    if (!initialized_ || !impl_) {
        CLLM_ERROR("[GPUBackend] Not initialized");
        return {};
    }
    
#ifdef GGML_USE_METAL
    if (impl_->ggmlBackend) {
        // 获取当前序列长度
        int startPos = 0;
        auto it = impl_->kvCacheLengths.find(requestId);
        if (it != impl_->kvCacheLengths.end()) {
            startPos = it->second;
        }
        
        // 使用最后一个 token 进行生成
        if (!inputIds.empty()) {
            int lastToken = inputIds.back();
            // TODO: 调用 GGMLGPUBackend 的 forward
            // auto logits = impl_->ggmlBackend->forward(lastToken, startPos);
            // impl_->kvCacheLengths[requestId] = startPos + 1;
            // return logits;
        }
    }
#endif
    
    CLLM_WARN("[GPUBackend] forward() not fully implemented yet");
    return {};
}

std::vector<std::vector<float>> GPUBackend::forwardBatch(
    const std::vector<std::vector<int32_t>>& batchInputIds,
    const std::vector<int>& requestIds
) {
    if (!initialized_ || !impl_) {
        CLLM_ERROR("[GPUBackend] Not initialized");
        return {};
    }
    
    // TODO: 实现 GPU 批量前向推理
    CLLM_WARN("[GPUBackend] forwardBatch() not implemented yet");
    return {};
}

void GPUBackend::resetKVCache(int requestId) {
    if (!impl_) return;
    
    auto it = impl_->kvCacheLengths.find(requestId);
    if (it != impl_->kvCacheLengths.end()) {
        it->second = 0;
    }
    
#ifdef GGML_USE_METAL
    if (impl_->ggmlBackend) {
        // TODO: 重置 GGML 后端的 KV Cache
    }
#endif
}

void GPUBackend::releaseKVCache(int requestId) {
    if (!impl_) return;
    impl_->kvCacheLengths.erase(requestId);
}

int GPUBackend::getKVCacheCurrentLength(int requestId) const {
    if (!impl_) return 0;
    
    auto it = impl_->kvCacheLengths.find(requestId);
    if (it != impl_->kvCacheLengths.end()) {
        return it->second;
    }
    return 0;
}

} // namespace kylin
} // namespace cllm
