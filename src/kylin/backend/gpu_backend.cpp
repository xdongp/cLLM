/**
 * @file gpu_backend.cpp
 * @brief GPU 计算后端实现 - 1/4 完整版本
 * 
 * 封装 GGMLGPUBackend，提供统一的 GPU 计算接口
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
    
    // 权重指针（用于上传）
    const float* embedTokens = nullptr;
    std::vector<LayerWeightsGPU> layerWeights;
    const float* finalNorm = nullptr;
    const float* lmHead = nullptr;
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
    
    // 初始化 GGML 后端
    if (!impl_->ggmlBackend->initialize(config)) {
        CLLM_ERROR("[GPUBackend] Failed to initialize GGML backend");
        return false;
    }
    
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
        // 注意：ModelWeights 使用 void* 指针
        // 需要 HFTransformerModel 提供具体的 float* 指针来上传权重
        // 暂时标记为已加载，实际权重上传通过 uploadWeights 方法完成
        weightsLoaded_ = true;
        CLLM_INFO("[GPUBackend] Weights placeholder loaded");
        return true;
    }
#endif
    
    return false;
}

bool GPUBackend::uploadWeights(
    const float* embedTokens,
    const std::vector<LayerWeightsGPU>& layers,
    const float* finalNorm,
    const float* lmHead
) {
    if (!impl_ || !impl_->ggmlBackend) {
        CLLM_ERROR("[GPUBackend] Not initialized");
        return false;
    }
    
#ifdef GGML_USE_METAL
    // 保存权重指针
    impl_->embedTokens = embedTokens;
    impl_->layerWeights = layers;
    impl_->finalNorm = finalNorm;
    impl_->lmHead = lmHead;
    
    // 上传到 GPU
    if (impl_->ggmlBackend->uploadWeights(embedTokens, layers, finalNorm, lmHead)) {
        weightsLoaded_ = true;
        CLLM_INFO("[GPUBackend] Weights uploaded to GPU: %zu layers", layers.size());
        return true;
    } else {
        CLLM_ERROR("[GPUBackend] Failed to upload weights to GPU");
        return false;
    }
#else
    CLLM_ERROR("[GPUBackend] GPU not compiled");
    return false;
#endif
}

std::vector<float> GPUBackend::forward(
    const std::vector<int32_t>& inputIds,
    int requestId
) {
    if (!initialized_ || !impl_) {
        CLLM_ERROR("[GPUBackend] Not initialized");
        return {};
    }
    
    if (!weightsLoaded_) {
        CLLM_ERROR("[GPUBackend] Weights not loaded");
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
            auto logits = impl_->ggmlBackend->forward(lastToken, startPos);
            
            if (!logits.empty()) {
                impl_->kvCacheLengths[requestId] = startPos + 1;
                return logits;
            } else {
                CLLM_WARN("[GPUBackend] Forward returned empty logits");
                return {};
            }
        }
    }
#endif
    
    CLLM_WARN("[GPUBackend] forward() not available");
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
    
    if (!weightsLoaded_) {
        CLLM_ERROR("[GPUBackend] Weights not loaded");
        return {};
    }
    
#ifdef GGML_USE_METAL
    if (impl_->ggmlBackend) {
        // 准备 token IDs 和 positions
        std::vector<int> tokenIds;
        std::vector<int> positions;
        
        for (size_t i = 0; i < batchInputIds.size(); ++i) {
            if (!batchInputIds[i].empty()) {
                int lastToken = batchInputIds[i].back();
                tokenIds.push_back(lastToken);
                
                int startPos = 0;
                auto it = impl_->kvCacheLengths.find(requestIds[i]);
                if (it != impl_->kvCacheLengths.end()) {
                    startPos = it->second;
                }
                positions.push_back(startPos);
            }
        }
        
        // 调用批量 forward
        auto results = impl_->ggmlBackend->forwardBatch(tokenIds, positions);
        
        // 更新 KV Cache 长度
        for (size_t i = 0; i < requestIds.size() && i < results.size(); ++i) {
            int startPos = positions[i];
            impl_->kvCacheLengths[requestIds[i]] = startPos + 1;
        }
        
        return results;
    }
#endif
    
    CLLM_WARN("[GPUBackend] forwardBatch() not available");
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
        // GGMLGPUBackend 的 KV Cache 管理是内部的
        // 这里只需要重置我们的跟踪状态
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
