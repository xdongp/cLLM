/**
 * @file gpu_backend.cpp
 * @brief GPU 后端实现
 * 
 * 注意：实际 GPU 实现在 hf/ggml_backend.cpp 中。
 * 这个文件作为 backend/gpu 目录的占位符，
 * 未来可以逐步迁移功能到这里。
 */

#include "cllm/kylin/backend/gpu/gpu_backend.h"
#include "cllm/common/logger.h"

namespace cllm {
namespace kylin {
namespace backend {

GPUBackend::GPUBackend() = default;

GPUBackend::~GPUBackend() {
    cleanup();
}

void GPUBackend::cleanup() {
    // 清理资源
    CLLM_INFO("[GPUBackend] Cleanup (actual implementation in hf/ggml_backend.cpp)");
}

bool GPUBackend::initialize(const HFModelConfig& config) {
    CLLM_INFO("[GPUBackend] Initialize (actual implementation in hf/ggml_backend.cpp)");
    // 实际实现使用 hf/ggml_backend.cpp 中的 GGMLGPUBackend
    (void)config;
    return true;
}

bool GPUBackend::uploadWeights(
    const float* embedTokens,
    const std::vector<LayerWeightsGPU>& layers,
    const float* finalNorm,
    const float* lmHead
) {
    (void)embedTokens;
    (void)layers;
    (void)finalNorm;
    (void)lmHead;
    // 实际实现使用 hf/ggml_backend.cpp 中的 GGMLGPUBackend
    return true;
}

std::vector<float> GPUBackend::forward(int tokenId, int position) {
    (void)tokenId;
    (void)position;
    // 实际实现使用 hf/ggml_backend.cpp 中的 GGMLGPUBackend
    return {};
}

std::vector<float> GPUBackend::forwardGraph(int tokenId, int position) {
    return forward(tokenId, position);
}

bool GPUBackend::buildGraph() {
    return true;
}

void GPUBackend::resetKVCache() {
    // 实际实现使用 hf/ggml_backend.cpp 中的 GGMLGPUBackend
}

// isInitialized() 在头文件中内联定义

bool GPUBackend::isGPUSupported() const {
#ifdef GGML_USE_METAL
    return true;
#else
    return false;
#endif
}

std::vector<std::vector<float>> GPUBackend::forwardBatch(
    const std::vector<int>& tokenIds,
    const std::vector<int>& positions,
    const std::vector<size_t>& requestIds
) {
    (void)tokenIds;
    (void)positions;
    (void)requestIds;
    // 实际实现使用 hf/ggml_backend.cpp 中的 GGMLGPUBackend
    return {};
}

} // namespace backend
} // namespace kylin
} // namespace cllm
