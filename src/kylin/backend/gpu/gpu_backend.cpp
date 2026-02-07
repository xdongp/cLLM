/**
 * @file gpu_backend.cpp
 * @brief GPU 后端实现 - 包装 GGMLGPUBackend
 * 
 * 这个文件作为 backend/gpu 目录的入口点，
 * 实际实现委托给 hf/ggml_backend.cpp 中的 GGMLGPUBackend。
 */

#include "cllm/kylin/backend/gpu/gpu_backend.h"
#include "cllm/kylin/hf/ggml_backend.h"
#include "cllm/common/logger.h"

namespace cllm {
namespace kylin {
namespace backend {

// ============================================================================
// 构造函数和析构函数
// ============================================================================

GPUBackend::GPUBackend() : impl_(std::make_unique<GGMLGPUBackend>()) {}

GPUBackend::~GPUBackend() = default;

// ============================================================================
// 初始化
// ============================================================================

bool GPUBackend::initialize(const HFModelConfig& config) {
    return impl_->initialize(config);
}

// ============================================================================
// 权重管理
// ============================================================================

bool GPUBackend::uploadWeights(
    const float* embedTokens,
    const std::vector<LayerWeightsGPU>& layers,
    const float* finalNorm,
    const float* lmHead
) {
    // 转换 LayerWeightsGPU 到 hf::LayerWeightsGPU
    std::vector<struct ::cllm::kylin::LayerWeightsGPU> hfLayers;
    hfLayers.reserve(layers.size());
    
    for (const auto& lw : layers) {
        struct ::cllm::kylin::LayerWeightsGPU hfLw;
        hfLw.inputLayernorm = lw.inputLayernorm;
        hfLw.qProj = lw.qProj;
        hfLw.kProj = lw.kProj;
        hfLw.vProj = lw.vProj;
        hfLw.oProj = lw.oProj;
        hfLw.qNorm = lw.qNorm;
        hfLw.kNorm = lw.kNorm;
        hfLw.postAttentionLayernorm = lw.postAttentionLayernorm;
        hfLw.gateProj = lw.gateProj;
        hfLw.upProj = lw.upProj;
        hfLw.downProj = lw.downProj;
        hfLayers.push_back(hfLw);
    }
    
    return impl_->uploadWeights(embedTokens, hfLayers, finalNorm, lmHead);
}

// ============================================================================
// Forward 实现
// ============================================================================

std::vector<float> GPUBackend::forward(int tokenId, int position) {
    return impl_->forward(tokenId, position);
}

std::vector<float> GPUBackend::forwardGraph(int tokenId, int position) {
    return impl_->forwardGraphMinimal(tokenId, position);
}

std::vector<float> GPUBackend::forwardCPU(int tokenId, int position) {
    return impl_->forwardCPU(tokenId, position);
}

// ============================================================================
// 批量处理和辅助功能
// ============================================================================

std::vector<std::vector<float>> GPUBackend::forwardBatch(
    const std::vector<int>& tokenIds,
    const std::vector<int>& positions,
    const std::vector<size_t>& requestIds
) {
    (void)requestIds;
    return impl_->forwardBatch(tokenIds, positions);
}

void GPUBackend::resetKVCache() {
    impl_->resetKVCache();
}

bool GPUBackend::buildGraph() {
    // 预构建计算图以优化性能
    // 实际构建在 forwardGraphMinimal 中按需进行
    return true;
}

bool GPUBackend::isGPUSupported() const {
#ifdef GGML_USE_METAL
    return true;
#else
    return false;
#endif
}

} // namespace backend
} // namespace kylin
} // namespace cllm
