/**
 * @file cpu_backend.cpp
 * @brief CPU 计算后端实现
 */

#include "cllm/kylin/backend/cpu_backend.h"
#include "cllm/common/logger.h"

namespace cllm {
namespace kylin {

CPUBackend::CPUBackend() = default;

CPUBackend::~CPUBackend() {
    shutdown();
}

bool CPUBackend::initialize(const HFModelConfig& config) {
    config_ = config;
    initialized_ = true;
    CLLM_INFO("[CPUBackend] Initialized with hidden_size=%d, num_layers=%d",
              config.hiddenSize, config.numHiddenLayers);
    return true;
}

void CPUBackend::shutdown() {
    initialized_ = false;
    weightsLoaded_ = false;
    CLLM_INFO("[CPUBackend] Shutdown");
}

bool CPUBackend::loadWeights(const ModelWeights& weights) {
    weights_ = weights;
    weightsLoaded_ = true;
    CLLM_INFO("[CPUBackend] Weights loaded");
    return true;
}

std::vector<float> CPUBackend::forward(
    const std::vector<int32_t>& inputIds,
    int requestId
) {
    // TODO: 实现 CPU 前向推理
    CLLM_WARN("[CPUBackend] forward() not implemented yet");
    return {};
}

std::vector<std::vector<float>> CPUBackend::forwardBatch(
    const std::vector<std::vector<int32_t>>& batchInputIds,
    const std::vector<int>& requestIds
) {
    // TODO: 实现 CPU 批量前向推理
    CLLM_WARN("[CPUBackend] forwardBatch() not implemented yet");
    return {};
}

void CPUBackend::resetKVCache(int requestId) {
    // TODO: 实现 KV Cache 重置
}

void CPUBackend::releaseKVCache(int requestId) {
    // TODO: 实现 KV Cache 释放
}

int CPUBackend::getKVCacheCurrentLength(int requestId) const {
    // TODO: 实现获取 KV Cache 长度
    return 0;
}

} // namespace kylin
} // namespace cllm
