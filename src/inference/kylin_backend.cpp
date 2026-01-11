/**
 * @file kylin_backend.cpp
 * @brief Kylin (麒麟) 自研推理后端实现
 */

#include "cllm/inference/kylin_backend.h"
#include "cllm/common/logger.h"

#include <stdexcept>

namespace cllm {
namespace inference {

namespace {

// 为权重张量填充简单的可重复模式，避免全零
inline void fill_tensor_with_pattern(kylin::Tensor &tensor, float scale) {
    const size_t n = tensor.size();
    for (size_t i = 0; i < n; ++i) {
        // 使用一个简单位移后的周期模式，保证数值稳定且非零
        float v = static_cast<float>((i % 31) - 15);
        tensor[i] = v * scale;
    }
}

} // namespace

KylinBackend::KylinBackend(const ModelConfig &config, const std::string &modelPath)
    : externalConfig_(config)
    , internalConfig_(config)
    , initialized_(false)
    , modelPath_(modelPath)
    , model_(config) {
    
    CLLM_INFO("[KylinBackend] Initializing Kylin (麒麟) inference backend");
    
    if (!modelPath_.empty()) {
        // 真实权重模式
        CLLM_INFO("[KylinBackend] Will load real weights from: {}", modelPath_);
        loader_ = std::make_unique<kylin::ModelLoader>(modelPath_, externalConfig_);
    } else {
        // 占位权重模式
        CLLM_INFO("[KylinBackend] Will use placeholder weights (test mode)");
        prepareInternalConfig();
    }

    // 预分配权重容器
    const size_t numLayers = internalConfig_.numLayers;
    wq_.resize(numLayers);
    wk_.resize(numLayers);
    wv_.resize(numLayers);
    wo_.resize(numLayers);
    wGate_.resize(numLayers);
    wUp_.resize(numLayers);
    wDown_.resize(numLayers);
    norm1_.resize(numLayers);
    norm2_.resize(numLayers);
}

void KylinBackend::prepareInternalConfig() {
    // 为避免一次性分配过大的矩阵，使用精简配置
    // 但保持 vocabSize 与外部配置一致，确保 logits 维度正确
    internalConfig_.hiddenSize = 128;
    internalConfig_.intermediateSize = 256;
    internalConfig_.numLayers = 2;
    internalConfig_.numAttentionHeads = 4; // headDim = 32
    
    CLLM_INFO("[KylinBackend] Using simplified config:");
    CLLM_INFO("  hiddenSize: {}", internalConfig_.hiddenSize);
    CLLM_INFO("  intermediateSize: {}", internalConfig_.intermediateSize);
    CLLM_INFO("  numLayers: {}", internalConfig_.numLayers);
    CLLM_INFO("  vocabSize: {} (from external)", internalConfig_.vocabSize);
}

bool KylinBackend::initialize() {
    if (initialized_) {
        CLLM_INFO("[KylinBackend] Already initialized");
        return true;
    }

    CLLM_INFO("[KylinBackend] Starting initialization...");

    // 1. 验证配置
    const size_t vocab = externalConfig_.vocabSize;
    const size_t hidden = internalConfig_.hiddenSize;
    const size_t inter = internalConfig_.intermediateSize;
    const size_t numLayers = internalConfig_.numLayers;

    if (vocab == 0 || hidden == 0 || inter == 0 || numLayers == 0) {
        throw std::runtime_error("KylinBackend::initialize: invalid model configuration");
    }

    // 2. 加载权重
    if (loader_) {
        if (!loadRealWeights()) {
            return false;
        }
    } else {
        // 分配占位权重
        CLLM_INFO("[KylinBackend] Allocating placeholder weights...");
        
        embedding_ = Tensor({vocab, hidden});
        lmHead_ = Tensor({hidden, vocab});
        finalNormWeight_ = Tensor({hidden});

        for (size_t layer = 0; layer < numLayers; ++layer) {
            wq_[layer] = Tensor({hidden, hidden});
            wk_[layer] = Tensor({hidden, hidden});
            wv_[layer] = Tensor({hidden, hidden});
            wo_[layer] = Tensor({hidden, hidden});

            wGate_[layer] = Tensor({hidden, inter});
            wUp_[layer] = Tensor({hidden, inter});
            wDown_[layer] = Tensor({inter, hidden});

            norm1_[layer] = Tensor({hidden});
            norm2_[layer] = Tensor({hidden});
        }

        initializePlaceholderWeights();
    }

    // 3. 绑定权重到模型
    bindWeightsToModel();

    initialized_ = true;
    CLLM_INFO("[KylinBackend] Initialization completed successfully");
    return true;
}

bool KylinBackend::loadRealWeights() {
    CLLM_INFO("[KylinBackend] Loading real weights...");

    // 1) 使用 ModelLoader 加载（会读取 <model>.json 元数据）
    if (!loader_->load()) {
        CLLM_ERROR("[KylinBackend] Failed to load model weights via ModelLoader");
        return false;
    }

    // 2) 用元数据中的真实结构参数覆盖外部配置（避免默认 llama 配置导致 shape mismatch）
    externalConfig_ = loader_->getConfig();
    internalConfig_ = externalConfig_;

    const size_t numLayers = internalConfig_.numLayers;

    // 确保权重容器大小正确
    wq_.resize(numLayers);
    wk_.resize(numLayers);
    wv_.resize(numLayers);
    wo_.resize(numLayers);
    wGate_.resize(numLayers);
    wUp_.resize(numLayers);
    wDown_.resize(numLayers);
    norm1_.resize(numLayers);
    norm2_.resize(numLayers);

    if (!loader_->loadInto(
            embedding_,
            wq_, wk_, wv_, wo_,
            wGate_, wUp_, wDown_,
            norm1_, norm2_,
            finalNormWeight_,
            lmHead_)) {
        CLLM_ERROR("[KylinBackend] Failed to map weights from ModelLoader");
        return false;
    }

    CLLM_INFO("[KylinBackend] Real weights loaded successfully");
    return true;
}

void KylinBackend::initializePlaceholderWeights() {
    CLLM_INFO("[KylinBackend] Initializing placeholder weights...");

    const float baseScale = 0.01f;
    const size_t numLayers = internalConfig_.numLayers;

    // Embedding 和 LM Head
    fill_tensor_with_pattern(embedding_, baseScale);
    fill_tensor_with_pattern(lmHead_, baseScale);
    fill_tensor_with_pattern(finalNormWeight_, 1.0f);

    // 每层权重
    for (size_t layer = 0; layer < numLayers; ++layer) {
        fill_tensor_with_pattern(wq_[layer], baseScale);
        fill_tensor_with_pattern(wk_[layer], baseScale);
        fill_tensor_with_pattern(wv_[layer], baseScale);
        fill_tensor_with_pattern(wo_[layer], baseScale);

        fill_tensor_with_pattern(wGate_[layer], baseScale);
        fill_tensor_with_pattern(wUp_[layer], baseScale);
        fill_tensor_with_pattern(wDown_[layer], baseScale);

        // RMSNorm 权重初始化为 1
        norm1_[layer].fill(1.0f);
        norm2_[layer].fill(1.0f);
    }

    CLLM_INFO("[KylinBackend] Placeholder weights initialized");
}

void KylinBackend::bindWeightsToModel() {
    CLLM_INFO("[KylinBackend] Binding weights to TransformerModel...");

    // 重建模型（使用当前配置）
    model_ = kylin::TransformerModel(internalConfig_);

    // 绑定 Embedding 和 LM Head
    model_.setEmbeddingWeight(embedding_);
    model_.setLmHeadWeight(lmHead_);

    // 绑定每层权重
    const size_t numLayers = internalConfig_.numLayers;
    for (size_t layer = 0; layer < numLayers; ++layer) {
        model_.setBlockWeights(
            layer,
            wq_[layer],
            wk_[layer],
            wv_[layer],
            wo_[layer],
            wGate_[layer],
            wUp_[layer],
            wDown_[layer],
            norm1_[layer],
            norm2_[layer]
        );
    }

    // 绑定 Final Norm
    model_.setFinalNormWeight(finalNormWeight_);

    CLLM_INFO("[KylinBackend] Weights bound to model successfully");
}

kylin::Tensor KylinBackend::forward(const std::vector<int> &inputIds) {
    if (!initialized_) {
        throw std::runtime_error("KylinBackend::forward: backend not initialized");
    }

    return model_.forward(inputIds);
}

kylin::Tensor KylinBackend::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize
) {
    if (!initialized_) {
        throw std::runtime_error("KylinBackend::forwardBatch: backend not initialized");
    }

    if (batchSize == 0) {
        throw std::invalid_argument("KylinBackend::forwardBatch: batchSize == 0");
    }
    if (requestPositions.size() != batchSize) {
        throw std::invalid_argument("KylinBackend::forwardBatch: requestPositions size mismatch");
    }

    const size_t totalTokens = flatInputIds.size();
    const size_t vocab = externalConfig_.vocabSize;

    if (totalTokens == 0) {
        throw std::invalid_argument("KylinBackend::forwardBatch: empty flatInputIds");
    }

    // 分配输出张量
    kylin::Tensor logits({totalTokens, vocab});

    // 逐请求调用 forward，并拼接结果
    for (size_t i = 0; i < batchSize; ++i) {
        const auto &pos = requestPositions[i];
        const size_t start = pos.first;
        const size_t end = pos.second;

        if (start > end || end > totalTokens) {
            throw std::out_of_range("KylinBackend::forwardBatch: invalid requestPositions range");
        }

        if (start == end) {
            continue; // 空请求，跳过
        }

        // 提取当前请求的输入
        std::vector<int> inputIds(
            flatInputIds.begin() + static_cast<std::ptrdiff_t>(start),
            flatInputIds.begin() + static_cast<std::ptrdiff_t>(end)
        );

        // 推理
        kylin::Tensor requestLogits = forward(inputIds); // [len, vocab]

        const size_t len = end - start;
        if (requestLogits.shape().size() != 2 ||
            requestLogits.shape()[0] != len ||
            requestLogits.shape()[1] != vocab) {
            throw std::runtime_error("KylinBackend::forwardBatch: request logits shape mismatch");
        }

        // 拷贝到输出张量
        const float *src = requestLogits.data();
        float *dst = logits.data();

        for (size_t t = 0; t < len; ++t) {
            size_t globalRow = start + t;
            size_t srcOffset = t * vocab;
            size_t dstOffset = globalRow * vocab;
            for (size_t v = 0; v < vocab; ++v) {
                dst[dstOffset + v] = src[srcOffset + v];
            }
        }
    }

    return logits;
}

} // namespace inference
} // namespace cllm
