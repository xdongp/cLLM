/**
 * @file kylin_backend.cpp
 * @brief Kylin (麒麟) 自研推理后端实现
 */

#include "cllm/inference/kylin_backend.h"
#include "cllm/model/loader_interface.h"
#include "cllm/common/logger.h"

#include <stdexcept>
#include <cmath>

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

// Xavier/Glorot 初始化：适用于线性层权重
// 对于形状为 [fan_in, fan_out] 的权重，使用 scale = sqrt(2.0 / (fan_in + fan_out))
inline void xavier_init(kylin::Tensor &tensor, size_t fan_in, size_t fan_out) {
    const size_t n = tensor.size();
    if (n == 0) return;
    
    // 使用简化的Xavier初始化：scale = sqrt(2.0 / (fan_in + fan_out))
    float scale = std::sqrt(2.0f / static_cast<float>(fan_in + fan_out));
    
    // 使用正态分布的近似：Box-Muller变换的简化版本
    // 为了简单，使用均匀分布乘以scale，然后添加小的随机偏移
    for (size_t i = 0; i < n; ++i) {
        // 使用周期模式模拟均匀分布 [-1, 1]
        float u = static_cast<float>((i % 100) - 50) / 50.0f;
        tensor[i] = u * scale;
    }
}

// 小值初始化：适用于embedding和某些特殊层
inline void small_value_init(kylin::Tensor &tensor, float scale = 0.1f) {
    const size_t n = tensor.size();
    for (size_t i = 0; i < n; ++i) {
        // 使用小的均匀分布值
        float v = static_cast<float>((i % 21) - 10) / 100.0f * scale;
        tensor[i] = v;
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
        // 真实权重模式 - 使用 ModelLoaderFactory 自动检测格式
        CLLM_INFO("[KylinBackend] Will load real weights from: %s", modelPath_.c_str());
        try {
            loader_ = ModelLoaderFactory::createLoader(modelPath_, externalConfig_);
            CLLM_INFO("[KylinBackend] Created loader for format: %s", 
                     ModelLoaderFactory::formatToString(ModelLoaderFactory::detectFormat(modelPath_)).c_str());
        } catch (const std::exception& e) {
            CLLM_ERROR("[KylinBackend] Failed to create model loader: %s", e.what());
            throw;
        }
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
    CLLM_INFO("  hiddenSize: %u", internalConfig_.hiddenSize);
    CLLM_INFO("  intermediateSize: %u", internalConfig_.intermediateSize);
    CLLM_INFO("  numLayers: %u", internalConfig_.numLayers);
    CLLM_INFO("  vocabSize: %u (from external)", internalConfig_.vocabSize);
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
    CLLM_INFO("[KylinBackend] About to call bindWeightsToModel()...");
    try {
    bindWeightsToModel();
    } catch (const std::exception& e) {
        CLLM_ERROR("[KylinBackend] Exception in bindWeightsToModel(): %s", e.what());
        throw;
    } catch (...) {
        CLLM_ERROR("[KylinBackend] Unknown exception in bindWeightsToModel()");
        throw;
    }
    CLLM_INFO("[KylinBackend] bindWeightsToModel() completed successfully");

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

    CLLM_INFO("[KylinBackend] Calling loadInto to map weights...");
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
    CLLM_INFO("[KylinBackend] Weight containers: wq_.size()=%zu, embedding_.shape().size()=%zu",
             wq_.size(), embedding_.shape().size());
    return true;
}

void KylinBackend::initializePlaceholderWeights() {
    CLLM_INFO("[KylinBackend] Initializing placeholder weights...");

    const size_t numLayers = internalConfig_.numLayers;
    const size_t hidden = internalConfig_.hiddenSize;
    const size_t inter = internalConfig_.intermediateSize;
    const size_t vocab = internalConfig_.vocabSize;

    // Embedding: 使用小值初始化
    // embedding_: [vocab, hidden]
    small_value_init(embedding_, 0.1f);
    
    // LM Head: 使用Xavier初始化
    // lmHead_: [hidden, vocab]
    xavier_init(lmHead_, hidden, vocab);
    
    // Final norm: 初始化为1
    finalNormWeight_.fill(1.0f);

    // 每层权重：使用Xavier初始化以获得更好的数值稳定性
    for (size_t layer = 0; layer < numLayers; ++layer) {
        // Attention权重: [hidden, hidden] 或 [hidden, qDim/kvDim]
        const auto& wqShape = wq_[layer].shape();
        const auto& wkShape = wk_[layer].shape();
        const auto& wvShape = wv_[layer].shape();
        const auto& woShape = wo_[layer].shape();
        
        if (wqShape.size() == 2) {
            xavier_init(wq_[layer], wqShape[0], wqShape[1]);
        } else {
            fill_tensor_with_pattern(wq_[layer], 0.1f);
        }
        
        if (wkShape.size() == 2) {
            xavier_init(wk_[layer], wkShape[0], wkShape[1]);
        } else {
            fill_tensor_with_pattern(wk_[layer], 0.1f);
        }
        
        if (wvShape.size() == 2) {
            xavier_init(wv_[layer], wvShape[0], wvShape[1]);
        } else {
            fill_tensor_with_pattern(wv_[layer], 0.1f);
        }
        
        if (woShape.size() == 2) {
            xavier_init(wo_[layer], woShape[0], woShape[1]);
        } else {
            fill_tensor_with_pattern(wo_[layer], 0.1f);
        }

        // FFN权重: [hidden, inter] 或 [inter, hidden]
        const auto& wGateShape = wGate_[layer].shape();
        const auto& wUpShape = wUp_[layer].shape();
        const auto& wDownShape = wDown_[layer].shape();
        
        if (wGateShape.size() == 2) {
            xavier_init(wGate_[layer], wGateShape[0], wGateShape[1]);
        } else {
            fill_tensor_with_pattern(wGate_[layer], 0.1f);
        }
        
        if (wUpShape.size() == 2) {
            xavier_init(wUp_[layer], wUpShape[0], wUpShape[1]);
        } else {
            fill_tensor_with_pattern(wUp_[layer], 0.1f);
        }
        
        if (wDownShape.size() == 2) {
            xavier_init(wDown_[layer], wDownShape[0], wDownShape[1]);
        } else {
            fill_tensor_with_pattern(wDown_[layer], 0.1f);
        }

        // RMSNorm 权重初始化为 1
        norm1_[layer].fill(1.0f);
        norm2_[layer].fill(1.0f);
    }

    CLLM_INFO("[KylinBackend] Placeholder weights initialized");
}

void KylinBackend::bindWeightsToModel() {
    CLLM_INFO("[KylinBackend] ===== bindWeightsToModel() START =====");
    CLLM_INFO("[KylinBackend] Binding weights to TransformerModel...");

    // 验证权重形状
    CLLM_INFO("[KylinBackend] Getting config values...");
    const size_t hidden = internalConfig_.hiddenSize;
    const size_t vocab = internalConfig_.vocabSize;
    const size_t numLayers = internalConfig_.numLayers;
    const size_t numHeads = internalConfig_.numAttentionHeads;
    
    if (numHeads == 0) {
        CLLM_ERROR("[KylinBackend] numHeads is 0!");
        throw std::runtime_error("numHeads is 0");
    }
    
    // 从实际权重形状推断 qDim、kvDim 和 inter（支持 GQA 和不同的 FFN 配置）
    // 默认值：假设 Q 和 KV 维度相同
    size_t qDim = hidden;
    size_t kvDim = hidden;
    size_t inter = internalConfig_.intermediateSize;
    
    // 从第一层的权重形状推断
    if (numLayers > 0 && wq_[0].shape().size() == 2) {
        qDim = wq_[0].shape()[1];  // wq: [hidden, qDim]
        CLLM_INFO("[KylinBackend] 从wq形状推断qDim: %zu", qDim);
    }
    if (numLayers > 0 && wk_[0].shape().size() == 2) {
        kvDim = wk_[0].shape()[1];  // wk: [hidden, kvDim]
        CLLM_INFO("[KylinBackend] 从wk形状推断kvDim: %zu", kvDim);
    }
    // 从第一层的 FFN 权重形状推断 intermediateSize
    if (numLayers > 0 && wGate_[0].shape().size() == 2) {
        inter = wGate_[0].shape()[1];  // wGate: [hidden, inter]
        CLLM_INFO("[KylinBackend] 从wGate形状推断intermediateSize: %zu", inter);
        // 更新配置，以便后续使用
        internalConfig_.intermediateSize = inter;
    }
    
    const size_t headDim = hidden / numHeads;
    
    CLLM_INFO("[KylinBackend] 验证权重形状: hidden=%zu, vocab=%zu, numLayers=%zu, numHeads=%zu, headDim=%zu, qDim=%zu, kvDim=%zu, inter=%zu",
             hidden, vocab, numLayers, numHeads, headDim, qDim, kvDim, inter);
    
    if (hidden == 0 || vocab == 0 || numLayers == 0 || inter == 0) {
        CLLM_ERROR("[KylinBackend] Invalid config values!");
        throw std::runtime_error("Invalid config values");
    }
    
    // 验证embedding形状
    if (embedding_.shape().size() != 2) {
        CLLM_ERROR("[KylinBackend] Embedding shape invalid: expected 2D, got %zuD", embedding_.shape().size());
        throw std::runtime_error("Embedding shape mismatch");
    }
    if (embedding_.shape()[0] != vocab || embedding_.shape()[1] != hidden) {
        CLLM_ERROR("[KylinBackend] Embedding shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                  vocab, hidden, embedding_.shape()[0], embedding_.shape()[1]);
        throw std::runtime_error("Embedding shape mismatch");
    }
    CLLM_INFO("[KylinBackend] Embedding shape: [%zu, %zu] ✓", embedding_.shape()[0], embedding_.shape()[1]);
    
    // 验证lmHead形状
    if (lmHead_.shape().size() != 2) {
        CLLM_ERROR("[KylinBackend] LMHead shape invalid: expected 2D, got %zuD", lmHead_.shape().size());
        throw std::runtime_error("LMHead shape mismatch");
    }
    if (lmHead_.shape()[0] != hidden || lmHead_.shape()[1] != vocab) {
        CLLM_ERROR("[KylinBackend] LMHead shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                  hidden, vocab, lmHead_.shape()[0], lmHead_.shape()[1]);
        throw std::runtime_error("LMHead shape mismatch");
    }
    CLLM_INFO("[KylinBackend] LMHead shape: [%zu, %zu] ✓", lmHead_.shape()[0], lmHead_.shape()[1]);
    
    // 验证层权重形状
    for (size_t layer = 0; layer < numLayers; ++layer) {
        if (wq_[layer].shape().size() != 2) {
            CLLM_ERROR("[KylinBackend] Layer %zu wq shape invalid: expected 2D", layer);
            throw std::runtime_error("Layer weight shape mismatch");
        }
        
        // 验证wq形状: 期望 [hidden, qDim]
        if (wq_[layer].shape()[0] != hidden || wq_[layer].shape()[1] != qDim) {
            CLLM_ERROR("[KylinBackend] Layer %zu wq shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, hidden, qDim, wq_[layer].shape()[0], wq_[layer].shape()[1]);
            throw std::runtime_error("Layer wq shape mismatch");
        }
        
        // 验证wk/wv形状: 期望 [hidden, kvDim] (支持GQA，KV维度可能不同于Q)
        if (wk_[layer].shape()[0] != hidden || wk_[layer].shape()[1] != kvDim) {
            CLLM_ERROR("[KylinBackend] Layer %zu wk shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, hidden, kvDim, wk_[layer].shape()[0], wk_[layer].shape()[1]);
            throw std::runtime_error("Layer wk shape mismatch");
        }
        if (wv_[layer].shape()[0] != hidden || wv_[layer].shape()[1] != kvDim) {
            CLLM_ERROR("[KylinBackend] Layer %zu wv shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, hidden, kvDim, wv_[layer].shape()[0], wv_[layer].shape()[1]);
            throw std::runtime_error("Layer wv shape mismatch");
        }
        
        // 验证wo形状: 期望 [qDim, hidden]
        if (wo_[layer].shape()[0] != qDim || wo_[layer].shape()[1] != hidden) {
            CLLM_ERROR("[KylinBackend] Layer %zu wo shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, qDim, hidden, wo_[layer].shape()[0], wo_[layer].shape()[1]);
            throw std::runtime_error("Layer wo shape mismatch");
        }
        
        // 验证FFN权重形状
        if (wGate_[layer].shape()[0] != hidden || wGate_[layer].shape()[1] != inter) {
            CLLM_ERROR("[KylinBackend] Layer %zu wGate shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, hidden, inter, wGate_[layer].shape()[0], wGate_[layer].shape()[1]);
            throw std::runtime_error("Layer wGate shape mismatch");
        }
        if (wUp_[layer].shape()[0] != hidden || wUp_[layer].shape()[1] != inter) {
            CLLM_ERROR("[KylinBackend] Layer %zu wUp shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, hidden, inter, wUp_[layer].shape()[0], wUp_[layer].shape()[1]);
            throw std::runtime_error("Layer wUp shape mismatch");
        }
        if (wDown_[layer].shape()[0] != inter || wDown_[layer].shape()[1] != hidden) {
            CLLM_ERROR("[KylinBackend] Layer %zu wDown shape mismatch: expected [%zu, %zu], got [%zu, %zu]",
                      layer, inter, hidden, wDown_[layer].shape()[0], wDown_[layer].shape()[1]);
            throw std::runtime_error("Layer wDown shape mismatch");
        }
        
        CLLM_DEBUG("[KylinBackend] Layer %zu: wq[%zu, %zu], wk[%zu, %zu], wv[%zu, %zu], wo[%zu, %zu] ✓",
                   layer,
                   wq_[layer].shape()[0], wq_[layer].shape()[1],
                   wk_[layer].shape()[0], wk_[layer].shape()[1],
                   wv_[layer].shape()[0], wv_[layer].shape()[1],
                   wo_[layer].shape()[0], wo_[layer].shape()[1]);
    }

    // 重建模型（使用当前配置）
    model_ = kylin::TransformerModel(internalConfig_);

    // 绑定 Embedding 和 LM Head
    model_.setEmbeddingWeight(embedding_);
    model_.setLmHeadWeight(lmHead_);

    // 绑定每层权重
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

bool KylinBackend::loadFromModelWeights(const model::ModelWeights &weights) {
    CLLM_INFO("[KylinBackend] Loading weights from ModelWeights");
    
    try {
        const size_t numLayers = weights.layers.size();
        if (numLayers == 0) {
            CLLM_ERROR("[KylinBackend] No layers found in ModelWeights");
            return false;
        }
        
        // 更新内部配置
        internalConfig_.numLayers = numLayers;
        internalConfig_.vocabSize = weights.embedding.shape[0];
        internalConfig_.hiddenSize = weights.embedding.shape[1];
        
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
        
        // 加载embedding权重
        embedding_ = Tensor(weights.embedding.shape);
        std::copy(weights.embedding.data.begin(), weights.embedding.data.end(), embedding_.data());
        
        // 加载lmHead权重
        lmHead_ = Tensor(weights.lmHead.shape);
        std::copy(weights.lmHead.data.begin(), weights.lmHead.data.end(), lmHead_.data());
        
        // 加载finalNorm权重
        finalNormWeight_ = Tensor(weights.finalNorm.shape);
        std::copy(weights.finalNorm.data.begin(), weights.finalNorm.data.end(), finalNormWeight_.data());
        
        // 加载每层的权重
        for (size_t layer = 0; layer < numLayers; ++layer) {
            const model::LayerWeights &layerWeights = weights.layers[layer];
            
            // Attention权重
            wq_[layer] = Tensor(layerWeights.wq.shape);
            std::copy(layerWeights.wq.data.begin(), layerWeights.wq.data.end(), wq_[layer].data());
            
            wk_[layer] = Tensor(layerWeights.wk.shape);
            std::copy(layerWeights.wk.data.begin(), layerWeights.wk.data.end(), wk_[layer].data());
            
            wv_[layer] = Tensor(layerWeights.wv.shape);
            std::copy(layerWeights.wv.data.begin(), layerWeights.wv.data.end(), wv_[layer].data());
            
            wo_[layer] = Tensor(layerWeights.wo.shape);
            std::copy(layerWeights.wo.data.begin(), layerWeights.wo.data.end(), wo_[layer].data());
            
            // FFN权重
            wGate_[layer] = Tensor(layerWeights.wGate.shape);
            std::copy(layerWeights.wGate.data.begin(), layerWeights.wGate.data.end(), wGate_[layer].data());
            
            wUp_[layer] = Tensor(layerWeights.wUp.shape);
            std::copy(layerWeights.wUp.data.begin(), layerWeights.wUp.data.end(), wUp_[layer].data());
            
            wDown_[layer] = Tensor(layerWeights.wDown.shape);
            std::copy(layerWeights.wDown.data.begin(), layerWeights.wDown.data.end(), wDown_[layer].data());
            
            // Norm权重
            norm1_[layer] = Tensor(layerWeights.norm1.shape);
            std::copy(layerWeights.norm1.data.begin(), layerWeights.norm1.data.end(), norm1_[layer].data());
            
            norm2_[layer] = Tensor(layerWeights.norm2.shape);
            std::copy(layerWeights.norm2.data.begin(), layerWeights.norm2.data.end(), norm2_[layer].data());
        }
        
        // 绑定权重到模型
        bindWeightsToModel();
        
        CLLM_INFO("[KylinBackend] Successfully loaded weights from ModelWeights");
        return true;
    } catch (const std::exception &e) {
        CLLM_ERROR("[KylinBackend] Failed to load weights from ModelWeights: %s", e.what());
        return false;
    }
}

} // namespace inference
} // namespace cllm
