/**
 * @file kylin_backend.cpp
 * @brief Kylin (麒麟) 自研推理后端实现
 */

#include "cllm/inference/kylin_backend.h"
#include "cllm/model/loader_interface.h"
#include "cllm/common/logger.h"
#include "cllm/common/config.h"
#include "cllm/kylin/core/quantization.h"

#include <stdexcept>
#include <cmath>
#include <filesystem>

namespace cllm {
namespace inference {

namespace {

// 为权重张量填充简单的可重复模式，避免全零
inline void fill_tensor_with_pattern(kylin::Tensor &tensor, float scale) {
    const size_t n = tensor.size();
    for (size_t i = 0; i < n; ++i) {
        // 使用有符号偏移避免 size_t 下溢
        int v = static_cast<int>(i % 31) - 15;
        tensor[i] = static_cast<float>(v) * scale;
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
        // 使用有符号偏移避免 size_t 下溢
        int u = static_cast<int>(i % 100) - 50;
        tensor[i] = static_cast<float>(u) / 50.0f * scale;
    }
}

// 小值初始化：适用于embedding和某些特殊层
inline void small_value_init(kylin::Tensor &tensor, float scale = 0.1f) {
    const size_t n = tensor.size();
    for (size_t i = 0; i < n; ++i) {
        // 使用小的均匀分布值
        int v = static_cast<int>(i % 21) - 10;
        tensor[i] = static_cast<float>(v) / 100.0f * scale;
    }
}

} // namespace

KylinBackend::KylinBackend(
    const ModelConfig &config, 
    const std::string &modelPath,
    kylin::OperatorBackend operatorBackend
)
    : externalConfig_(config)
    , internalConfig_(config)
    , initialized_(false)
    , modelPath_(modelPath)
    , model_(config)
    , operatorBackendType_(operatorBackend)
    , deviceBackendType_(kylin::BackendType::CPU)
    , useGGMLDirect_(false)
    , ggmlModel_(nullptr)
    , useHFModel_(false)
    , hfModel_(nullptr) {
    
    CLLM_INFO("[KylinBackend] Initializing Kylin (麒麟) inference backend");
    
    // 检测模型格式
    if (!modelPath_.empty()) {
        std::filesystem::path path(modelPath_);
        
        // 检测是否为 HuggingFace 模型目录（包含 config.json 和 *.safetensors）
        if (std::filesystem::is_directory(path)) {
            std::filesystem::path configJson = path / "config.json";
            std::filesystem::path safetensors = path / "model.safetensors";
            
            if (std::filesystem::exists(configJson) && std::filesystem::exists(safetensors)) {
                useHFModel_ = true;
                CLLM_INFO("[KylinBackend] Detected HuggingFace model directory, will use HFTransformerModel");
            } else {
                CLLM_INFO("[KylinBackend] Directory does not contain HuggingFace model files");
            }
        }
        // 检测是否为 GGUF 格式
        else if (path.extension() == ".gguf") {
            useGGMLDirect_ = true;
            CLLM_INFO("[KylinBackend] Detected GGUF format, will use GGMLTransformerModel (direct GGML inference)");
        } else {
            CLLM_INFO("[KylinBackend] Will use TransformerModel with operator backend");
        }
    }
    
    // 读取设备后端配置
    std::string deviceBackendStr = Config::instance().backendKylinDeviceBackend();
    deviceBackendType_ = kylin::BackendType::CPU;
    
    if (deviceBackendStr == "metal" || deviceBackendStr == "Metal") {
        deviceBackendType_ = kylin::BackendType::Metal;
        CLLM_INFO("[KylinBackend] Using Metal (Apple GPU) device backend");
    } else if (deviceBackendStr == "cuda" || deviceBackendStr == "CUDA") {
        deviceBackendType_ = kylin::BackendType::CUDA;
        CLLM_INFO("[KylinBackend] Using CUDA (NVIDIA GPU) device backend");
    } else if (deviceBackendStr == "auto" || deviceBackendStr == "Auto") {
        deviceBackendType_ = kylin::BackendType::Auto;
        CLLM_INFO("[KylinBackend] Auto-detecting best device backend");
    } else {
        CLLM_INFO("[KylinBackend] Using CPU device backend");
    }
    
    // 如果使用 GGML 直接推理，不需要算子
    if (!useGGMLDirect_) {
        // 创建算子实例（传递设备后端）
        op_ = kylin::OperatorFactory::create(operatorBackend, deviceBackendType_);
        CLLM_INFO("[KylinBackend] Using operator backend: %s", op_->getName().c_str());
    }
    
    if (!modelPath_.empty()) {
        if (useHFModel_) {
            // HuggingFace 模型：HFTransformerModel 在 initialize() 中创建
            CLLM_INFO("[KylinBackend] Will use HFTransformerModel for HuggingFace model: %s", modelPath_.c_str());
        } else if (useGGMLDirect_) {
            // GGUF 文件：使用 GGMLTransformerModel（传递设备后端）
            CLLM_INFO("[KylinBackend] Will use GGMLTransformerModel for GGUF file: %s", modelPath_.c_str());
            CLLM_INFO("[KylinBackend] GGMLTransformerModel will use device backend: %s",
                     deviceBackendType_ == kylin::BackendType::Metal ? "Metal" :
                     deviceBackendType_ == kylin::BackendType::CUDA ? "CUDA" : "CPU");
            ggmlModel_ = std::make_unique<kylin::GGMLTransformerModel>(deviceBackendType_);
        } else {
            // 其他格式：使用 ModelLoader + TransformerModel
            CLLM_INFO("[KylinBackend] Will load real weights from: %s", modelPath_.c_str());
            try {
                loader_ = ModelLoaderFactory::createLoader(modelPath_, externalConfig_);
                CLLM_INFO("[KylinBackend] Created loader for format: %s", 
                         ModelLoaderFactory::formatToString(ModelLoaderFactory::detectFormat(modelPath_)).c_str());
            } catch (const std::exception& e) {
                CLLM_ERROR("[KylinBackend] Failed to create model loader: %s", e.what());
                throw;
            }
        }
    } else {
        // 占位权重模式
        CLLM_INFO("[KylinBackend] Will use placeholder weights (test mode)");
        prepareInternalConfig();
    }

    // 预分配权重容器（仅在非 HuggingFace 模式下需要）
    if (!useHFModel_) {
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
        attnQNorm_.resize(numLayers);
        attnKNorm_.resize(numLayers);
    }
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

    // ========== HuggingFace 模型（safetensors 目录）==========
    if (useHFModel_) {
        return initializeHFModel();
    }

    // ========== GGML 直接推理模式（GGUF 文件）==========
    if (useGGMLDirect_) {
        return initializeGGMLDirect();
    }
    
    // ========== TransformerModel 模式（.bin 文件或占位权重）==========

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

bool KylinBackend::initializeHFModel() {
    CLLM_INFO("[KylinBackend] Initializing HuggingFace model (safetensors)...");
    
    try {
        // 转换设备类型
        kylin::DeviceType deviceType = kylin::DeviceType::CPU;
        if (deviceBackendType_ == kylin::BackendType::Metal) {
            deviceType = kylin::DeviceType::Metal;
            CLLM_INFO("[KylinBackend] Requesting Metal GPU acceleration for HF model");
        } else if (deviceBackendType_ == kylin::BackendType::CUDA) {
            deviceType = kylin::DeviceType::CUDA;
            CLLM_INFO("[KylinBackend] Requesting CUDA GPU acceleration for HF model");
        }
        
        // 从全局配置读取量化类型
        kylin::QuantType quantType = kylin::QuantType::FP32;  // 默认 FP32
        std::string quantStr = Config::instance().serverQuantization();
        if (!quantStr.empty()) {
            quantType = kylin::parseQuantType(quantStr);
            CLLM_INFO("[KylinBackend] Using quantization: %s", kylin::quantTypeName(quantType));
        }
        
        // 创建并加载 HuggingFace 模型
        hfModel_ = std::make_unique<kylin::HFTransformerModel>(modelPath_, deviceType, quantType);
        
        if (!hfModel_->isLoaded()) {
            CLLM_ERROR("[KylinBackend] Failed to load HuggingFace model: %s", modelPath_.c_str());
            return false;
        }
        
        // 更新配置
        const auto& hfConfig = hfModel_->config();
        externalConfig_.vocabSize = hfConfig.vocabSize;
        externalConfig_.hiddenSize = hfConfig.hiddenSize;
        externalConfig_.numLayers = hfConfig.numHiddenLayers;
        externalConfig_.numAttentionHeads = hfConfig.numAttentionHeads;
        externalConfig_.numKeyValueHeads = hfConfig.getNumKVHeads();
        externalConfig_.intermediateSize = hfConfig.intermediateSize;
        
        CLLM_INFO("[KylinBackend] HuggingFace model loaded successfully:");
        CLLM_INFO("  Model: %s", hfConfig.modelType.c_str());
        CLLM_INFO("  Vocab: %u", externalConfig_.vocabSize);
        CLLM_INFO("  Hidden: %u", externalConfig_.hiddenSize);
        CLLM_INFO("  Layers: %u", externalConfig_.numLayers);
        CLLM_INFO("  Heads: %u (KV: %u)", externalConfig_.numAttentionHeads, externalConfig_.numKeyValueHeads);
        CLLM_INFO("  Dtype: %s", hfConfig.torchDtype.c_str());
        
        initialized_ = true;
        CLLM_INFO("[KylinBackend] HuggingFace model initialized successfully");
        return true;
        
    } catch (const std::exception& e) {
        CLLM_ERROR("[KylinBackend] Exception loading HuggingFace model: %s", e.what());
        return false;
    }
}

bool KylinBackend::initializeGGMLDirect() {
    CLLM_INFO("[KylinBackend] Initializing GGML direct inference mode...");
    
    if (!ggmlModel_) {
        CLLM_ERROR("[KylinBackend] GGMLTransformerModel not created");
        return false;
    }
    
    // 加载 GGUF 模型
    if (!ggmlModel_->loadFromGGUF(modelPath_)) {
        CLLM_ERROR("[KylinBackend] Failed to load GGUF model: %s", modelPath_.c_str());
        return false;
    }
    
    // 更新配置
    const auto& ggmlConfig = ggmlModel_->getConfig();
    externalConfig_.vocabSize = ggmlConfig.vocabSize;
    externalConfig_.hiddenSize = ggmlConfig.embeddingLength;
    externalConfig_.numLayers = ggmlConfig.blockCount;
    externalConfig_.numAttentionHeads = ggmlConfig.headCount;
    externalConfig_.numKeyValueHeads = ggmlConfig.headCountKV;
    externalConfig_.intermediateSize = ggmlConfig.feedForwardLength;
    externalConfig_.maxSequenceLength = ggmlConfig.contextLength;
    
    CLLM_INFO("[KylinBackend] GGML model loaded successfully:");
    CLLM_INFO("  Vocab: %u", externalConfig_.vocabSize);
    CLLM_INFO("  Hidden: %u", externalConfig_.hiddenSize);
    CLLM_INFO("  Layers: %u", externalConfig_.numLayers);
    CLLM_INFO("  Heads: %u (KV: %u)", externalConfig_.numAttentionHeads, externalConfig_.numKeyValueHeads);
    
    initialized_ = true;
    CLLM_INFO("[KylinBackend] GGML direct inference initialized successfully");
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
    attnQNorm_.resize(numLayers);
    attnKNorm_.resize(numLayers);

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

    // 尝试加载 Q/K 归一化权重（如果使用 GGUF 格式且存在）
    // 注意：loadInto 接口不支持这些权重，需要通过 ModelWeights 结构加载
    // 这里先尝试通过 IModelLoader 接口加载（如果支持）
    if (loader_) {
        model::ModelWeights modelWeights;
        if (loader_->loadWeights(modelWeights, false)) {  // 不立即加载数据，只创建结构
            // 尝试加载 Q/K 归一化权重
            for (size_t layer = 0; layer < numLayers && layer < modelWeights.layers.size(); ++layer) {
                const auto& layerWeights = modelWeights.layers[layer];
                
                // 如果 Q/K 归一化权重存在，加载它们
                if (!layerWeights.attnQNorm.shape.empty() && layerWeights.attnQNorm.data.size() > 0) {
                    attnQNorm_[layer] = kylin::Tensor(layerWeights.attnQNorm.shape);
                    std::copy(layerWeights.attnQNorm.data.begin(), 
                             layerWeights.attnQNorm.data.end(), 
                             attnQNorm_[layer].data());
                    CLLM_DEBUG("[KylinBackend] Loaded attnQNorm for layer %zu (shape size: %zu)", 
                              layer, layerWeights.attnQNorm.shape.size());
                }
                
                if (!layerWeights.attnKNorm.shape.empty() && layerWeights.attnKNorm.data.size() > 0) {
                    attnKNorm_[layer] = kylin::Tensor(layerWeights.attnKNorm.shape);
                    std::copy(layerWeights.attnKNorm.data.begin(), 
                             layerWeights.attnKNorm.data.end(), 
                             attnKNorm_[layer].data());
                    CLLM_DEBUG("[KylinBackend] Loaded attnKNorm for layer %zu (shape size: %zu)", 
                              layer, layerWeights.attnKNorm.shape.size());
                }
            }
        }
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
    size_t numHeads = internalConfig_.numAttentionHeads;  // 可能被修正
    
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
    
    // P1修复：禁用"标准公式推断 heads"覆盖模型配置
    // 但是，如果配置中的 numQHeads 明显错误（导致 Q/K head_dim 不一致），需要修正
    // 根据对比文档，llama.cpp 的 Qwen3 实现要求 Q/K head_dim 一致
    //
    // 策略：
    // 1. 先计算 qHeadDim = qDim / numHeads
    // 2. 验证 kvDim 是否能被 qHeadDim 整除（Qwen3 要求 Q/K head_dim 一致）
    // 3. 如果不能整除，说明配置的 numQHeads 可能错误，尝试从权重推断正确的值
    
    size_t standardHeadDim = hidden / numHeads;  // 标准 head_dim（仅用于对比和验证）
    
    // 验证 qDim 是否能被 numHeads 整除
    if (numHeads > 0 && qDim > 0) {
        if (qDim % numHeads != 0) {
            CLLM_ERROR("[KylinBackend] ❌ qDim (%zu) 不能被 numQHeads (%zu) 整除，配置或权重可能有问题",
                      qDim, numHeads);
            throw std::runtime_error("qDim must be divisible by numQHeads");
        }
        
        // 计算实际的 qHeadDim（从权重形状推断，支持扩展 head_dim）
        size_t actualQHeadDim = qDim / numHeads;
        
        // P1修复增强：验证 Q/K head_dim 一致性
        // llama.cpp 要求 Q/K head_dim 必须一致（n_embd_head_v == n_embd_head_k）
        // 如果配置的 numQHeads 导致 Q/K head_dim 不一致，需要推断正确的值
        if (kvDim > 0 && kvDim % actualQHeadDim != 0) {
            // 配置的 numQHeads 导致 Q/K head_dim 不一致，尝试推断正确的值
            // llama.cpp 要求 Q/K head_dim 必须一致，所以需要找到使得两者一致的 numQHeads
            // 策略：找到同时能被 qDim 和 kvDim 整除的 head_dim，然后反推 numQHeads
            
            CLLM_WARN("[KylinBackend] ⚠️ 配置的 numQHeads (%zu) 导致 Q/K head_dim 不一致：qHeadDim=%zu, kvDim=%zu，尝试推断正确的值",
                     numHeads, actualQHeadDim, kvDim);
            
            // 尝试推断：找到使得 Q 和 K 的 head_dim 一致的 head_dim
            // 常见的 head_dim 值：64, 128, 256
            size_t inferredHeadDim = 0;
            size_t inferredNumQHeads = 0;
            
            for (size_t candidateHeadDim = 64; candidateHeadDim <= 256; candidateHeadDim *= 2) {
                if (qDim % candidateHeadDim == 0 && kvDim % candidateHeadDim == 0) {
                    size_t candidateNumQHeads = qDim / candidateHeadDim;
                    size_t candidateNumKVHeads = kvDim / candidateHeadDim;
                    // 验证 GQA 约束：numQHeads 必须能被 numKVHeads 整除
                    if (candidateNumQHeads > 0 && candidateNumKVHeads > 0 && 
                        candidateNumQHeads % candidateNumKVHeads == 0 &&
                        candidateNumQHeads <= 128) {  // 合理的 head 数量范围
                        inferredHeadDim = candidateHeadDim;
                        inferredNumQHeads = candidateNumQHeads;
                        break;
                    }
                }
            }
            
            if (inferredHeadDim > 0 && inferredNumQHeads != numHeads) {
                CLLM_WARN("[KylinBackend] ⚠️ 推断正确的 numQHeads=%zu (qDim=%zu / headDim=%zu)，原配置为 %zu，将更新配置",
                         inferredNumQHeads, qDim, inferredHeadDim, numHeads);
                internalConfig_.numAttentionHeads = inferredNumQHeads;
                numHeads = inferredNumQHeads;  // 更新局部变量
                actualQHeadDim = inferredHeadDim;  // 更新 actualQHeadDim
                standardHeadDim = hidden / numHeads;  // 重新计算 standardHeadDim
            } else if (inferredHeadDim == 0) {
                CLLM_ERROR("[KylinBackend] ❌ 无法推断正确的 numQHeads，Q/K head_dim 不一致：qHeadDim=%zu (qDim=%zu / numQHeads=%zu), kvDim=%zu",
                          actualQHeadDim, qDim, numHeads, kvDim);
                throw std::runtime_error("Cannot infer correct numQHeads: Q/K head_dim inconsistent");
            }
        }
        
        // 如果 actualQHeadDim != standardHeadDim，说明使用了扩展 head_dim（如 Qwen3）
        if (actualQHeadDim != standardHeadDim) {
            size_t expansionFactor = actualQHeadDim / standardHeadDim;
            CLLM_INFO("[KylinBackend] ✓ 检测到扩展 head_dim：qHeadDim=%zu (标准=%zu, 扩展因子=%zu)，使用配置的 numQHeads=%zu",
                     actualQHeadDim, standardHeadDim, expansionFactor, numHeads);
        } else {
            CLLM_INFO("[KylinBackend] ✓ 使用标准 head_dim：qHeadDim=%zu，numQHeads=%zu",
                     actualQHeadDim, numHeads);
        }
    }
    
    // 从 kvDim 推断 KV heads（GQA 支持）
    // 关键点：Qwen3（llama.cpp 实现）要求 Q/K 的 head_dim 一致（n_embd_head_v == n_embd_head_k）。
    // 在这类模型中，head_dim 不能用 hidden/num_heads 的“标准公式”推断，而应以投影权重的实际形状为准。
    // 因此 KV heads 应该使用 qHeadDimFromWeights 反推：numKVHeads = kvDim / qHeadDimFromWeights。
    size_t qHeadDimFromWeights = 0;
    if (internalConfig_.numAttentionHeads > 0 && qDim % internalConfig_.numAttentionHeads == 0) {
        qHeadDimFromWeights = qDim / internalConfig_.numAttentionHeads;
    }

    // P1修复：验证 kvDim 是否能被 qHeadDim 整除（Qwen3 要求 Q/K head_dim 一致）
    // 优先相信配置中的 numKeyValueHeads，只在配置不合理时才推断
    bool kvHeadsInferred = false;
    if (qHeadDimFromWeights > 0 && kvDim > 0) {
        if (kvDim % qHeadDimFromWeights != 0) {
            CLLM_ERROR("[KylinBackend] ❌ kvDim (%zu) 不能被 qHeadDim (%zu) 整除，Q/K head_dim 不一致！",
                      kvDim, qHeadDimFromWeights);
            throw std::runtime_error("kvDim must be divisible by qHeadDim (Qwen3 requires Q/K head_dim to be consistent)");
        }
        
        size_t inferredNumKVHeads = kvDim / qHeadDimFromWeights;
        if (inferredNumKVHeads > 0 &&
            inferredNumKVHeads <= internalConfig_.numAttentionHeads &&
            (internalConfig_.numAttentionHeads % inferredNumKVHeads == 0)) {
            size_t configKVHeadDim = 0;
            if (internalConfig_.numKeyValueHeads > 0 && kvDim % internalConfig_.numKeyValueHeads == 0) {
                configKVHeadDim = kvDim / internalConfig_.numKeyValueHeads;
            }
            // 如果配置不合理或导致 Q/K head_dim 不一致，使用推断值
            if (internalConfig_.numKeyValueHeads == 0 ||
                internalConfig_.numKeyValueHeads > internalConfig_.numAttentionHeads ||
                configKVHeadDim != qHeadDimFromWeights) {
                CLLM_WARN("[KylinBackend] ⚠️ 配置的 numKVHeads=%zu 导致 kvHeadDim=%zu，与 qHeadDim=%zu 不一致，改为 %zu",
                         internalConfig_.numKeyValueHeads, configKVHeadDim, qHeadDimFromWeights, inferredNumKVHeads);
                internalConfig_.numKeyValueHeads = inferredNumKVHeads;
                kvHeadsInferred = true;
            } else {
                CLLM_INFO("[KylinBackend] ✓ KV heads 配置与权重一致：numKVHeads=%zu, kvHeadDim=%zu",
                         internalConfig_.numKeyValueHeads, qHeadDimFromWeights);
                kvHeadsInferred = true;
            }
        } else {
            CLLM_ERROR("[KylinBackend] ❌ 推断出的 numKVHeads=%zu 不合法（numQHeads=%zu）",
                      inferredNumKVHeads, internalConfig_.numAttentionHeads);
            throw std::runtime_error("Invalid inferred numKVHeads");
        }
    }

    if (!kvHeadsInferred && (internalConfig_.numKeyValueHeads == 0 ||
                             internalConfig_.numKeyValueHeads > internalConfig_.numAttentionHeads)) {
        // 如果推断失败或配置不合理，默认等于 Q heads（MHA）
        internalConfig_.numKeyValueHeads = internalConfig_.numAttentionHeads;
        CLLM_INFO("[KylinBackend] 设置 numKeyValueHeads = numAttentionHeads = %zu (MHA)",
                 internalConfig_.numAttentionHeads);
    }
    
    // 使用更新后的配置
    const size_t actualNumHeads = internalConfig_.numAttentionHeads;
    const size_t actualNumKVHeads = internalConfig_.numKeyValueHeads;
    
    // W1修复：使用实际的 qHeadDim 和 kvHeadDim，而不是标准公式计算的 headDim
    // 实际的 head_dim 应该从权重形状推断，而不是假设 headDim = hidden / numHeads
    const size_t actualQHeadDim = (actualNumHeads > 0) ? (qDim / actualNumHeads) : (hidden / actualNumHeads);
    const size_t actualKVHeadDim = (actualNumKVHeads > 0) ? (kvDim / actualNumKVHeads) : (hidden / actualNumKVHeads);
    // 重新计算 standardHeadDim（基于更新后的 actualNumHeads）
    standardHeadDim = hidden / actualNumHeads;  // 标准公式（仅用于对比）
    
    CLLM_INFO("[KylinBackend] 验证权重形状: hidden=%zu, vocab=%zu, numLayers=%zu, numQHeads=%zu, numKVHeads=%zu, actualQHeadDim=%zu, actualKVHeadDim=%zu, standardHeadDim=%zu, qDim=%zu, kvDim=%zu, inter=%zu",
             hidden, vocab, numLayers, actualNumHeads, actualNumKVHeads, actualQHeadDim, actualKVHeadDim, standardHeadDim, qDim, kvDim, inter);
    
    // 验证 head 数量的一致性（使用实际的 head_dim）
    // 修复：应该使用实际的 head_dim 进行验证，不应该再出现维度不匹配的警告（除非真的有问题）
    if (qDim != actualNumHeads * actualQHeadDim) {
        CLLM_WARN("[KylinBackend] ⚠️ qDim (%zu) != numQHeads * actualQHeadDim (%zu * %zu = %zu)，维度不匹配！",
                 qDim, actualNumHeads, actualQHeadDim, actualNumHeads * actualQHeadDim);
    } else if (actualQHeadDim != standardHeadDim) {
        // 如果 qDim 与标准公式不同，说明使用了扩展 head_dim（这是正常的，比如 Qwen3）
        size_t expansionFactor = actualQHeadDim / standardHeadDim;
        CLLM_INFO("[KylinBackend] ✓ 检测到扩展 head_dim：Q head_dim=%zu (标准=%zu, 扩展因子=%zu)，维度匹配",
                 actualQHeadDim, standardHeadDim, expansionFactor);
    } else {
        CLLM_INFO("[KylinBackend] ✓ Q 维度匹配：qDim=%zu = numQHeads * headDim (%zu * %zu)",
                 qDim, actualNumHeads, actualQHeadDim);
    }
    if (kvDim != actualNumKVHeads * actualKVHeadDim) {
        CLLM_WARN("[KylinBackend] ⚠️ kvDim (%zu) != numKVHeads * actualKVHeadDim (%zu * %zu = %zu)，维度不匹配！",
                 kvDim, actualNumKVHeads, actualKVHeadDim, actualNumKVHeads * actualKVHeadDim);
    } else if (actualKVHeadDim != standardHeadDim) {
        // 如果 kvDim 与标准公式不同，说明使用了不同的 head_dim（这是正常的）
        CLLM_INFO("[KylinBackend] ✓ KV head_dim=%zu (标准=%zu)，维度匹配",
                 actualKVHeadDim, standardHeadDim);
    } else {
        CLLM_INFO("[KylinBackend] ✓ KV 维度匹配：kvDim=%zu = numKVHeads * headDim (%zu * %zu)",
                 kvDim, actualNumKVHeads, actualKVHeadDim);
    }
    
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
    CLLM_INFO("[KylinBackend] Binding LM head...");
    model_.setLmHeadWeight(lmHead_);
    CLLM_INFO("[KylinBackend] LM head bound");

    // 绑定每层权重
    for (size_t layer = 0; layer < numLayers; ++layer) {
        CLLM_INFO("[KylinBackend] Binding layer %zu weights...", layer);
        // 检查是否有 Q/K 归一化权重
        kylin::Tensor attnQNorm = attnQNorm_[layer].shape().empty() ? kylin::Tensor() : attnQNorm_[layer];
        kylin::Tensor attnKNorm = attnKNorm_[layer].shape().empty() ? kylin::Tensor() : attnKNorm_[layer];
        
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
            norm2_[layer],
            attnQNorm,  // Q 归一化权重（可选）
            attnKNorm   // K 归一化权重（可选）
        );
        CLLM_INFO("[KylinBackend] Layer %zu weights bound", layer);
    }

    // 绑定 Final Norm
    model_.setFinalNormWeight(finalNormWeight_);

    CLLM_INFO("[KylinBackend] Weights bound to model successfully");
}

kylin::Tensor KylinBackend::forward(const std::vector<int> &inputIds) {
    if (!initialized_) {
        throw std::runtime_error("KylinBackend::forward: backend not initialized");
    }
    if (inputIds.empty()) {
        throw std::invalid_argument("KylinBackend::forward: empty inputIds");
    }
    const size_t vocabSize = externalConfig_.vocabSize;
    if (vocabSize == 0) {
        throw std::runtime_error("KylinBackend::forward: vocabSize is 0");
    }
    for (int id : inputIds) {
        if (id < 0 || static_cast<size_t>(id) >= vocabSize) {
            throw std::out_of_range("KylinBackend::forward: token id out of range");
        }
    }

    // ========== HuggingFace 模型 ==========
    if (useHFModel_ && hfModel_) {
        const size_t seqLen = inputIds.size();
        
        CLLM_DEBUG("[KylinBackend::forward] HF mode: seqLen=%zu, vocabSize=%zu", seqLen, vocabSize);
        
        // HFTransformerModel 只返回最后一个 token 的 logits (vocab_size 个元素)
        // 对于自回归生成，这是正确的行为
        // 对于 prefill（seq_len > 1），需要逐 token 处理以正确构建 KV cache
        
        if (seqLen == 1) {
            // 单 token 推理（增量生成）：直接调用 forward，KV cache 会自动累积
            std::vector<int32_t> inputIds32(inputIds.begin(), inputIds.end());
            std::vector<float> logitsFlat = hfModel_->forward(inputIds32);
            
            kylin::Tensor result({1, vocabSize});
            std::copy(logitsFlat.begin(), logitsFlat.end(), result.data());
            return result;
        } else {
            // 多 token 推理（prefill）：清除 KV cache 并逐 token 处理
            // 使用 requestId = 0 作为默认请求
            hfModel_->releaseKVCache(0);
            CLLM_DEBUG("[KylinBackend::forward] Prefill: seqLen=%zu, reset KV cache", seqLen);
            
            // 逐 token 处理以正确构建 KV cache
            kylin::Tensor result({seqLen, vocabSize});
            std::fill(result.data(), result.data() + seqLen * vocabSize, 0.0f);
            
            for (size_t i = 0; i < seqLen; ++i) {
                std::vector<int32_t> singleToken = {static_cast<int32_t>(inputIds[i])};
                std::vector<float> logits = hfModel_->forward(singleToken);
                
                // 复制当前 token 的 logits 到结果中
                std::copy(logits.begin(), logits.end(), 
                         result.data() + i * vocabSize);
            }
            
            return result;
        }
    }

    // ========== GGML 直接推理模式 ==========
    if (useGGMLDirect_ && ggmlModel_) {
        // 注意：独立的 forward() 调用（非通过 forwardBatch）
        // 根据输入长度判断是否需要清除 KV cache：
        // - 如果 inputIds.size() > 1，说明是新请求的 prompt，清除 KV cache
        // - 如果 inputIds.size() == 1，可能是增量生成，保留 KV cache
        // 这个启发式方法不完美，但在没有 sequenceId 信息时是合理的默认行为
        if (inputIds.size() > 1) {
            ggmlModel_->clearKVCache();
            currentSequenceId_ = SIZE_MAX;  // 重置序列 ID
            CLLM_DEBUG("[KylinBackend] Cleared KV cache for new prompt (len=%zu)", inputIds.size());
        }
        
        // 转换 int 到 int32_t
        std::vector<int32_t> inputIds32(inputIds.begin(), inputIds.end());
        
        // 使用 GGMLTransformerModel 推理
        std::vector<float> logitsFlat = ggmlModel_->forward(inputIds32);
        
        // 转换为 kylin::Tensor
        const size_t seqLen = inputIds.size();
        const size_t vocabSize = externalConfig_.vocabSize;
        kylin::Tensor result({seqLen, vocabSize});
        std::copy(logitsFlat.begin(), logitsFlat.end(), result.data());
        
        return result;
    }
    
    // ========== TransformerModel 模式 ==========
    return model_.forward(inputIds);
}

kylin::Tensor KylinBackend::forwardBatch(
    const std::vector<int> &flatInputIds,
    const std::vector<std::pair<size_t, size_t>> &requestPositions,
    size_t batchSize,
    const std::vector<size_t> &sequenceIds
) {
    // sequenceIds 参数未使用（KylinBackend 不需要序列ID管理）
    (void)sequenceIds;
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
    
    // ========== HuggingFace 模型 ==========
    if (useHFModel_ && hfModel_) {
        // 使用 per-request KV Cache 实现真正的并发
        CLLM_DEBUG("[KylinBackend] HuggingFace mode: processing %zu requests with per-request KV cache", batchSize);
        
        // 优化：如果所有请求都是单 token 推理，使用 GPU 批处理加速
        bool allSingleToken = true;
        std::vector<int> singleTokenIds;
        std::vector<int> positions;
        std::vector<size_t> requestIds;
        
        // 检查是否所有请求都是单 token
        for (size_t i = 0; i < batchSize; ++i) {
            const auto &pos = requestPositions[i];
            const size_t start = pos.first;
            const size_t end = pos.second;
            
            if (start > end || end > totalTokens) {
                CLLM_ERROR("[KylinBackend] Invalid requestPositions range for request %zu", i);
                continue;
            }
            
            const size_t reqLen = end - start;
            size_t requestId = (i < sequenceIds.size()) ? sequenceIds[i] : i;
            
            if (reqLen == 1) {
                singleTokenIds.push_back(flatInputIds[start]);
                positions.push_back(hfModel_->getKVCacheCurrentLength(requestId)); // 获取当前位置
                requestIds.push_back(requestId);
            } else {
                // 有多 token 请求，不能使用 GPU 批处理
                allSingleToken = false;
                break;
            }
        }
        
        // 使用 forwardWithRequestId 以确保正确的 per-request KV Cache 管理
        // 注：GPU 批处理已禁用，因为 GGMLGPUBackend::forwardGraphMinimal 没有 per-request KV Cache 支持
        {
             // 混合请求或有多个 token 的请求，使用原来的处理方式
             #pragma omp parallel for schedule(dynamic) if(batchSize > 1)
             for (size_t i = 0; i < batchSize; ++i) {
                 const auto &pos = requestPositions[i];
                 const size_t start = pos.first;
                 const size_t end = pos.second;

                 if (start > end || end > totalTokens) {
                     CLLM_ERROR("[KylinBackend] Invalid requestPositions range for request %zu", i);
                     continue;
                 }
                 
                 // 获取请求 ID（用于 per-request KV Cache）
                 size_t requestId = (i < sequenceIds.size()) ? sequenceIds[i] : i;
                 
                 const size_t reqLen = end - start;
                 
                 // 🔥 关键优化：使用批量 prefill，一次性处理所有 tokens
                 // CPUBackend::forward 现在支持多 token 输入
                 if (reqLen > 1) {
                     // 新请求（多 token）：先释放旧的 KV Cache
                     hfModel_->releaseKVCache(requestId);
                 }
                 
                 // 一次性传入所有 tokens（单 token 或多 token）
                 std::vector<int32_t> allTokens(flatInputIds.begin() + start, flatInputIds.begin() + end);
                 std::vector<float> requestLogits = hfModel_->forwardWithRequestId(allTokens, requestId);
                 
                 // 复制最后一个 token 的 logits 到输出
                 // 对于多 token 输入，只返回最后一个 token 的 logits
                 std::copy(requestLogits.begin(), requestLogits.end(), 
                          logits.data() + (end - 1) * vocab);
             }
        }
        
        return logits;
    }
    
    // ========== GGML 直接推理模式 ==========
    if (useGGMLDirect_ && ggmlModel_) {
        // GGML 模式：逐请求推理（暂不支持真正的批处理）
        // 由于 GGMLTransformerModel 只有一个全局 KV cache，不支持多序列并发
        // 我们使用 sequenceId 来管理 KV cache：
        // - 如果 sequenceId 与当前序列不同，说明是新请求，清除 KV cache
        // - 如果 sequenceId 相同，说明是同一序列的增量生成，保留 KV cache
        CLLM_DEBUG("[KylinBackend] GGML direct mode: processing %zu requests sequentially", batchSize);
        
        for (size_t i = 0; i < batchSize; ++i) {
            const auto &pos = requestPositions[i];
            const size_t start = pos.first;
            const size_t end = pos.second;

            if (start > end || end > totalTokens) {
                throw std::out_of_range("KylinBackend::forwardBatch: invalid requestPositions range");
            }
            
            // 获取当前请求的序列ID
            size_t seqId = (i < sequenceIds.size()) ? sequenceIds[i] : SIZE_MAX;
            const size_t reqLen = end - start;
            
            // KV cache 管理：
            // 1. 如果序列ID变化，清除 KV cache（新请求）
            // 2. 如果输入长度 > 1，也清除 KV cache（新 prompt）
            bool needClearCache = (seqId != currentSequenceId_) || (reqLen > 1);
            
            if (needClearCache) {
                ggmlModel_->clearKVCache();
                currentSequenceId_ = seqId;
                CLLM_DEBUG("[KylinBackend] Cleared KV cache for sequence %zu (prev=%zu, reqLen=%zu)", 
                          seqId, currentSequenceId_, reqLen);
            } else {
                CLLM_DEBUG("[KylinBackend] Reusing KV cache for sequence %zu (reqLen=%zu)", 
                          seqId, reqLen);
            }
            
            // 提取当前请求的 token IDs
            std::vector<int32_t> requestIds32(flatInputIds.begin() + start, flatInputIds.begin() + end);
            
            // 推理
            std::vector<float> requestLogits = ggmlModel_->forward(requestIds32);
            
            // 复制到输出
            std::copy(requestLogits.begin(), requestLogits.end(), 
                     logits.data() + start * vocab);
        }
        
        return logits;
    }

    // ========== TransformerModel 模式 ==========
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
        attnQNorm_.resize(numLayers);
        attnKNorm_.resize(numLayers);
        
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
            
            // Q/K 独立归一化权重（可选）
            if (!layerWeights.attnQNorm.shape.empty() && layerWeights.attnQNorm.data.size() > 0) {
                attnQNorm_[layer] = Tensor(layerWeights.attnQNorm.shape);
                std::copy(layerWeights.attnQNorm.data.begin(), 
                         layerWeights.attnQNorm.data.end(), 
                         attnQNorm_[layer].data());
                CLLM_DEBUG("[KylinBackend] Loaded attnQNorm for layer %zu from ModelWeights", layer);
            }
            
            if (!layerWeights.attnKNorm.shape.empty() && layerWeights.attnKNorm.data.size() > 0) {
                attnKNorm_[layer] = Tensor(layerWeights.attnKNorm.shape);
                std::copy(layerWeights.attnKNorm.data.begin(), 
                         layerWeights.attnKNorm.data.end(), 
                         attnKNorm_[layer].data());
                CLLM_DEBUG("[KylinBackend] Loaded attnKNorm for layer %zu from ModelWeights", layer);
            }
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

kylin::OperatorBackend KylinBackend::getOperatorBackend() const {
    if (op_) {
        return op_->getBackend();
    }
    return kylin::OperatorBackend::Native;
}

std::string KylinBackend::getOperatorBackendName() const {
    if (op_) {
        return op_->getName();
    }
    return "None";
}

bool KylinBackend::cleanupKVCache(size_t requestId) {
    if (useHFModel_ && hfModel_) {
        hfModel_->releaseKVCache(requestId);
        CLLM_DEBUG("[KylinBackend] Released KV cache for request %zu", requestId);
        return true;
    }
    // 其他模式暂不支持 per-request KV cache
    return false;
}

} // namespace inference
} // namespace cllm
