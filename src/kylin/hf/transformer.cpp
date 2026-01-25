/**
 * @file hf_transformer.cpp
 * @brief HuggingFace Transformer 模型实现
 * 
 * 使用 SIMD 优化内核加速推理
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/common/logger.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <thread>
#include <future>

#ifdef _OPENMP
#include <omp.h>
#endif

// SIMD 支持
#if defined(__ARM_NEON) || defined(__aarch64__)
    #define USE_NEON 1
    #include <arm_neon.h>
#endif

namespace cllm {
namespace kylin {

HFTransformerModel::HFTransformerModel(const std::string& modelDir, DeviceType device, QuantType quantType)
    : deviceType_(device), useGPU_(false), quantType_(quantType) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    CLLM_INFO("[HFTransformer] Requested device: %s, quantization: %s", 
              device == DeviceType::Metal ? "Metal GPU" : 
              device == DeviceType::CUDA ? "CUDA GPU" : "CPU",
              quantTypeName(quantType));
    
    // 初始化计算内核
    ggml_kernels::initialize(device);
    
    // 对于 Metal，尝试初始化 GPU 后端
    if (device == DeviceType::Metal) {
#ifdef GGML_USE_METAL
        gpuBackend_ = std::make_unique<GGMLGPUBackend>();
        CLLM_INFO("[HFTransformer] GPU backend created, will initialize after loading weights");
#else
        CLLM_WARN("[HFTransformer] Metal not compiled, falling back to CPU");
        deviceType_ = DeviceType::CPU;
#endif
    }
    
    // 加载配置
    config_ = loadHFConfigFromDir(modelDir);
    if (!config_.isValid()) {
        CLLM_ERROR("[HFTransformer] Invalid model config");
        return;
    }
    config_.print();
    
    // 加载 safetensors
    std::string safetensorsPath = modelDir;
    if (safetensorsPath.back() != '/') safetensorsPath += '/';
    safetensorsPath += "model.safetensors";
    
    loader_ = std::make_unique<SafetensorsLoader>(safetensorsPath);
    if (!loader_->isValid()) {
        CLLM_ERROR("[HFTransformer] Failed to load safetensors");
        return;
    }
    
    // 加载权重
    if (!loadWeights()) {
        CLLM_ERROR("[HFTransformer] Failed to load weights");
        return;
    }
    
    // 预计算 RoPE 频率
    int headDim = config_.getHeadDim();
    ropeFreqsCos_.resize(kMaxSeqLen * headDim / 2);
    ropeFreqsSin_.resize(kMaxSeqLen * headDim / 2);
    
    for (int pos = 0; pos < kMaxSeqLen; ++pos) {
        for (int i = 0; i < headDim / 2; ++i) {
            float freq = 1.0f / std::pow(config_.ropeTheta, 2.0f * i / headDim);
            float angle = pos * freq;
            ropeFreqsCos_[pos * headDim / 2 + i] = std::cos(angle);
            ropeFreqsSin_[pos * headDim / 2 + i] = std::sin(angle);
        }
    }
    
    // 初始化 KV Cache
    int kvHeads = config_.getNumKVHeads();
    size_t kvSize = static_cast<size_t>(config_.numHiddenLayers) * kMaxSeqLen * kvHeads * headDim;
    kCache_.resize(kvSize, 0.0f);
    vCache_.resize(kvSize, 0.0f);
    
    // 分配工作缓冲区（预分配避免运行时分配）
    hiddenStates_.resize(config_.hiddenSize);
    residual_.resize(config_.hiddenSize);
    normOutput_.resize(config_.hiddenSize);
    attnOutput_.resize(config_.hiddenSize);
    ffnOutput_.resize(config_.hiddenSize);
    
    // QKV 缓冲区
    int qSize = config_.numAttentionHeads * headDim;
    int kvSize2 = kvHeads * headDim;
    qkvBuffer_.resize(qSize + 2 * kvSize2);
    
    // Attention 工作缓冲区（预分配）
    qBuffer_.resize(qSize);
    kBuffer_.resize(kvSize2);
    vBuffer_.resize(kvSize2);
    attnScores_.resize(config_.numAttentionHeads * kMaxSeqLen);  // 每个 head 独立
    attnOutBuffer_.resize(qSize);
    
    // FFN 工作缓冲区
    gateBuffer_.resize(config_.intermediateSize);
    upBuffer_.resize(config_.intermediateSize);
    gateUpBuffer_.resize(config_.intermediateSize * 2);
    
    // Norm 权重缓冲区（避免每层重复分配）
    normWeightBuffer_.resize(config_.hiddenSize);
    qkNormBuffer_.resize(headDim);
    
    // 预转换权重到目标精度（消除运行时转换开销）
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::FP16) {
            convertWeightsToFP16();
        } else if (quantType_ == QuantType::INT8) {
            convertWeightsToINT8();
        } else {
            preconvertWeights();  // 默认转为 FP32
        }
    }
    
    // 初始化 GPU 后端并上传权重
    if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
        if (gpuBackend_->initialize(config_)) {
            // 准备层权重
            std::vector<LayerWeightsGPU> layerWeights(config_.numHiddenLayers);
            for (int i = 0; i < config_.numHiddenLayers; ++i) {
                layerWeights[i].inputLayernorm = layersF32_[i].inputLayernorm.data();
                layerWeights[i].qProj = layersF32_[i].qProj.data();
                layerWeights[i].kProj = layersF32_[i].kProj.data();
                layerWeights[i].vProj = layersF32_[i].vProj.data();
                layerWeights[i].oProj = layersF32_[i].oProj.data();
                layerWeights[i].qNorm = layersF32_[i].qNorm.empty() ? nullptr : layersF32_[i].qNorm.data();
                layerWeights[i].kNorm = layersF32_[i].kNorm.empty() ? nullptr : layersF32_[i].kNorm.data();
                layerWeights[i].postAttentionLayernorm = layersF32_[i].postAttentionLayernorm.data();
                layerWeights[i].gateProj = layersF32_[i].gateProj.data();
                layerWeights[i].upProj = layersF32_[i].upProj.data();
                layerWeights[i].downProj = layersF32_[i].downProj.data();
            }
            
            // 上传权重到 GPU
            if (gpuBackend_->uploadWeights(
                    embedTokensF32_.data(),
                    layerWeights,
                    finalNormWeightF32_.data(),
                    config_.tieWordEmbeddings ? nullptr : lmHeadWeightF32_.data())) {
                useGPU_ = true;
                CLLM_INFO("[HFTransformer] ✅ GPU backend ready, weights uploaded");
            } else {
                CLLM_WARN("[HFTransformer] Failed to upload weights to GPU, using CPU");
                useGPU_ = false;
            }
        } else {
            CLLM_WARN("[HFTransformer] GPU backend initialization failed, using CPU");
            useGPU_ = false;
        }
    }
    
    // 初始化 Per-Request KV Cache Pool
    kvCachePool_ = std::make_unique<KVCachePool>(
        kMaxConcurrentRequests,
        config_.numHiddenLayers,
        kMaxSeqLen,
        config_.getNumKVHeads(),
        headDim
    );
    
    // 初始化工作缓冲区池
    workBufferPool_ = std::make_unique<WorkBufferPool>(
        kMaxConcurrentRequests,
        config_.hiddenSize,
        config_.intermediateSize,
        config_.vocabSize,
        config_.numAttentionHeads,
        config_.getNumKVHeads(),
        headDim,
        kMaxSeqLen
    );
    
    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully (buffers pre-allocated, preconverted=%s, GPU=%s, maxConcurrent=%d)",
             usePreconvertedWeights_ ? "true" : "false",
             useGPU_ ? "true" : "false",
             kMaxConcurrentRequests);
}

HFTransformerModel::~HFTransformerModel() = default;

bool HFTransformerModel::loadWeights() {
    // 加载嵌入层
    embedTokens_ = static_cast<const uint16_t*>(
        loader_->getTensorData("model.embed_tokens.weight"));
    if (!embedTokens_) {
        CLLM_ERROR("[HFTransformer] Missing embed_tokens.weight");
        return false;
    }
    
    // 加载 LM Head（可能与 embed_tokens 共享）
    if (config_.tieWordEmbeddings) {
        lmHeadWeight_ = embedTokens_;
        CLLM_INFO("[HFTransformer] LM head tied with embed_tokens");
    } else {
        lmHeadWeight_ = static_cast<const uint16_t*>(
            loader_->getTensorData("lm_head.weight"));
        if (!lmHeadWeight_) {
            CLLM_ERROR("[HFTransformer] Missing lm_head.weight");
            return false;
        }
    }
    
    // 加载最终 norm
    finalNormWeight_ = static_cast<const uint16_t*>(
        loader_->getTensorData("model.norm.weight"));
    if (!finalNormWeight_) {
        CLLM_ERROR("[HFTransformer] Missing model.norm.weight");
        return false;
    }
    
    // 加载每一层
    layers_.resize(config_.numHiddenLayers);
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        std::string prefix = "model.layers." + std::to_string(i);
        LayerWeightsBF16& layer = layers_[i];
        
        layer.inputLayernorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".input_layernorm.weight"));
        layer.qProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.q_proj.weight"));
        layer.kProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.k_proj.weight"));
        layer.vProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.v_proj.weight"));
        layer.oProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.o_proj.weight"));
        layer.qNorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.q_norm.weight"));
        layer.kNorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".self_attn.k_norm.weight"));
        layer.postAttentionLayernorm = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".post_attention_layernorm.weight"));
        layer.gateProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.gate_proj.weight"));
        layer.upProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.up_proj.weight"));
        layer.downProj = static_cast<const uint16_t*>(
            loader_->getTensorData(prefix + ".mlp.down_proj.weight"));
        
        // 验证必需的权重
        if (!layer.inputLayernorm || !layer.qProj || !layer.kProj || !layer.vProj ||
            !layer.oProj || !layer.postAttentionLayernorm ||
            !layer.gateProj || !layer.upProj || !layer.downProj) {
            CLLM_ERROR("[HFTransformer] Missing weights for layer %d", i);
            return false;
        }
    }
    
    CLLM_INFO("[HFTransformer] All weights loaded");
    return true;
}

void HFTransformerModel::preconvertWeights() {
    CLLM_INFO("[HFTransformer] Pre-converting BF16 weights to F32...");
    
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int vocabSize = config_.vocabSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // 转换嵌入层
    size_t embedSize = static_cast<size_t>(vocabSize) * hiddenSize;
    embedTokensF32_.resize(embedSize);
    ggml_kernels::convert_bf16_to_f32(embedTokens_, embedTokensF32_.data(), embedSize);
    CLLM_DEBUG("[HFTransformer] Converted embed_tokens: %zu elements", embedSize);
    
    // 转换 LM Head（如果不共享）
    if (!config_.tieWordEmbeddings) {
        lmHeadWeightF32_.resize(embedSize);
        ggml_kernels::convert_bf16_to_f32(lmHeadWeight_, lmHeadWeightF32_.data(), embedSize);
    } else {
        // 共享时指向相同数据
        lmHeadWeightF32_ = embedTokensF32_;
    }
    
    // 转换最终 norm
    finalNormWeightF32_.resize(hiddenSize);
    ggml_kernels::convert_bf16_to_f32(finalNormWeight_, finalNormWeightF32_.data(), hiddenSize);
    
    // 转换每一层
    layersF32_.resize(config_.numHiddenLayers);
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        LayerWeightsF32& dst = layersF32_[i];
        const LayerWeightsBF16& src = layers_[i];
        
        // Attention norms
        dst.inputLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.inputLayernorm, dst.inputLayernorm.data(), hiddenSize);
        
        dst.postAttentionLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.postAttentionLayernorm, dst.postAttentionLayernorm.data(), hiddenSize);
        
        // Q/K/V projections
        dst.qProj.resize(qSize * hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.qProj, dst.qProj.data(), qSize * hiddenSize);
        
        dst.kProj.resize(kvSize * hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.kProj, dst.kProj.data(), kvSize * hiddenSize);
        
        dst.vProj.resize(kvSize * hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.vProj, dst.vProj.data(), kvSize * hiddenSize);
        
        dst.oProj.resize(hiddenSize * qSize);
        ggml_kernels::convert_bf16_to_f32(src.oProj, dst.oProj.data(), hiddenSize * qSize);

        // 融合 QKV 权重，减少 matmul 次数
        dst.qkvProj.resize((qSize + 2 * kvSize) * hiddenSize);
        std::memcpy(dst.qkvProj.data(),
                    dst.qProj.data(),
                    static_cast<size_t>(qSize) * hiddenSize * sizeof(float));
        std::memcpy(dst.qkvProj.data() + static_cast<size_t>(qSize) * hiddenSize,
                    dst.kProj.data(),
                    static_cast<size_t>(kvSize) * hiddenSize * sizeof(float));
        std::memcpy(dst.qkvProj.data() + static_cast<size_t>(qSize + kvSize) * hiddenSize,
                    dst.vProj.data(),
                    static_cast<size_t>(kvSize) * hiddenSize * sizeof(float));
        
        // Q/K Norm (optional)
        if (src.qNorm) {
            dst.qNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(src.qNorm, dst.qNorm.data(), headDim);
        }
        if (src.kNorm) {
            dst.kNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(src.kNorm, dst.kNorm.data(), headDim);
        }
        
        // FFN projections
        dst.gateProj.resize(intermediateSize * hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.gateProj, dst.gateProj.data(), intermediateSize * hiddenSize);
        
        dst.upProj.resize(intermediateSize * hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.upProj, dst.upProj.data(), intermediateSize * hiddenSize);
        
        dst.downProj.resize(hiddenSize * intermediateSize);
        ggml_kernels::convert_bf16_to_f32(src.downProj, dst.downProj.data(), hiddenSize * intermediateSize);

        // 融合 Gate/Up 权重，减少 matmul 次数
        dst.gateUpProj.resize(static_cast<size_t>(intermediateSize) * hiddenSize * 2);
        std::memcpy(dst.gateUpProj.data(),
                    dst.gateProj.data(),
                    static_cast<size_t>(intermediateSize) * hiddenSize * sizeof(float));
        std::memcpy(dst.gateUpProj.data() + static_cast<size_t>(intermediateSize) * hiddenSize,
                    dst.upProj.data(),
                    static_cast<size_t>(intermediateSize) * hiddenSize * sizeof(float));
        
        CLLM_DEBUG("[HFTransformer] Converted layer %d weights", i);
    }
    
    // 计算内存使用
    size_t totalBytes = embedSize * sizeof(float);
    if (!config_.tieWordEmbeddings) totalBytes += embedSize * sizeof(float);
    totalBytes += hiddenSize * sizeof(float);  // final norm
    
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        totalBytes += 2 * hiddenSize * sizeof(float);  // norms
        totalBytes += (qSize + 2 * kvSize + hiddenSize) * hiddenSize * sizeof(float);  // attn
        totalBytes += 3 * intermediateSize * hiddenSize * sizeof(float);  // ffn
    }
    
    CLLM_INFO("[HFTransformer] Pre-conversion complete: %.2f MB F32 weights",
             totalBytes / (1024.0 * 1024.0));
}

void HFTransformerModel::convertWeightsToFP16() {
    CLLM_INFO("[HFTransformer] Converting BF16 weights to FP16 (half memory)...");
    
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int vocabSize = config_.vocabSize;
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // BF16 -> FP32 -> FP16 转换（复用临时 FP32 缓冲区）
    auto convertBF16toFP16 = [](const uint16_t* bf16, uint16_t* fp16, size_t count) {
        // 分批处理，减少峰值内存
        static constexpr size_t BATCH_SIZE = 65536;
        std::vector<float> tmpF32(std::min(count, BATCH_SIZE));
        
        for (size_t offset = 0; offset < count; offset += BATCH_SIZE) {
            size_t batchCount = std::min(BATCH_SIZE, count - offset);
            
            // BF16 -> FP32
            for (size_t i = 0; i < batchCount; ++i) {
                uint32_t bits = static_cast<uint32_t>(bf16[offset + i]) << 16;
                std::memcpy(&tmpF32[i], &bits, sizeof(float));
            }
            
            // FP32 -> FP16
            quant_kernels::convert_f32_to_fp16(tmpF32.data(), fp16 + offset, batchCount);
        }
    };
    
    // 转换嵌入层
    size_t embedSize = static_cast<size_t>(vocabSize) * hiddenSize;
    embedTokensFP16_.resize(embedSize);
    convertBF16toFP16(embedTokens_, embedTokensFP16_.data(), embedSize);
    CLLM_DEBUG("[HFTransformer] Converted embed_tokens to FP16: %zu elements", embedSize);
    
    // 转换 LM Head
    if (!config_.tieWordEmbeddings) {
        lmHeadWeightFP16_.resize(embedSize);
        convertBF16toFP16(lmHeadWeight_, lmHeadWeightFP16_.data(), embedSize);
    } else {
        lmHeadWeightFP16_ = embedTokensFP16_;
    }
    
    // 转换最终 norm（norm 权重保持 FP32 精度，因为很小）
    finalNormWeightF32_.resize(hiddenSize);
    ggml_kernels::convert_bf16_to_f32(finalNormWeight_, finalNormWeightF32_.data(), hiddenSize);
    
    // 转换每一层
    layersFP16_.resize(config_.numHiddenLayers);
    layersF32_.resize(config_.numHiddenLayers);  // norm 权重仍需要 FP32
    
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        LayerWeightsFP16& dst16 = layersFP16_[i];
        LayerWeightsF32& dstF32 = layersF32_[i];
        const LayerWeightsBF16& src = layers_[i];
        
        // Norm 权重保持 FP32（小且对精度敏感）
        dstF32.inputLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.inputLayernorm, dstF32.inputLayernorm.data(), hiddenSize);
        
        dstF32.postAttentionLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.postAttentionLayernorm, dstF32.postAttentionLayernorm.data(), hiddenSize);
        
        // Q/K Norm (optional, 保持 FP32)
        if (src.qNorm) {
            dstF32.qNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(src.qNorm, dstF32.qNorm.data(), headDim);
        }
        if (src.kNorm) {
            dstF32.kNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(src.kNorm, dstF32.kNorm.data(), headDim);
        }
        
        // 大权重矩阵使用 FP16
        // Q/K/V/O projections
        dst16.qProj.resize(qSize * hiddenSize);
        convertBF16toFP16(src.qProj, dst16.qProj.data(), qSize * hiddenSize);
        
        dst16.kProj.resize(kvSize * hiddenSize);
        convertBF16toFP16(src.kProj, dst16.kProj.data(), kvSize * hiddenSize);
        
        dst16.vProj.resize(kvSize * hiddenSize);
        convertBF16toFP16(src.vProj, dst16.vProj.data(), kvSize * hiddenSize);
        
        dst16.oProj.resize(hiddenSize * qSize);
        convertBF16toFP16(src.oProj, dst16.oProj.data(), hiddenSize * qSize);
        
        // 融合 QKV（FP16）
        dst16.qkvProj.resize((qSize + 2 * kvSize) * hiddenSize);
        std::memcpy(dst16.qkvProj.data(),
                    dst16.qProj.data(),
                    static_cast<size_t>(qSize) * hiddenSize * sizeof(uint16_t));
        std::memcpy(dst16.qkvProj.data() + static_cast<size_t>(qSize) * hiddenSize,
                    dst16.kProj.data(),
                    static_cast<size_t>(kvSize) * hiddenSize * sizeof(uint16_t));
        std::memcpy(dst16.qkvProj.data() + static_cast<size_t>(qSize + kvSize) * hiddenSize,
                    dst16.vProj.data(),
                    static_cast<size_t>(kvSize) * hiddenSize * sizeof(uint16_t));
        
        // FFN projections (FP16)
        dst16.gateProj.resize(intermediateSize * hiddenSize);
        convertBF16toFP16(src.gateProj, dst16.gateProj.data(), intermediateSize * hiddenSize);
        
        dst16.upProj.resize(intermediateSize * hiddenSize);
        convertBF16toFP16(src.upProj, dst16.upProj.data(), intermediateSize * hiddenSize);
        
        dst16.downProj.resize(hiddenSize * intermediateSize);
        convertBF16toFP16(src.downProj, dst16.downProj.data(), hiddenSize * intermediateSize);
        
        // 融合 Gate/Up（FP16）
        dst16.gateUpProj.resize(static_cast<size_t>(intermediateSize) * hiddenSize * 2);
        std::memcpy(dst16.gateUpProj.data(),
                    dst16.gateProj.data(),
                    static_cast<size_t>(intermediateSize) * hiddenSize * sizeof(uint16_t));
        std::memcpy(dst16.gateUpProj.data() + static_cast<size_t>(intermediateSize) * hiddenSize,
                    dst16.upProj.data(),
                    static_cast<size_t>(intermediateSize) * hiddenSize * sizeof(uint16_t));
        
        CLLM_DEBUG("[HFTransformer] Converted layer %d to FP16", i);
    }
    
    // 计算内存使用（FP16 是 FP32 的一半）
    size_t totalBytes = embedSize * sizeof(uint16_t);
    if (!config_.tieWordEmbeddings) totalBytes += embedSize * sizeof(uint16_t);
    totalBytes += hiddenSize * sizeof(float);  // final norm (FP32)
    
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        totalBytes += 2 * hiddenSize * sizeof(float);  // norms (FP32)
        totalBytes += (qSize + 2 * kvSize + hiddenSize) * hiddenSize * sizeof(uint16_t);  // attn (FP16)
        totalBytes += 3 * static_cast<size_t>(intermediateSize) * hiddenSize * sizeof(uint16_t);  // ffn (FP16)
    }
    
    CLLM_INFO("[HFTransformer] FP16 conversion complete: %.2f MB (vs %.2f MB FP32)",
             totalBytes / (1024.0 * 1024.0),
             totalBytes * 2.0 / (1024.0 * 1024.0));
}

void HFTransformerModel::matmulFP16(const uint16_t* weight, const float* input,
                                     float* output, int outFeatures, int inFeatures) {
    quant_kernels::matmul_fp16_f32(weight, input, output, outFeatures, inFeatures);
}

void HFTransformerModel::matmulINT8(const int8_t* weight, const float* input,
                                     float* output, int outFeatures, int inFeatures,
                                     float scale, int32_t zeroPoint) {
    quant_kernels::matmul_int8_f32(weight, input, output, outFeatures, inFeatures, scale, zeroPoint);
}

void HFTransformerModel::convertWeightsToINT8() {
    CLLM_INFO("[HFTransformer] Converting BF16 weights to INT8 (quarter memory)...");
    
    const int hiddenSize = config_.hiddenSize;
    const int intermediateSize = config_.intermediateSize;
    const int vocabSize = config_.vocabSize;
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int headDim = hiddenSize / nHeads;
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // BF16 -> F32 -> INT8 转换辅助函数
    auto convertBF16toINT8 = [](const uint16_t* bf16, int8_t* int8, size_t count,
                                 float& outScale, int32_t& outZeroPoint) {
        // 先转为 F32
        std::vector<float> tmpF32(count);
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
            std::memcpy(&tmpF32[i], &bits, sizeof(float));
        }
        
        // 计算量化参数
        quant_kernels::compute_int8_params(tmpF32.data(), count, outScale, outZeroPoint);
        
        // 量化为 INT8
        quant_kernels::quantize_f32_to_int8(tmpF32.data(), int8, count, outScale, outZeroPoint);
    };
    
    // 注意：embedding 和 lm_head 保持 F32（查表操作，INT8 收益不大）
    // 只量化 projection 权重（计算密集型）
    
    // 预转换 embedding 到 F32（用于查表）
    size_t embedSize = static_cast<size_t>(vocabSize) * hiddenSize;
    embedTokensF32_.resize(embedSize);
    ggml_kernels::convert_bf16_to_f32(embedTokens_, embedTokensF32_.data(), embedSize);
    
    // 预转换 lm_head 到 F32
    if (!config_.tieWordEmbeddings) {
        lmHeadWeightF32_.resize(embedSize);
        ggml_kernels::convert_bf16_to_f32(lmHeadWeight_, lmHeadWeightF32_.data(), embedSize);
    } else {
        lmHeadWeightF32_ = embedTokensF32_;
    }
    
    // 预转换 final norm（保持 F32）
    finalNormWeightF32_.resize(hiddenSize);
    ggml_kernels::convert_bf16_to_f32(finalNormWeight_, finalNormWeightF32_.data(), hiddenSize);
    
    // 转换每一层的 projection 权重
    layersINT8_.resize(config_.numHiddenLayers);
    layersF32_.resize(config_.numHiddenLayers);  // norm 权重仍需要 F32
    
    size_t totalINT8Bytes = 0;
    
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        LayerWeightsINT8& dst8 = layersINT8_[i];
        LayerWeightsF32& dstF32 = layersF32_[i];
        const LayerWeightsBF16& src = layers_[i];
        
        // Norm 权重保持 F32（精度敏感）
        dstF32.inputLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.inputLayernorm, dstF32.inputLayernorm.data(), hiddenSize);
        
        dstF32.postAttentionLayernorm.resize(hiddenSize);
        ggml_kernels::convert_bf16_to_f32(src.postAttentionLayernorm, dstF32.postAttentionLayernorm.data(), hiddenSize);
        
        // Q/K norm 权重保持 F32
        if (src.qNorm && src.kNorm) {
            dstF32.qNorm.resize(headDim);
            dstF32.kNorm.resize(headDim);
            ggml_kernels::convert_bf16_to_f32(src.qNorm, dstF32.qNorm.data(), headDim);
            ggml_kernels::convert_bf16_to_f32(src.kNorm, dstF32.kNorm.data(), headDim);
        }
        
        // Q/K/V/O projections -> INT8
        size_t qProjSize = static_cast<size_t>(qSize) * hiddenSize;
        size_t kvProjSize = static_cast<size_t>(kvSize) * hiddenSize;
        size_t oProjSize = static_cast<size_t>(hiddenSize) * qSize;
        
        dst8.qProj.resize(qProjSize);
        convertBF16toINT8(src.qProj, dst8.qProj.data(), qProjSize, dst8.qProjScale, dst8.qProjZP);
        
        dst8.kProj.resize(kvProjSize);
        convertBF16toINT8(src.kProj, dst8.kProj.data(), kvProjSize, dst8.kProjScale, dst8.kProjZP);
        
        dst8.vProj.resize(kvProjSize);
        convertBF16toINT8(src.vProj, dst8.vProj.data(), kvProjSize, dst8.vProjScale, dst8.vProjZP);
        
        dst8.oProj.resize(oProjSize);
        convertBF16toINT8(src.oProj, dst8.oProj.data(), oProjSize, dst8.oProjScale, dst8.oProjZP);
        
        totalINT8Bytes += qProjSize + 2 * kvProjSize + oProjSize;
        
        // FFN projections -> INT8
        size_t ffnProjSize = static_cast<size_t>(intermediateSize) * hiddenSize;
        size_t downProjSize = static_cast<size_t>(hiddenSize) * intermediateSize;
        
        dst8.gateProj.resize(ffnProjSize);
        convertBF16toINT8(src.gateProj, dst8.gateProj.data(), ffnProjSize, dst8.gateProjScale, dst8.gateProjZP);
        
        dst8.upProj.resize(ffnProjSize);
        convertBF16toINT8(src.upProj, dst8.upProj.data(), ffnProjSize, dst8.upProjScale, dst8.upProjZP);
        
        dst8.downProj.resize(downProjSize);
        convertBF16toINT8(src.downProj, dst8.downProj.data(), downProjSize, dst8.downProjScale, dst8.downProjZP);
        
        totalINT8Bytes += 2 * ffnProjSize + downProjSize;
        
        CLLM_DEBUG("[HFTransformer] Converted layer %d to INT8", i);
    }
    
    // 计算总内存
    size_t totalF32Bytes = embedSize * sizeof(float);  // embed
    if (!config_.tieWordEmbeddings) totalF32Bytes += embedSize * sizeof(float);  // lm_head
    totalF32Bytes += hiddenSize * sizeof(float);  // final norm
    totalF32Bytes += config_.numHiddenLayers * 2 * hiddenSize * sizeof(float);  // layer norms
    
    size_t totalBytes = totalINT8Bytes + totalF32Bytes;
    size_t fp32Equivalent = totalINT8Bytes * 4 + totalF32Bytes;
    
    CLLM_INFO("[HFTransformer] INT8 conversion complete: %.2f MB (INT8: %.2f MB + F32: %.2f MB)",
             totalBytes / (1024.0 * 1024.0),
             totalINT8Bytes / (1024.0 * 1024.0),
             totalF32Bytes / (1024.0 * 1024.0));
    CLLM_INFO("[HFTransformer] Memory savings: %.2f MB -> %.2f MB (%.1f%% reduction)",
             fp32Equivalent / (1024.0 * 1024.0),
             totalBytes / (1024.0 * 1024.0),
             100.0 * (1.0 - static_cast<double>(totalBytes) / fp32Equivalent));
}

void HFTransformerModel::matmulF32(const float* weight, const float* input,
                                    float* output, int outFeatures, int inFeatures) {
    // 如果使用 GPU，则调用 GPU 版本
    if (deviceType_ == DeviceType::Metal && ggml_kernels::isGPUAvailable()) {
        ggml_kernels::matmul_gpu(weight, input, output, outFeatures, inFeatures);
    } else {
        ggml_kernels::matmul_f32(weight, input, output, outFeatures, inFeatures);
    }
}

void HFTransformerModel::resetKVCache() {
    kvCacheLen_ = 0;
    std::fill(kCache_.begin(), kCache_.end(), 0.0f);
    std::fill(vCache_.begin(), vCache_.end(), 0.0f);
}

std::vector<float> HFTransformerModel::forward(const std::vector<int32_t>& inputIds) {
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    int seqLen = static_cast<int>(inputIds.size());
    int startPos = kvCacheLen_;
    
    CLLM_DEBUG("[HFTransformer] Forward: seq_len=%d, start_pos=%d, token_id=%d, GPU=%s", 
               seqLen, startPos, inputIds.empty() ? -1 : inputIds[0],
               useGPU_ ? "true" : "false");
    
    // GPU forward 暂时禁用 - CPU BLAS 已经很快
    // 真正的 GPU 加速需要使用 GGML 计算图，但 API 复杂
    // 当前 CPU 实现约 25 tok/s，已达到目标
    // if (useGPU_ && gpuBackend_ && seqLen == 1) {
    //     auto logits = gpuBackend_->forward(inputIds[0], startPos);
    //     if (!logits.empty()) {
    //         kvCacheLen_ = startPos + 1;
    //         return logits;
    //     }
    // }
    
    // 目前只支持单 token 推理
    if (seqLen != 1) {
        CLLM_WARN("[HFTransformer] Currently only supports single token inference");
    }
    
    // CPU Forward 路径
    // Embedding
    embedding(inputIds, hiddenStates_);
    
    // Debug: 检查 embedding 统计
    {
        float minVal = hiddenStates_[0], maxVal = hiddenStates_[0], sum = 0;
        for (int i = 0; i < config_.hiddenSize; ++i) {
            if (hiddenStates_[i] < minVal) minVal = hiddenStates_[i];
            if (hiddenStates_[i] > maxVal) maxVal = hiddenStates_[i];
            sum += hiddenStates_[i];
        }
        CLLM_DEBUG("[HFTransformer] Embedding stats: min=%.4f, max=%.4f, mean=%.4f", 
                   minVal, maxVal, sum / config_.hiddenSize);
    }
    
    // 保存残差（使用 memcpy 更快）
    memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
    
    // Transformer 层
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        // 1. RMS Norm (input_layernorm)
        const float* normWeight = usePreconvertedWeights_ 
            ? layersF32_[i].inputLayernorm.data()
            : (bf16ToF32Array(layers_[i].inputLayernorm, normWeightBuffer_.data(), config_.hiddenSize),
               normWeightBuffer_.data());
        rmsNorm(hiddenStates_.data(), normWeight, normOutput_.data(), 
                config_.hiddenSize, config_.rmsNormEps);
        
        // 2. Self-Attention
        attention(i, normOutput_.data(), attnOutput_.data(), seqLen, startPos);
        
        // 3. Residual Add（使用 SIMD 优化）
        ggml_kernels::vector_add(residual_.data(), attnOutput_.data(), 
                                 hiddenStates_.data(), config_.hiddenSize);
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
        
        // 4. RMS Norm (post_attention_layernorm)
        const float* postNormWeight = usePreconvertedWeights_
            ? layersF32_[i].postAttentionLayernorm.data()
            : (bf16ToF32Array(layers_[i].postAttentionLayernorm, normWeightBuffer_.data(), config_.hiddenSize),
               normWeightBuffer_.data());
        rmsNorm(hiddenStates_.data(), postNormWeight, normOutput_.data(),
                config_.hiddenSize, config_.rmsNormEps);
        
        // 5. FFN
        ffn(i, normOutput_.data(), ffnOutput_.data());
        
        // 6. Residual Add
        ggml_kernels::vector_add(residual_.data(), ffnOutput_.data(),
                                 hiddenStates_.data(), config_.hiddenSize);
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
    }
    
    // Final RMS Norm
    const float* finalNormW = usePreconvertedWeights_
        ? finalNormWeightF32_.data()
        : (bf16ToF32Array(finalNormWeight_, normWeightBuffer_.data(), config_.hiddenSize),
           normWeightBuffer_.data());
    rmsNorm(hiddenStates_.data(), finalNormW, normOutput_.data(),
            config_.hiddenSize, config_.rmsNormEps);
    
    // LM Head
    std::vector<float> logits(config_.vocabSize);
    lmHead(normOutput_.data(), logits.data());
    
    // 更新 KV Cache 长度
    kvCacheLen_ += seqLen;
    
    return logits;
}

void HFTransformerModel::embedding(const std::vector<int32_t>& inputIds, 
                                   std::vector<float>& output) {
    // 只处理最后一个 token（或第一个，对于单 token 输入）
    int tokenId = inputIds.back();
    if (tokenId < 0 || tokenId >= config_.vocabSize) {
        CLLM_ERROR("[HFTransformer] Invalid token ID: %d", tokenId);
        std::fill(output.begin(), output.end(), 0.0f);
        return;
    }
    
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::FP16 && !embedTokensFP16_.empty()) {
            // 从 FP16 嵌入表查找并转为 FP32
            const size_t offset = static_cast<size_t>(tokenId) * config_.hiddenSize;
            if (offset + static_cast<size_t>(config_.hiddenSize) > embedTokensFP16_.size()) {
                CLLM_ERROR("[HFTransformer] FP16 embedding offset out of range: token=%d, offset=%zu, size=%zu",
                           tokenId, offset, embedTokensFP16_.size());
                std::fill(output.begin(), output.end(), 0.0f);
                return;
            }
            const uint16_t* embRow = embedTokensFP16_.data() + offset;
            quant_kernels::convert_fp16_to_f32(embRow, output.data(), config_.hiddenSize);
        } else {
            // 从预转换的 F32 嵌入表中查找（直接复制）
            const float* embRow = embedTokensF32_.data() + tokenId * config_.hiddenSize;
            std::copy(embRow, embRow + config_.hiddenSize, output.data());
        }
    } else {
        // 从 BF16 嵌入表中查找（需要转换）
        const uint16_t* embRow = embedTokens_ + tokenId * config_.hiddenSize;
        bf16ToF32Array(embRow, output.data(), config_.hiddenSize);
    }
}

void HFTransformerModel::rmsNorm(const float* input, const float* weight, 
                                  float* output, int size, float eps) {
    // 使用 SIMD 优化的 RMS Norm
    ggml_kernels::rms_norm(input, weight, output, size, eps);
}

void HFTransformerModel::attention(int layerIdx, const float* input, 
                                    float* output, int seqLen, int startPos) {
    const LayerWeightsBF16& layerBF16 = layers_[layerIdx];
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    
    // Q, K, V 投影 - 使用预分配缓冲区
    float* q = qBuffer_.data();
    float* k = kBuffer_.data();
    float* v = vBuffer_.data();
    
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::INT8 && !layersINT8_.empty()) {
            // INT8 路径
            const LayerWeightsINT8& layer8 = layersINT8_[layerIdx];
            matmulINT8(layer8.qProj.data(), input, q, qSize, config_.hiddenSize, layer8.qProjScale, layer8.qProjZP);
            matmulINT8(layer8.kProj.data(), input, k, kvSize, config_.hiddenSize, layer8.kProjScale, layer8.kProjZP);
            matmulINT8(layer8.vProj.data(), input, v, kvSize, config_.hiddenSize, layer8.vProjScale, layer8.vProjZP);
        } else if (quantType_ == QuantType::FP16 && !layersFP16_.empty()) {
            // FP16 路径
            const LayerWeightsFP16& layer16 = layersFP16_[layerIdx];
            if (!layer16.qkvProj.empty()) {
                float* qkv = qkvBuffer_.data();
                matmulFP16(layer16.qkvProj.data(), input, qkv, qSize + 2 * kvSize, config_.hiddenSize);
                q = qkv;
                k = qkv + qSize;
                v = k + kvSize;
            } else {
                matmulFP16(layer16.qProj.data(), input, q, qSize, config_.hiddenSize);
                matmulFP16(layer16.kProj.data(), input, k, kvSize, config_.hiddenSize);
                matmulFP16(layer16.vProj.data(), input, v, kvSize, config_.hiddenSize);
            }
        } else {
            // FP32 路径（默认）
            const LayerWeightsF32& layer = layersF32_[layerIdx];
            if (!layer.qkvProj.empty()) {
                float* qkv = qkvBuffer_.data();
                matmulF32(layer.qkvProj.data(), input, qkv, qSize + 2 * kvSize, config_.hiddenSize);
                q = qkv;
                k = qkv + qSize;
                v = k + kvSize;
            } else {
                matmulF32(layer.qProj.data(), input, q, qSize, config_.hiddenSize);
                matmulF32(layer.kProj.data(), input, k, kvSize, config_.hiddenSize);
                matmulF32(layer.vProj.data(), input, v, kvSize, config_.hiddenSize);
            }
        }
    } else {
        matmulBF16(layerBF16.qProj, input, q, qSize, config_.hiddenSize);
        matmulBF16(layerBF16.kProj, input, k, kvSize, config_.hiddenSize);
        matmulBF16(layerBF16.vProj, input, v, kvSize, config_.hiddenSize);
    }
    
    // Q/K Norm (Qwen3 特有) - 使用预分配缓冲区
    bool hasQKNorm = usePreconvertedWeights_ 
        ? !layersF32_[layerIdx].qNorm.empty()
        : (layerBF16.qNorm && layerBF16.kNorm);
    
    if (hasQKNorm) {
        const float* qNormW = usePreconvertedWeights_
            ? layersF32_[layerIdx].qNorm.data()
            : (bf16ToF32Array(layerBF16.qNorm, qkNormBuffer_.data(), headDim), qkNormBuffer_.data());
        
        // 对每个 Q 头进行 RMS Norm（SIMD）
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q + h * headDim;
            // 使用 dot_product 计算平方和
            float sumSq = ggml_kernels::dot_product(qHead, qHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + config_.rmsNormEps);
            // 应用 norm
            for (int i = 0; i < headDim; i += 4) {
                qHead[i] = qHead[i] * invRms * qNormW[i];
                qHead[i+1] = qHead[i+1] * invRms * qNormW[i+1];
                qHead[i+2] = qHead[i+2] * invRms * qNormW[i+2];
                qHead[i+3] = qHead[i+3] * invRms * qNormW[i+3];
            }
        }
        
        const float* kNormW = usePreconvertedWeights_
            ? layersF32_[layerIdx].kNorm.data()
            : (bf16ToF32Array(layerBF16.kNorm, qkNormBuffer_.data(), headDim), qkNormBuffer_.data());
        
        // 对每个 K 头进行 RMS Norm（SIMD）
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k + h * headDim;
            float sumSq = ggml_kernels::dot_product(kHead, kHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + config_.rmsNormEps);
            for (int i = 0; i < headDim; i += 4) {
                kHead[i] = kHead[i] * invRms * kNormW[i];
                kHead[i+1] = kHead[i+1] * invRms * kNormW[i+1];
                kHead[i+2] = kHead[i+2] * invRms * kNormW[i+2];
                kHead[i+3] = kHead[i+3] * invRms * kNormW[i+3];
            }
        }
    }
    
    // RoPE
    applyRoPE(q, k, headDim, nHeads, nKVHeads, seqLen, startPos);
    
    // 存储 K, V 到 cache（使用 memcpy 更快）
    const int cacheOffset = layerIdx * kMaxSeqLen * nKVHeads * headDim + startPos * nKVHeads * headDim;
    memcpy(kCache_.data() + cacheOffset, k, kvSize * sizeof(float));
    memcpy(vCache_.data() + cacheOffset, v, kvSize * sizeof(float));
    
    // Attention 计算
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    const int totalLen = startPos + seqLen;
    
    // 使用预分配缓冲区
    float* attnOut = attnOutBuffer_.data();
    std::fill(attnOut, attnOut + qSize, 0.0f);
    float* scores = attnScores_.data();
    
    // GQA: 每 gqa 个 Q 头共享一个 KV 头
    const int gqa = nHeads / nKVHeads;
    const float* kCacheBase = kCache_.data() + layerIdx * kMaxSeqLen * nKVHeads * headDim;
    const float* vCacheBase = vCache_.data() + layerIdx * kMaxSeqLen * nKVHeads * headDim;
    
    // 并行处理每个 attention head
    #pragma omp parallel for schedule(static) if(nHeads >= 4)
    for (int h = 0; h < nHeads; ++h) {
        const int kvHead = h / gqa;
        const float* qHead = q + h * headDim;
        float* localScores = scores + h * kMaxSeqLen;  // 每个 head 使用独立的 scores 缓冲区
        
        // 计算 attention scores + softmax（融合计算减少内存访问）
        float maxScore = -1e30f;
        
        // 第一遍：计算 scores 并找 max
        for (int t = 0; t < totalLen; ++t) {
            const float* kRow = kCacheBase + t * nKVHeads * headDim + kvHead * headDim;
            float dot = ggml_kernels::dot_product(qHead, kRow, headDim) * scale;
            localScores[t] = dot;
            maxScore = (dot > maxScore) ? dot : maxScore;
        }
        
        // 第二遍：exp 和 sum
        float sumExp = 0.0f;
        for (int t = 0; t < totalLen; ++t) {
            float e = std::exp(localScores[t] - maxScore);
            localScores[t] = e;
            sumExp += e;
        }
        
        // 第三遍：归一化
        const float invSum = 1.0f / sumExp;
        for (int t = 0; t < totalLen; ++t) {
            localScores[t] *= invSum;
        }
        
        // Weighted sum of V（SIMD 优化）
        float* outHead = attnOut + h * headDim;
        memset(outHead, 0, headDim * sizeof(float));
        
        const int vStride = nKVHeads * headDim;
        
#if USE_NEON
        // NEON 优化：4 个时间步 + 4 维度并行
        int t = 0;
        for (; t + 3 < totalLen; t += 4) {
            const float* v0 = vCacheBase + t * vStride + kvHead * headDim;
            const float* v1 = vCacheBase + (t+1) * vStride + kvHead * headDim;
            const float* v2 = vCacheBase + (t+2) * vStride + kvHead * headDim;
            const float* v3 = vCacheBase + (t+3) * vStride + kvHead * headDim;
            
            float32x4_t vw0 = vdupq_n_f32(localScores[t]);
            float32x4_t vw1 = vdupq_n_f32(localScores[t+1]);
            float32x4_t vw2 = vdupq_n_f32(localScores[t+2]);
            float32x4_t vw3 = vdupq_n_f32(localScores[t+3]);
            
            for (int d = 0; d < headDim; d += 4) {
                float32x4_t out = vld1q_f32(outHead + d);
                float32x4_t a0 = vld1q_f32(v0 + d);
                float32x4_t a1 = vld1q_f32(v1 + d);
                float32x4_t a2 = vld1q_f32(v2 + d);
                float32x4_t a3 = vld1q_f32(v3 + d);
                
                out = vmlaq_f32(out, a0, vw0);
                out = vmlaq_f32(out, a1, vw1);
                out = vmlaq_f32(out, a2, vw2);
                out = vmlaq_f32(out, a3, vw3);
                
                vst1q_f32(outHead + d, out);
            }
        }
        // 处理剩余时间步
        for (; t < totalLen; ++t) {
            const float* vRow = vCacheBase + t * vStride + kvHead * headDim;
            float32x4_t vw = vdupq_n_f32(localScores[t]);
            for (int d = 0; d < headDim; d += 4) {
                float32x4_t out = vld1q_f32(outHead + d);
                float32x4_t a = vld1q_f32(vRow + d);
                out = vmlaq_f32(out, a, vw);
                vst1q_f32(outHead + d, out);
            }
        }
#else
        // 标量回退（带循环展开）
        int t = 0;
        for (; t + 3 < totalLen; t += 4) {
            const float* v0 = vCacheBase + t * vStride + kvHead * headDim;
            const float* v1 = vCacheBase + (t+1) * vStride + kvHead * headDim;
            const float* v2 = vCacheBase + (t+2) * vStride + kvHead * headDim;
            const float* v3 = vCacheBase + (t+3) * vStride + kvHead * headDim;
            const float w0 = localScores[t], w1 = localScores[t+1];
            const float w2 = localScores[t+2], w3 = localScores[t+3];
            
            for (int d = 0; d < headDim; d += 4) {
                outHead[d] += w0*v0[d] + w1*v1[d] + w2*v2[d] + w3*v3[d];
                outHead[d+1] += w0*v0[d+1] + w1*v1[d+1] + w2*v2[d+1] + w3*v3[d+1];
                outHead[d+2] += w0*v0[d+2] + w1*v1[d+2] + w2*v2[d+2] + w3*v3[d+2];
                outHead[d+3] += w0*v0[d+3] + w1*v1[d+3] + w2*v2[d+3] + w3*v3[d+3];
            }
        }
        for (; t < totalLen; ++t) {
            const float* vRow = vCacheBase + t * vStride + kvHead * headDim;
            const float weight = localScores[t];
            for (int d = 0; d < headDim; d += 4) {
                outHead[d] += weight * vRow[d];
                outHead[d+1] += weight * vRow[d+1];
                outHead[d+2] += weight * vRow[d+2];
                outHead[d+3] += weight * vRow[d+3];
            }
        }
#endif
    }
    
    // O 投影
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::INT8 && !layersINT8_.empty()) {
            const LayerWeightsINT8& layer8 = layersINT8_[layerIdx];
            matmulINT8(layer8.oProj.data(), attnOut, output, config_.hiddenSize, qSize, layer8.oProjScale, layer8.oProjZP);
        } else if (quantType_ == QuantType::FP16 && !layersFP16_.empty()) {
            matmulFP16(layersFP16_[layerIdx].oProj.data(), attnOut, output, config_.hiddenSize, qSize);
        } else {
            matmulF32(layersF32_[layerIdx].oProj.data(), attnOut, output, config_.hiddenSize, qSize);
        }
    } else {
        matmulBF16(layerBF16.oProj, attnOut, output, config_.hiddenSize, qSize);
    }
}

void HFTransformerModel::ffn(int layerIdx, const float* input, float* output) {
    const int intermediateSize = config_.intermediateSize;
    const int hiddenSize = config_.hiddenSize;
    
    // Gate 和 Up 投影
    float* gate = gateBuffer_.data();
    float* up = upBuffer_.data();
    
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::INT8 && !layersINT8_.empty()) {
            // INT8 路径
            const LayerWeightsINT8& layer8 = layersINT8_[layerIdx];
            matmulINT8(layer8.gateProj.data(), input, gate, intermediateSize, hiddenSize, layer8.gateProjScale, layer8.gateProjZP);
            matmulINT8(layer8.upProj.data(), input, up, intermediateSize, hiddenSize, layer8.upProjScale, layer8.upProjZP);
            ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
            matmulINT8(layer8.downProj.data(), gate, output, hiddenSize, intermediateSize, layer8.downProjScale, layer8.downProjZP);
            return;
        } else if (quantType_ == QuantType::FP16 && !layersFP16_.empty()) {
            // FP16 路径
            const LayerWeightsFP16& layer16 = layersFP16_[layerIdx];
            if (!layer16.gateUpProj.empty()) {
                float* gateUp = gateUpBuffer_.data();
                matmulFP16(layer16.gateUpProj.data(), input, gateUp, intermediateSize * 2, hiddenSize);
                ggml_kernels::silu_mul_fused(gateUp, intermediateSize);
                matmulFP16(layer16.downProj.data(), gateUp, output, hiddenSize, intermediateSize);
                return;
            } else {
                matmulFP16(layer16.gateProj.data(), input, gate, intermediateSize, hiddenSize);
                matmulFP16(layer16.upProj.data(), input, up, intermediateSize, hiddenSize);
            }
        } else {
            // FP32 路径（默认）
            const LayerWeightsF32& layer = layersF32_[layerIdx];
            if (!layer.gateUpProj.empty()) {
                // 优化：使用融合 matmul + 原地 SiLU，消除 memcpy
                float* gateUp = gateUpBuffer_.data();
                matmulF32(layer.gateUpProj.data(), input, gateUp, intermediateSize * 2, hiddenSize);
                
                // 使用融合版本，直接在 gateUp 上操作，结果写入 gateUp[0:intermediateSize]
                ggml_kernels::silu_mul_fused(gateUp, intermediateSize);
                
                // Down 投影，直接从 gateUp 读取
                matmulF32(layer.downProj.data(), gateUp, output, hiddenSize, intermediateSize);
                return;
            } else {
                // 分离路径：顺序执行
                matmulF32(layer.gateProj.data(), input, gate, intermediateSize, hiddenSize);
                matmulF32(layer.upProj.data(), input, up, intermediateSize, hiddenSize);
            }
        }
    } else {
        const LayerWeightsBF16& layer = layers_[layerIdx];
        matmulBF16(layer.gateProj, input, gate, intermediateSize, hiddenSize);
        matmulBF16(layer.upProj, input, up, intermediateSize, hiddenSize);
    }
    
    // SwiGLU: silu(gate) * up - 使用 SIMD 优化
    ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
    
    // Down 投影
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::INT8 && !layersINT8_.empty()) {
            const LayerWeightsINT8& layer8 = layersINT8_[layerIdx];
            matmulINT8(layer8.downProj.data(), gate, output, hiddenSize, intermediateSize, layer8.downProjScale, layer8.downProjZP);
        } else if (quantType_ == QuantType::FP16 && !layersFP16_.empty()) {
            matmulFP16(layersFP16_[layerIdx].downProj.data(), gate, output, hiddenSize, intermediateSize);
        } else {
            matmulF32(layersF32_[layerIdx].downProj.data(), gate, output, hiddenSize, intermediateSize);
        }
    } else {
        matmulBF16(layers_[layerIdx].downProj, gate, output, hiddenSize, intermediateSize);
    }
}

void HFTransformerModel::lmHead(const float* input, float* output) {
    // 使用完整 BLAS 计算
    // 对于 vocabSize ~150K, hiddenSize 512，BLAS (cblas_sgemv) 可以在 <1ms 完成
    // 完整计算保证生成质量，避免稀疏采样可能的遗漏
    
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::FP16 && !lmHeadWeightFP16_.empty()) {
            matmulFP16(lmHeadWeightFP16_.data(), input, output, config_.vocabSize, config_.hiddenSize);
        } else {
            matmulF32(lmHeadWeightF32_.data(), input, output, config_.vocabSize, config_.hiddenSize);
        }
    } else {
        matmulBF16(lmHeadWeight_, input, output, config_.vocabSize, config_.hiddenSize);
    }
}

void HFTransformerModel::applyRoPE(float* q, float* k, int headDim, 
                                    int nHeads, int nKVHeads, int seqLen, int startPos) {
    const int halfDim = headDim / 2;
    
    // 对每个位置应用 RoPE
    for (int pos = 0; pos < seqLen; ++pos) {
        const int actualPos = startPos + pos;
        const float* cosPtr = ropeFreqsCos_.data() + actualPos * halfDim;
        const float* sinPtr = ropeFreqsSin_.data() + actualPos * halfDim;
        
        // Q 头
        for (int h = 0; h < nHeads; ++h) {
            float* head = q + h * headDim;
            for (int i = 0; i < halfDim; i += 2) {
                // 展开循环，减少循环开销
                float x0_0 = head[i], x1_0 = head[i + halfDim];
                float x0_1 = head[i+1], x1_1 = head[i+1 + halfDim];
                head[i] = x0_0 * cosPtr[i] - x1_0 * sinPtr[i];
                head[i + halfDim] = x0_0 * sinPtr[i] + x1_0 * cosPtr[i];
                head[i+1] = x0_1 * cosPtr[i+1] - x1_1 * sinPtr[i+1];
                head[i+1 + halfDim] = x0_1 * sinPtr[i+1] + x1_1 * cosPtr[i+1];
            }
        }
        
        // K 头
        for (int h = 0; h < nKVHeads; ++h) {
            float* head = k + h * headDim;
            for (int i = 0; i < halfDim; i += 2) {
                float x0_0 = head[i], x1_0 = head[i + halfDim];
                float x0_1 = head[i+1], x1_1 = head[i+1 + halfDim];
                head[i] = x0_0 * cosPtr[i] - x1_0 * sinPtr[i];
                head[i + halfDim] = x0_0 * sinPtr[i] + x1_0 * cosPtr[i];
                head[i+1] = x0_1 * cosPtr[i+1] - x1_1 * sinPtr[i+1];
                head[i+1 + halfDim] = x0_1 * sinPtr[i+1] + x1_1 * cosPtr[i+1];
            }
        }
    }
}

void HFTransformerModel::matmulBF16(const uint16_t* weight, const float* input, 
                                     float* output, int outFeatures, int inFeatures,
                                     int batchSize) {
    // weight: [outFeatures, inFeatures] in row-major (BF16)
    // input: [inFeatures] (F32)
    // output: [outFeatures] (F32)
    
    // 使用 SIMD 优化的 BF16 矩阵乘法
    ggml_kernels::matmul_bf16_f32(weight, input, output, outFeatures, inFeatures);
}

// ========== Per-Request KV Cache 支持 ==========

std::vector<float> HFTransformerModel::forwardWithRequestId(
    const std::vector<int32_t>& inputIds, size_t requestId) {
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    std::vector<float> logits(config_.vocabSize);
    forwardSingle(inputIds, requestId, logits);
    return logits;
}

std::vector<std::vector<float>> HFTransformerModel::forwardBatch(
    const std::vector<std::vector<int32_t>>& batchInputIds,
    const std::vector<size_t>& requestIds) {
    
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    size_t batchSize = batchInputIds.size();
    if (batchSize != requestIds.size()) {
        CLLM_ERROR("[HFTransformer] Batch size mismatch");
        return {};
    }
    
    std::vector<std::vector<float>> results(batchSize);
    
    // 并行处理每个请求
    #pragma omp parallel for schedule(dynamic) if(batchSize > 1)
    for (size_t i = 0; i < batchSize; ++i) {
        results[i].resize(config_.vocabSize);
        forwardSingle(batchInputIds[i], requestIds[i], results[i]);
    }
    
    return results;
}

void HFTransformerModel::forwardSingle(
    const std::vector<int32_t>& inputIds,
    size_t requestId,
    std::vector<float>& logits) {
    
    // 获取或分配 KV Cache 槽位
    KVCacheSlot* kvSlot = kvCachePool_->getOrAllocate(requestId);
    if (!kvSlot) {
        CLLM_ERROR("[HFTransformer] Failed to allocate KV cache for request %zu", requestId);
        return;
    }
    
    // 获取工作缓冲区
    WorkBufferSlot* workBuf = workBufferPool_->allocate();
    if (!workBuf) {
        CLLM_ERROR("[HFTransformer] Failed to allocate work buffer for request %zu", requestId);
        return;
    }
    
    int seqLen = static_cast<int>(inputIds.size());
    int startPos = kvSlot->currentLen;
    
    CLLM_DEBUG("[HFTransformer] ForwardSingle: request=%zu, seq_len=%d, start_pos=%d",
               requestId, seqLen, startPos);
    
    // Embedding
    embedding(inputIds, workBuf->hiddenStates);
    
    // 保存残差
    memcpy(workBuf->residual.data(), workBuf->hiddenStates.data(), 
           config_.hiddenSize * sizeof(float));
    
    // Transformer 层
    for (int i = 0; i < config_.numHiddenLayers; ++i) {
        // 1. RMS Norm (input_layernorm)
        const float* normWeight = layersF32_[i].inputLayernorm.data();
        rmsNorm(workBuf->hiddenStates.data(), normWeight, workBuf->normOutput.data(),
                config_.hiddenSize, config_.rmsNormEps);
        
        // 2. Self-Attention（使用独立 KV Cache）
        attentionWithKVCache(i, workBuf->normOutput.data(), workBuf->attnOutput.data(),
                             seqLen, startPos, kvSlot, workBuf);
        
        // 3. Residual Add
        ggml_kernels::vector_add(workBuf->residual.data(), workBuf->attnOutput.data(),
                                 workBuf->hiddenStates.data(), config_.hiddenSize);
        memcpy(workBuf->residual.data(), workBuf->hiddenStates.data(),
               config_.hiddenSize * sizeof(float));
        
        // 4. RMS Norm (post_attention_layernorm)
        const float* postNormWeight = layersF32_[i].postAttentionLayernorm.data();
        rmsNorm(workBuf->hiddenStates.data(), postNormWeight, workBuf->normOutput.data(),
                config_.hiddenSize, config_.rmsNormEps);
        
        // 5. FFN
        ffnWithBuffer(i, workBuf->normOutput.data(), workBuf->ffnOutput.data(), workBuf);
        
        // 6. Residual Add
        ggml_kernels::vector_add(workBuf->residual.data(), workBuf->ffnOutput.data(),
                                 workBuf->hiddenStates.data(), config_.hiddenSize);
        memcpy(workBuf->residual.data(), workBuf->hiddenStates.data(),
               config_.hiddenSize * sizeof(float));
    }
    
    // Final RMS Norm
    rmsNorm(workBuf->hiddenStates.data(), finalNormWeightF32_.data(),
            workBuf->normOutput.data(), config_.hiddenSize, config_.rmsNormEps);
    
    // LM Head（复用统一逻辑，避免 FP16 模式下使用未初始化的 F32 权重）
    lmHead(workBuf->normOutput.data(), logits.data());
    
    // 更新 KV Cache 长度
    kvSlot->currentLen += seqLen;
    
    // 释放工作缓冲区
    workBufferPool_->release(workBuf);
}

void HFTransformerModel::attentionWithKVCache(
    int layerIdx, const float* input, float* output,
    int seqLen, int startPos,
    KVCacheSlot* kvSlot, WorkBufferSlot* workBuf) {
    
    const int headDim = config_.getHeadDim();
    const int nHeads = config_.numAttentionHeads;
    const int nKVHeads = config_.getNumKVHeads();
    const int qSize = nHeads * headDim;
    const int kvSize = nKVHeads * headDim;
    const size_t perLayerCacheSize = kvCachePool_->perLayerCacheSize();
    
    // Q, K, V 投影
    float* q = workBuf->qBuffer.data();
    float* k = workBuf->kBuffer.data();
    float* v = workBuf->vBuffer.data();
    
    const LayerWeightsF32& layerF32 = layersF32_[layerIdx];
    const LayerWeightsBF16& layerBF16 = layers_[layerIdx];
    const LayerWeightsFP16* layer16 = (usePreconvertedWeights_ && quantType_ == QuantType::FP16 && !layersFP16_.empty())
        ? &layersFP16_[layerIdx]
        : nullptr;
    const LayerWeightsINT8* layer8 = (usePreconvertedWeights_ && quantType_ == QuantType::INT8 && !layersINT8_.empty())
        ? &layersINT8_[layerIdx]
        : nullptr;

    if (usePreconvertedWeights_) {
        if (layer8) {
            // INT8 路径
            matmulINT8(layer8->qProj.data(), input, q, qSize, config_.hiddenSize, layer8->qProjScale, layer8->qProjZP);
            matmulINT8(layer8->kProj.data(), input, k, kvSize, config_.hiddenSize, layer8->kProjScale, layer8->kProjZP);
            matmulINT8(layer8->vProj.data(), input, v, kvSize, config_.hiddenSize, layer8->vProjScale, layer8->vProjZP);
        } else if (layer16) {
            if (!layer16->qkvProj.empty()) {
                float* qkv = workBuf->qkvBuffer.data();
                matmulFP16(layer16->qkvProj.data(), input, qkv, qSize + 2 * kvSize, config_.hiddenSize);
                memcpy(q, qkv, qSize * sizeof(float));
                memcpy(k, qkv + qSize, kvSize * sizeof(float));
                memcpy(v, qkv + qSize + kvSize, kvSize * sizeof(float));
            } else {
                matmulFP16(layer16->qProj.data(), input, q, qSize, config_.hiddenSize);
                matmulFP16(layer16->kProj.data(), input, k, kvSize, config_.hiddenSize);
                matmulFP16(layer16->vProj.data(), input, v, kvSize, config_.hiddenSize);
            }
        } else {
            if (!layerF32.qkvProj.empty()) {
                float* qkv = workBuf->qkvBuffer.data();
                matmulF32(layerF32.qkvProj.data(), input, qkv, qSize + 2 * kvSize, config_.hiddenSize);
                memcpy(q, qkv, qSize * sizeof(float));
                memcpy(k, qkv + qSize, kvSize * sizeof(float));
                memcpy(v, qkv + qSize + kvSize, kvSize * sizeof(float));
            } else {
                matmulF32(layerF32.qProj.data(), input, q, qSize, config_.hiddenSize);
                matmulF32(layerF32.kProj.data(), input, k, kvSize, config_.hiddenSize);
                matmulF32(layerF32.vProj.data(), input, v, kvSize, config_.hiddenSize);
            }
        }
    } else {
        matmulBF16(layerBF16.qProj, input, q, qSize, config_.hiddenSize);
        matmulBF16(layerBF16.kProj, input, k, kvSize, config_.hiddenSize);
        matmulBF16(layerBF16.vProj, input, v, kvSize, config_.hiddenSize);
    }
    
    // Q/K Norm
    bool hasQKNorm = usePreconvertedWeights_ ? !layerF32.qNorm.empty() : (layerBF16.qNorm && layerBF16.kNorm);
    if (hasQKNorm) {
        const float* qNormW = nullptr;
        const float* kNormW = nullptr;
        std::vector<float> qNormTmp;
        std::vector<float> kNormTmp;

        if (usePreconvertedWeights_) {
            qNormW = layerF32.qNorm.data();
            kNormW = layerF32.kNorm.data();
        } else {
            qNormTmp.resize(headDim);
            kNormTmp.resize(headDim);
            bf16ToF32Array(layerBF16.qNorm, qNormTmp.data(), headDim);
            bf16ToF32Array(layerBF16.kNorm, kNormTmp.data(), headDim);
            qNormW = qNormTmp.data();
            kNormW = kNormTmp.data();
        }
        
        for (int h = 0; h < nHeads; ++h) {
            float* qHead = q + h * headDim;
            float sumSq = ggml_kernels::dot_product(qHead, qHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + config_.rmsNormEps);
            for (int i = 0; i < headDim; i += 4) {
                qHead[i] = qHead[i] * invRms * qNormW[i];
                qHead[i+1] = qHead[i+1] * invRms * qNormW[i+1];
                qHead[i+2] = qHead[i+2] * invRms * qNormW[i+2];
                qHead[i+3] = qHead[i+3] * invRms * qNormW[i+3];
            }
        }
        
        for (int h = 0; h < nKVHeads; ++h) {
            float* kHead = k + h * headDim;
            float sumSq = ggml_kernels::dot_product(kHead, kHead, headDim);
            float invRms = 1.0f / std::sqrt(sumSq / headDim + config_.rmsNormEps);
            for (int i = 0; i < headDim; i += 4) {
                kHead[i] = kHead[i] * invRms * kNormW[i];
                kHead[i+1] = kHead[i+1] * invRms * kNormW[i+1];
                kHead[i+2] = kHead[i+2] * invRms * kNormW[i+2];
                kHead[i+3] = kHead[i+3] * invRms * kNormW[i+3];
            }
        }
    }
    
    // RoPE
    applyRoPE(q, k, headDim, nHeads, nKVHeads, seqLen, startPos);
    
    // 存储 K, V 到独立的 KV Cache
    float* kCacheLayer = kvSlot->kCache.data() + layerIdx * perLayerCacheSize;
    float* vCacheLayer = kvSlot->vCache.data() + layerIdx * perLayerCacheSize;
    const int cacheOffset = startPos * nKVHeads * headDim;
    memcpy(kCacheLayer + cacheOffset, k, kvSize * sizeof(float));
    memcpy(vCacheLayer + cacheOffset, v, kvSize * sizeof(float));
    
    // Attention 计算
    const float scale = 1.0f / std::sqrt(static_cast<float>(headDim));
    const int totalLen = startPos + seqLen;
    
    float* attnOut = workBuf->attnOutBuffer.data();
    std::fill(attnOut, attnOut + qSize, 0.0f);
    float* scores = workBuf->attnScores.data();
    
    const int gqa = nHeads / nKVHeads;
    const int vStride = nKVHeads * headDim;
    
    for (int h = 0; h < nHeads; ++h) {
        const int kvHead = h / gqa;
        const float* qHead = q + h * headDim;
        float* localScores = scores + h * kMaxSeqLen;
        
        // 计算 attention scores
        float maxScore = -1e30f;
        for (int t = 0; t < totalLen; ++t) {
            const float* kRow = kCacheLayer + t * vStride + kvHead * headDim;
            float dot = ggml_kernels::dot_product(qHead, kRow, headDim) * scale;
            localScores[t] = dot;
            maxScore = (dot > maxScore) ? dot : maxScore;
        }
        
        // Softmax
        float sumExp = 0.0f;
        for (int t = 0; t < totalLen; ++t) {
            float e = std::exp(localScores[t] - maxScore);
            localScores[t] = e;
            sumExp += e;
        }
        const float invSum = 1.0f / sumExp;
        for (int t = 0; t < totalLen; ++t) {
            localScores[t] *= invSum;
        }
        
        // Weighted sum of V
        float* outHead = attnOut + h * headDim;
        memset(outHead, 0, headDim * sizeof(float));
        
#if USE_NEON
        int t = 0;
        for (; t + 3 < totalLen; t += 4) {
            const float* v0 = vCacheLayer + t * vStride + kvHead * headDim;
            const float* v1 = vCacheLayer + (t+1) * vStride + kvHead * headDim;
            const float* v2 = vCacheLayer + (t+2) * vStride + kvHead * headDim;
            const float* v3 = vCacheLayer + (t+3) * vStride + kvHead * headDim;
            
            float32x4_t vw0 = vdupq_n_f32(localScores[t]);
            float32x4_t vw1 = vdupq_n_f32(localScores[t+1]);
            float32x4_t vw2 = vdupq_n_f32(localScores[t+2]);
            float32x4_t vw3 = vdupq_n_f32(localScores[t+3]);
            
            for (int d = 0; d < headDim; d += 4) {
                float32x4_t out = vld1q_f32(outHead + d);
                float32x4_t a0 = vld1q_f32(v0 + d);
                float32x4_t a1 = vld1q_f32(v1 + d);
                float32x4_t a2 = vld1q_f32(v2 + d);
                float32x4_t a3 = vld1q_f32(v3 + d);
                
                out = vmlaq_f32(out, a0, vw0);
                out = vmlaq_f32(out, a1, vw1);
                out = vmlaq_f32(out, a2, vw2);
                out = vmlaq_f32(out, a3, vw3);
                
                vst1q_f32(outHead + d, out);
            }
        }
        for (; t < totalLen; ++t) {
            const float* vRow = vCacheLayer + t * vStride + kvHead * headDim;
            float32x4_t vw = vdupq_n_f32(localScores[t]);
            for (int d = 0; d < headDim; d += 4) {
                float32x4_t out = vld1q_f32(outHead + d);
                float32x4_t a = vld1q_f32(vRow + d);
                out = vmlaq_f32(out, a, vw);
                vst1q_f32(outHead + d, out);
            }
        }
#else
        for (int t = 0; t < totalLen; ++t) {
            const float* vRow = vCacheLayer + t * vStride + kvHead * headDim;
            const float weight = localScores[t];
            for (int d = 0; d < headDim; ++d) {
                outHead[d] += weight * vRow[d];
            }
        }
#endif
    }
    
    // O 投影
    if (usePreconvertedWeights_) {
        if (layer8) {
            matmulINT8(layer8->oProj.data(), attnOut, output, config_.hiddenSize, qSize, layer8->oProjScale, layer8->oProjZP);
        } else if (layer16) {
            matmulFP16(layer16->oProj.data(), attnOut, output, config_.hiddenSize, qSize);
        } else {
            matmulF32(layerF32.oProj.data(), attnOut, output, config_.hiddenSize, qSize);
        }
    } else {
        matmulBF16(layerBF16.oProj, attnOut, output, config_.hiddenSize, qSize);
    }
}

void HFTransformerModel::ffnWithBuffer(int layerIdx, const float* input, float* output,
                                       WorkBufferSlot* workBuf) {
    const int intermediateSize = config_.intermediateSize;
    const int hiddenSize = config_.hiddenSize;
    
    const LayerWeightsF32& layerF32 = layersF32_[layerIdx];
    const LayerWeightsBF16& layerBF16 = layers_[layerIdx];
    const LayerWeightsFP16* layer16 = (usePreconvertedWeights_ && quantType_ == QuantType::FP16 && !layersFP16_.empty())
        ? &layersFP16_[layerIdx]
        : nullptr;
    const LayerWeightsINT8* layer8 = (usePreconvertedWeights_ && quantType_ == QuantType::INT8 && !layersINT8_.empty())
        ? &layersINT8_[layerIdx]
        : nullptr;

    if (usePreconvertedWeights_) {
        if (layer8) {
            // INT8 路径
            float* gate = workBuf->gateBuffer.data();
            float* up = workBuf->upBuffer.data();
            matmulINT8(layer8->gateProj.data(), input, gate, intermediateSize, hiddenSize, layer8->gateProjScale, layer8->gateProjZP);
            matmulINT8(layer8->upProj.data(), input, up, intermediateSize, hiddenSize, layer8->upProjScale, layer8->upProjZP);
            ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
            matmulINT8(layer8->downProj.data(), gate, output, hiddenSize, intermediateSize, layer8->downProjScale, layer8->downProjZP);
        } else if (layer16) {
            if (!layer16->gateUpProj.empty()) {
                float* gateUp = workBuf->gateUpBuffer.data();
                matmulFP16(layer16->gateUpProj.data(), input, gateUp, intermediateSize * 2, hiddenSize);
                ggml_kernels::silu_mul_fused(gateUp, intermediateSize);
                matmulFP16(layer16->downProj.data(), gateUp, output, hiddenSize, intermediateSize);
            } else {
                float* gate = workBuf->gateBuffer.data();
                float* up = workBuf->upBuffer.data();
                matmulFP16(layer16->gateProj.data(), input, gate, intermediateSize, hiddenSize);
                matmulFP16(layer16->upProj.data(), input, up, intermediateSize, hiddenSize);
                ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
                matmulFP16(layer16->downProj.data(), gate, output, hiddenSize, intermediateSize);
            }
        } else {
            if (!layerF32.gateUpProj.empty()) {
                float* gateUp = workBuf->gateUpBuffer.data();
                matmulF32(layerF32.gateUpProj.data(), input, gateUp, intermediateSize * 2, hiddenSize);
                ggml_kernels::silu_mul_fused(gateUp, intermediateSize);
                matmulF32(layerF32.downProj.data(), gateUp, output, hiddenSize, intermediateSize);
            } else {
                float* gate = workBuf->gateBuffer.data();
                float* up = workBuf->upBuffer.data();
                matmulF32(layerF32.gateProj.data(), input, gate, intermediateSize, hiddenSize);
                matmulF32(layerF32.upProj.data(), input, up, intermediateSize, hiddenSize);
                ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
                matmulF32(layerF32.downProj.data(), gate, output, hiddenSize, intermediateSize);
            }
        }
    } else {
        float* gate = workBuf->gateBuffer.data();
        float* up = workBuf->upBuffer.data();
        matmulBF16(layerBF16.gateProj, input, gate, intermediateSize, hiddenSize);
        matmulBF16(layerBF16.upProj, input, up, intermediateSize, hiddenSize);
        ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
        matmulBF16(layerBF16.downProj, gate, output, hiddenSize, intermediateSize);
    }
}

void HFTransformerModel::releaseKVCache(size_t requestId) {
    if (kvCachePool_) {
        kvCachePool_->release(requestId);
    }
}

int HFTransformerModel::getAvailableKVSlots() const {
    return kvCachePool_ ? kvCachePool_->availableSlots() : 0;
}

int HFTransformerModel::getUsedKVSlots() const {
    return kvCachePool_ ? kvCachePool_->usedSlots() : 0;
}

} // namespace kylin
} // namespace cllm
