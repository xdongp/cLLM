/**
 * @file hf_transformer.cpp
 * @brief HuggingFace Transformer 模型实现
 * 
 * 使用 SIMD 优化内核加速推理
 */

#include "cllm/kylin/hf/transformer.h"
#include "cllm/kylin/core/ggml_kernels.h"
#include "cllm/kylin/backend/backend_interface.h"
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
    
    // 创建新的统一后端接口（实验性）
    backend_ = BackendFactory::create(device);
    if (backend_) {
        CLLM_INFO("[HFTransformer] Backend created: %s", backend_->getName().c_str());
        // 初始化后端（将在加载权重后完成）
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
    
    // 初始化新的后端接口（实验性）
    if (backend_) {
        backend_->initialize(config_);
        // TODO: 转换并加载权重到后端
        // ModelWeights weights = convertToModelWeights();
        // backend_->loadWeights(weights);
        CLLM_INFO("[HFTransformer] Backend initialized");
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
    // 注意：如果使用 GPU，需要 FP32 权重用于上传，即使 CPU 使用 FP16/INT8
    if (usePreconvertedWeights_) {
        if (quantType_ == QuantType::FP16) {
            convertWeightsToFP16();
            // GPU 需要 FP32 权重，所以还要预转换一份（避免 FP16->FP32 精度损失）
            if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
                CLLM_INFO("[HFTransformer] GPU mode with FP16: also preconverting to FP32 for GPU upload");
                preconvertWeights();
            }
        } else if (quantType_ == QuantType::INT8) {
            convertWeightsToINT8();
            // GPU 需要 FP32 权重，所以还要预转换一份
            if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
                CLLM_INFO("[HFTransformer] GPU mode with INT8: also preconverting to FP32 for GPU upload");
                preconvertWeights();
            }
        } else {
            preconvertWeights();  // 默认转为 FP32
        }
    }
    
    // 初始化 GPU 后端并上传权重
    if (gpuBackend_ && deviceType_ == DeviceType::Metal) {
        if (gpuBackend_->initialize(config_)) {
            // 准备层权重 - 直接使用 layersF32_ 中的 FP32 权重
            // 注意：在 FP16/INT8 模式下，convertWeightsToFP16/INT8() 已经创建了 layersF32_
            std::vector<LayerWeightsGPU> layerWeights(config_.numHiddenLayers);
            for (int i = 0; i < config_.numHiddenLayers; ++i) {
                layerWeights[i].inputLayernorm = layersF32_[i].inputLayernorm.data();
                layerWeights[i].postAttentionLayernorm = layersF32_[i].postAttentionLayernorm.data();
                layerWeights[i].qNorm = layersF32_[i].qNorm.empty() ? nullptr : layersF32_[i].qNorm.data();
                layerWeights[i].kNorm = layersF32_[i].kNorm.empty() ? nullptr : layersF32_[i].kNorm.data();
                
                // 直接使用 layersF32_ 中的 FP32 权重（无论是 FP16/INT8/FP32 模式都已准备好）
                // 注意：INT8 模式下如果没有调用 preconvertWeights()，这些指针可能为空！
                if (layersF32_[i].qProj.empty()) {
                    CLLM_ERROR("[HFTransformer] Layer %d qProj is empty! Cannot upload to GPU.", i);
                }
                layerWeights[i].qProj = layersF32_[i].qProj.data();
                layerWeights[i].kProj = layersF32_[i].kProj.data();
                layerWeights[i].vProj = layersF32_[i].vProj.data();
                layerWeights[i].oProj = layersF32_[i].oProj.data();
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
            
            // 临时缓冲区会在函数结束时自动释放
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

HFTransformerModel::~HFTransformerModel() {
    // 关闭后端
    if (backend_) {
        backend_->shutdown();
    }
}

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
    
    // 同时创建 FP32 版本用于 GPU 上传
    embedTokensF32_.resize(embedSize);
    quant_kernels::convert_fp16_to_f32(embedTokensFP16_.data(), embedTokensF32_.data(), embedSize);
    
    // 转换 LM Head
    if (!config_.tieWordEmbeddings) {
        lmHeadWeightFP16_.resize(embedSize);
        convertBF16toFP16(lmHeadWeight_, lmHeadWeightFP16_.data(), embedSize);
        // 同时创建 FP32 版本用于 GPU 上传
        lmHeadWeightF32_.resize(embedSize);
        quant_kernels::convert_fp16_to_f32(lmHeadWeightFP16_.data(), lmHeadWeightF32_.data(), embedSize);
    } else {
        lmHeadWeightFP16_ = embedTokensFP16_;
        lmHeadWeightF32_ = embedTokensF32_;  // 共享相同的 FP32 数据
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
        
        // 大权重矩阵使用 FP16，但同时保留 FP32 版本供 GPU 上传使用
        // Q/K/V/O projections
        size_t qProjSize = static_cast<size_t>(qSize) * hiddenSize;
        size_t kvProjSize = static_cast<size_t>(kvSize) * hiddenSize;
        size_t oProjSize = static_cast<size_t>(hiddenSize) * qSize;
        
        // FP16 版本（用于 CPU 推理）
        dst16.qProj.resize(qProjSize);
        convertBF16toFP16(src.qProj, dst16.qProj.data(), qProjSize);
        
        dst16.kProj.resize(kvProjSize);
        convertBF16toFP16(src.kProj, dst16.kProj.data(), kvProjSize);
        
        dst16.vProj.resize(kvProjSize);
        convertBF16toFP16(src.vProj, dst16.vProj.data(), kvProjSize);
        
        dst16.oProj.resize(oProjSize);
        convertBF16toFP16(src.oProj, dst16.oProj.data(), oProjSize);
        
        // FP32 版本（用于 GPU 上传）
        dstF32.qProj.resize(qProjSize);
        quant_kernels::convert_fp16_to_f32(dst16.qProj.data(), dstF32.qProj.data(), qProjSize);
        
        dstF32.kProj.resize(kvProjSize);
        quant_kernels::convert_fp16_to_f32(dst16.kProj.data(), dstF32.kProj.data(), kvProjSize);
        
        dstF32.vProj.resize(kvProjSize);
        quant_kernels::convert_fp16_to_f32(dst16.vProj.data(), dstF32.vProj.data(), kvProjSize);
        
        dstF32.oProj.resize(oProjSize);
        quant_kernels::convert_fp16_to_f32(dst16.oProj.data(), dstF32.oProj.data(), oProjSize);
        
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
        
        // FFN projections (FP16 + FP32)
        size_t ffnSize = static_cast<size_t>(intermediateSize) * hiddenSize;
        size_t downProjSize = static_cast<size_t>(hiddenSize) * intermediateSize;
        
        dst16.gateProj.resize(ffnSize);
        convertBF16toFP16(src.gateProj, dst16.gateProj.data(), ffnSize);
        
        dst16.upProj.resize(ffnSize);
        convertBF16toFP16(src.upProj, dst16.upProj.data(), ffnSize);
        
        dst16.downProj.resize(downProjSize);
        convertBF16toFP16(src.downProj, dst16.downProj.data(), downProjSize);
        
        // FP32 版本（用于 GPU 上传）
        dstF32.gateProj.resize(ffnSize);
        quant_kernels::convert_fp16_to_f32(dst16.gateProj.data(), dstF32.gateProj.data(), ffnSize);
        
        dstF32.upProj.resize(ffnSize);
        quant_kernels::convert_fp16_to_f32(dst16.upProj.data(), dstF32.upProj.data(), ffnSize);
        
        dstF32.downProj.resize(downProjSize);
        quant_kernels::convert_fp16_to_f32(dst16.downProj.data(), dstF32.downProj.data(), downProjSize);
        
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
    
    int actualTokenId = inputIds.empty() ? -1 : inputIds.back();
    CLLM_DEBUG("[HFTransformer] Forward: seq_len=%d, start_pos=%d, token_id=%d (last), first_token=%d, GPU=%s", 
               seqLen, startPos, actualTokenId, inputIds.empty() ? -1 : inputIds[0],
               useGPU_ ? "true" : "false");
    
    // GPU forward 已启用 - Flash Attention 形状问题已修复
    // 在生成模式下，我们总是只处理最后一个 token，因为之前的 token 已经在 KV Cache 中
    if (useGPU_ && gpuBackend_) {
        auto logits = gpuBackend_->forward(inputIds.back(), startPos);
        if (!logits.empty()) {
            kvCacheLen_ = startPos + 1;
            return logits;
        }
    }
    
    // CPU Forward 路径
    // 在生成模式下，只处理最后一个 token，因为之前的 token 已经在 KV Cache 中
    // 将 seqLen 设置为 1，确保 applyRoPE 不会循环多次
    seqLen = 1;
    
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
        int debugTokenId = inputIds.empty() ? -1 : inputIds.back();
        CLLM_INFO("[LAYER_DEBUG] Embedding: token_id=%d (last), min=%.6f, max=%.6f, mean=%.6f, device=%s",
                   debugTokenId, minVal, maxVal, sum / config_.hiddenSize,
                   useGPU_ ? "GPU" : "CPU");
        
        // 输出前 10 个 embedding 值用于对比
        CLLM_INFO("[LAYER_DEBUG] Embedding first 10 values: [%.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f, %.6f]",
                   hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4],
                   hiddenStates_[5], hiddenStates_[6], hiddenStates_[7], hiddenStates_[8], hiddenStates_[9]);
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

        // DEBUG: 打印 Layer 0 和 Layer 1 的 RMS Norm 前输入
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            CLLM_INFO("[CPU DEBUG] Layer %d Before RMS Norm (pos=%d) - Input first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4]);
        }

        // DEBUG: 计算并打印 RMS Norm 后的输出（未乘以权重）
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            float sumSq = 0.0f;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                sumSq += hiddenStates_[j] * hiddenStates_[j];
            }
            float scale = 1.0f / std::sqrt(sumSq / config_.hiddenSize + config_.rmsNormEps);
            CLLM_INFO("[CPU DEBUG] Layer %d RMS Norm scale (pos=%d, inv_rms): %.6f", i, startPos, scale);
            CLLM_INFO("[CPU DEBUG] Layer %d After RMS Norm (pos=%d, before weight mul) - Output first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, hiddenStates_[0] * scale, hiddenStates_[1] * scale, hiddenStates_[2] * scale,
                      hiddenStates_[3] * scale, hiddenStates_[4] * scale);
        }

        rmsNorm(hiddenStates_.data(), normWeight, normOutput_.data(),
                config_.hiddenSize, config_.rmsNormEps);

        // DEBUG: 打印 Layer 0 和 Layer 1 的 RMS Norm 后输出（乘以权重后）
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            CLLM_INFO("[CPU DEBUG] Layer %d After RMS Norm (pos=%d, after weight mul) - Output first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, normOutput_[0], normOutput_[1], normOutput_[2], normOutput_[3], normOutput_[4]);
        }

        // DEBUG: 保存所有层的 RMS Norm 后输出统计信息
        if (startPos == 0) {
            float minVal = normOutput_[0], maxVal = normOutput_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (normOutput_[j] < minVal) minVal = normOutput_[j];
                if (normOutput_[j] > maxVal) maxVal = normOutput_[j];
                sum += normOutput_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] RMS Norm Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, minVal, maxVal, mean, normOutput_[0], normOutput_[1], normOutput_[2], normOutput_[3], normOutput_[4]);
        }

        // 2. Self-Attention
        // DEBUG: 打印 Layer 0 和 Layer 1 的 Attention 输入 (position 0 和 1)
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            CLLM_INFO("[CPU DEBUG] Layer %d Attention Input (pos=%d) - Input first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, normOutput_[0], normOutput_[1], normOutput_[2], normOutput_[3], normOutput_[4]);
        }
        
        attention(i, normOutput_.data(), attnOutput_.data(), seqLen, startPos);

        // DEBUG: 打印 Layer 0 和 Layer 1 的 Attention 输出 (position 0 和 1)
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            CLLM_INFO("[CPU DEBUG] Layer %d Attention Output (pos=%d) - Output first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, attnOutput_[0], attnOutput_[1], attnOutput_[2], attnOutput_[3], attnOutput_[4]);
        }

        // DEBUG: Attention 输出统计
        if (startPos == 0) {
            float minVal = attnOutput_[0], maxVal = attnOutput_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (attnOutput_[j] < minVal) minVal = attnOutput_[j];
                if (attnOutput_[j] > maxVal) maxVal = attnOutput_[j];
                sum += attnOutput_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] Attention Output Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, minVal, maxVal, mean, attnOutput_[0], attnOutput_[1], attnOutput_[2], attnOutput_[3], attnOutput_[4]);
        }

        // 3. Residual Add（使用 SIMD 优化）
        ggml_kernels::vector_add(residual_.data(), attnOutput_.data(),
                                 hiddenStates_.data(), config_.hiddenSize);
        
        // DEBUG: 打印 Layer 0 和 Layer 1 的 Residual + Attention Output (position 0 和 1)
        if ((i == 0 || i == 1) && (startPos == 0 || startPos == 1)) {
            CLLM_INFO("[CPU DEBUG] Layer %d After Residual+Attention (pos=%d) - Output first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4]);
        }
        
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));

        // DEBUG: 第一次残差连接后统计
        if (startPos == 0) {
            float minVal = hiddenStates_[0], maxVal = hiddenStates_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (hiddenStates_[j] < minVal) minVal = hiddenStates_[j];
                if (hiddenStates_[j] > maxVal) maxVal = hiddenStates_[j];
                sum += hiddenStates_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] After 1st Residual Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, minVal, maxVal, mean, hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4]);
        }

        // 4. RMS Norm (post_attention_layernorm)
        const float* postNormWeight = usePreconvertedWeights_
            ? layersF32_[i].postAttentionLayernorm.data()
            : (bf16ToF32Array(layers_[i].postAttentionLayernorm, normWeightBuffer_.data(), config_.hiddenSize),
               normWeightBuffer_.data());
        rmsNorm(hiddenStates_.data(), postNormWeight, normOutput_.data(),
                config_.hiddenSize, config_.rmsNormEps);

        // DEBUG: Post-Attention RMS Norm 后统计
        if (startPos == 0) {
            float minVal = normOutput_[0], maxVal = normOutput_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (normOutput_[j] < minVal) minVal = normOutput_[j];
                if (normOutput_[j] > maxVal) maxVal = normOutput_[j];
                sum += normOutput_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] Post-Attention RMS Norm Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, minVal, maxVal, mean, normOutput_[0], normOutput_[1], normOutput_[2], normOutput_[3], normOutput_[4]);
        }

        // 5. FFN
        ffn(i, normOutput_.data(), ffnOutput_.data());

        // DEBUG: FFN 输出统计
        if (startPos == 0) {
            float minVal = ffnOutput_[0], maxVal = ffnOutput_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (ffnOutput_[j] < minVal) minVal = ffnOutput_[j];
                if (ffnOutput_[j] > maxVal) maxVal = ffnOutput_[j];
                sum += ffnOutput_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] FFN Output Stats: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, minVal, maxVal, mean, ffnOutput_[0], ffnOutput_[1], ffnOutput_[2], ffnOutput_[3], ffnOutput_[4]);
        }

        // 6. Residual Add
        ggml_kernels::vector_add(residual_.data(), ffnOutput_.data(),
                                 hiddenStates_.data(), config_.hiddenSize);
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));

        // DEBUG: 第二次残差连接后统计（层输出）
        if (startPos == 0 || startPos == 1) {
            float minVal = hiddenStates_[0], maxVal = hiddenStates_[0];
            double sum = 0;
            for (int j = 0; j < config_.hiddenSize; ++j) {
                if (hiddenStates_[j] < minVal) minVal = hiddenStates_[j];
                if (hiddenStates_[j] > maxVal) maxVal = hiddenStates_[j];
                sum += hiddenStates_[j];
            }
            float mean = sum / config_.hiddenSize;
            CLLM_INFO("[CPU Layer %2d] Layer Output Stats (pos=%d, after 2nd residual): min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      i, startPos, minVal, maxVal, mean, hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4]);
        }
    }
    
    // DEBUG: 最终隐藏状态统计（所有层之后）
    if (startPos == 0) {
        float minVal = hiddenStates_[0], maxVal = hiddenStates_[0];
        double sum = 0;
        for (int j = 0; j < config_.hiddenSize; ++j) {
            if (hiddenStates_[j] < minVal) minVal = hiddenStates_[j];
            if (hiddenStates_[j] > maxVal) maxVal = hiddenStates_[j];
            sum += hiddenStates_[j];
        }
        float mean = sum / config_.hiddenSize;
        CLLM_INFO("[CPU FINAL] Hidden States after all layers: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  minVal, maxVal, mean, hiddenStates_[0], hiddenStates_[1], hiddenStates_[2], hiddenStates_[3], hiddenStates_[4]);
    }

    // Final RMS Norm
    const float* finalNormW = usePreconvertedWeights_
        ? finalNormWeightF32_.data()
        : (bf16ToF32Array(finalNormWeight_, normWeightBuffer_.data(), config_.hiddenSize),
           normWeightBuffer_.data());
    rmsNorm(hiddenStates_.data(), finalNormW, normOutput_.data(),
            config_.hiddenSize, config_.rmsNormEps);

    // DEBUG: Final RMS Norm 后统计
    if (startPos == 0) {
        float minVal = normOutput_[0], maxVal = normOutput_[0];
        double sum = 0;
        for (int j = 0; j < config_.hiddenSize; ++j) {
            if (normOutput_[j] < minVal) minVal = normOutput_[j];
            if (normOutput_[j] > maxVal) maxVal = normOutput_[j];
            sum += normOutput_[j];
        }
        float mean = sum / config_.hiddenSize;
        CLLM_INFO("[CPU FINAL] After Final RMS Norm: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  minVal, maxVal, mean, normOutput_[0], normOutput_[1], normOutput_[2], normOutput_[3], normOutput_[4]);
    }

    // LM Head
    std::vector<float> logits(config_.vocabSize);
    lmHead(normOutput_.data(), logits.data());

    // DEBUG: Logits 统计
    if (startPos == 0) {
        float minVal = logits[0], maxVal = logits[0];
        double sum = 0;
        for (int j = 0; j < config_.vocabSize; ++j) {
            if (logits[j] < minVal) minVal = logits[j];
            if (logits[j] > maxVal) maxVal = logits[j];
            sum += logits[j];
        }
        float mean = sum / config_.vocabSize;
        CLLM_INFO("[CPU FINAL] Logits: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  minVal, maxVal, mean, logits[0], logits[1], logits[2], logits[3], logits[4]);
    }

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
    
    // Debug: Q/K/V 投影后的统计
    if (layerIdx == 0 || layerIdx == config_.numHiddenLayers - 1) {
        float qMin = q[0], qMax = q[0], qSum = 0;
        float kMin = k[0], kMax = k[0], kSum = 0;
        float vMin = v[0], vMax = v[0], vSum = 0;
        
        for (int i = 0; i < qSize; ++i) {
            if (q[i] < qMin) qMin = q[i];
            if (q[i] > qMax) qMax = q[i];
            qSum += q[i];
        }
        
        for (int i = 0; i < kvSize; ++i) {
            if (k[i] < kMin) kMin = k[i];
            if (k[i] > kMax) kMax = k[i];
            kSum += k[i];
            if (v[i] < vMin) vMin = v[i];
            if (v[i] > vMax) vMax = v[i];
            vSum += v[i];
        }
        
        CLLM_INFO("[LAYER_DEBUG] Layer %d QKV Projection: Q[min=%.6f,max=%.6f,mean=%.6f], K[min=%.6f,max=%.6f,mean=%.6f], V[min=%.6f,max=%.6f,mean=%.6f]",
                   layerIdx, qMin, qMax, qSum / qSize, kMin, kMax, kSum / kvSize, vMin, vMax, vSum / kvSize);
    }
    
    // DEBUG: 打印 Layer 0 和 Layer 1 的 Q/K Projection 前5个值 (position 0 和 1)
    if ((layerIdx == 0 || layerIdx == 1) && (startPos == 0 || startPos == 1)) {
        CLLM_INFO("[CPU DEBUG] Layer %d After Q/K Projection (pos=%d) - Q first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, q[0], q[1], q[2], q[3], q[4]);
        CLLM_INFO("[CPU DEBUG] Layer %d After Q/K Projection (pos=%d) - K first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, k[0], k[1], k[2], k[3], k[4]);
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
    
    // DEBUG: 打印 RoPE 前的 Q/K 值 (position 0 和 1 都打印)
    if ((layerIdx == 0 || layerIdx == 1) && (startPos == 0 || startPos == 1)) {
        CLLM_INFO("[CPU DEBUG] Layer %d Before RoPE (pos=%d) - Q first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, q[0], q[1], q[2], q[3], q[4]);
        CLLM_INFO("[CPU DEBUG] Layer %d Before RoPE (pos=%d) - Q halfDim=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, q[headDim/2], q[headDim/2+1], q[headDim/2+2], q[headDim/2+3], q[headDim/2+4]);
        CLLM_INFO("[CPU DEBUG] Layer %d Before RoPE (pos=%d) - K first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, k[0], k[1], k[2], k[3], k[4]);
        CLLM_INFO("[CPU DEBUG] Layer %d Before RoPE (pos=%d) - K halfDim=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, k[headDim/2], k[headDim/2+1], k[headDim/2+2], k[headDim/2+3], k[headDim/2+4]);
    }

    // RoPE
    applyRoPE(q, k, headDim, nHeads, nKVHeads, seqLen, startPos);

    // DEBUG: 打印 RoPE 后的 Q/K 值 (position 0 和 1 都打印)
    if ((layerIdx == 0 || layerIdx == 1) && (startPos == 0 || startPos == 1)) {
        CLLM_INFO("[CPU DEBUG] Layer %d After RoPE (pos=%d) - q=%p, Q first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, (void*)q, q[0], q[1], q[2], q[3], q[4]);
        CLLM_INFO("[CPU DEBUG] Layer %d After RoPE (pos=%d) - Q halfDim=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, q[headDim/2], q[headDim/2+1], q[headDim/2+2], q[headDim/2+3], q[headDim/2+4]);
        CLLM_INFO("[CPU DEBUG] Layer %d After RoPE (pos=%d) - K first 5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, k[0], k[1], k[2], k[3], k[4]);
        CLLM_INFO("[CPU DEBUG] Layer %d After RoPE (pos=%d) - K halfDim=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, k[headDim/2], k[headDim/2+1], k[headDim/2+2], k[headDim/2+3], k[headDim/2+4]);
    }
    
    // 存储 K, V 到 cache（使用 memcpy 更快）
    const int cacheOffset = layerIdx * kMaxSeqLen * nKVHeads * headDim + startPos * nKVHeads * headDim;
    memcpy(kCache_.data() + cacheOffset, k, kvSize * sizeof(float));
    memcpy(vCache_.data() + cacheOffset, v, kvSize * sizeof(float));
    
    // DEBUG: 打印 KV Cache 统计信息
    if ((layerIdx == 0 || layerIdx == 1) && startPos < 3) {
        float kMin = k[0], kMax = k[0];
        float vMin = v[0], vMax = v[0];
        double kSum = 0, vSum = 0;
        for (int i = 0; i < kvSize; ++i) {
            if (k[i] < kMin) kMin = k[i];
            if (k[i] > kMax) kMax = k[i];
            if (v[i] < vMin) vMin = v[i];
            if (v[i] > vMax) vMax = v[i];
            kSum += k[i];
            vSum += v[i];
        }
        CLLM_INFO("[CPU KV DEBUG] Layer %d position %d - K: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, kMin, kMax, kSum / kvSize, k[0], k[1], k[2], k[3], k[4]);
        CLLM_INFO("[CPU KV DEBUG] Layer %d position %d - V: min=%.6f, max=%.6f, mean=%.6f, first5=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layerIdx, startPos, vMin, vMax, vSum / kvSize, v[0], v[1], v[2], v[3], v[4]);
    }
    
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
    
    // Debug: Attention 输出统计
    if (layerIdx == 0 || layerIdx == config_.numHiddenLayers - 1) {
        float outMin = output[0], outMax = output[0], outSum = 0;
        for (int i = 0; i < config_.hiddenSize; ++i) {
            if (output[i] < outMin) outMin = output[i];
            if (output[i] > outMax) outMax = output[i];
            outSum += output[i];
        }
        CLLM_INFO("[LAYER_DEBUG] Layer %d Attention Output: min=%.6f, max=%.6f, mean=%.6f",
                   layerIdx, outMin, outMax, outSum / config_.hiddenSize);
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
    
    // Debug: FFN 输出统计
    if (layerIdx == 0 || layerIdx == config_.numHiddenLayers - 1) {
        float outMin = output[0], outMax = output[0], outSum = 0;
        for (int i = 0; i < config_.hiddenSize; ++i) {
            if (output[i] < outMin) outMin = output[i];
            if (output[i] > outMax) outMax = output[i];
            outSum += output[i];
        }
        CLLM_INFO("[LAYER_DEBUG] Layer %d FFN Output: min=%.6f, max=%.6f, mean=%.6f",
                   layerIdx, outMin, outMax, outSum / config_.hiddenSize);
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
    
    // DEBUG: 检查 logits 统计
    {
        float minVal = output[0], maxVal = output[0];
        double sum = 0;
        size_t nanCount = 0, infCount = 0;
        
        for (size_t i = 0; i < static_cast<size_t>(config_.vocabSize); ++i) {
            float v = output[i];
            if (std::isnan(v)) { nanCount++; continue; }
            if (std::isinf(v)) { infCount++; continue; }
            if (v < minVal) minVal = v;
            if (v > maxVal) maxVal = v;
            sum += v;
        }
        
        CLLM_DEBUG("[HFTransformer] LM Head logits: vocab=%d, min=%.4f, max=%.4f, mean=%.4f, NaN=%zu, Inf=%zu",
                   config_.vocabSize, minVal, maxVal, sum / config_.vocabSize, nanCount, infCount);
        
        // 检查关键 token 的 logit
        CLLM_DEBUG("  Key tokens: EOS(151645)=%.4f, BOS(151643)=%.4f, <|im_end|>=%.4f",
                   output[151645], output[151643], output[151645]);
        
        // 找到 top 5 tokens
        std::vector<std::pair<float, int>> topTokens;
        for (int i = 0; i < config_.vocabSize && i < 200000; ++i) {
            topTokens.push_back({output[i], i});
        }
        std::partial_sort(topTokens.begin(), topTokens.begin() + std::min(5, (int)topTokens.size()),
                          topTokens.end(), std::greater<>());
        CLLM_DEBUG("  Top 5 tokens: ");
        for (int i = 0; i < std::min(5, (int)topTokens.size()); ++i) {
            CLLM_DEBUG("    [%d]: logit=%.4f", topTokens[i].second, topTokens[i].first);
        }
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
        
        // DEBUG: 打印 Position 1 的 RoPE 频率值和计算过程
        if (actualPos == 1 && nHeads > 0) {
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE params: headDim=%d, halfDim=%d", headDim, halfDim);
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE freqs - cos[0..4]=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      cosPtr[0], cosPtr[1], cosPtr[2], cosPtr[3], cosPtr[4]);
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE freqs - sin[0..4]=[%.6f, %.6f, %.6f, %.6f, %.6f]",
                      sinPtr[0], sinPtr[1], sinPtr[2], sinPtr[3], sinPtr[4]);
            
            // DEBUG: 手动计算 i=0 的 RoPE 结果
            float* head = q;
            float x0 = head[0];
            float x1 = head[halfDim];
            float newX0 = x0 * cosPtr[0] - x1 * sinPtr[0];
            float newX1 = x0 * sinPtr[0] + x1 * cosPtr[0];
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE manual calc i=0: x0=%.6f, x1=%.6f (head[%d]), cos=%.6f, sin=%.6f",
                      x0, x1, halfDim, cosPtr[0], sinPtr[0]);
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE manual calc i=0: newX0=%.6f, newX1=%.6f",
                      newX0, newX1);
        }
        
        // Q 头 - 与 GGML 的 RoPE 实现保持一致
        // GGML 使用 i0 = 0, 2, 4, ... 作为频率索引
        // 每个元素 i 使用频率索引 i0 = 2*i，即 cosPtr[i] 对应频率 2*i/headDim
        for (int h = 0; h < nHeads; ++h) {
            float* head = q + h * headDim;
            // DEBUG: 打印第一个 head 的指针和值
            if (actualPos == 1 && h == 0) {
                CLLM_INFO("[CPU DEBUG] Position 1 RoPE loop h=0: head=%p, q=%p, head[0]=%.6f, head[64]=%.6f",
                          (void*)head, (void*)q, head[0], head[64]);
            }
            for (int i = 0; i < halfDim; ++i) {
                // 频率索引：GGML 使用 i0 = 2*i，所以 freqIdx = i
                const int freqIdx = i;
                float x0 = head[i];
                float x1 = head[i + halfDim];
                float newX0 = x0 * cosPtr[freqIdx] - x1 * sinPtr[freqIdx];
                float newX1 = x0 * sinPtr[freqIdx] + x1 * cosPtr[freqIdx];
                // DEBUG: 打印 i=0 的计算过程
                if (actualPos == 1 && h == 0 && i == 0) {
                    CLLM_INFO("[CPU DEBUG] Position 1 RoPE loop h=0 i=0: x0=%.6f, x1=%.6f, newX0=%.6f, newX1=%.6f",
                              x0, x1, newX0, newX1);
                }
                head[i] = newX0;
                head[i + halfDim] = newX1;
                // DEBUG: 打印 i=0 的赋值结果
                if (actualPos == 1 && h == 0 && i == 0) {
                    CLLM_INFO("[CPU DEBUG] Position 1 RoPE loop h=0 i=0 after: head[0]=%.6f, head[64]=%.6f",
                              head[0], head[64]);
                }
            }
        }
        
        // K 头
        for (int h = 0; h < nKVHeads; ++h) {
            float* head = k + h * headDim;
            for (int i = 0; i < halfDim; ++i) {
                const int freqIdx = i;
                float x0 = head[i];
                float x1 = head[i + halfDim];
                float newX0 = x0 * cosPtr[freqIdx] - x1 * sinPtr[freqIdx];
                float newX1 = x0 * sinPtr[freqIdx] + x1 * cosPtr[freqIdx];
                head[i] = newX0;
                head[i + halfDim] = newX1;
            }
        }
        
        // DEBUG: K头处理后打印q的值
        if (actualPos == 1 && nHeads > 0) {
            CLLM_INFO("[CPU DEBUG] Position 1 RoPE after K heads: q=%p, q[0]=%.6f, q[64]=%.6f",
                      (void*)q, q[0], q[64]);
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

int HFTransformerModel::getKVCacheCurrentLength(size_t requestId) const {
    if (!kvCachePool_) {
        return -1;
    }
    
    KVCacheSlot* slot = kvCachePool_->get(requestId);
    if (!slot) {
        return -1;  // 请求不存在
    }
    
    return slot->currentLen;
}

std::vector<std::vector<float>> HFTransformerModel::forwardBatchGPU(
    const std::vector<int>& tokenIds,
    const std::vector<int>& positions,
    const std::vector<size_t>& requestIds) {
    
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    if (tokenIds.size() != positions.size() || tokenIds.size() != requestIds.size()) {
        CLLM_ERROR("[HFTransformer] Input sizes mismatch");
        return {};
    }
    
    const size_t batchSize = tokenIds.size();
    if (batchSize == 0) {
        return {};
    }
    
    // 检查是否可以使用 GPU 后端进行批处理
    if (useGPU_ && gpuBackend_) {
        CLLM_INFO("[HFTransformer] Using GPU accelerated batch processing for %zu requests", batchSize);
        
        // 打印positions用于调试
        for (size_t i = 0; i < tokenIds.size(); ++i) {
            CLLM_INFO("[HFTransformer] Batch[%zu]: tokenId=%d, position=%d, requestId=%zu", 
                      i, tokenIds[i], positions[i], requestIds[i]);
        }
        
        // 使用 GPU 后端的批处理功能
        auto gpuResults = gpuBackend_->forwardBatch(tokenIds, positions);
        
        // 验证结果大小
        if (gpuResults.size() != batchSize) {
            CLLM_ERROR("[HFTransformer] GPU batch results size mismatch");
            return {};
        }
        
        return gpuResults;
    } else {
        CLLM_DEBUG("[HFTransformer] GPU not available, falling back to CPU batch processing");
        
        // 如果 GPU 不可用，使用 CPU 批处理
        std::vector<std::vector<float>> results(batchSize);
        
        #pragma omp parallel for schedule(dynamic) if(batchSize > 1)
        for (size_t i = 0; i < batchSize; ++i) {
            results[i] = forwardWithRequestId({tokenIds[i]}, requestIds[i]);
        }
        
        return results;
    }
}

// ============================================================================
// 调试功能：CPU 前向推理并导出中间结果
// ============================================================================
std::vector<float> HFTransformerModel::forwardWithDebugCPU(
    const std::vector<int32_t>& inputIds,
    std::vector<LayerDebugOutput>& layerOutputs,
    std::vector<float>& embeddingOutput,
    std::vector<float>& finalNormOutput
) {
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    int seqLen = static_cast<int>(inputIds.size());
    int startPos = kvCacheLen_;
    
    CLLM_INFO("[DEBUG_CPU] Starting forward with debug, seqLen=%d, startPos=%d", seqLen, startPos);
    
    // 清空输出容器
    layerOutputs.clear();
    layerOutputs.reserve(config_.numHiddenLayers);
    
    // Embedding
    embedding(inputIds, hiddenStates_);
    embeddingOutput = hiddenStates_;  // 保存 Embedding 输出
    
    CLLM_INFO("[DEBUG_CPU] Embedding output shape: [%d], first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]",
              config_.hiddenSize,
              embeddingOutput[0], embeddingOutput[1], embeddingOutput[2], 
              embeddingOutput[3], embeddingOutput[4]);
    
    // DEBUG: 验证权重
    if (usePreconvertedWeights_) {
        CLLM_INFO("[DEBUG_CPU] Layer 0 q_proj first 5 weights: [%.6f, %.6f, %.6f, %.6f, %.6f]",
                  layersF32_[0].qProj[0], layersF32_[0].qProj[1], layersF32_[0].qProj[2],
                  layersF32_[0].qProj[3], layersF32_[0].qProj[4]);
    }
    
    // 保存残差
    memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
    
    // Transformer 层
    for (int layerIdx = 0; layerIdx < config_.numHiddenLayers; ++layerIdx) {
        LayerDebugOutput layerDebug;
        layerDebug.layerIdx = layerIdx;
        
        // 1. RMS Norm (input_layernorm)
        const float* normWeight = usePreconvertedWeights_ 
            ? layersF32_[layerIdx].inputLayernorm.data()
            : (bf16ToF32Array(layers_[layerIdx].inputLayernorm, normWeightBuffer_.data(), config_.hiddenSize),
               normWeightBuffer_.data());
        rmsNorm(hiddenStates_.data(), normWeight, normOutput_.data(), 
                config_.hiddenSize, config_.rmsNormEps);
        layerDebug.inputNormOutput = normOutput_;  // 保存 Input Norm 输出
        
        // 2. QKV Projection
        int qSize = config_.numAttentionHeads * config_.getHeadDim();
        int kvSize = config_.getNumKVHeads() * config_.getHeadDim();
        std::vector<float> qkvOutput(qSize + 2 * kvSize);
        
        if (usePreconvertedWeights_ && !layersF32_[layerIdx].qkvProj.empty()) {
            // 使用融合的 QKV
            matmulF32(layersF32_[layerIdx].qkvProj.data(), normOutput_.data(), 
                     qkvOutput.data(), qSize + 2 * kvSize, config_.hiddenSize);
        } else {
            // 分别计算 Q, K, V
            std::vector<float> q(qSize), k(kvSize), v(kvSize);
            if (usePreconvertedWeights_) {
                matmulF32(layersF32_[layerIdx].qProj.data(), normOutput_.data(), 
                         q.data(), qSize, config_.hiddenSize);
                matmulF32(layersF32_[layerIdx].kProj.data(), normOutput_.data(), 
                         k.data(), kvSize, config_.hiddenSize);
                matmulF32(layersF32_[layerIdx].vProj.data(), normOutput_.data(), 
                         v.data(), kvSize, config_.hiddenSize);
            } else {
                matmulBF16(layers_[layerIdx].qProj, normOutput_.data(), 
                          q.data(), qSize, config_.hiddenSize);
                matmulBF16(layers_[layerIdx].kProj, normOutput_.data(), 
                          k.data(), kvSize, config_.hiddenSize);
                matmulBF16(layers_[layerIdx].vProj, normOutput_.data(), 
                          v.data(), kvSize, config_.hiddenSize);
            }
            memcpy(qkvOutput.data(), q.data(), qSize * sizeof(float));
            memcpy(qkvOutput.data() + qSize, k.data(), kvSize * sizeof(float));
            memcpy(qkvOutput.data() + qSize + kvSize, v.data(), kvSize * sizeof(float));
        }
        layerDebug.qkvOutput = qkvOutput;  // 保存 QKV 输出
        
        // 3. Self-Attention
        attention(layerIdx, normOutput_.data(), attnOutput_.data(), seqLen, startPos);
        layerDebug.attentionOutput.assign(attnOutput_.data(), 
                                          attnOutput_.data() + config_.hiddenSize);  // 保存 Attention 输出
        
        // 4. Residual Add
        ggml_kernels::vector_add(residual_.data(), attnOutput_.data(), 
                                 hiddenStates_.data(), config_.hiddenSize);
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
        
        // 5. RMS Norm (post_attention_layernorm)
        const float* postNormWeight = usePreconvertedWeights_
            ? layersF32_[layerIdx].postAttentionLayernorm.data()
            : (bf16ToF32Array(layers_[layerIdx].postAttentionLayernorm, normWeightBuffer_.data(), config_.hiddenSize),
               normWeightBuffer_.data());
        rmsNorm(hiddenStates_.data(), postNormWeight, normOutput_.data(),
                config_.hiddenSize, config_.rmsNormEps);
        layerDebug.postNormOutput = normOutput_;  // 保存 Post Norm 输出
        
        // 6. FFN
        ffn(layerIdx, normOutput_.data(), ffnOutput_.data());
        layerDebug.ffnOutput.assign(ffnOutput_.data(), 
                                    ffnOutput_.data() + config_.hiddenSize);  // 保存 FFN 输出
        
        // 7. Residual Add
        ggml_kernels::vector_add(residual_.data(), ffnOutput_.data(),
                                 hiddenStates_.data(), config_.hiddenSize);
        memcpy(residual_.data(), hiddenStates_.data(), config_.hiddenSize * sizeof(float));
        
        // 保存该层的所有输出
        layerOutputs.push_back(std::move(layerDebug));
        
        CLLM_INFO("[DEBUG_CPU] Layer %d completed", layerIdx);
    }
    
    // Final RMS Norm
    const float* finalNormW = usePreconvertedWeights_
        ? finalNormWeightF32_.data()
        : (bf16ToF32Array(finalNormWeight_, normWeightBuffer_.data(), config_.hiddenSize),
           normWeightBuffer_.data());
    rmsNorm(hiddenStates_.data(), finalNormW, normOutput_.data(),
            config_.hiddenSize, config_.rmsNormEps);
    finalNormOutput = normOutput_;  // 保存 Final Norm 输出
    
    CLLM_INFO("[DEBUG_CPU] Final norm output shape: [%d], first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]",
              config_.hiddenSize,
              finalNormOutput[0], finalNormOutput[1], finalNormOutput[2],
              finalNormOutput[3], finalNormOutput[4]);
    
    // LM Head
    std::vector<float> logits(config_.vocabSize);
    lmHead(normOutput_.data(), logits.data());
    
    CLLM_INFO("[DEBUG_CPU] Logits shape: [%d], first 5 values: [%.6f, %.6f, %.6f, %.6f, %.6f]",
              config_.vocabSize,
              logits[0], logits[1], logits[2], logits[3], logits[4]);
    
    // 更新 KV Cache 长度
    kvCacheLen_ = startPos + seqLen;
    
    return logits;
}

// ============================================================================
// 调试功能：GPU 前向推理并导出中间结果
// ============================================================================
std::vector<float> HFTransformerModel::forwardWithDebugGPU(
    const std::vector<int32_t>& inputIds,
    std::vector<GGMLGPUBackend::LayerOutput>& layerOutputs,
    std::vector<float>& embeddingOutput,
    std::vector<float>& finalNormOutput
) {
    if (!loaded_) {
        CLLM_ERROR("[HFTransformer] Model not loaded");
        return {};
    }
    
    if (!useGPU_ || !gpuBackend_) {
        CLLM_ERROR("[HFTransformer] GPU backend not available");
        return {};
    }
    
    int seqLen = static_cast<int>(inputIds.size());
    if (seqLen != 1) {
        CLLM_ERROR("[HFTransformer] GPU debug only supports single token inference");
        return {};
    }
    
    int startPos = kvCacheLen_;
    CLLM_INFO("[DEBUG_GPU] Starting forward with debug, tokenId=%d, startPos=%d", 
              inputIds[0], startPos);
    
    // 使用 GPU 后端的 forwardWithDebug
    auto logits = gpuBackend_->forwardWithDebug(
        inputIds[0], startPos,
        &layerOutputs,
        &embeddingOutput,
        &finalNormOutput
    );
    
    if (!logits.empty()) {
        kvCacheLen_ = startPos + 1;
        CLLM_INFO("[DEBUG_GPU] Forward completed successfully, logits shape: [%zu]", logits.size());
    } else {
        CLLM_ERROR("[DEBUG_GPU] Forward failed");
    }
    
    return logits;
}

} // namespace kylin
} // namespace cllm
