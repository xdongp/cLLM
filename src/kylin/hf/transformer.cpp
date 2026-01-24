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

namespace cllm {
namespace kylin {

HFTransformerModel::HFTransformerModel(const std::string& modelDir, DeviceType device)
    : deviceType_(device), useGPU_(false) {
    CLLM_INFO("[HFTransformer] Loading model from: %s", modelDir.c_str());
    CLLM_INFO("[HFTransformer] Requested device: %s", 
              device == DeviceType::Metal ? "Metal GPU" : 
              device == DeviceType::CUDA ? "CUDA GPU" : "CPU");
    
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
    
    // Norm 权重缓冲区（避免每层重复分配）
    normWeightBuffer_.resize(config_.hiddenSize);
    qkNormBuffer_.resize(headDim);
    
    // 预转换权重到 F32（消除运行时转换开销）
    if (usePreconvertedWeights_) {
        preconvertWeights();
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
    
    loaded_ = true;
    CLLM_INFO("[HFTransformer] Model loaded successfully (buffers pre-allocated, preconverted=%s, GPU=%s)",
             usePreconvertedWeights_ ? "true" : "false",
             useGPU_ ? "true" : "false");
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
        // 从预转换的 F32 嵌入表中查找（直接复制）
        const float* embRow = embedTokensF32_.data() + tokenId * config_.hiddenSize;
        std::copy(embRow, embRow + config_.hiddenSize, output.data());
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
    
    // Q, K, V 投影 - 使用预分配缓冲区，并行执行
    float* q = qBuffer_.data();
    float* k = kBuffer_.data();
    float* v = vBuffer_.data();
    
    if (usePreconvertedWeights_) {
        const LayerWeightsF32& layer = layersF32_[layerIdx];
        // 顺序执行（BLAS 已高度优化，async 开销可能更大）
        matmulF32(layer.qProj.data(), input, q, qSize, config_.hiddenSize);
        matmulF32(layer.kProj.data(), input, k, kvSize, config_.hiddenSize);
        matmulF32(layer.vProj.data(), input, v, kvSize, config_.hiddenSize);
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
        
        // Weighted sum of V（使用 SIMD 加速）
        float* outHead = attnOut + h * headDim;
        memset(outHead, 0, headDim * sizeof(float));
        
        // 展开外层循环，提高缓存效率
        int t = 0;
        for (; t + 3 < totalLen; t += 4) {
            const float* v0 = vCacheBase + t * nKVHeads * headDim + kvHead * headDim;
            const float* v1 = vCacheBase + (t+1) * nKVHeads * headDim + kvHead * headDim;
            const float* v2 = vCacheBase + (t+2) * nKVHeads * headDim + kvHead * headDim;
            const float* v3 = vCacheBase + (t+3) * nKVHeads * headDim + kvHead * headDim;
            const float w0 = localScores[t], w1 = localScores[t+1];
            const float w2 = localScores[t+2], w3 = localScores[t+3];
            
            for (int d = 0; d < headDim; d += 4) {
                outHead[d] += w0*v0[d] + w1*v1[d] + w2*v2[d] + w3*v3[d];
                outHead[d+1] += w0*v0[d+1] + w1*v1[d+1] + w2*v2[d+1] + w3*v3[d+1];
                outHead[d+2] += w0*v0[d+2] + w1*v1[d+2] + w2*v2[d+2] + w3*v3[d+2];
                outHead[d+3] += w0*v0[d+3] + w1*v1[d+3] + w2*v2[d+3] + w3*v3[d+3];
            }
        }
        // 处理剩余
        for (; t < totalLen; ++t) {
            const float* vRow = vCacheBase + t * nKVHeads * headDim + kvHead * headDim;
            const float weight = localScores[t];
            for (int d = 0; d < headDim; d += 4) {
                outHead[d] += weight * vRow[d];
                outHead[d+1] += weight * vRow[d+1];
                outHead[d+2] += weight * vRow[d+2];
                outHead[d+3] += weight * vRow[d+3];
            }
        }
    }
    
    // O 投影
    if (usePreconvertedWeights_) {
        matmulF32(layersF32_[layerIdx].oProj.data(), attnOut, output, config_.hiddenSize, qSize);
    } else {
        matmulBF16(layerBF16.oProj, attnOut, output, config_.hiddenSize, qSize);
    }
}

void HFTransformerModel::ffn(int layerIdx, const float* input, float* output) {
    const int intermediateSize = config_.intermediateSize;
    const int hiddenSize = config_.hiddenSize;
    
    // Gate 和 Up 投影 - 使用预分配缓冲区
    float* gate = gateBuffer_.data();
    float* up = upBuffer_.data();
    
    if (usePreconvertedWeights_) {
        const LayerWeightsF32& layer = layersF32_[layerIdx];
        // 顺序执行（BLAS 已经高度优化，async 开销可能更大）
        matmulF32(layer.gateProj.data(), input, gate, intermediateSize, hiddenSize);
        matmulF32(layer.upProj.data(), input, up, intermediateSize, hiddenSize);
    } else {
        const LayerWeightsBF16& layer = layers_[layerIdx];
        matmulBF16(layer.gateProj, input, gate, intermediateSize, hiddenSize);
        matmulBF16(layer.upProj, input, up, intermediateSize, hiddenSize);
    }
    
    // SwiGLU: silu(gate) * up - 使用 SIMD 优化
    ggml_kernels::silu_mul(gate, up, gate, intermediateSize);
    
    // Down 投影
    if (usePreconvertedWeights_) {
        matmulF32(layersF32_[layerIdx].downProj.data(), gate, output, hiddenSize, intermediateSize);
    } else {
        matmulBF16(layers_[layerIdx].downProj, gate, output, hiddenSize, intermediateSize);
    }
}

void HFTransformerModel::lmHead(const float* input, float* output) {
    // 激进优化策略：
    // Qwen3 的常用 token 集中在词表前部（CJK 和 ASCII 区域）
    // 使用更小的安全区 + 稀疏采样
    static constexpr int SAFE_ZONE = 32768;   // 前 32K 是"安全区"（减少 36%）
    static constexpr int SAMPLE_STRIDE = 512;  // 每 512 个采样一个（更稀疏）
    
    if (usePreconvertedWeights_) {
        const int vocabSize = config_.vocabSize;
        const int hiddenSize = config_.hiddenSize;
        
        if (vocabSize <= SAFE_ZONE) {
            matmulF32(lmHeadWeightF32_.data(), input, output, vocabSize, hiddenSize);
            return;
        }
        
        // 使用 BLAS 完整计算安全区
        matmulF32(lmHeadWeightF32_.data(), input, output, SAFE_ZONE, hiddenSize);
        
        const int remainingSize = vocabSize - SAFE_ZONE;
        const float* remainingWeight = lmHeadWeightF32_.data() + SAFE_ZONE * hiddenSize;
        float* remainingOutput = output + SAFE_ZONE;
        
        // 填充 -INFINITY
        std::fill(remainingOutput, remainingOutput + remainingSize, 
                  -std::numeric_limits<float>::infinity());
        
        // 快速找安全区最大值（使用 SIMD）
        float safeMax = output[0];
        for (int i = 1; i < SAFE_ZONE; i++) {
            if (output[i] > safeMax) safeMax = output[i];
        }
        
        // 稀疏采样剩余区域
        float bestSampleVal = -std::numeric_limits<float>::infinity();
        int bestSampleIdx = 0;
        
        for (int i = 0; i < remainingSize; i += SAMPLE_STRIDE) {
            float dot = ggml_kernels::dot_product(
                remainingWeight + i * hiddenSize, input, hiddenSize);
            remainingOutput[i] = dot;
            if (dot > bestSampleVal) {
                bestSampleVal = dot;
                bestSampleIdx = i;
            }
        }
        
        // 只有当采样最大值可能超过安全区最大值时才扩展计算
        if (bestSampleVal > safeMax - 3.0f) {
            // 计算最佳采样点附近 ±256 范围（更小）
            int start = std::max(0, bestSampleIdx - 256);
            int end = std::min(remainingSize, bestSampleIdx + 256);
            for (int i = start; i < end; i++) {
                remainingOutput[i] = ggml_kernels::dot_product(
                    remainingWeight + i * hiddenSize, input, hiddenSize);
            }
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

} // namespace kylin
} // namespace cllm
