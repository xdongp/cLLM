/**
 * @file gguf_loader.cpp
 * @brief GGUF 模型加载器实现
 */

#include "cllm/kylin/gguf/loader.h"
#include "cllm/kylin/core/tensor.h"
#include "cllm/kylin/core/quantization.h"
#include "cllm/common/logger.h"

#include <stdexcept>
#include <cstring>
#include <cstdio>
#include <limits>
#include <cmath>

namespace cllm {
namespace kylin {

GGUFLoader::GGUFLoader(const std::string& path)
    : path_(path)
    , ggufCtx_(nullptr)
    , dataCtx_(nullptr) {
    
    CLLM_INFO("[GGUFLoader] Loading GGUF file: %s", path.c_str());
    
    // 初始化参数：不自动分配张量数据
    struct gguf_init_params params = {
        .no_alloc = true,
        .ctx = &dataCtx_,
    };
    
    ggufCtx_ = gguf_init_from_file(path.c_str(), params);
    
    if (!ggufCtx_) {
        CLLM_ERROR("[GGUFLoader] Failed to load GGUF file: %s", path.c_str());
        throw std::runtime_error("Failed to load GGUF file: " + path);
    }
    
    CLLM_INFO("[GGUFLoader] GGUF version: %u, tensors: %lld, kv pairs: %lld",
              gguf_get_version(ggufCtx_),
              static_cast<long long>(gguf_get_n_tensors(ggufCtx_)),
              static_cast<long long>(gguf_get_n_kv(ggufCtx_)));
}

GGUFLoader::~GGUFLoader() {
    if (dataCtx_) {
        ggml_free(dataCtx_);
        dataCtx_ = nullptr;
    }
    if (ggufCtx_) {
        gguf_free(ggufCtx_);
        ggufCtx_ = nullptr;
    }
}

bool GGUFLoader::isValid() const {
    return ggufCtx_ != nullptr;
}

uint32_t GGUFLoader::getVersion() const {
    if (!ggufCtx_) return 0;
    return gguf_get_version(ggufCtx_);
}

std::string GGUFLoader::getArchKey(const std::string& arch, const std::string& suffix) const {
    return arch + "." + suffix;
}

GGUFModelConfig GGUFLoader::loadConfig() {
    if (!ggufCtx_) {
        throw std::runtime_error("GGUF context not initialized");
    }
    
    GGUFModelConfig config;
    
    // 获取架构名称
    auto archOpt = getMetaString("general.architecture");
    if (archOpt) {
        config.architecture = *archOpt;
    } else {
        CLLM_WARN("[GGUFLoader] No architecture specified, defaulting to 'llama'");
        config.architecture = "llama";
    }
    
    // 获取模型名称
    auto nameOpt = getMetaString("general.name");
    if (nameOpt) {
        config.name = *nameOpt;
    }
    
    CLLM_INFO("[GGUFLoader] Architecture: %s, Name: %s", 
              config.architecture.c_str(), 
              config.name.c_str());
    
    // 解析架构特定配置
    parseArchConfig(config, config.architecture);
    
    // 获取量化类型（从第一个张量推断）
    if (gguf_get_n_tensors(ggufCtx_) > 0) {
        config.quantType = gguf_get_tensor_type(ggufCtx_, 0);
    }
    
    CLLM_INFO("[GGUFLoader] Config loaded: embedding=%u, layers=%u, heads=%u, kv_heads=%u, vocab=%u",
              config.embeddingLength, config.blockCount, config.headCount, 
              config.headCountKV, config.vocabSize);
    
    return config;
}

void GGUFLoader::parseArchConfig(GGUFModelConfig& config, const std::string& arch) {
    // 上下文长度
    auto ctxLen = getMetaInt(getArchKey(arch, "context_length"));
    if (ctxLen) config.contextLength = static_cast<uint32_t>(*ctxLen);
    
    // 隐藏层维度
    auto embLen = getMetaInt(getArchKey(arch, "embedding_length"));
    if (embLen) config.embeddingLength = static_cast<uint32_t>(*embLen);
    
    // 层数
    auto blocks = getMetaInt(getArchKey(arch, "block_count"));
    if (blocks) config.blockCount = static_cast<uint32_t>(*blocks);
    
    // 注意力头数
    auto heads = getMetaInt(getArchKey(arch, "attention.head_count"));
    if (heads) config.headCount = static_cast<uint32_t>(*heads);
    
    // KV 头数（GQA）
    auto kvHeads = getMetaInt(getArchKey(arch, "attention.head_count_kv"));
    if (kvHeads) {
        config.headCountKV = static_cast<uint32_t>(*kvHeads);
    } else {
        config.headCountKV = config.headCount;  // 默认等于 Q 头数
    }
    
    // FFN 中间层维度
    auto ffLen = getMetaInt(getArchKey(arch, "feed_forward_length"));
    if (ffLen) config.feedForwardLength = static_cast<uint32_t>(*ffLen);
    
    // K/V 每头维度（用于 GQA 等架构）
    auto keyLen = getMetaInt(getArchKey(arch, "attention.key_length"));
    if (keyLen) {
        config.keyLength = static_cast<uint32_t>(*keyLen);
        CLLM_INFO("[GGUFLoader] Key length from metadata: %u", config.keyLength);
    }
    
    // RMS Norm epsilon
    auto rmsEps = getMetaFloat(getArchKey(arch, "attention.layer_norm_rms_epsilon"));
    if (rmsEps) config.rmsNormEps = *rmsEps;
    
    // RoPE 频率基数
    auto ropeBase = getMetaFloat(getArchKey(arch, "rope.freq_base"));
    if (ropeBase) config.ropeFreqBase = *ropeBase;
    
    // RoPE 类型 - 根据架构推断
    // GGML_ROPE_TYPE_NORMAL = 0, GGML_ROPE_TYPE_NEOX = 2
    // Qwen 系列和大多数现代模型使用 NEOX 风格
    if (arch == "qwen3" || arch == "qwen2" || arch == "qwen" ||
        arch == "qwen3moe" || arch == "qwen2moe" ||
        arch == "llama" || arch == "mistral" || arch == "gemma" ||
        arch == "phi3" || arch == "internlm2" || arch == "stablelm") {
        config.ropeType = 2;  // GGML_ROPE_TYPE_NEOX
        CLLM_INFO("[GGUFLoader] RoPE type set to NEOX (2) for architecture: %s", arch.c_str());
    } else {
        config.ropeType = 0;  // GGML_ROPE_TYPE_NORMAL
        CLLM_INFO("[GGUFLoader] RoPE type set to NORMAL (0) for architecture: %s", arch.c_str());
    }
    
    // 词表大小（多种来源）
    auto vocabSize = getMetaInt("tokenizer.ggml.vocab_size");
    if (vocabSize && *vocabSize > 0) {
        config.vocabSize = static_cast<uint32_t>(*vocabSize);
    } else {
        // 尝试从 tokenizer.ggml.tokens 数组长度获取
        int64_t tokensKeyId = gguf_find_key(ggufCtx_, "tokenizer.ggml.tokens");
        if (tokensKeyId >= 0) {
            size_t n = gguf_get_arr_n(ggufCtx_, tokensKeyId);
            if (n > 0) {
                config.vocabSize = static_cast<uint32_t>(n);
                CLLM_INFO("[GGUFLoader] Vocab size from tokens array: %u", config.vocabSize);
            }
        }
        
        // 如果仍然没有，尝试从 embedding 张量形状推断
        if (config.vocabSize == 0) {
            auto names = getTensorNames();
            for (const auto& name : names) {
                if (name.find("token_embd") != std::string::npos ||
                    name.find("embed_tokens") != std::string::npos) {
                    auto shape = getTensorShape(name);
                    if (!shape.empty() && shape[0] > 1000) {  // 合理的词表大小应该 > 1000
                        config.vocabSize = static_cast<uint32_t>(shape[0]);
                        CLLM_INFO("[GGUFLoader] Vocab size from embedding tensor: %u", config.vocabSize);
                        break;
                    }
                }
            }
        }
    }
}

std::optional<TokenizerInfo> GGUFLoader::getTokenizerInfo() {
    if (!ggufCtx_) return std::nullopt;
    
    TokenizerInfo info;
    
    // Tokenizer 类型
    auto model = getMetaString("tokenizer.ggml.model");
    if (model) {
        info.model = *model;
    } else {
        return std::nullopt;  // 没有内嵌 tokenizer
    }
    
    // 特殊 token IDs
    auto bos = getMetaInt("tokenizer.ggml.bos_token_id");
    if (bos) info.bosId = static_cast<int32_t>(*bos);
    
    auto eos = getMetaInt("tokenizer.ggml.eos_token_id");
    if (eos) info.eosId = static_cast<int32_t>(*eos);
    
    auto pad = getMetaInt("tokenizer.ggml.padding_token_id");
    if (pad) info.padId = static_cast<int32_t>(*pad);
    
    auto unk = getMetaInt("tokenizer.ggml.unknown_token_id");
    if (unk) info.unkId = static_cast<int32_t>(*unk);
    
    // 词表（tokens 数组）
    int64_t tokensKeyId = gguf_find_key(ggufCtx_, "tokenizer.ggml.tokens");
    if (tokensKeyId >= 0) {
        size_t n = gguf_get_arr_n(ggufCtx_, tokensKeyId);
        info.tokens.reserve(n);
        for (size_t i = 0; i < n; ++i) {
            const char* token = gguf_get_arr_str(ggufCtx_, tokensKeyId, i);
            if (token) {
                info.tokens.emplace_back(token);
            }
        }
    }
    
    // Token 分数
    int64_t scoresKeyId = gguf_find_key(ggufCtx_, "tokenizer.ggml.scores");
    if (scoresKeyId >= 0) {
        size_t n = gguf_get_arr_n(ggufCtx_, scoresKeyId);
        const float* scores = static_cast<const float*>(gguf_get_arr_data(ggufCtx_, scoresKeyId));
        if (scores) {
            info.scores.assign(scores, scores + n);
        }
    }
    
    CLLM_INFO("[GGUFLoader] Tokenizer info: model=%s, vocab=%zu, bos=%d, eos=%d",
              info.model.c_str(), info.tokens.size(), info.bosId, info.eosId);
    
    return info;
}

int64_t GGUFLoader::getTensorCount() const {
    if (!ggufCtx_) return 0;
    return gguf_get_n_tensors(ggufCtx_);
}

std::vector<std::string> GGUFLoader::getTensorNames() const {
    std::vector<std::string> names;
    if (!ggufCtx_) return names;
    
    int64_t n = gguf_get_n_tensors(ggufCtx_);
    names.reserve(static_cast<size_t>(n));
    
    for (int64_t i = 0; i < n; ++i) {
        const char* name = gguf_get_tensor_name(ggufCtx_, i);
        if (name) {
            names.emplace_back(name);
        }
    }
    
    return names;
}

ggml_type GGUFLoader::getTensorType(const std::string& name) const {
    if (!ggufCtx_) return GGML_TYPE_F32;
    
    int64_t id = gguf_find_tensor(ggufCtx_, name.c_str());
    if (id < 0) {
        CLLM_WARN("[GGUFLoader] Tensor not found: %s", name.c_str());
        return GGML_TYPE_F32;
    }
    
    return gguf_get_tensor_type(ggufCtx_, id);
}

std::vector<int64_t> GGUFLoader::getTensorShape(const std::string& name) const {
    std::vector<int64_t> shape;
    
    if (!ggufCtx_) return shape;
    
    // 首先尝试从 GGUF 张量信息获取
    int64_t tensorIdx = gguf_find_tensor(ggufCtx_, name.c_str());
    if (tensorIdx >= 0 && dataCtx_) {
        // 在数据上下文中查找张量（GGUF 会创建张量元信息）
        ggml_tensor* tensor = ggml_get_tensor(dataCtx_, name.c_str());
        if (tensor) {
            // 获取形状
            for (int i = 0; i < GGML_MAX_DIMS; ++i) {
                if (tensor->ne[i] > 1 || i == 0) {
                    shape.push_back(tensor->ne[i]);
                } else {
                    break;
                }
            }
        }
    }
    
    return shape;
}

void GGUFLoader::loadTensors(GGMLContext* ctx, std::map<std::string, ggml_tensor*>& tensors) {
    if (!ggufCtx_ || !ctx) {
        throw std::runtime_error("Invalid GGUF or GGML context");
    }
    
    auto names = getTensorNames();
    CLLM_INFO("[GGUFLoader] Loading %zu tensors...", names.size());
    
    for (const auto& name : names) {
        ggml_tensor* tensor = loadTensor(ctx, name);
        if (tensor) {
            tensors[name] = tensor;
        }
    }
    
    CLLM_INFO("[GGUFLoader] Loaded %zu tensors", tensors.size());
}

ggml_tensor* GGUFLoader::loadTensor(GGMLContext* ctx, const std::string& name) {
    if (!ggufCtx_ || !ctx || !dataCtx_) {
        return nullptr;
    }
    
    // 在 GGUF 数据上下文中查找张量
    ggml_tensor* srcTensor = ggml_get_tensor(dataCtx_, name.c_str());
    if (!srcTensor) {
        CLLM_DEBUG("[GGUFLoader] Tensor not found in data context: %s", name.c_str());
        return nullptr;
    }
    
    // 在目标上下文中创建张量（保持相同类型和形状）
    ggml_tensor* dstTensor = nullptr;
    
    int nDims = ggml_n_dims(srcTensor);
    switch (nDims) {
        case 1:
            dstTensor = ctx->newTensor1D(srcTensor->type, srcTensor->ne[0]);
            break;
        case 2:
            dstTensor = ctx->newTensor2D(srcTensor->type, srcTensor->ne[0], srcTensor->ne[1]);
            break;
        case 3:
            dstTensor = ctx->newTensor3D(srcTensor->type, srcTensor->ne[0], srcTensor->ne[1], srcTensor->ne[2]);
            break;
        case 4:
            dstTensor = ctx->newTensor4D(srcTensor->type, srcTensor->ne[0], srcTensor->ne[1], 
                                         srcTensor->ne[2], srcTensor->ne[3]);
            break;
        default:
            CLLM_WARN("[GGUFLoader] Unsupported tensor dimensions: %d", nDims);
            return nullptr;
    }
    
    if (!dstTensor) {
        CLLM_ERROR("[GGUFLoader] Failed to create tensor: %s", name.c_str());
        return nullptr;
    }
    
    // 设置张量名称
    ggml_set_name(dstTensor, name.c_str());
    
    // 从 GGUF 文件读取张量数据
    // 由于 no_alloc = true，srcTensor->data 为 nullptr，需要从文件读取
    int64_t tensorIdx = gguf_find_tensor(ggufCtx_, name.c_str());
    if (tensorIdx >= 0) {
        // 获取张量在文件中的偏移量
        size_t dataOffset = gguf_get_data_offset(ggufCtx_) + gguf_get_tensor_offset(ggufCtx_, tensorIdx);
        size_t dataSize = ggml_nbytes(srcTensor);
        
        // 从文件读取数据
        FILE* fp = fopen(path_.c_str(), "rb");
        if (fp) {
            fseek(fp, static_cast<long>(dataOffset), SEEK_SET);
            size_t bytesRead = fread(dstTensor->data, 1, dataSize, fp);
            fclose(fp);
            
            if (bytesRead != dataSize) {
                CLLM_ERROR("[GGUFLoader] Failed to read tensor data: %s (read %zu of %zu bytes)",
                          name.c_str(), bytesRead, dataSize);
                return nullptr;
            }
        } else {
            CLLM_ERROR("[GGUFLoader] Failed to open file for tensor data: %s", path_.c_str());
            return nullptr;
        }
    } else if (srcTensor->data) {
        // 如果有内存中的数据，直接复制
        size_t dataSize = ggml_nbytes(srcTensor);
        std::memcpy(dstTensor->data, srcTensor->data, dataSize);
    }
    
    CLLM_DEBUG("[GGUFLoader] Loaded tensor: %s, type=%s, shape=[%lld, %lld, %lld, %lld]",
               name.c_str(),
               GGMLContext::typeToString(srcTensor->type).c_str(),
               static_cast<long long>(srcTensor->ne[0]),
               static_cast<long long>(srcTensor->ne[1]),
               static_cast<long long>(srcTensor->ne[2]),
               static_cast<long long>(srcTensor->ne[3]));
    
    return dstTensor;
}

std::optional<std::string> GGUFLoader::getMetaString(const std::string& key) const {
    if (!ggufCtx_) return std::nullopt;
    
    int64_t keyId = gguf_find_key(ggufCtx_, key.c_str());
    if (keyId < 0) return std::nullopt;
    
    gguf_type type = gguf_get_kv_type(ggufCtx_, keyId);
    if (type != GGUF_TYPE_STRING) return std::nullopt;
    
    const char* val = gguf_get_val_str(ggufCtx_, keyId);
    if (!val) return std::nullopt;
    
    return std::string(val);
}

std::optional<int64_t> GGUFLoader::getMetaInt(const std::string& key) const {
    if (!ggufCtx_) return std::nullopt;
    
    int64_t keyId = gguf_find_key(ggufCtx_, key.c_str());
    if (keyId < 0) return std::nullopt;
    
    gguf_type type = gguf_get_kv_type(ggufCtx_, keyId);
    
    switch (type) {
        case GGUF_TYPE_UINT8:  return static_cast<int64_t>(gguf_get_val_u8(ggufCtx_, keyId));
        case GGUF_TYPE_INT8:   return static_cast<int64_t>(gguf_get_val_i8(ggufCtx_, keyId));
        case GGUF_TYPE_UINT16: return static_cast<int64_t>(gguf_get_val_u16(ggufCtx_, keyId));
        case GGUF_TYPE_INT16:  return static_cast<int64_t>(gguf_get_val_i16(ggufCtx_, keyId));
        case GGUF_TYPE_UINT32: return static_cast<int64_t>(gguf_get_val_u32(ggufCtx_, keyId));
        case GGUF_TYPE_INT32:  return static_cast<int64_t>(gguf_get_val_i32(ggufCtx_, keyId));
        case GGUF_TYPE_UINT64: return static_cast<int64_t>(gguf_get_val_u64(ggufCtx_, keyId));
        case GGUF_TYPE_INT64:  return gguf_get_val_i64(ggufCtx_, keyId);
        default: return std::nullopt;
    }
}

std::optional<float> GGUFLoader::getMetaFloat(const std::string& key) const {
    if (!ggufCtx_) return std::nullopt;
    
    int64_t keyId = gguf_find_key(ggufCtx_, key.c_str());
    if (keyId < 0) return std::nullopt;
    
    gguf_type type = gguf_get_kv_type(ggufCtx_, keyId);
    
    switch (type) {
        case GGUF_TYPE_FLOAT32: return gguf_get_val_f32(ggufCtx_, keyId);
        case GGUF_TYPE_FLOAT64: return static_cast<float>(gguf_get_val_f64(ggufCtx_, keyId));
        default: return std::nullopt;
    }
}

} // namespace kylin
} // namespace cllm

// ========== 模板方法实现 ==========

namespace cllm {
namespace kylin {

template<typename Tensor>
bool GGUFLoader::loadInto(
    Tensor& embedding,
    std::vector<Tensor>& wq,
    std::vector<Tensor>& wk,
    std::vector<Tensor>& wv,
    std::vector<Tensor>& wo,
    std::vector<Tensor>& wGate,
    std::vector<Tensor>& wUp,
    std::vector<Tensor>& wDown,
    std::vector<Tensor>& norm1,
    std::vector<Tensor>& norm2,
    Tensor& finalNorm,
    Tensor& lmHead
) {
    if (!ggufCtx_ || !dataCtx_) {
        CLLM_ERROR("[GGUFLoader::loadInto] GGUF context not initialized");
        return false;
    }
    
    CLLM_INFO("[GGUFLoader::loadInto] Loading GGUF weights into Kylin tensors");
    
    // Lambda: 加载并反量化张量
    auto loadAndDequantize = [&](const std::string& name, Tensor& dst) -> bool {
        // 在 dataCtx_ 中查找张量
        ggml_tensor* srcTensor = ggml_get_tensor(dataCtx_, name.c_str());
        if (!srcTensor) {
            CLLM_DEBUG("[GGUFLoader::loadInto] Tensor not found in dataCtx: %s", name.c_str());
            return false;
        }
        
        // 获取张量形状和类型
        ggml_type srcType = srcTensor->type;
        size_t numElements = static_cast<size_t>(ggml_nelements(srcTensor));
        
        // 打印形状信息用于调试
        CLLM_DEBUG("[GGUFLoader::loadInto] Tensor %s: type=%d, elements=%zu, shape=[%lld,%lld,%lld,%lld]",
                  name.c_str(), static_cast<int>(srcType), numElements,
                  srcTensor->ne[0], srcTensor->ne[1], srcTensor->ne[2], srcTensor->ne[3]);
        
        // 确保目标张量大小匹配
        if (dst.size() != numElements) {
            CLLM_ERROR("[GGUFLoader::loadInto] Tensor size mismatch: %s (src=%zu, dst=%zu)",
                      name.c_str(), numElements, dst.size());
            CLLM_ERROR("  src shape: [%lld,%lld,%lld,%lld]",
                      srcTensor->ne[0], srcTensor->ne[1], srcTensor->ne[2], srcTensor->ne[3]);
            return false;
        }
        
        // 从文件读取量化数据
        int64_t tensorIdx = gguf_find_tensor(ggufCtx_, name.c_str());
        if (tensorIdx < 0) {
            CLLM_ERROR("[GGUFLoader::loadInto] Tensor index not found: %s", name.c_str());
            return false;
        }
        
        size_t dataOffset = gguf_get_data_offset(ggufCtx_) + gguf_get_tensor_offset(ggufCtx_, tensorIdx);
        size_t dataSize = ggml_nbytes(srcTensor);
        
        // 分配临时缓冲区读取量化数据
        std::vector<uint8_t> quantizedData(dataSize);
        
        FILE* fp = fopen(path_.c_str(), "rb");
        if (!fp) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to open file: %s", path_.c_str());
            return false;
        }
        
        fseek(fp, static_cast<long>(dataOffset), SEEK_SET);
        size_t bytesRead = fread(quantizedData.data(), 1, dataSize, fp);
        fclose(fp);
        
        if (bytesRead != dataSize) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to read tensor data: %s (read %zu of %zu bytes)",
                      name.c_str(), bytesRead, dataSize);
            return false;
        }
        
        // 根据类型反量化
        if (srcType == GGML_TYPE_F32) {
            // 已经是 FP32，直接复制
            std::memcpy(dst.data(), quantizedData.data(), numElements * sizeof(float));
        } else if (srcType == GGML_TYPE_F16) {
            // FP16 转 FP32
            const ggml_fp16_t* src = reinterpret_cast<const ggml_fp16_t*>(quantizedData.data());
            for (size_t i = 0; i < numElements; ++i) {
                dst[i] = ggml_fp16_to_fp32(src[i]);
            }
        } else if (srcType == GGML_TYPE_Q4_K) {
            // Q4_K 反量化
            quantization::dequantize_q4_K_to_f32(quantizedData.data(), dst.data(), numElements);
        } else if (srcType == GGML_TYPE_Q6_K) {
            // Q6_K 反量化
            quantization::dequantize_q6_K_to_f32(quantizedData.data(), dst.data(), numElements);
        } else {
            CLLM_ERROR("[GGUFLoader::loadInto] Unsupported quantization type for tensor %s: %d",
                      name.c_str(), static_cast<int>(srcType));
            return false;
        }
        
        // 验证反量化后的数据
        size_t nanCount = 0;
        size_t infCount = 0;
        float minVal = std::numeric_limits<float>::max();
        float maxVal = std::numeric_limits<float>::lowest();
        
        for (size_t i = 0; i < std::min(numElements, size_t(100)); ++i) {
            float val = dst[i];
            if (std::isnan(val)) nanCount++;
            else if (std::isinf(val)) infCount++;
            else {
                minVal = std::min(minVal, val);
                maxVal = std::max(maxVal, val);
            }
        }
        
        if (nanCount > 0 || infCount > 0) {
            CLLM_WARN("[GGUFLoader::loadInto] Tensor %s contains %zu NaN, %zu Inf (sampled first 100)",
                     name.c_str(), nanCount, infCount);
        }
        
        CLLM_DEBUG("[GGUFLoader::loadInto] ✓ Loaded %s: %zu elements, type=%s, range=[%.4f, %.4f]",
                  name.c_str(), numElements, GGMLContext::typeToString(srcType).c_str(),
                  minVal, maxVal);
        return true;
    };
    
    // 1. 加载 token embedding
    if (!loadAndDequantize("token_embd.weight", embedding)) {
        // 尝试备用名称
        if (!loadAndDequantize("model.embed_tokens.weight", embedding)) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load token embedding");
            return false;
        }
    }
    
    // 2. 加载每层权重
    const size_t numLayers = wq.size();
    for (size_t i = 0; i < numLayers; ++i) {
        std::string prefix = "blk." + std::to_string(i);
        std::string altPrefix = "model.layers." + std::to_string(i);
        
        // Attention权重
        if (!loadAndDequantize(prefix + ".attn_q.weight", wq[i]) &&
            !loadAndDequantize(altPrefix + ".self_attn.q_proj.weight", wq[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Q weight for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".attn_k.weight", wk[i]) &&
            !loadAndDequantize(altPrefix + ".self_attn.k_proj.weight", wk[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load K weight for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".attn_v.weight", wv[i]) &&
            !loadAndDequantize(altPrefix + ".self_attn.v_proj.weight", wv[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load V weight for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".attn_output.weight", wo[i]) &&
            !loadAndDequantize(altPrefix + ".self_attn.o_proj.weight", wo[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load O weight for layer %zu", i);
            return false;
        }
        
        // FFN权重
        if (!loadAndDequantize(prefix + ".ffn_gate.weight", wGate[i]) &&
            !loadAndDequantize(altPrefix + ".mlp.gate_proj.weight", wGate[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Gate weight for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".ffn_up.weight", wUp[i]) &&
            !loadAndDequantize(altPrefix + ".mlp.up_proj.weight", wUp[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Up weight for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".ffn_down.weight", wDown[i]) &&
            !loadAndDequantize(altPrefix + ".mlp.down_proj.weight", wDown[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Down weight for layer %zu", i);
            return false;
        }
        
        // Norm权重
        if (!loadAndDequantize(prefix + ".attn_norm.weight", norm1[i]) &&
            !loadAndDequantize(altPrefix + ".input_layernorm.weight", norm1[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Norm1 for layer %zu", i);
            return false;
        }
        
        if (!loadAndDequantize(prefix + ".ffn_norm.weight", norm2[i]) &&
            !loadAndDequantize(altPrefix + ".post_attention_layernorm.weight", norm2[i])) {
            CLLM_ERROR("[GGUFLoader::loadInto] Failed to load Norm2 for layer %zu", i);
            return false;
        }
        
        if ((i + 1) % 10 == 0 || i == numLayers - 1) {
            CLLM_INFO("[GGUFLoader::loadInto] Loaded layer %zu/%zu", i + 1, numLayers);
        }
    }
    
    // 3. 加载 final norm
    if (!loadAndDequantize("output_norm.weight", finalNorm) &&
        !loadAndDequantize("model.norm.weight", finalNorm)) {
        CLLM_ERROR("[GGUFLoader::loadInto] Failed to load final norm");
        return false;
    }
    
    // 4. 加载 LM head
    bool lmHeadLoaded = loadAndDequantize("output.weight", lmHead) ||
                        loadAndDequantize("lm_head.weight", lmHead);
    
    if (!lmHeadLoaded) {
        // LM head 可能与 embedding 共享，这种情况下复制 embedding
        CLLM_WARN("[GGUFLoader::loadInto] LM head not found, using tied embedding");
        if (lmHead.size() == embedding.size()) {
            std::copy(embedding.data(), embedding.data() + embedding.size(), lmHead.data());
        } else {
            CLLM_ERROR("[GGUFLoader::loadInto] LM head size mismatch with embedding");
            return false;
        }
    }
    
    CLLM_INFO("[GGUFLoader::loadInto] Successfully loaded all weights from GGUF file");
    return true;
}

// 显式实例化模板
template bool GGUFLoader::loadInto<Tensor>(
    Tensor&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&,
    std::vector<Tensor>&, std::vector<Tensor>&, std::vector<Tensor>&,
    std::vector<Tensor>&, std::vector<Tensor>&,
    Tensor&, Tensor&
);

} // namespace kylin
} // namespace cllm
