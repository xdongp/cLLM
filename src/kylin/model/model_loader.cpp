#include "cllm/kylin/model/model_loader.h"
#include "cllm/model/gguf_loader_new.h"
#include "cllm/common/json.h"
#include "cllm/common/logger.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <filesystem>

namespace cllm {
namespace kylin {

ModelLoader::ModelLoader(const std::string &modelPath, const ModelConfig &config)
    : modelPath_(modelPath)
    , config_(config)
    , weights_()
    , dtype_(WeightDType::FP32)
    , int8Scale_(1.0f)
    , actualQProjDim_(0)
    , actualKVProjDim_(0)
    , ggufLoader_(nullptr)
    , isGGUFFormat_(false) {}

bool ModelLoader::detectGGUFFormat() const {
    // 检查文件扩展名
    std::filesystem::path path(modelPath_);
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    
    if (ext == ".gguf") {
        return true;
    }
    
    // 检查文件头（魔数）
    std::ifstream file(modelPath_, std::ios::binary);
    if (!file.is_open()) {
        CLLM_DEBUG("ModelLoader: failed to open file for GGUF detection: %s", modelPath_.c_str());
        return false;
    }
    
    uint32_t magic = 0;
    if (!file.read(reinterpret_cast<char*>(&magic), sizeof(magic)) || 
        file.gcount() != sizeof(magic)) {
        CLLM_DEBUG("ModelLoader: failed to read magic number from file: %s", modelPath_.c_str());
        file.close();
        return false;
    }
    file.close();
    
    // GGUF魔数: 0x46554747 ("GGUF")
    return (magic == 0x46554747);
}

WeightDType ModelLoader::detectDType() const {
    // 先检查是否为GGUF格式
    if (detectGGUFFormat()) {
        // GGUF格式需要加载后才能确定具体量化类型
        // 这里先返回一个占位符，实际类型在loadGGUF中确定
        return WeightDType::GGUF_F32;  // 默认，实际会在loadGGUF中更新
    }
    
    if (modelPath_.find("fp16") != std::string::npos) {
        return WeightDType::FP16;
    }
    if (modelPath_.find("int8") != std::string::npos) {
        return WeightDType::INT8;
    }
    return WeightDType::FP32;
}

bool ModelLoader::loadMetadata() {
    const std::string metaPath = modelPath_ + ".json";
    std::ifstream metaFile(metaPath);
    if (!metaFile.is_open()) {
        CLLM_WARN("ModelLoader: metadata file not found at %s, will use config-based proj dims", metaPath.c_str());
        return true;  // 元数据可选，除非 int8
    }

    std::string json((std::istreambuf_iterator<char>(metaFile)), std::istreambuf_iterator<char>());
    metaFile.close();

    try {
        auto obj = JsonParser::parse(json);
        
        // 读取 model_config
        auto mcIt = obj.find("model_config");
        if (mcIt != obj.end()) {
            auto& mcObj = mcIt.value();

            // ✅ 读取真实结构参数（用于推导张量形状，避免依赖默认 ModelConfig）
            auto vocabIt = mcObj.find("vocab_size");
            if (vocabIt != mcObj.end()) {
                config_.vocabSize = static_cast<size_t>(vocabIt.value());
            }
            auto hiddenIt = mcObj.find("hidden_size");
            if (hiddenIt != mcObj.end()) {
                config_.hiddenSize = static_cast<size_t>(hiddenIt.value());
            }
            auto layersIt = mcObj.find("num_layers");
            if (layersIt != mcObj.end()) {
                config_.numLayers = static_cast<size_t>(layersIt.value());
            }
            auto interIt = mcObj.find("intermediate_size");
            if (interIt != mcObj.end()) {
                config_.intermediateSize = static_cast<size_t>(interIt.value());
            }
            auto headsIt = mcObj.find("num_attention_heads");
            if (headsIt != mcObj.end()) {
                config_.numAttentionHeads = static_cast<size_t>(headsIt.value());
            }
            auto kvHeadsIt = mcObj.find("num_key_value_heads");
            if (kvHeadsIt != mcObj.end()) {
                config_.numKeyValueHeads = static_cast<size_t>(kvHeadsIt.value());
            }

            CLLM_INFO("ModelLoader: loaded model_config: vocab=%zu hidden=%zu layers=%zu heads=%zu kv_heads=%zu inter=%zu",
                      config_.vocabSize, config_.hiddenSize, config_.numLayers,
                      config_.numAttentionHeads, config_.numKeyValueHeads, config_.intermediateSize);

            // 读取实际投影维度
            auto qProjIt = mcObj.find("actual_q_proj_dim");
            if (qProjIt != mcObj.end()) {
                actualQProjDim_ = static_cast<size_t>(qProjIt.value());
                CLLM_INFO("ModelLoader: loaded actual_q_proj_dim=%zu", actualQProjDim_);
            }

            auto kvProjIt = mcObj.find("actual_kv_proj_dim");
            if (kvProjIt != mcObj.end()) {
                actualKVProjDim_ = static_cast<size_t>(kvProjIt.value());
                CLLM_INFO("ModelLoader: loaded actual_kv_proj_dim=%zu", actualKVProjDim_);
            }
        }
        
        // 处理 int8 元数据
        if (dtype_ == WeightDType::INT8) {
            auto it = obj.find("int8");
            if (it == obj.end()) {
                CLLM_ERROR("ModelLoader: int8 metadata missing 'int8' key");
                return false;
            }

            auto& subObj = it.value();
            auto scaleIt = subObj.find("scale");
            if (scaleIt == subObj.end()) {
                CLLM_ERROR("ModelLoader: int8 metadata missing 'scale' field");
                return false;
            }

            int8Scale_ = scaleIt.value();
            CLLM_INFO("ModelLoader: loaded int8 scale=%f", int8Scale_);
        }
    } catch (const std::exception &e) {
        CLLM_ERROR("ModelLoader: failed to parse metadata: %s", e.what());
        if (dtype_ == WeightDType::INT8) {
            return false;  // int8 必须有元数据
        }
    }

    return true;
}

bool ModelLoader::load() {
    if (!weights_.empty()) {
        return true;
    }

    // 检测格式
    isGGUFFormat_ = detectGGUFFormat();
    
    if (isGGUFFormat_) {
        CLLM_INFO("ModelLoader: detected GGUF format");
        return loadGGUF();
    }

    dtype_ = detectDType();
    CLLM_INFO("ModelLoader: detected dtype=%s", 
              (dtype_ == WeightDType::FP32 ? "fp32" :
                  dtype_ == WeightDType::FP16 ? "fp16" : "int8"));

    if (!loadMetadata()) {
        return false;
    }

    return loadBinaryFile();
}

bool ModelLoader::loadGGUF() {
    try {
        // 创建GGUF加载器
        ggufLoader_ = std::make_unique<GGUFLoader>(modelPath_, true, true);
        
        // 加载GGUF文件
        if (!ggufLoader_->load()) {
            CLLM_ERROR("ModelLoader: failed to load GGUF file");
            return false;
        }
        
        // 更新配置
        config_ = ggufLoader_->getConfig();
        
        // 检测量化类型（通过检查张量类型）
        // 这里简化处理，实际应该检查所有权重张量的类型
        dtype_ = WeightDType::GGUF_F32;  // 默认，实际使用时会根据张量类型判断
        
        CLLM_INFO("ModelLoader: GGUF model loaded successfully");
        CLLM_INFO("  - Vocab size: %zu", config_.vocabSize);
        CLLM_INFO("  - Hidden size: %zu", config_.hiddenSize);
        CLLM_INFO("  - Num layers: %zu", config_.numLayers);
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("ModelLoader: exception while loading GGUF: %s", e.what());
        return false;
    }
}

bool ModelLoader::loadBinaryFile() {
    std::ifstream file(modelPath_, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        CLLM_ERROR("ModelLoader: failed to open model file: %s", modelPath_.c_str());
        return false;
    }

    std::streamsize fileSize = file.tellg();
    if (fileSize <= 0) {
        CLLM_ERROR("ModelLoader: empty model file: %s", modelPath_.c_str());
        return false;
    }

    file.seekg(0, std::ios::beg);

    if (dtype_ == WeightDType::FP32) {
        if (fileSize % static_cast<std::streamsize>(sizeof(float)) != 0) {
            CLLM_ERROR("ModelLoader: fp32 file size not multiple of sizeof(float)");
            return false;
        }
        size_t numFloats = static_cast<size_t>(fileSize / sizeof(float));
        weights_.resize(numFloats);
        if (!file.read(reinterpret_cast<char*>(weights_.data()), fileSize)) {
            CLLM_ERROR("ModelLoader: failed to read fp32 file");
            weights_.clear();
            return false;
        }
    } else if (dtype_ == WeightDType::FP16) {
        size_t numHalf = static_cast<size_t>(fileSize / sizeof(uint16_t));
        std::vector<uint16_t> halfData(numHalf);
        if (!file.read(reinterpret_cast<char*>(halfData.data()), fileSize)) {
            CLLM_ERROR("ModelLoader: failed to read fp16 file");
            return false;
        }

        // 转为 fp32
        weights_.resize(numHalf);
        for (size_t i = 0; i < numHalf; ++i) {
            uint16_t h = halfData[i];
            // 简单 fp16 -> fp32 转换（IEEE 754 half precision）
            uint32_t sign = (h & 0x8000u) << 16;
            uint32_t exp = (h & 0x7C00u) >> 10;
            uint32_t frac = (h & 0x03FFu);

            uint32_t f32;
            if (exp == 0) {
                if (frac == 0) {
                    f32 = sign;  // ±0
                } else {
                    // subnormal
                    exp = 127 - 15;
                    while ((frac & 0x0400u) == 0) {
                        frac <<= 1;
                        exp--;
                    }
                    frac &= 0x03FFu;
                    f32 = sign | (exp << 23) | (frac << 13);
                }
            } else if (exp == 0x1F) {
                f32 = sign | 0x7F800000u | (frac << 13);  // inf/NaN
            } else {
                f32 = sign | ((exp - 15 + 127) << 23) | (frac << 13);
            }
            std::memcpy(&weights_[i], &f32, sizeof(float));
        }
    } else if (dtype_ == WeightDType::INT8) {
        size_t numInt8 = static_cast<size_t>(fileSize);
        std::vector<int8_t> int8Data(numInt8);
        if (!file.read(reinterpret_cast<char*>(int8Data.data()), fileSize)) {
            CLLM_ERROR("ModelLoader: failed to read int8 file");
            return false;
        }

        // 反量化为 fp32
        weights_.resize(numInt8);
        for (size_t i = 0; i < numInt8; ++i) {
            weights_[i] = static_cast<float>(int8Data[i]) * int8Scale_;
        }
    }

    return true;
}

bool ModelLoader::loadInto(
    Tensor &embedding,
    std::vector<Tensor> &wq,
    std::vector<Tensor> &wk,
    std::vector<Tensor> &wv,
    std::vector<Tensor> &wo,
    std::vector<Tensor> &wGate,
    std::vector<Tensor> &wUp,
    std::vector<Tensor> &wDown,
    std::vector<Tensor> &norm1,
    std::vector<Tensor> &norm2,
    Tensor &finalNorm,
    Tensor &lmHead
) const {
    // 如果是GGUF格式，使用GGUFLoader的loadInto方法
    if (isGGUFFormat_ && ggufLoader_) {
        return ggufLoader_->loadInto(
            embedding, wq, wk, wv, wo,
            wGate, wUp, wDown,
            norm1, norm2,
            finalNorm, lmHead
        );
    }
    
    if (weights_.empty()) {
        CLLM_ERROR("ModelLoader::loadInto called before load()");
        return false;
    }

    const size_t vocab = config_.vocabSize;
    const size_t hidden = config_.hiddenSize;
    const size_t inter = config_.intermediateSize;
    const size_t numLayers = config_.numLayers;
    const size_t numHeads = config_.numAttentionHeads;
    const size_t numKVHeads = config_.numKeyValueHeads;

    if (vocab == 0 || hidden == 0 || inter == 0 || numLayers == 0 || numHeads == 0 || numKVHeads == 0) {
        CLLM_ERROR("ModelLoader: invalid ModelConfig values");
        return false;
    }

    // 计算 head_dim 和 Q/KV 投影维度
    const size_t headDim = hidden / numHeads;
    
    // 优先使用元数据中的实际值，否则使用 config 推导
    const size_t qDim = (actualQProjDim_ > 0) ? actualQProjDim_ : (numHeads * headDim);
    const size_t kvDim = (actualKVProjDim_ > 0) ? actualKVProjDim_ : (numKVHeads * headDim);

    const size_t embedCount = vocab * hidden;
    const size_t perLayerCount =
        hidden * qDim +            // wq
        hidden * kvDim +           // wk
        hidden * kvDim +           // wv
        qDim * hidden +            // wo
        2 * hidden * inter +       // wGate, wUp
        inter * hidden +           // wDown
        2 * hidden;                // norm1, norm2
    const size_t finalNormCount = hidden;
    const size_t lmHeadCount = hidden * vocab;

    const size_t totalExpected =
        embedCount +
        numLayers * perLayerCount +
        finalNormCount +
        lmHeadCount;

    if (totalExpected != weights_.size()) {
        CLLM_ERROR("ModelLoader: model file size mismatch. expected floats=%zu, actual=%zu\n  Config: hidden=%zu, numHeads=%zu, numKVHeads=%zu, headDim=%zu\n  qDim=%zu, kvDim=%zu", 
                   totalExpected, weights_.size(), hidden, numHeads, numKVHeads, headDim, qDim, kvDim);
        return false;
    }

    // 确保权重向量大小正确
    if (wq.size() != numLayers) wq.resize(numLayers);
    if (wk.size() != numLayers) wk.resize(numLayers);
    if (wv.size() != numLayers) wv.resize(numLayers);
    if (wo.size() != numLayers) wo.resize(numLayers);
    if (wGate.size() != numLayers) wGate.resize(numLayers);
    if (wUp.size() != numLayers) wUp.resize(numLayers);
    if (wDown.size() != numLayers) wDown.resize(numLayers);
    if (norm1.size() != numLayers) norm1.resize(numLayers);
    if (norm2.size() != numLayers) norm2.resize(numLayers);

    // 按配置分配张量形状（支持 GQA）
    embedding = Tensor({vocab, hidden});
    for (size_t layer = 0; layer < numLayers; ++layer) {
        wq[layer] = Tensor({hidden, qDim});
        wk[layer] = Tensor({hidden, kvDim});
        wv[layer] = Tensor({hidden, kvDim});
        wo[layer] = Tensor({qDim, hidden});

        wGate[layer] = Tensor({hidden, inter});
        wUp[layer] = Tensor({hidden, inter});
        wDown[layer] = Tensor({inter, hidden});

        norm1[layer] = Tensor({hidden});
        norm2[layer] = Tensor({hidden});
    }
    finalNorm = Tensor({hidden});
    lmHead = Tensor({hidden, vocab});

    // 逐段拷贝数据
    size_t offset = 0;
    auto copyBlock = [&](Tensor &t) {
        const size_t count = t.size();
        std::copy(
            weights_.data() + offset,
            weights_.data() + offset + count,
            t.data()
        );
        offset += count;
    };

    // 1. embedding
    copyBlock(embedding);

    // 2. each layer
    for (size_t layer = 0; layer < numLayers; ++layer) {
        copyBlock(wq[layer]);
        copyBlock(wk[layer]);
        copyBlock(wv[layer]);
        copyBlock(wo[layer]);

        copyBlock(wGate[layer]);
        copyBlock(wUp[layer]);
        copyBlock(wDown[layer]);

        copyBlock(norm1[layer]);
        copyBlock(norm2[layer]);
    }

    // 3. final norm
    copyBlock(finalNorm);

    // 4. lm head
    copyBlock(lmHead);

    if (offset != weights_.size()) {
        CLLM_ERROR("ModelLoader: internal offset mismatch after loadInto (offset=%zu, total=%zu)", offset, weights_.size());
        return false;
    }

    return true;
}

} // namespace kylin
} // namespace cllm
