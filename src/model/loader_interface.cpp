/**
 * @file loader_interface.cpp
 * @brief Implementation of model loader interface and factory
 * @author cLLM Team
 * @date 2026-01-13
 */
#include "cllm/model/loader_interface.h"
#include "cllm/common/logger.h"
#include "cllm/model/gguf_loader_new.h"
#include "cllm/kylin/model/model_loader.h"
#include <filesystem>
#include <stdexcept>
#include <algorithm>
#include <fstream>

namespace cllm {

// ModelLoaderFactory implementation

std::unique_ptr<IModelLoader> ModelLoaderFactory::createLoader(
    const std::string& modelPath,
    const ModelConfig& config
) {
    if (modelPath.empty()) {
        throw std::runtime_error("ModelLoaderFactory: model path cannot be empty");
    }
    
    std::filesystem::path filePath(modelPath);
    
    // Check if file exists
    if (!std::filesystem::exists(filePath)) {
        throw std::runtime_error("ModelLoaderFactory: model file does not exist: " + modelPath);
    }
    
    // Check if path is a regular file (not a directory)
    if (!std::filesystem::is_regular_file(filePath)) {
        throw std::runtime_error("ModelLoaderFactory: path is not a regular file: " + modelPath);
    }
    
    // Check if file is readable
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("ModelLoaderFactory: cannot open model file for reading: " + modelPath);
    }
    file.close();
    
    ModelFormat format = detectFormat(modelPath);
    
    if (format == ModelFormat::UNKNOWN) {
        throw std::runtime_error("ModelLoaderFactory: unknown model format for file: " + modelPath);
    }
    
    if (!isFormatSupported(format)) {
        throw std::runtime_error("ModelLoaderFactory: unsupported model format '" + formatToString(format) + "' for file: " + modelPath);
    }
    
    switch (format) {
        case ModelFormat::BINARY:
            CLLM_INFO("ModelLoaderFactory: Creating BinaryModelLoader for %s", modelPath.c_str());
            return std::make_unique<BinaryModelLoader>(modelPath, config);
            
        case ModelFormat::GGUF:
            CLLM_INFO("ModelLoaderFactory: Creating GGUFLoader for %s", modelPath.c_str());
            return std::make_unique<GGUFLoader>(modelPath);
            
        case ModelFormat::SAFETENSORS:
            CLLM_INFO("ModelLoaderFactory: SafeTensors format detected for %s", modelPath.c_str());
            throw std::runtime_error("ModelLoaderFactory: SafeTensors format support not yet implemented");
            
        default:
            throw std::runtime_error("ModelLoaderFactory: unknown model format '" + formatToString(format) + "' for file: " + modelPath);
    }
}

ModelFormat ModelLoaderFactory::detectFormat(const std::string& modelPath) {
    if (modelPath.empty()) {
        return ModelFormat::UNKNOWN;
    }
    
    std::filesystem::path path(modelPath);
    std::string extension = path.extension().string();
    
    // Convert to lowercase for case-insensitive comparison
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    if (extension == ".bin") {
        return ModelFormat::BINARY;
    } else if (extension == ".gguf") {
        return ModelFormat::GGUF;
    } else if (extension == ".safetensors") {
        return ModelFormat::SAFETENSORS;
    } else {
        CLLM_WARN("ModelLoaderFactory: Unknown file extension '%s' for file: %s", extension.c_str(), modelPath.c_str());
        return ModelFormat::UNKNOWN;
    }
}

bool ModelLoaderFactory::isFormatSupported(ModelFormat format) {
    switch (format) {
        case ModelFormat::BINARY:
            return true;  // Already implemented
        case ModelFormat::GGUF:
            return true; // GGUF format support implemented
        case ModelFormat::SAFETENSORS:
            return false; // Future implementation
        case ModelFormat::UNKNOWN:
        default:
            return false;
    }
}

std::string ModelLoaderFactory::formatToString(ModelFormat format) {
    switch (format) {
        case ModelFormat::BINARY:
            return "BINARY";
        case ModelFormat::GGUF:
            return "GGUF";
        case ModelFormat::SAFETENSORS:
            return "SAFETENSORS";
        case ModelFormat::UNKNOWN:
        default:
            return "UNKNOWN";
    }
}

// BinaryModelLoader implementation

BinaryModelLoader::BinaryModelLoader(const std::string& modelPath, const ModelConfig& config)
    : loader_(std::make_unique<kylin::ModelLoader>(modelPath, config)) {
    CLLM_DEBUG("BinaryModelLoader: Created for model path: %s", modelPath.c_str());
}

bool BinaryModelLoader::load() {
    if (!loader_) {
        CLLM_ERROR("BinaryModelLoader: Internal loader is null");
        return false;
    }
    
    bool success = loader_->load();
    if (success) {
        CLLM_INFO("BinaryModelLoader: Successfully loaded model from %s", getModelPath().c_str());
    } else {
        CLLM_ERROR("BinaryModelLoader: Failed to load model from %s", getModelPath().c_str());
    }
    
    return success;
}

bool BinaryModelLoader::loadWeights(model::ModelWeights& weights, bool loadAll) {
    if (!loader_) {
        CLLM_ERROR("BinaryModelLoader: Internal loader is null");
        return false;
    }
    
    // 确保模型已加载
    if (!loader_->load()) {
        CLLM_ERROR("BinaryModelLoader::loadWeights: Failed to load model");
        return false;
    }
    
    const ModelConfig& config = loader_->getConfig();
    const size_t vocab = config.vocabSize;
    const size_t hidden = config.hiddenSize;
    const size_t inter = config.intermediateSize;
    const size_t numLayers = config.numLayers;
    const size_t numHeads = config.numAttentionHeads;
    const size_t numKVHeads = config.numKeyValueHeads;
    
    if (vocab == 0 || hidden == 0 || inter == 0 || numLayers == 0 || numHeads == 0 || numKVHeads == 0) {
        CLLM_ERROR("BinaryModelLoader::loadWeights: Invalid ModelConfig values");
        return false;
    }
    
    // 计算 head_dim 和 Q/KV 投影维度
    const size_t headDim = hidden / numHeads;
    const size_t qDim = numHeads * headDim;
    const size_t kvDim = numKVHeads * headDim;
    
    // 1. 设置embedding
    weights.embedding.name = "embedding";
    weights.embedding.shape = {vocab, hidden};
    weights.embedding.dtype = loader_->getDType();
    weights.embedding.data.resize(vocab * hidden);
    
    // 2. 为每层分配空间
    weights.layers.resize(numLayers);
    for (size_t layer = 0; layer < numLayers; ++layer) {
        model::LayerWeights& layerWeights = weights.layers[layer];
        std::string layerPrefix = "layers." + std::to_string(layer) + ".";
        
        // Attention weights
        layerWeights.wq.name = layerPrefix + "attention.wq.weight";
        layerWeights.wq.shape = {hidden, qDim};
        layerWeights.wq.dtype = loader_->getDType();
        layerWeights.wq.data.resize(hidden * qDim);
        
        layerWeights.wk.name = layerPrefix + "attention.wk.weight";
        layerWeights.wk.shape = {hidden, kvDim};
        layerWeights.wk.dtype = loader_->getDType();
        layerWeights.wk.data.resize(hidden * kvDim);
        
        layerWeights.wv.name = layerPrefix + "attention.wv.weight";
        layerWeights.wv.shape = {hidden, kvDim};
        layerWeights.wv.dtype = loader_->getDType();
        layerWeights.wv.data.resize(hidden * kvDim);
        
        layerWeights.wo.name = layerPrefix + "attention.wo.weight";
        layerWeights.wo.shape = {qDim, hidden};
        layerWeights.wo.dtype = loader_->getDType();
        layerWeights.wo.data.resize(qDim * hidden);
        
        // FFN weights
        layerWeights.wGate.name = layerPrefix + "feed_forward.wGate.weight";
        layerWeights.wGate.shape = {hidden, inter};
        layerWeights.wGate.dtype = loader_->getDType();
        layerWeights.wGate.data.resize(hidden * inter);
        
        layerWeights.wUp.name = layerPrefix + "feed_forward.wUp.weight";
        layerWeights.wUp.shape = {hidden, inter};
        layerWeights.wUp.dtype = loader_->getDType();
        layerWeights.wUp.data.resize(hidden * inter);
        
        layerWeights.wDown.name = layerPrefix + "feed_forward.wDown.weight";
        layerWeights.wDown.shape = {inter, hidden};
        layerWeights.wDown.dtype = loader_->getDType();
        layerWeights.wDown.data.resize(inter * hidden);
        
        // Norm weights
        layerWeights.norm1.name = layerPrefix + "attention_norm.weight";
        layerWeights.norm1.shape = {hidden};
        layerWeights.norm1.dtype = loader_->getDType();
        layerWeights.norm1.data.resize(hidden);
        
        layerWeights.norm2.name = layerPrefix + "ffn_norm.weight";
        layerWeights.norm2.shape = {hidden};
        layerWeights.norm2.dtype = loader_->getDType();
        layerWeights.norm2.data.resize(hidden);
    }
    
    // 3. 设置finalNorm
    weights.finalNorm.name = "finalNorm";
    weights.finalNorm.shape = {hidden};
    weights.finalNorm.dtype = loader_->getDType();
    weights.finalNorm.data.resize(hidden);
    
    // 4. 设置lmHead
    weights.lmHead.name = "lmHead";
    weights.lmHead.shape = {hidden, vocab};
    weights.lmHead.dtype = loader_->getDType();
    weights.lmHead.data.resize(hidden * vocab);
    
    // 如果需要加载权重数据
    if (loadAll) {
        // 从kylin loader复制权重数据
        const std::vector<float>& rawWeights = loader_->getWeights();
        size_t offset = 0;
        
        // 复制embedding
        std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + vocab * hidden, weights.embedding.data.begin());
        offset += vocab * hidden;
        
        // 复制每一层的权重
        for (size_t layer = 0; layer < numLayers; ++layer) {
            model::LayerWeights& layerWeights = weights.layers[layer];
            
            // Attention weights
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * qDim, layerWeights.wq.data.begin());
            offset += hidden * qDim;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * kvDim, layerWeights.wk.data.begin());
            offset += hidden * kvDim;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * kvDim, layerWeights.wv.data.begin());
            offset += hidden * kvDim;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + qDim * hidden, layerWeights.wo.data.begin());
            offset += qDim * hidden;
            
            // FFN weights
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * inter, layerWeights.wGate.data.begin());
            offset += hidden * inter;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * inter, layerWeights.wUp.data.begin());
            offset += hidden * inter;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + inter * hidden, layerWeights.wDown.data.begin());
            offset += inter * hidden;
            
            // Norm weights
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden, layerWeights.norm1.data.begin());
            offset += hidden;
            
            std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden, layerWeights.norm2.data.begin());
            offset += hidden;
        }
        
        // 复制finalNorm
        std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden, weights.finalNorm.data.begin());
        offset += hidden;
        
        // 复制lmHead
        std::copy(rawWeights.begin() + offset, rawWeights.begin() + offset + hidden * vocab, weights.lmHead.data.begin());
        offset += hidden * vocab;
        
        if (offset != rawWeights.size()) {
            CLLM_ERROR("BinaryModelLoader::loadWeights: Offset mismatch after copying weights. Expected %zu, got %zu", rawWeights.size(), offset);
            return false;
        }
    } else {
        // 不加载权重数据，只分配结构
        CLLM_INFO("BinaryModelLoader::loadWeights: Skipping data load (loadAll = false)");
    }
    
    CLLM_INFO("BinaryModelLoader::loadWeights: Successfully loaded weights into ModelWeights structure");
    return true;
}

bool BinaryModelLoader::loadWeightByName(const std::string& name, model::WeightData& weight) {
    // BinaryModelLoader不支持按需加载单个权重
    CLLM_WARN("BinaryModelLoader::loadWeightByName: Not supported for binary model format");
    return false;
}

bool BinaryModelLoader::hasWeight(const std::string& name) {
    // BinaryModelLoader不支持权重名称查询
    CLLM_WARN("BinaryModelLoader::hasWeight: Not supported for binary model format");
    return false;
}

bool BinaryModelLoader::loadInto(
    kylin::Tensor& embedding,
    std::vector<kylin::Tensor>& wq,
    std::vector<kylin::Tensor>& wk,
    std::vector<kylin::Tensor>& wv,
    std::vector<kylin::Tensor>& wo,
    std::vector<kylin::Tensor>& wGate,
    std::vector<kylin::Tensor>& wUp,
    std::vector<kylin::Tensor>& wDown,
    std::vector<kylin::Tensor>& norm1,
    std::vector<kylin::Tensor>& norm2,
    kylin::Tensor& finalNorm,
    kylin::Tensor& lmHead
) {
    if (!loader_) {
        CLLM_ERROR("BinaryModelLoader: Internal loader is null");
        return false;
    }
    
    bool success = loader_->loadInto(
        embedding, wq, wk, wv, wo, wGate, wUp, wDown,
        norm1, norm2, finalNorm, lmHead
    );
    
    if (success) {
        CLLM_DEBUG("BinaryModelLoader: Successfully loaded weights into tensors");
    } else {
        CLLM_ERROR("BinaryModelLoader: Failed to load weights into tensors");
    }
    
    return success;
}

const ModelConfig& BinaryModelLoader::getConfig() const {
    if (!loader_) {
        throw std::runtime_error("BinaryModelLoader: Internal loader is null");
    }
    return loader_->getConfig();
}

const std::string& BinaryModelLoader::getModelPath() const {
    if (!loader_) {
        throw std::runtime_error("BinaryModelLoader: Internal loader is null");
    }
    return loader_->getModelPath();
}

kylin::WeightDType BinaryModelLoader::getDType() const {
    if (!loader_) {
        throw std::runtime_error("BinaryModelLoader: Internal loader is null");
    }
    return loader_->getDType();
}

} // namespace cllm