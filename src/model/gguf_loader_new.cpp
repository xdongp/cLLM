#include "cllm/model/gguf_loader_new.h"
#include "cllm/model/gguf_dequantization.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>
#include <limits>

namespace cllm {

GGUFLoader::GGUFLoader(const std::string& filepath, bool useMemoryMap, bool allowMultipleMappings)
    : filepath_(filepath),
      file_(nullptr),
      useMemoryMap_(useMemoryMap),
      currentPosition_(0),
      needByteOrderSwap_(false),
      allowMultipleMappings_(allowMultipleMappings),
      ggufVersion_(0),
      alignment_(32) // 默认对齐值32
#ifdef _WIN32
    , fileHandle_(INVALID_HANDLE_VALUE),
      mapHandle_(nullptr),
      mappedMemory_(nullptr)
#else
    , fileDescriptor_(-1),
      mappedMemory_(nullptr)
#endif
{
    // 打开文件
    file_ = fopen(filepath.c_str(), "rb");
    if (!file_) {
        throw std::runtime_error("无法打开GGUF文件: " + filepath);
    }
    
    // 获取文件大小
    fseek(file_, 0, SEEK_END);
    fileSize_ = ftell(file_);
    fseek(file_, 0, SEEK_SET);
    
    // 检查文件大小是否足够大
    if (fileSize_ < sizeof(GGUFHeader)) {
        fclose(file_);
        file_ = nullptr;
        throw std::runtime_error("GGUF文件太小，无法包含完整的文件头");
    }
    
    // 检查系统字节序
    needByteOrderSwap_ = !isSystemLittleEndian();
    CLLM_INFO(needByteOrderSwap_ ? "系统是大端字节序，需要进行字节序转换" : "系统是小端字节序，不需要进行字节序转换");
    
    // 初始化内存映射（如果启用）
    if (useMemoryMap_) {
        try {
            initializeMemoryMap();
        } catch (const std::exception& e) {
            CLLM_WARN("初始化内存映射失败: %s，将使用传统文件I/O", e.what());
            useMemoryMap_ = false;
            // 不需要调用releaseMemoryMap()，因为initializeMemoryMap()在失败时已经清理了资源
        }
    }
}

GGUFLoader::~GGUFLoader() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
    releaseMemoryMap();
}

bool GGUFLoader::load() {
    try {
        // 解析文件头
        GGUFHeader header = parseHeader();
        CLLM_INFO("GGUF文件头解析成功: 版本 %u, 张量数量 %zu, 元数据数量 %zu", 
                  header.version, header.tensorCount, header.metadataCount);
        
        // 解析元数据
        parseMetadata(header.metadataCount);
        CLLM_INFO("元数据解析完成，共 %zu 项", metadata_.size());
        
        // 提取模型配置
        extractModelConfig();
        
        // 解析张量信息
        parseTensorInfos(header.tensorCount);
        CLLM_INFO("张量信息解析完成，共 %zu 个张量", tensorInfos_.size());
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF模型失败: %s", e.what());
        return false;
    }
}

bool GGUFLoader::loadWeights(cllm::model::ModelWeights& weights, bool loadAll) {
    try {
        // 如果还没有加载文件，先加载
        if (metadata_.empty() || tensorInfos_.empty()) {
            if (!load()) {
                return false;
            }
        }
        
        // 为权重分配基本属性
        weights.embedding.name = "embedding";
        weights.embedding.dtype = kylin::WeightDType::FP32;
        
        weights.finalNorm.name = "finalNorm";
        weights.finalNorm.dtype = kylin::WeightDType::FP32;
        
        weights.lmHead.name = "lmHead";
        weights.lmHead.dtype = kylin::WeightDType::FP32;
        
        // 根据配置创建层
        for (size_t i = 0; i < config_.numLayers; ++i) {
            model::LayerWeights layer;
            
            // 设置层权重的基本属性
            layer.wq.name = "layers." + std::to_string(i) + ".attention.wq.weight";
            layer.wq.dtype = kylin::WeightDType::FP32;
            
            layer.wk.name = "layers." + std::to_string(i) + ".attention.wk.weight";
            layer.wk.dtype = kylin::WeightDType::FP32;
            
            layer.wv.name = "layers." + std::to_string(i) + ".attention.wv.weight";
            layer.wv.dtype = kylin::WeightDType::FP32;
            
            layer.wo.name = "layers." + std::to_string(i) + ".attention.wo.weight";
            layer.wo.dtype = kylin::WeightDType::FP32;
            
            layer.wGate.name = "layers." + std::to_string(i) + ".feed_forward.wGate.weight";
            layer.wGate.dtype = kylin::WeightDType::FP32;
            
            layer.wUp.name = "layers." + std::to_string(i) + ".feed_forward.wUp.weight";
            layer.wUp.dtype = kylin::WeightDType::FP32;
            
            layer.wDown.name = "layers." + std::to_string(i) + ".feed_forward.wDown.weight";
            layer.wDown.dtype = kylin::WeightDType::FP32;
            
            layer.norm1.name = "layers." + std::to_string(i) + ".attention_norm.weight";
            layer.norm1.dtype = kylin::WeightDType::FP32;
            
            layer.norm2.name = "layers." + std::to_string(i) + ".ffn_norm.weight";
            layer.norm2.dtype = kylin::WeightDType::FP32;
            
            weights.layers.push_back(layer);
        }
        
        // 如果需要立即加载所有权重数据
        if (loadAll) {
            // 加载embedding
            loadWeightByName("embedding", weights.embedding);
            
            // 加载finalNorm
            loadWeightByName("finalNorm", weights.finalNorm);
            
            // 加载lmHead
            loadWeightByName("lmHead", weights.lmHead);
            
            // 加载所有层的权重
            for (size_t i = 0; i < config_.numLayers; ++i) {
                model::LayerWeights& layer = weights.layers[i];
                
                loadWeightByName("layers." + std::to_string(i) + ".attention.wq.weight", layer.wq);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wk.weight", layer.wk);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wv.weight", layer.wv);
                loadWeightByName("layers." + std::to_string(i) + ".attention.wo.weight", layer.wo);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wGate.weight", layer.wGate);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wUp.weight", layer.wUp);
                loadWeightByName("layers." + std::to_string(i) + ".feed_forward.wDown.weight", layer.wDown);
                loadWeightByName("layers." + std::to_string(i) + ".attention_norm.weight", layer.norm1);
                loadWeightByName("layers." + std::to_string(i) + ".ffn_norm.weight", layer.norm2);
            }
        }
        
        // 更新权重映射
        weights.updateWeightMap();
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF权重失败: %s", e.what());
        return false;
    }
}

bool GGUFLoader::loadWeightByName(const std::string& name, cllm::model::WeightData& weight) {
    try {
        auto it = tensorNameMap_.find(name);
        if (it == tensorNameMap_.end()) {
            CLLM_WARN("权重 '%s' 不存在", name.c_str());
            return false;
        }
        
        const GGULTensorInfo& tensorInfo = tensorInfos_[it->second];
        
        // 保存当前位置
        uint64_t savedPos = getCurrentFilePosition();
        
        // 定位到张量数据位置
        setFilePosition(tensorInfo.offset);
        
        // 设置权重的元数据
        weight.name = name;
        weight.shape.clear();
        weight.shape.reserve(tensorInfo.shape.size());
        for (uint64_t dim : tensorInfo.shape) {
            weight.shape.push_back(static_cast<size_t>(dim));
        }
        
        // 根据张量类型设置数据类型
        weight.dtype = kylin::WeightDType::FP32;
        
        // 计算权重元素数量
        size_t elementCount = weight.elementCount();
        
        // 预分配输出缓冲区
        weight.data.reserve(elementCount);
        weight.data.resize(elementCount);
        
        // 根据张量类型读取并反量化权重数据
        switch (static_cast<uint32_t>(tensorInfo.type)) {
            case 0: // F32
                // 直接读取
                readValues(weight.data.data(), elementCount);
                break;
            case 1: // F16
                {
                    // 预分配F16数据缓冲区
                    std::vector<uint16_t> f16Data;
                    f16Data.reserve(elementCount);
                    f16Data.resize(elementCount);
                    
                    // 批量读取F16数据
                    readValues(f16Data.data(), elementCount);
                    
                    // 使用SIMD优化的反量化
                    dequantizeF16ToF32(f16Data.data(), weight.data.data(), elementCount);
                }
                break;
            case 2: // Q8_0
                {
                    // 预分配Q8_0数据缓冲区
                    std::vector<int8_t> q8Data;
                    q8Data.reserve(elementCount);
                    q8Data.resize(elementCount);
                    
                    // 批量读取Q8_0数据
                    readValues(q8Data.data(), elementCount);
                    
                    // 使用SIMD优化的反量化
                    dequantizeQ8ToF32(q8Data.data(), weight.data.data(), elementCount);
                }
                break;
            case 13: // Q4_K_M
                {
                    // Q4_K_M格式: 每个块包含1个缩放因子 + 8个4位值 (共6字节/块)
                    size_t blockCount = (elementCount + 7) / 8;
                    size_t q4DataSize = blockCount * 6;
                    
                    // 预分配Q4_K_M数据缓冲区
                    std::vector<uint8_t> q4Data;
                    q4Data.reserve(q4DataSize);
                    q4Data.resize(q4DataSize);
                    
                    // 批量读取Q4_K_M数据
                    readValues(q4Data.data(), q4DataSize);
                    
                    // 使用SIMD优化的反量化
                    dequantizeQ4KMF32(q4Data.data(), weight.data.data(), weight.shape);
                }
                break;
            case 14: // Q5_K_M
                {
                    // Q5_K_M格式: 每个块包含1个缩放因子 + 2位高位数据 + 4位低位数据 (共7字节/块)
                    size_t blockCount = (elementCount + 7) / 8;
                    size_t q5DataSize = blockCount * 7;
                    
                    // 预分配Q5_K_M数据缓冲区
                    std::vector<uint8_t> q5Data;
                    q5Data.reserve(q5DataSize);
                    q5Data.resize(q5DataSize);
                    
                    // 批量读取Q5_K_M数据
                    readValues(q5Data.data(), q5DataSize);
                    
                    // 使用SIMD优化的反量化
                    dequantizeQ5KMF32(q5Data.data(), weight.data.data(), weight.shape);
                }
                break;
            default:
                CLLM_ERROR("不支持的张量类型: %u", static_cast<uint32_t>(tensorInfo.type));
                setFilePosition(savedPos);
                return false;
        }
        
        // 恢复当前位置
        setFilePosition(savedPos);
        
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载张量 %s 失败: %s", name.c_str(), e.what());
        return false;
    }
}

bool GGUFLoader::hasWeight(const std::string& name) {
    return tensorNameMap_.find(name) != tensorNameMap_.end();
}

bool GGUFLoader::loadInto(
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
    try {
        // 实现加载到指定张量结构的逻辑
        return true;
    } catch (const std::exception& e) {
        CLLM_ERROR("加载GGUF模型到张量结构失败: %s", e.what());
        return false;
    }
}

const ModelConfig& GGUFLoader::getConfig() const {
    return config_;
}

const std::string& GGUFLoader::getModelPath() const {
    return filepath_;
}

kylin::WeightDType GGUFLoader::getDType() const {
    // 根据模型配置返回数据类型
    return kylin::WeightDType::FP32; // 默认返回FP32，需要根据实际情况修改
}

GGUFHeader GGUFLoader::parseHeader() {
    GGUFHeader header;
    
    // 读取文件头
    readValues(&header.magic, 1);
    
    // 验证魔数
    if (header.magic != 0x46554747) { // 'GGUF'
        throw std::runtime_error("无效的GGUF文件格式: 魔数不匹配");
    }
    
    // 读取版本号
    readValues(&header.version, 1);
    ggufVersion_ = header.version;
    
    // GGUF版本3在版本号字段(uint32_t, 4字节)后有3个填充字节，使版本号字段对齐到8字节
    // 必须在读取tensorCount之前处理填充字节
    if (header.version == 3) {
        // 检查边界
        if (currentPosition_ + 3 > fileSize_) {
            throw std::runtime_error("文件大小不足，无法跳过版本号填充字节: 位置 " + 
                                   std::to_string(currentPosition_) + " + 3 > 文件大小 " + 
                                   std::to_string(fileSize_));
        }
        
        // 跳过3个填充字节
        if (useMemoryMap_) {
            currentPosition_ += 3;
        } else {
            if (fseek(file_, 3, SEEK_CUR) != 0) {
                throw std::runtime_error("无法跳过版本号填充字节: fseek失败");
            }
            currentPosition_ += 3;
        }
    }
    
    // 验证版本号
    if (header.version != 3) {
        CLLM_WARN("GGUF版本号 %u 与当前实现的版本 3 不匹配，可能会导致兼容性问题", header.version);
    }
    
    // 读取张量数量和元数据数量（这些字段在版本3中是8字节对齐的）
    readValues(&header.tensorCount, 1);
    readValues(&header.metadataCount, 1);
    
    return header;
}

void GGUFLoader::parseMetadata(uint64_t metadataCount) {
    for (uint64_t i = 0; i < metadataCount; ++i) {
        GGUFMetadata metadata;
        
        // 保存当前文件位置，以便在解析失败时恢复
        uint64_t savedPosition = getCurrentFilePosition();
        
        try {
            // 读取键名
            metadata.key = readString();
            
            // 读取值类型
            uint32_t valueTypeRaw = readValue<uint32_t>();
            
            // 验证值类型是否在合理范围内
            // 注意：GGUF规范可能会随着版本更新而添加新的类型
            // 这里使用一个更宽松的检查，允许更多类型
            if (valueTypeRaw > 30) { // 使用一个更宽松的上限
                CLLM_WARN("元数据值类型超出常见范围: %u", valueTypeRaw);
                // 不抛出异常，而是尝试跳过这个元数据项
                // 恢复文件位置
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前元数据项", savedPosition);
                continue;
            }
            
            metadata.type = static_cast<GGUFValueType>(valueTypeRaw);
            
            // 读取值
            try {
                readMetadataValue(metadata);
            } catch (const std::exception& e) {
                CLLM_WARN("读取元数据值失败: %s，将跳过此元数据项", e.what());
                // 恢复文件位置
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前元数据项", savedPosition);
                continue;
            }
            
            // 存储元数据
            metadata_[metadata.key] = metadata;
            
            // 检查是否是对齐值
            if (metadata.key == "general.alignment") {
                // 根据值类型获取对齐值
                switch (metadata.type) {
                    case GGUFValueType::UINT32:
                        alignment_ = metadata.value.u32_val;
                        break;
                    case GGUFValueType::UINT64:
                        alignment_ = static_cast<uint32_t>(metadata.value.u64_val);
                        break;
                    default:
                        CLLM_WARN("general.alignment 元数据的类型 %u 不是预期的整数类型，将使用默认对齐值 %u", 
                                  valueTypeRaw, alignment_);
                }
                
                // 验证对齐值是否为2的幂
                if (alignment_ != 0 && (alignment_ & (alignment_ - 1)) != 0) {
                    CLLM_WARN("对齐值 %u 不是2的幂，可能导致问题。将使用默认对齐值32", alignment_);
                    alignment_ = 32;
                } else {
                    CLLM_INFO("使用元数据中指定的对齐值: %u", alignment_);
                }
            }
            
        } catch (const std::exception& e) {
            CLLM_ERROR("解析元数据第 %zu 项失败: %s，位置: %zu", i, e.what(), savedPosition);
            
            // 恢复文件位置，避免后续数据错位
            try {
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前元数据项", savedPosition);
            } catch (const std::exception& restoreError) {
                CLLM_ERROR("无法恢复文件位置: %s，解析可能已损坏", restoreError.what());
                // 如果无法恢复位置，抛出异常，因为继续解析会导致数据错位
                throw std::runtime_error("元数据解析失败且无法恢复文件位置: " + std::string(e.what()));
            }
            
            // 跳过当前元数据项，继续解析下一个
            continue;
        }
    }
}

void GGUFLoader::parseTensorInfos(uint64_t tensorCount) {
    tensorInfos_.reserve(tensorCount);
    tensorNameMap_.reserve(tensorCount);
    
    for (uint64_t i = 0; i < tensorCount; ++i) {
        GGULTensorInfo tensorInfo;
        
        // 保存当前文件位置，以便在解析失败时恢复
        uint64_t savedPosition = getCurrentFilePosition();
        
        try {
            // 读取张量名称
            tensorInfo.name = readString();
            
            // 读取维度数
            tensorInfo.dimensions = readValue<uint32_t>();
            
            // 验证维度数是否在合理范围内 (0-8)
            // 注意：维度数为0是有效的（标量），但超过8通常不合理
            if (tensorInfo.dimensions > 8) {
                throw std::runtime_error("张量维度数异常: " + std::to_string(tensorInfo.dimensions) + 
                                        " (合理范围: 0-8)");
            }
            
            // 读取形状
            tensorInfo.shape.resize(tensorInfo.dimensions);
            for (uint32_t j = 0; j < tensorInfo.dimensions; ++j) {
                tensorInfo.shape[j] = readValue<uint64_t>();
                
                // 验证每个维度的大小是否合理（不能为0，除非是标量）
                if (tensorInfo.shape[j] == 0 && tensorInfo.dimensions > 0) {
                    throw std::runtime_error("张量 " + tensorInfo.name + " 的第 " + std::to_string(j) + 
                                            " 个维度大小为0，这是无效的");
                }
            }
            
            // 读取张量类型
            uint32_t tensorTypeRaw = readValue<uint32_t>();
            
            // 验证张量类型是否在有效范围内
            // GGMLType的最大值是COUNT-1，但实际使用的类型值可能更小
            // 根据GGUF规范，有效的类型值通常在0-40之间
            const uint32_t MAX_VALID_GGML_TYPE = 40;
            if (tensorTypeRaw > MAX_VALID_GGML_TYPE) {
                throw std::runtime_error("张量类型值异常: " + std::to_string(tensorTypeRaw) + 
                                        " (有效范围: 0-" + std::to_string(MAX_VALID_GGML_TYPE) + ")");
            }
            
            tensorInfo.type = static_cast<GGMLType>(tensorTypeRaw);
            
            // 读取偏移量
            tensorInfo.offset = readValue<uint64_t>();
            
            // 验证偏移量是否超出文件大小
            if (tensorInfo.offset >= fileSize_) {
                throw std::runtime_error("张量 " + tensorInfo.name + " 的偏移量 " + 
                                        std::to_string(tensorInfo.offset) + " 超出文件大小 " + 
                                        std::to_string(fileSize_));
            }
            
            // 验证偏移量是否对齐
            if (tensorInfo.offset % alignment_ != 0) {
                CLLM_WARN("张量 %s 的偏移量 %zu 不是对齐值 %u 的倍数，可能会影响性能", 
                          tensorInfo.name.c_str(), tensorInfo.offset, alignment_);
            }
            
            // 存储张量信息
            tensorNameMap_[tensorInfo.name] = tensorInfos_.size();
            tensorInfos_.push_back(tensorInfo);
            
        } catch (const std::exception& e) {
            CLLM_ERROR("解析张量信息第 %zu 项失败: %s，位置: %zu", i, e.what(), savedPosition);
            
            // 恢复文件位置，避免后续数据错位
            try {
                setFilePosition(savedPosition);
                CLLM_WARN("已恢复文件位置到 %zu，跳过当前张量信息项", savedPosition);
            } catch (const std::exception& restoreError) {
                CLLM_ERROR("无法恢复文件位置: %s，解析可能已损坏", restoreError.what());
                // 如果无法恢复位置，抛出异常，因为继续解析会导致数据错位
                throw std::runtime_error("张量信息解析失败且无法恢复文件位置: " + std::string(e.what()));
            }
            
            // 跳过当前张量信息，继续解析下一个
            continue;
        }
    }
    
    // 对齐文件位置到下一个对齐边界
    uint64_t alignedPosition = alignOffset(currentPosition_);
    if (alignedPosition > currentPosition_) {
        if (useMemoryMap_) {
            currentPosition_ = alignedPosition;
        } else {
            if (fseek(file_, alignedPosition, SEEK_SET) != 0) {
                throw std::runtime_error("无法对齐文件位置");
            }
            currentPosition_ = alignedPosition;
        }
        CLLM_INFO("将文件位置从 %zu 对齐到 %zu (对齐值 %u)", currentPosition_, alignedPosition, alignment_);
    }
}

// 辅助函数：从元数据中提取uint32_t值
template<typename T>
uint32_t GGUFLoader::extractUInt32(const GGUFMetadata& metadata) {
    switch (metadata.type) {
        case GGUFValueType::UINT32: return metadata.value.u32_val;
        case GGUFValueType::UINT64: return static_cast<uint32_t>(metadata.value.u64_val);
        case GGUFValueType::INT32: return static_cast<uint32_t>(metadata.value.i32_val);
        case GGUFValueType::INT64: return static_cast<uint32_t>(metadata.value.i64_val);
        case GGUFValueType::UINT16: return static_cast<uint32_t>(metadata.value.u16_val);
        case GGUFValueType::INT16: return static_cast<uint32_t>(metadata.value.i16_val);
        case GGUFValueType::UINT8: return static_cast<uint32_t>(metadata.value.u8_val);
        case GGUFValueType::INT8: return static_cast<uint32_t>(metadata.value.i8_val);
        case GGUFValueType::BOOL: return metadata.value.bool_val ? 1 : 0;
        default:
            CLLM_WARN("无法从元数据类型 %u 提取uint32_t值", static_cast<uint32_t>(metadata.type));
            return 0;
    }
}

// 辅助函数：尝试从多个元数据键中提取uint32_t值
bool GGUFLoader::tryExtractUInt32(uint32_t& result, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = metadata_.find(key);
        if (it != metadata_.end()) {
            result = extractUInt32<uint32_t>(it->second);
            return true;
        }
    }
    return false;
}

// 辅助函数：尝试从多个元数据键中提取字符串值
bool GGUFLoader::tryExtractString(std::string& result, const std::vector<std::string>& keys) {
    for (const auto& key : keys) {
        auto it = metadata_.find(key);
        if (it != metadata_.end() && it->second.type == GGUFValueType::STRING) {
            result = it->second.string_val;
            return true;
        }
    }
    return false;
}

void GGUFLoader::extractModelConfig() {
    // 从元数据中提取模型配置
    // 设置默认值
    config_.modelType = "llama";
    config_.vocabSize = 32000;
    config_.hiddenSize = 768;
    config_.numLayers = 6;
    config_.numAttentionHeads = 8;
    config_.numKeyValueHeads = 8;
    config_.maxSequenceLength = 512;
    config_.intermediateSize = 3072;
    config_.useKVCache = true;
    config_.useQuantization = false;
    config_.useMemoryCompression = false;
    config_.quantizationType = "";
    
    // 提取模型类型
    std::vector<std::string> modelTypeKeys = {
        "general.architecture",
        "model_type",
        "tokenizer.ggml.model"
    };
    tryExtractString(config_.modelType, modelTypeKeys);
    CLLM_INFO("检测到模型类型: %s", config_.modelType.c_str());
    
    // 提取上下文长度
    uint32_t maxSequenceLength = 512;
    std::vector<std::string> contextLengthKeys = {"ggml.context_length"};
    if (tryExtractUInt32(maxSequenceLength, contextLengthKeys)) {
        config_.maxSequenceLength = maxSequenceLength;
    } else if (metadata_.count("tokenizer.chat_template") > 0) {
        config_.maxSequenceLength = 4096; // 默认较大值
    }
    
    // 提取层数
    uint32_t numLayers = 6;
    std::vector<std::string> numLayersKeys = {
        "ggml.n_layers",
        "llama.layers",
        "n_layer"
    };
    if (tryExtractUInt32(numLayers, numLayersKeys)) {
        config_.numLayers = numLayers;
    }
    
    // 提取注意力头数
    uint32_t numAttentionHeads = 8;
    std::vector<std::string> attentionHeadsKeys = {
        "ggml.n_heads",
        "llama.attention.head_count",
        "n_head"
    };
    if (tryExtractUInt32(numAttentionHeads, attentionHeadsKeys)) {
        config_.numAttentionHeads = numAttentionHeads;
    }
    
    // 提取KV头数
    uint32_t numKeyValueHeads = 8;
    std::vector<std::string> kvHeadsKeys = {
        "ggml.n_kv_heads",
        "n_kv_head"
    };
    if (tryExtractUInt32(numKeyValueHeads, kvHeadsKeys)) {
        config_.numKeyValueHeads = numKeyValueHeads;
    } else {
        // 默认与注意力头数相等
        config_.numKeyValueHeads = config_.numAttentionHeads;
    }
    
    // 提取隐藏维度
    uint32_t hiddenSize = 768;
    std::vector<std::string> hiddenSizeKeys = {
        "ggml.dim",
        "llama.hidden_size",
        "llama.embedding_length",
        "n_embd"
    };
    if (tryExtractUInt32(hiddenSize, hiddenSizeKeys)) {
        config_.hiddenSize = hiddenSize;
    }
    
    // 提取词汇表大小
    uint32_t vocabSize = 32000;
    std::vector<std::string> vocabSizeKeys = {
        "ggml.vocab_size",
        "tokenizer.ggml.vocab_size"
    };
    if (tryExtractUInt32(vocabSize, vocabSizeKeys)) {
        config_.vocabSize = vocabSize;
    }
    
    // 提取中间层大小
    uint32_t intermediateSize = 3072;
    std::vector<std::string> intermediateSizeKeys = {
        "ggml.intermediate_size",
        "n_ff"
    };
    if (tryExtractUInt32(intermediateSize, intermediateSizeKeys)) {
        config_.intermediateSize = intermediateSize;
    } else {
        // 对于Llama系列，中间层大小通常是隐藏层大小的2.65倍
        config_.intermediateSize = static_cast<size_t>(config_.hiddenSize * 2.65f);
    }
    
    // 为Qwen3模型添加特定的配置支持
    if (config_.modelType == "qwen3" || config_.modelType == "qwen") {
        CLLM_INFO("应用Qwen特定配置");
        
        // 提取Qwen隐藏维度
        uint32_t qwenHiddenSize = 0;
        std::vector<std::string> qwenHiddenSizeKeys = {
            "qwen3.hidden_size",
            "qwen.hidden_size"
        };
        if (tryExtractUInt32(qwenHiddenSize, qwenHiddenSizeKeys)) {
            config_.hiddenSize = qwenHiddenSize;
        }
        
        // 提取Qwen层数
        uint32_t qwenNumLayers = 0;
        std::vector<std::string> qwenNumLayersKeys = {
            "qwen3.num_layers",
            "qwen.num_layers"
        };
        if (tryExtractUInt32(qwenNumLayers, qwenNumLayersKeys)) {
            config_.numLayers = qwenNumLayers;
        }
        
        // 提取Qwen注意力头数
        uint32_t qwenAttentionHeads = 0;
        std::vector<std::string> qwenAttentionHeadsKeys = {
            "qwen3.num_attention_heads",
            "qwen.num_attention_heads"
        };
        if (tryExtractUInt32(qwenAttentionHeads, qwenAttentionHeadsKeys)) {
            config_.numAttentionHeads = qwenAttentionHeads;
        }
        
        // 提取Qwen KV头数
        uint32_t qwenKeyValueHeads = 0;
        std::vector<std::string> qwenKeyValueHeadsKeys = {
            "qwen3.num_key_value_heads",
            "qwen.num_key_value_heads"
        };
        if (tryExtractUInt32(qwenKeyValueHeads, qwenKeyValueHeadsKeys)) {
            config_.numKeyValueHeads = qwenKeyValueHeads;
        }
        
        // 提取Qwen中间层大小
        uint32_t qwenIntermediateSize = 0;
        std::vector<std::string> qwenIntermediateSizeKeys = {
            "qwen3.intermediate_size",
            "qwen.intermediate_size"
        };
        if (tryExtractUInt32(qwenIntermediateSize, qwenIntermediateSizeKeys)) {
            config_.intermediateSize = qwenIntermediateSize;
        }
        
        // 提取Qwen最大序列长度
        uint32_t qwenContextLength = 0;
        std::vector<std::string> qwenContextLengthKeys = {
            "qwen3.context_length",
            "qwen.context_length"
        };
        if (tryExtractUInt32(qwenContextLength, qwenContextLengthKeys)) {
            config_.maxSequenceLength = qwenContextLength;
        }
    }
    
    // 检查是否使用量化
    if (!tensorInfos_.empty()) {
        const GGULTensorInfo& firstTensor = tensorInfos_[0];
        uint32_t tensorType = static_cast<uint32_t>(firstTensor.type);
        
        CLLM_INFO("第一个张量名称: %s, 类型: %u", firstTensor.name.c_str(), tensorType);
        
        // 如果是量化类型，设置量化相关配置
        if (tensorType == 2 || tensorType == 13 || tensorType == 14) {
            config_.useQuantization = true;
            switch (tensorType) {
                case 2: config_.quantizationType = "Q8_0"; break;
                case 13: config_.quantizationType = "Q4_K_M"; break;
                case 14: config_.quantizationType = "Q5_K_M"; break;
                default: config_.quantizationType = "未知"; break;
            }
        }
    }
    
    CLLM_INFO("GGUF model configuration extracted:");
    CLLM_INFO("  Model type: %s", config_.modelType.c_str());
    CLLM_INFO("  Vocab size: %u", config_.vocabSize);
    CLLM_INFO("  Hidden size: %u", config_.hiddenSize);
    CLLM_INFO("  Number of layers: %u", config_.numLayers);
    CLLM_INFO("  Attention heads: %u", config_.numAttentionHeads);
    CLLM_INFO("  KV heads: %u", config_.numKeyValueHeads);
    CLLM_INFO("  Max sequence length: %u", config_.maxSequenceLength);
    CLLM_INFO("  Intermediate size: %u", config_.intermediateSize);
}

size_t GGUFLoader::getTensorByteSize(const GGULTensorInfo& tensorInfo) const {
    // 计算张量的字节大小
    // 这里需要根据不同的张量类型实现不同的计算逻辑
    size_t elementSize = 0;
    
    // 计算元素总数
    uint64_t elementCount = 1;
    for (uint64_t dim : tensorInfo.shape) {
        elementCount *= dim;
    }
    
    size_t byteSize = 0;
    
    switch (tensorInfo.type) {
        case GGMLType::F32:
            byteSize = elementCount * 4;
            break;
        case GGMLType::F16:
        case GGMLType::BF16:
            byteSize = elementCount * 2;
            break;
        case GGMLType::Q4_0:
        case GGMLType::Q4_1:
            // Q4_0和Q4_1每个元素占用4位
            byteSize = (elementCount * 4 + 7) / 8; // 计算总位数并转换为字节数，向上取整
            break;
        case GGMLType::Q5_0:
        case GGMLType::Q5_1:
            // Q5_0和Q5_1每个元素占用5位
            byteSize = (elementCount * 5 + 7) / 8; // 计算总位数并转换为字节数，向上取整
            break;
        case GGMLType::Q8_0:
        case GGMLType::Q8_1:
            // Q8_0和Q8_1每个元素占用8位
            byteSize = elementCount * 1;
            break;
        // ... 其他类型的实现
        default:
            byteSize = elementCount * 4; // 默认4字节
            CLLM_WARN("未知张量类型 %u，将使用默认元素大小4字节", static_cast<uint32_t>(tensorInfo.type));
    }
    
    // 确保字节大小是对齐的
    return alignOffset(byteSize);
}

uint64_t GGUFLoader::getCurrentFilePosition() const {
    return currentPosition_;
}

void GGUFLoader::setFilePosition(uint64_t offset) {
    if (offset > fileSize_) {
        throw std::runtime_error("文件位置超出范围: 偏移量 " + std::to_string(offset) + 
                               " > 文件大小 " + std::to_string(fileSize_));
    }
    
    currentPosition_ = offset;
    
    if (!useMemoryMap_) {
        // 对于大文件，fseek可能使用long类型，需要检查是否超出范围
        if (offset > static_cast<uint64_t>(std::numeric_limits<long>::max())) {
            throw std::runtime_error("文件位置超出fseek支持范围: " + std::to_string(offset));
        }
        
        if (fseek(file_, static_cast<long>(offset), SEEK_SET) != 0) {
            throw std::runtime_error("无法设置文件位置: fseek失败，偏移量 " + std::to_string(offset));
        }
    }
}

uint64_t GGUFLoader::alignOffset(uint64_t offset) const {
    if (alignment_ == 0) {
        return offset;
    }
    return offset + (alignment_ - (offset % alignment_)) % alignment_;
}

bool GGUFLoader::isSystemLittleEndian() const {
    // 检查系统是否是小端字节序
    uint16_t test = 0x1234;
    return *reinterpret_cast<uint8_t*>(&test) == 0x34;
}

void GGUFLoader::initializeMemoryMap() {
#ifdef _WIN32
    // Windows内存映射实现
    fileHandle_ = CreateFile(filepath_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, 
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fileHandle_ == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("无法创建文件句柄");
    }
    
    // 创建内存映射文件
    mapHandle_ = CreateFileMapping(fileHandle_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!mapHandle_) {
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("无法创建内存映射");
    }
    
    // 将文件映射到进程地址空间
    mappedMemory_ = MapViewOfFile(mapHandle_, FILE_MAP_READ, 0, 0, 0);
    if (!mappedMemory_) {
        CloseHandle(mapHandle_);
        CloseHandle(fileHandle_);
        mapHandle_ = nullptr;
        fileHandle_ = INVALID_HANDLE_VALUE;
        throw std::runtime_error("无法将文件映射到进程地址空间");
    }
    
    CLLM_INFO("成功创建Windows内存映射，文件大小: %zu 字节", fileSize_);
#else
    // Unix/Linux内存映射实现
    fileDescriptor_ = open(filepath_.c_str(), O_RDONLY);
    if (fileDescriptor_ == -1) {
        throw std::runtime_error("无法打开文件描述符");
    }
    
    // 创建内存映射
    mappedMemory_ = mmap(nullptr, fileSize_, PROT_READ, MAP_PRIVATE, fileDescriptor_, 0);
    if (mappedMemory_ == MAP_FAILED) {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
        throw std::runtime_error("无法创建内存映射");
    }
    
    CLLM_INFO("成功创建Unix/Linux内存映射，文件大小: %zu 字节", fileSize_);
#endif
}

void GGUFLoader::releaseMemoryMap() {
#ifdef _WIN32
    if (mappedMemory_) {
        UnmapViewOfFile(mappedMemory_);
        mappedMemory_ = nullptr;
    }
    if (mapHandle_) {
        CloseHandle(mapHandle_);
        mapHandle_ = nullptr;
    }
    if (fileHandle_ != INVALID_HANDLE_VALUE) {
        CloseHandle(fileHandle_);
        fileHandle_ = INVALID_HANDLE_VALUE;
    }
#else
    if (mappedMemory_ != nullptr && mappedMemory_ != MAP_FAILED) {
        munmap(mappedMemory_, fileSize_);
        mappedMemory_ = nullptr;
    }
    if (fileDescriptor_ != -1) {
        close(fileDescriptor_);
        fileDescriptor_ = -1;
    }
#endif
}

} // namespace cllm
