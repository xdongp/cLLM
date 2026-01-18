#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include "cllm/common/logger.h"
#include "cllm/model/weight_data.h"
#include "cllm/model/loader_interface.h"

// 跨平台内存映射支持
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace cllm {

// GGUF张量类型枚举 (与GGUF规范一致)
enum class GGMLType : uint32_t {
    F32     = 0,
    F16     = 1,
    Q4_0    = 2,
    Q4_1    = 3,
    // Q4_2 = 4, 已移除
    // Q4_3 = 5, 已移除
    Q5_0    = 6,
    Q5_1    = 7,
    Q8_0    = 8,
    Q8_1    = 9,
    Q2_K    = 10,
    Q3_K    = 11,
    Q4_K    = 12,
    Q5_K    = 13,
    Q6_K    = 14,
    Q8_K    = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8      = 24,
    I16     = 25,
    I32     = 26,
    I64     = 27,
    F64     = 28,
    IQ1_M   = 29,
    BF16    = 30,
    TQ1_0   = 34,
    TQ2_0   = 35,
    MXFP4   = 39,
    COUNT   = 40
};

// GGUF元数据值类型枚举 (与GGUF规范一致)
enum class GGUFValueType : uint32_t {
    UINT8    = 0,
    INT8     = 1,
    UINT16   = 2,
    INT16    = 3,
    UINT32   = 4,
    INT32    = 5,
    FLOAT32  = 6,
    BOOL     = 7,
    STRING   = 8,
    ARRAY    = 9,
    UINT64   = 10,
    INT64    = 11,
    FLOAT64  = 12,
    COUNT    = 13  // 用于范围检查
};

// GGUF文件头结构
struct GGUFHeader {
    uint32_t magic;           // 魔数 0x46554747 (GGUF)
    uint32_t version;         // 版本号 (规范中版本3引入大端支持)
    uint64_t tensorCount;     // 张量数量
    uint64_t metadataCount;   // 元数据数量
};

// GGUF元数据项
struct GGUFMetadata {
    std::string key;
    GGUFValueType type;
    
    // 存储不同类型的值
    union {
        uint8_t u8_val;
        int8_t i8_val;
        uint16_t u16_val;
        int16_t i16_val;
        uint32_t u32_val;
        int32_t i32_val;
        float f32_val;
        uint64_t u64_val;
        int64_t i64_val;
        double f64_val;
        bool bool_val;
    } value;
    
    // 字符串值
    std::string string_val;
    
    // 数组值 (支持嵌套数组)
    struct ArrayData {
        GGUFValueType elementType;
        uint64_t elementCount;
        std::vector<GGUFMetadata> elements;
    } array_val;
};

// GGUF张量信息结构
struct GGULTensorInfo {
    std::string name;
    uint32_t dimensions;
    std::vector<uint64_t> shape;
    GGMLType type;
    uint64_t offset; // 张量数据在“数据段(data section)”起点的相对偏移量（GGUF规范）
};

// GGUFLoader类，实现IModelLoader接口
class GGUFLoader : public cllm::IModelLoader {
public:
    GGUFLoader(const std::string& filepath, bool useMemoryMap = true, bool allowMultipleMappings = true);
    ~GGUFLoader() override;
    
    // 实现IModelLoader接口
    bool load() override;
    bool loadWeights(cllm::model::ModelWeights& weights, bool loadAll = true) override;
    bool loadWeightByName(const std::string& name, cllm::model::WeightData& weight) override;
    bool hasWeight(const std::string& name) override;
    bool loadInto(
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
    ) override;
    const ModelConfig& getConfig() const override;
    const std::string& getModelPath() const override;
    kylin::WeightDType getDType() const override;
    
    // 获取元数据（供Tokenizer使用）
    const std::unordered_map<std::string, GGUFMetadata>& getMetadata() const {
        return metadata_;
    }
    
    // 获取张量信息
    const std::vector<GGULTensorInfo>& getTensorInfos() const {
        return tensorInfos_;
    }
    
    // 兼容性方法：解析张量数据（返回旧格式的映射）
    std::unordered_map<std::string, uint64_t> parseTensorData(uint64_t tensorCount) {
        if (tensorInfos_.empty() && tensorCount > 0) {
            parseTensorInfos(tensorCount);
        }
        
        std::unordered_map<std::string, uint64_t> tensorOffsets;
        for (const auto& tensorInfo : tensorInfos_) {
            tensorOffsets[tensorInfo.name] = tensorInfo.offset;
        }
        return tensorOffsets;
    }
    
    // 兼容性方法：获取张量偏移量（供Tokenizer使用）
    const std::unordered_map<std::string, uint64_t> getTensorOffsets() const {
        std::unordered_map<std::string, uint64_t> tensorOffsets;
        for (const auto& tensorInfo : tensorInfos_) {
            tensorOffsets[tensorInfo.name] = tensorInfo.offset;
        }
        return tensorOffsets;
    }
    
    // 加载Tokenizer元数据（兼容性方法）
    void loadTokenizerMetadata() {
        // 从GGUF元数据中提取Tokenizer配置
        try {
            // 检查是否已经解析了元数据
            if (metadata_.empty()) {
                // 如果没有解析元数据，先解析文件头获取元数据数量
                GGUFHeader header = parseHeader();
                parseMetadata(header.metadataCount);
            }
            
            // 提取Tokenizer类型信息
            auto it = metadata_.find("tokenizer.ggml.model");
            if (it != metadata_.end() && it->second.type == GGUFValueType::STRING) {
                CLLM_INFO("Tokenizer model type: %s", it->second.string_val.c_str());
            }
            
            // 提取词汇表大小
            it = metadata_.find("tokenizer.ggml.vocab_size");
            if (it != metadata_.end()) {
                switch (it->second.type) {
                    case GGUFValueType::UINT32:
                        CLLM_INFO("Vocab size: %u", it->second.value.u32_val);
                        break;
                    case GGUFValueType::UINT64:
                        CLLM_INFO("Vocab size: %llu", it->second.value.u64_val);
                        break;
                    case GGUFValueType::INT32:
                        CLLM_INFO("Vocab size: %d", it->second.value.i32_val);
                        break;
                    case GGUFValueType::INT64:
                        CLLM_INFO("Vocab size: %lld", it->second.value.i64_val);
                        break;
                    default:
                        break;
                }
            }
            
            // 提取特殊Token信息
            it = metadata_.find("tokenizer.ggml.bos_token_id");
            if (it != metadata_.end()) {
                switch (it->second.type) {
                    case GGUFValueType::UINT32:
                        CLLM_INFO("BOS token ID: %u", it->second.value.u32_val);
                        break;
                    case GGUFValueType::INT32:
                        CLLM_INFO("BOS token ID: %d", it->second.value.i32_val);
                        break;
                    default:
                        break;
                }
            }
            
            it = metadata_.find("tokenizer.ggml.eos_token_id");
            if (it != metadata_.end()) {
                switch (it->second.type) {
                    case GGUFValueType::UINT32:
                        CLLM_INFO("EOS token ID: %u", it->second.value.u32_val);
                        break;
                    case GGUFValueType::INT32:
                        CLLM_INFO("EOS token ID: %d", it->second.value.i32_val);
                        break;
                    default:
                        break;
                }
            }
            
            it = metadata_.find("tokenizer.ggml.unk_token_id");
            if (it != metadata_.end()) {
                switch (it->second.type) {
                    case GGUFValueType::UINT32:
                        CLLM_INFO("UNK token ID: %u", it->second.value.u32_val);
                        break;
                    case GGUFValueType::INT32:
                        CLLM_INFO("UNK token ID: %d", it->second.value.i32_val);
                        break;
                    default:
                        break;
                }
            }
            
            it = metadata_.find("tokenizer.ggml.pad_token_id");
            if (it != metadata_.end()) {
                switch (it->second.type) {
                    case GGUFValueType::UINT32:
                        CLLM_INFO("PAD token ID: %u", it->second.value.u32_val);
                        break;
                    case GGUFValueType::INT32:
                        CLLM_INFO("PAD token ID: %d", it->second.value.i32_val);
                        break;
                    default:
                        break;
                }
            }
            
            // 提取词汇表数据
            it = metadata_.find("tokenizer.ggml.vocab");
            if (it != metadata_.end() && it->second.type == GGUFValueType::ARRAY) {
                CLLM_INFO("Vocabulary data found");
            }
            
            // 提取合并规则
            it = metadata_.find("tokenizer.ggml.merges");
            if (it != metadata_.end() && it->second.type == GGUFValueType::ARRAY) {
                CLLM_INFO("Merge rules found");
            }
            
            // 提取其他可能的Tokenizer配置
            it = metadata_.find("tokenizer.ggml.tokens");
            if (it != metadata_.end()) {
                CLLM_INFO("Tokens data found, type: %u", static_cast<uint32_t>(it->second.type));
            }
            
            it = metadata_.find("tokenizer.ggml.scores");
            if (it != metadata_.end()) {
                CLLM_INFO("Scores data found, type: %u", static_cast<uint32_t>(it->second.type));
            }
            
            // 提取BPE相关配置
            it = metadata_.find("tokenizer.ggml.bpe_ranks");
            if (it != metadata_.end()) {
                CLLM_INFO("BPE ranks data found, type: %u", static_cast<uint32_t>(it->second.type));
            }
            
        } catch (const std::exception& e) {
            CLLM_ERROR("GGUFLoader::loadTokenizerMetadata: %s", e.what());
            throw;
        }
    }

private:
    std::string filepath_;
    FILE* file_;
    ModelConfig config_;
    std::unordered_map<std::string, GGUFMetadata> metadata_;
    std::vector<GGULTensorInfo> tensorInfos_; // 存储所有张量信息
    std::unordered_map<std::string, size_t> tensorNameMap_; // 张量名称到索引的映射
    
    // GGUF版本号和对齐值
    uint32_t ggufVersion_;
    uint32_t alignment_; // 全局对齐值，默认32

    // GGUF数据段(data section)在文件中的起始位置
    // GGUF张量的offset是相对该位置的偏移
    uint64_t dataSectionOffset_ = 0;
    
    // 内存映射相关变量
    bool useMemoryMap_;
    size_t fileSize_;
    uint64_t currentPosition_;
    bool needByteOrderSwap_; // 是否需要字节序转换
    bool allowMultipleMappings_; // 是否允许多个映射共享同一文件
    
#ifdef _WIN32
    HANDLE fileHandle_;
    HANDLE mapHandle_;
    LPVOID mappedMemory_;
#else
    int fileDescriptor_;
    void* mappedMemory_;
#endif
    
    // 初始化内存映射
    void initializeMemoryMap();
    
    // 释放内存映射
    void releaseMemoryMap();
    
    // 获取当前文件位置
    uint64_t getCurrentFilePosition() const;
    
    // 设置文件位置
    void setFilePosition(uint64_t offset);
    
    // 批量读取数据（支持文件I/O和内存映射）
    template<typename T>
    size_t readValues(T* buffer, size_t count) {
        if (count == 0) {
            return 0;
        }
        
        size_t totalBytes = count * sizeof(T);
        
        if (useMemoryMap_) {
            // 从内存映射中批量读取
            if (currentPosition_ + totalBytes > fileSize_) {
                throw std::runtime_error("内存映射读取越界: " + std::to_string(currentPosition_) + " + " + std::to_string(totalBytes) + " > " + std::to_string(fileSize_));
            }
            
            // 内存映射模式下，使用memcpy进行批量复制，利用CPU缓存行
            if (totalBytes >= 64) { // 大于等于一个缓存行大小
                // 确保目标缓冲区对齐，提高缓存效率
                if (reinterpret_cast<uintptr_t>(buffer) % 64 != 0) {
                    // 如果目标缓冲区不对齐，使用临时对齐缓冲区
                    T* tempBuffer = static_cast<T*>(aligned_alloc(64, totalBytes));
                    if (tempBuffer) {
                        memcpy(tempBuffer, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, totalBytes);
                        memcpy(buffer, tempBuffer, totalBytes);
                        free(tempBuffer);
                    } else {
                        // 对齐失败，使用普通memcpy
                        memcpy(buffer, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, totalBytes);
                    }
                } else {
                    // 目标缓冲区已经对齐，直接复制
                    memcpy(buffer, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, totalBytes);
                }
            } else {
                // 小数据量，直接复制
                memcpy(buffer, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, totalBytes);
            }
            
            currentPosition_ += totalBytes;
        } else {
            // 文件I/O模式下，先检查边界
            if (currentPosition_ + totalBytes > fileSize_) {
                throw std::runtime_error("文件I/O读取越界: 位置 " + std::to_string(currentPosition_) + " + 大小 " + std::to_string(totalBytes) + " > 文件大小 " + std::to_string(fileSize_));
            }
            
            // 使用fread的批量读取功能
            // 可以让系统自动管理缓冲区，不需要手动分配
            // setvbuf(file_, nullptr, _IOFBF, BUFSIZ);
            
            // 批量读取数据
            size_t result = fread(buffer, sizeof(T), count, file_);
            if (result != count) {
                throw std::runtime_error("无法读取批量数据: 尝试读取 " + std::to_string(count) + " 个元素，实际读取 " + std::to_string(result) + " 个元素");
            }
            
            // 手动更新currentPosition_，不依赖ftell()
            currentPosition_ += result * sizeof(T);
        }
        
        // 进行字节序转换（如果需要）
        if (needByteOrderSwap_) {
            convertByteOrder(buffer, count);
        }
        
        return count;
    }
    
    // 读取单个值（支持文件I/O和内存映射）
    template<typename T>
    T readValue() {
        T value;
        readValues(&value, 1);
        return value;
    }
    
    // 读取GGUF字符串（包含长度前缀）
    std::string readString() {
        // GGUF格式中，字符串的长度前缀是uint64_t类型
        uint64_t length;
        
        // 直接读取原始字节，手动处理字节序
        uint8_t length_bytes[8];
        if (useMemoryMap_) {
            if (currentPosition_ + 8 > fileSize_) {
                throw std::runtime_error("文件太小，无法读取字符串长度");
            }
            memcpy(length_bytes, static_cast<uint8_t*>(mappedMemory_) + currentPosition_, 8);
            currentPosition_ += 8;
        } else {
            if (fread(length_bytes, 1, 8, file_) != 8) {
                throw std::runtime_error("无法读取字符串长度");
            }
            currentPosition_ += 8;
        }
        
        // 根据字节序转换长度
        if (needByteOrderSwap_) {
            // 大端到小端转换
            length = (static_cast<uint64_t>(length_bytes[0]) << 56) |
                    (static_cast<uint64_t>(length_bytes[1]) << 48) |
                    (static_cast<uint64_t>(length_bytes[2]) << 40) |
                    (static_cast<uint64_t>(length_bytes[3]) << 32) |
                    (static_cast<uint64_t>(length_bytes[4]) << 24) |
                    (static_cast<uint64_t>(length_bytes[5]) << 16) |
                    (static_cast<uint64_t>(length_bytes[6]) << 8) |
                    (static_cast<uint64_t>(length_bytes[7]) << 0);
        } else {
            // 小端直接转换（第一个字节是最低有效位）
            length = (static_cast<uint64_t>(length_bytes[0]) << 0) |
                    (static_cast<uint64_t>(length_bytes[1]) << 8) |
                    (static_cast<uint64_t>(length_bytes[2]) << 16) |
                    (static_cast<uint64_t>(length_bytes[3]) << 24) |
                    (static_cast<uint64_t>(length_bytes[4]) << 32) |
                    (static_cast<uint64_t>(length_bytes[5]) << 40) |
                    (static_cast<uint64_t>(length_bytes[6]) << 48) |
                    (static_cast<uint64_t>(length_bytes[7]) << 56);
        }
        
        // 验证字符串长度是否合理，防止恶意文件或损坏文件导致内存分配过大
        // 设置合理的上限：1MB，对于键名和元数据字符串来说已经足够大
        const uint64_t MAX_STRING_LENGTH = 1024 * 1024;
        if (length > MAX_STRING_LENGTH) {
            throw std::runtime_error("字符串长度异常: " + std::to_string(length) + 
                                   " 超过最大允许长度 " + std::to_string(MAX_STRING_LENGTH));
        }
        
        // 检查长度是否超出文件剩余大小
        if (currentPosition_ + length > fileSize_) {
            throw std::runtime_error("字符串长度超出文件大小: 位置 " + std::to_string(currentPosition_) + 
                                   " + 长度 " + std::to_string(length) + " > 文件大小 " + 
                                   std::to_string(fileSize_));
        }
        
        return readRawString(length);
    }
    
    // 直接读取指定长度的字符串（不包含长度前缀）
    std::string readRawString(uint64_t length) {
        if (useMemoryMap_) {
            // 从内存映射中读取字符串
            if (currentPosition_ + length > fileSize_) {
                throw std::runtime_error("内存映射读取越界: " + std::to_string(currentPosition_) + " + " + std::to_string(length) + " > " + std::to_string(fileSize_));
            }
            
            // 直接使用内存映射中的数据构造字符串，避免额外的内存分配
            const char* strPtr = static_cast<const char*>(mappedMemory_) + currentPosition_;
            std::string str(strPtr, length);
            currentPosition_ += length;
            return str;
        } else {
            // 从文件中读取字符串
            // 先检查边界，防止越界访问
            if (currentPosition_ + length > fileSize_) {
                throw std::runtime_error("文件I/O读取字符串越界: 位置 " + std::to_string(currentPosition_) + 
                                       " + 长度 " + std::to_string(length) + " > 文件大小 " + 
                                       std::to_string(fileSize_));
            }
            
            // 预分配字符串容量，避免多次扩容
            std::string str;
            str.reserve(length);
            str.resize(length);
            
            size_t result = fread(&str[0], 1, length, file_);
            if (result != length) {
                throw std::runtime_error("无法读取字符串: 尝试读取 " + std::to_string(length) + 
                                       " 个字符，实际读取 " + std::to_string(result) + " 个字符");
            }
            
            // 手动更新currentPosition_，不依赖ftell()
            currentPosition_ += result;
            
            return str;
        }
    }
    
    // 读取数组
    GGUFMetadata::ArrayData readArray() {
        // 数组类型存储格式：类型 + 长度 + 元素
        GGUFValueType elementType = static_cast<GGUFValueType>(readValue<uint32_t>());
        uint64_t elementCount = readValue<uint64_t>();
        
        GGUFMetadata::ArrayData array;
        array.elementType = elementType;
        array.elementCount = elementCount;
        array.elements.reserve(elementCount);
        
        for (uint64_t j = 0; j < elementCount; ++j) {
            GGUFMetadata element;
            element.type = elementType;
            
            // 根据数组元素类型读取值
            readMetadataValue(element);
            
            array.elements.push_back(element);
        }
        
        return array;
    }
    
    // 读取元数据值
    void readMetadataValue(GGUFMetadata& metadata) {
        switch (metadata.type) {
            case GGUFValueType::UINT8:
                metadata.value.u8_val = readValue<uint8_t>();
                break;
            case GGUFValueType::INT8:
                metadata.value.i8_val = readValue<int8_t>();
                break;
            case GGUFValueType::UINT16:
                metadata.value.u16_val = readValue<uint16_t>();
                break;
            case GGUFValueType::INT16:
                metadata.value.i16_val = readValue<int16_t>();
                break;
            case GGUFValueType::UINT32:
                metadata.value.u32_val = readValue<uint32_t>();
                break;
            case GGUFValueType::INT32:
                metadata.value.i32_val = readValue<int32_t>();
                break;
            case GGUFValueType::FLOAT32:
                metadata.value.f32_val = readValue<float>();
                break;
            case GGUFValueType::BOOL:
                // BOOL类型存储为1字节
                metadata.value.bool_val = readValue<uint8_t>() != 0;
                break;
            case GGUFValueType::STRING:
                metadata.string_val = readString();
                break;
            case GGUFValueType::ARRAY:
                metadata.array_val = readArray();
                break;
            case GGUFValueType::UINT64:
                metadata.value.u64_val = readValue<uint64_t>();
                break;
            case GGUFValueType::INT64:
                metadata.value.i64_val = readValue<int64_t>();
                break;
            case GGUFValueType::FLOAT64:
                metadata.value.f64_val = readValue<double>();
                break;
            default:
                throw std::runtime_error("读取元数据时遇到未知类型: " + std::to_string(static_cast<uint32_t>(metadata.type)));
        }
    }
    
    // 解析GGUF文件头（公开方法，供测试使用）
    GGUFHeader parseHeader();
    
    // 解析元数据（公开方法，供测试使用）
    void parseMetadata(uint64_t metadataCount);
    
    // 解析张量信息（私有方法）
    void parseTensorInfos(uint64_t tensorCount);
    
    // 计算对齐后的偏移量
    uint64_t alignOffset(uint64_t offset) const;
    
    // 转换字节序（仅在需要时使用）
    template<typename T>
    void convertByteOrder(T* buffer, size_t count) {
        // 检查T是否是基本类型
        static_assert(std::is_arithmetic<T>::value, "Byte order conversion only supports arithmetic types");
        
        if (count == 0 || !needByteOrderSwap_) {
            return;
        }
        
        for (size_t i = 0; i < count; ++i) {
            buffer[i] = swapByteOrder(buffer[i]);
        }
    }
    
    // 交换单个值的字节序
    template<typename T>
    T swapByteOrder(T value) {
        static_assert(std::is_arithmetic<T>::value, "Byte order conversion only supports arithmetic types");
        
        // 对于1字节类型，无需转换
        if (sizeof(T) == 1) {
            return value;
        }
        
        // 创建一个字节数组并复制值
        uint8_t bytes[sizeof(T)];
        std::memcpy(bytes, &value, sizeof(T));
        
        // 反转字节数组
        for (size_t i = 0; i < sizeof(T) / 2; ++i) {
            std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
        }
        
        // 将反转后的字节复制回值
        std::memcpy(&value, bytes, sizeof(T));
        
        return value;
    }
    
    // 检查系统字节序
    bool isSystemLittleEndian() const;
    
    // 从元数据中提取模型配置
    void extractModelConfig();
    
    // 辅助函数：从元数据中提取uint32_t值
    template<typename T>
    uint32_t extractUInt32(const GGUFMetadata& metadata);
    
    // 辅助函数：尝试从多个元数据键中提取uint32_t值
    bool tryExtractUInt32(uint32_t& result, const std::vector<std::string>& keys);
    
    // 辅助函数：尝试从多个元数据键中提取字符串值
    bool tryExtractString(std::string& result, const std::vector<std::string>& keys);
    
    // 辅助函数：尝试从多个元数据键中提取浮点数值
    bool tryExtractFloat32(float& result, const std::vector<std::string>& keys);
    
    // 获取张量大小（字节数）
    size_t getTensorByteSize(const GGULTensorInfo& tensorInfo) const;
};

} // namespace cllm
