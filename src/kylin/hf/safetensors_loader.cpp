/**
 * @file safetensors_loader.cpp
 * @brief Safetensors 格式加载器实现
 */

#include "cllm/kylin/hf/safetensors_loader.h"
#include "cllm/common/logger.h"

#include <fstream>
#include <cstring>
#include <stdexcept>
#include <algorithm>

// 系统头文件
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

// JSON 解析（简单实现）
#include <sstream>

namespace cllm {
namespace kylin {

// ========== 简单的 JSON 解析辅助函数 ==========

static std::string trimQuotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"') {
        return s.substr(1, s.size() - 2);
    }
    return s;
}

static bool parseJsonTensorEntry(const std::string& json, 
                                 const std::string& key,
                                 TensorMeta& meta) {
    // 查找 "key": {...}
    std::string pattern = "\"" + key + "\"";
    size_t keyPos = json.find(pattern);
    if (keyPos == std::string::npos) return false;
    
    // 找到对应的 {}
    size_t braceStart = json.find('{', keyPos);
    if (braceStart == std::string::npos) return false;
    
    int braceCount = 1;
    size_t braceEnd = braceStart + 1;
    while (braceEnd < json.size() && braceCount > 0) {
        if (json[braceEnd] == '{') braceCount++;
        else if (json[braceEnd] == '}') braceCount--;
        braceEnd++;
    }
    
    std::string entry = json.substr(braceStart, braceEnd - braceStart);
    meta.name = key;
    
    // 解析 dtype
    size_t dtypePos = entry.find("\"dtype\"");
    if (dtypePos != std::string::npos) {
        size_t colonPos = entry.find(':', dtypePos);
        size_t valueStart = entry.find('"', colonPos);
        size_t valueEnd = entry.find('"', valueStart + 1);
        if (valueStart != std::string::npos && valueEnd != std::string::npos) {
            meta.dtype = entry.substr(valueStart + 1, valueEnd - valueStart - 1);
        }
    }
    
    // 解析 shape
    size_t shapePos = entry.find("\"shape\"");
    if (shapePos != std::string::npos) {
        size_t arrayStart = entry.find('[', shapePos);
        size_t arrayEnd = entry.find(']', arrayStart);
        if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
            std::string shapeStr = entry.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
            std::istringstream iss(shapeStr);
            std::string token;
            while (std::getline(iss, token, ',')) {
                // 去除空格
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty()) {
                    meta.shape.push_back(std::stoll(token));
                }
            }
        }
    }
    
    // 解析 data_offsets
    size_t offsetsPos = entry.find("\"data_offsets\"");
    if (offsetsPos != std::string::npos) {
        size_t arrayStart = entry.find('[', offsetsPos);
        size_t arrayEnd = entry.find(']', arrayStart);
        if (arrayStart != std::string::npos && arrayEnd != std::string::npos) {
            std::string offsetsStr = entry.substr(arrayStart + 1, arrayEnd - arrayStart - 1);
            std::istringstream iss(offsetsStr);
            std::string token;
            std::vector<size_t> offsets;
            while (std::getline(iss, token, ',')) {
                token.erase(std::remove_if(token.begin(), token.end(), ::isspace), token.end());
                if (!token.empty()) {
                    offsets.push_back(std::stoull(token));
                }
            }
            if (offsets.size() >= 2) {
                meta.dataOffset = offsets[0];
                meta.dataSize = offsets[1] - offsets[0];
            }
        }
    }
    
    return true;
}

// ========== SafetensorsLoader 实现 ==========

SafetensorsLoader::SafetensorsLoader(const std::string& path)
    : path_(path) {
    
    CLLM_INFO("[SafetensorsLoader] Loading: %s", path.c_str());
    
    // 打开文件
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        CLLM_ERROR("[SafetensorsLoader] Failed to open file: %s", path.c_str());
        return;
    }
    
    // 获取文件大小
    struct stat st;
    if (fstat(fd_, &st) < 0) {
        CLLM_ERROR("[SafetensorsLoader] Failed to get file size");
        close(fd_);
        fd_ = -1;
        return;
    }
    mappedSize_ = static_cast<size_t>(st.st_size);
    
    // 内存映射
    mappedData_ = mmap(nullptr, mappedSize_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mappedData_ == MAP_FAILED) {
        CLLM_ERROR("[SafetensorsLoader] Failed to mmap file");
        close(fd_);
        fd_ = -1;
        mappedData_ = nullptr;
        return;
    }
    
    // 解析头部
    if (!parseHeader()) {
        munmap(mappedData_, mappedSize_);
        close(fd_);
        fd_ = -1;
        mappedData_ = nullptr;
        return;
    }
    
    CLLM_INFO("[SafetensorsLoader] Loaded %zu tensors", tensors_.size());
}

SafetensorsLoader::~SafetensorsLoader() {
    if (mappedData_) {
        munmap(mappedData_, mappedSize_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool SafetensorsLoader::parseHeader() {
    if (!mappedData_ || mappedSize_ < 8) {
        return false;
    }
    
    // 读取头部大小（前 8 字节，little-endian uint64）
    const uint8_t* data = static_cast<const uint8_t*>(mappedData_);
    headerSize_ = 0;
    for (int i = 0; i < 8; ++i) {
        headerSize_ |= static_cast<uint64_t>(data[i]) << (i * 8);
    }
    
    if (headerSize_ > mappedSize_ - 8) {
        CLLM_ERROR("[SafetensorsLoader] Invalid header size: %zu", headerSize_);
        return false;
    }
    
    // 读取 JSON 头部
    std::string headerJson(reinterpret_cast<const char*>(data + 8), headerSize_);
    
    // 数据区起始位置
    dataStart_ = reinterpret_cast<const char*>(data + 8 + headerSize_);
    
    // 解析每个张量
    // 遍历查找所有 "name": { 模式
    size_t pos = 0;
    while (pos < headerJson.size()) {
        // 找下一个 "
        size_t nameStart = headerJson.find('"', pos);
        if (nameStart == std::string::npos) break;
        
        size_t nameEnd = headerJson.find('"', nameStart + 1);
        if (nameEnd == std::string::npos) break;
        
        std::string name = headerJson.substr(nameStart + 1, nameEnd - nameStart - 1);
        
        // 跳过 __metadata__
        if (name == "__metadata__") {
            pos = nameEnd + 1;
            // 跳过整个 metadata 对象
            size_t braceStart = headerJson.find('{', pos);
            if (braceStart != std::string::npos) {
                int braceCount = 1;
                size_t braceEnd = braceStart + 1;
                while (braceEnd < headerJson.size() && braceCount > 0) {
                    if (headerJson[braceEnd] == '{') braceCount++;
                    else if (headerJson[braceEnd] == '}') braceCount--;
                    braceEnd++;
                }
                pos = braceEnd;
            }
            continue;
        }
        
        // 尝试解析为张量
        TensorMeta meta;
        if (parseJsonTensorEntry(headerJson, name, meta)) {
            tensors_[name] = meta;
        }
        
        pos = nameEnd + 1;
    }
    
    return !tensors_.empty();
}

std::vector<std::string> SafetensorsLoader::getTensorNames() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& pair : tensors_) {
        names.push_back(pair.first);
    }
    std::sort(names.begin(), names.end());
    return names;
}

const TensorMeta* SafetensorsLoader::getTensorMeta(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? &it->second : nullptr;
}

const void* SafetensorsLoader::getTensorData(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !dataStart_) {
        return nullptr;
    }
    return dataStart_ + it->second.dataOffset;
}

std::vector<int64_t> SafetensorsLoader::getTensorShape(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? it->second.shape : std::vector<int64_t>{};
}

std::string SafetensorsLoader::getTensorDtype(const std::string& name) const {
    auto it = tensors_.find(name);
    return (it != tensors_.end()) ? it->second.dtype : "";
}

size_t SafetensorsLoader::getTensorNumElements(const std::string& name) const {
    auto shape = getTensorShape(name);
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto dim : shape) {
        n *= static_cast<size_t>(dim);
    }
    return n;
}

std::vector<float> SafetensorsLoader::getTensorAsF32(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end() || !dataStart_) {
        return {};
    }
    
    const TensorMeta& meta = it->second;
    size_t numElements = getTensorNumElements(name);
    std::vector<float> result(numElements);
    
    const void* data = dataStart_ + meta.dataOffset;
    
    if (meta.dtype == "BF16" || meta.dtype == "bfloat16") {
        bf16ToF32Array(static_cast<const uint16_t*>(data), result.data(), numElements);
    } else if (meta.dtype == "F16" || meta.dtype == "float16") {
        f16ToF32Array(static_cast<const uint16_t*>(data), result.data(), numElements);
    } else if (meta.dtype == "F32" || meta.dtype == "float32") {
        std::memcpy(result.data(), data, numElements * sizeof(float));
    } else {
        CLLM_WARN("[SafetensorsLoader] Unsupported dtype: %s", meta.dtype.c_str());
        return {};
    }
    
    return result;
}

// ========== BF16/F16 转换实现 ==========

void bf16ToF32Array(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = bf16ToF32(src[i]);
    }
}

float f16ToF32(uint16_t f16) {
    // F16: 1 sign + 5 exp + 10 mantissa
    // F32: 1 sign + 8 exp + 23 mantissa
    
    uint32_t sign = (f16 >> 15) & 0x1;
    uint32_t exp = (f16 >> 10) & 0x1F;
    uint32_t mantissa = f16 & 0x3FF;
    
    uint32_t f32_bits;
    
    if (exp == 0) {
        if (mantissa == 0) {
            // Zero
            f32_bits = sign << 31;
        } else {
            // Subnormal
            exp = 1;
            while ((mantissa & 0x400) == 0) {
                mantissa <<= 1;
                exp--;
            }
            mantissa &= 0x3FF;
            f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
        }
    } else if (exp == 31) {
        // Inf or NaN
        f32_bits = (sign << 31) | (0xFF << 23) | (mantissa << 13);
    } else {
        // Normal
        f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (mantissa << 13);
    }
    
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

void f16ToF32Array(const uint16_t* src, float* dst, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        dst[i] = f16ToF32(src[i]);
    }
}

} // namespace kylin
} // namespace cllm
