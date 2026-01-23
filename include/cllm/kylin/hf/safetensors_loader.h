/**
 * @file safetensors_loader.h
 * @brief Safetensors 格式加载器
 * 
 * 支持直接加载 HuggingFace safetensors 格式的模型权重
 */

#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <memory>

namespace cllm {
namespace kylin {

/**
 * @brief 张量元信息
 */
struct TensorMeta {
    std::string name;
    std::string dtype;           // "BF16", "F16", "F32", etc.
    std::vector<int64_t> shape;
    size_t dataOffset = 0;       // 在数据区的偏移
    size_t dataSize = 0;         // 数据大小（字节）
};

/**
 * @brief Safetensors 文件加载器
 * 
 * 使用内存映射高效加载 safetensors 格式文件
 */
class SafetensorsLoader {
public:
    explicit SafetensorsLoader(const std::string& path);
    ~SafetensorsLoader();
    
    // 禁止拷贝
    SafetensorsLoader(const SafetensorsLoader&) = delete;
    SafetensorsLoader& operator=(const SafetensorsLoader&) = delete;
    
    /**
     * @brief 检查文件是否有效
     */
    bool isValid() const { return mappedData_ != nullptr; }
    
    /**
     * @brief 获取所有张量名
     */
    std::vector<std::string> getTensorNames() const;
    
    /**
     * @brief 获取张量元信息
     */
    const TensorMeta* getTensorMeta(const std::string& name) const;
    
    /**
     * @brief 获取张量原始数据指针（零拷贝）
     * 
     * @param name 张量名
     * @return 指向原始数据的指针，类型取决于 dtype
     */
    const void* getTensorData(const std::string& name) const;
    
    /**
     * @brief 获取张量形状
     */
    std::vector<int64_t> getTensorShape(const std::string& name) const;
    
    /**
     * @brief 获取张量数据类型
     */
    std::string getTensorDtype(const std::string& name) const;
    
    /**
     * @brief 将张量数据转换为 F32
     * 
     * @param name 张量名
     * @return F32 数据向量
     */
    std::vector<float> getTensorAsF32(const std::string& name) const;
    
    /**
     * @brief 获取张量元素数量
     */
    size_t getTensorNumElements(const std::string& name) const;
    
private:
    bool parseHeader();
    
    std::string path_;
    int fd_ = -1;
    void* mappedData_ = nullptr;
    size_t mappedSize_ = 0;
    size_t headerSize_ = 0;
    const char* dataStart_ = nullptr;
    
    std::unordered_map<std::string, TensorMeta> tensors_;
};

// ========== BF16 工具函数 ==========

/**
 * @brief BF16 转 F32（单个值）
 */
inline float bf16ToF32(uint16_t bf16) {
    uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &f32_bits, sizeof(float));
    return result;
}

/**
 * @brief BF16 数组转 F32 数组
 */
void bf16ToF32Array(const uint16_t* src, float* dst, size_t count);

/**
 * @brief F16 转 F32（单个值）
 */
float f16ToF32(uint16_t f16);

/**
 * @brief F16 数组转 F32 数组
 */
void f16ToF32Array(const uint16_t* src, float* dst, size_t count);

} // namespace kylin
} // namespace cllm
