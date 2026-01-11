/**
 * @file tensor.h
 * @brief 简化版张量类，用于自研推理引擎MVP阶段
 */
#pragma once

#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace cllm {
namespace kylin {

/**
 * @brief 数据类型枚举（MVP阶段仅支持FP32）
 */
enum class DataType {
    FP32
};

/**
 * @brief 设备类型枚举（MVP阶段仅支持CPU）
 */
enum class Device {
    CPU
};

/**
 * @brief 简化版张量类
 *
 * MVP阶段的目标是提供一个足够承载 Transformer 前向计算的最小实现：
 * - 仅支持 float 数据类型
 * - 仅支持 CPU 设备
 * - 以 row-major 方式存储
 * - 形状信息通过 std::vector<size_t> 维护
 */
class Tensor {
public:
    /// 默认构造，得到一个空张量
    Tensor() = default;

    /// 通过形状构造张量
    explicit Tensor(const std::vector<size_t>& shape)
        : shape_(shape) {
        allocate();
    }

    /// 通过初始化列表构造张量，例如 Tensor({batch, seq, hidden})
    Tensor(std::initializer_list<size_t> shape)
        : shape_(shape) {
        allocate();
    }

    /// 获取张量形状
    const std::vector<size_t>& shape() const {
        return shape_;
    }

    /// 获取维度个数
    size_t ndim() const {
        return shape_.size();
    }

    /// 获取元素总数
    size_t size() const {
        return data_.size();
    }

    /// 获取底层数据指针
    float* data() {
        return data_.data();
    }

    /// 获取常量数据指针
    const float* data() const {
        return data_.data();
    }

    /// 按一维索引访问元素（调用方需保证索引合法）
    float& operator[](size_t index) {
        return data_.at(index);
    }

    /// 按一维索引访问元素（常量版本）
    const float& operator[](size_t index) const {
        return data_.at(index);
    }

    /// 重新设置形状并重新分配内存
    void resize(const std::vector<size_t>& newShape) {
        shape_ = newShape;
        allocate();
    }

    /// 将所有元素填充为指定值
    void fill(float value) {
        std::fill(data_.begin(), data_.end(), value);
    }

private:
    std::vector<size_t> shape_;
    std::vector<float> data_;

    void allocate() {
        size_t total = 1;
        for (size_t dim : shape_) {
            total *= dim;
        }
        data_.assign(total, 0.0f);
    }
};

}  // namespace kylin
}  // namespace cllm
