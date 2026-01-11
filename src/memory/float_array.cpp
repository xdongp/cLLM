/**
 * @file float_array.cpp
 * @brief 浮点数组RAII包装器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/memory/float_array.h"
#include <cstring>
#include <stdexcept>

namespace cllm {

FloatArray::FloatArray(size_t size)
    : data_(new float[size > 0 ? size : 1]), size_(size) {
}

FloatArray::~FloatArray() {
    if (data_ != nullptr) {
        delete[] data_;
    }
}

FloatArray::FloatArray(const FloatArray& other)
    : data_(nullptr), size_(other.size_) {
    if (size_ > 0) {
        data_ = new float[size_];
        std::memcpy(data_, other.data_, size_ * sizeof(float));
    }
}

FloatArray& FloatArray::operator=(const FloatArray& other) {
    if (this != &other) {
        delete[] data_;
        size_ = other.size_;
        if (size_ > 0) {
            data_ = new float[size_];
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        } else {
            data_ = nullptr;
        }
    }
    return *this;
}

FloatArray::FloatArray(FloatArray&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

FloatArray& FloatArray::operator=(FloatArray&& other) noexcept {
    if (this != &other) {
        delete[] data_;
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void FloatArray::resize(size_t newSize) {
    if (newSize == size_) {
        return;
    }
    
    float* newData = nullptr;
    if (newSize > 0) {
        try {
            newData = new float[newSize];
        } catch (const std::bad_alloc& e) {
            throw std::runtime_error("Failed to allocate memory for FloatArray");
        }
    }
    
    if (data_ != nullptr && newData != nullptr) {
        size_t copySize = (size_ < newSize) ? size_ : newSize;
        std::memcpy(newData, data_, copySize * sizeof(float));
    }
    
    delete[] data_;
    data_ = newData;
    size_ = newSize;
}

float* FloatArray::data() {
    return data_;
}

const float* FloatArray::data() const {
    return data_;
}

size_t FloatArray::size() const {
    return size_;
}

bool FloatArray::empty() const {
    return size_ == 0;
}

float& FloatArray::operator[](size_t index) {
    if (index >= size_) {
        throw std::out_of_range("FloatArray index out of range");
    }
    return data_[index];
}

const float& FloatArray::operator[](size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("FloatArray index out of range");
    }
    return data_[index];
}

void FloatArray::clear() {
    delete[] data_;
    data_ = nullptr;
    size_ = 0;
}

}  // namespace cllm
