/**
 * @file memory_utils.cpp
 * @brief 通用内存管理工具类的实现
 * @author cLLM Team
 * @date 2026-01-10
 */

#include "cllm/common/memory_utils.h"
#include "cllm/common/logger.h"
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace cllm {

// FloatArray implementation
FloatArray::FloatArray(size_t size) : data_(nullptr), size_(size) {
    if (size > 0) {
        try {
            data_ = new float[size];
        } catch (const std::bad_alloc& e) {
            CLLM_ERROR("Failed to allocate FloatArray of size %zu: %s", size, e.what());
            throw;
        }
    }
}

FloatArray::~FloatArray() {
    delete[] data_;
}

FloatArray::FloatArray(const FloatArray& other) : data_(nullptr), size_(other.size_) {
    if (size_ > 0) {
        try {
            data_ = new float[size_];
            std::memcpy(data_, other.data_, size_ * sizeof(float));
        } catch (const std::bad_alloc& e) {
            CLLM_ERROR("Failed to copy FloatArray of size %zu: %s", size_, e.what());
            throw;
        }
    }
}

FloatArray& FloatArray::operator=(const FloatArray& other) {
    if (this != &other) {
        delete[] data_;
        size_ = other.size_;
        data_ = nullptr;
        
        if (size_ > 0) {
            try {
                data_ = new float[size_];
                std::memcpy(data_, other.data_, size_ * sizeof(float));
            } catch (const std::bad_alloc& e) {
                CLLM_ERROR("Failed to assign FloatArray of size %zu: %s", size_, e.what());
                throw;
            }
        }
    }
    return *this;
}

FloatArray::FloatArray(FloatArray&& other) noexcept : data_(other.data_), size_(other.size_) {
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
            CLLM_ERROR("Failed to resize FloatArray to size %zu: %s", newSize, e.what());
            throw;
        }
        
        if (data_ != nullptr) {
            size_t minSize = std::min(size_, newSize);
            std::memcpy(newData, data_, minSize * sizeof(float));
        }
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

// MemoryMonitor implementation
MemoryMonitor::MemoryMonitor() : usedMemory_(0), peakMemory_(0), memoryLimit_(0) {
    CLLM_DEBUG("MemoryMonitor initialized");
}

MemoryMonitor& MemoryMonitor::instance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::setLimit(size_t limitBytes) {
    memoryLimit_.store(limitBytes);
    CLLM_DEBUG("Memory limit set to %zu bytes", limitBytes);
}

size_t MemoryMonitor::getLimit() const {
    return memoryLimit_.load();
}

void MemoryMonitor::allocate(size_t bytes) {
    size_t limit = memoryLimit_.load();
    if (limit > 0) {
        size_t currentUsed = usedMemory_.load();
        if (currentUsed + bytes > limit) {
            if (limitCallback_) {
                limitCallback_(currentUsed, limit);
            }
            throw std::runtime_error("Memory allocation would exceed limit");
        }
    }
    
    size_t prevUsed = usedMemory_.fetch_add(bytes);
    
    // Update peak memory if necessary
    size_t currentPeak = peakMemory_.load();
    size_t newPotentialPeak = prevUsed + bytes;
    while (newPotentialPeak > currentPeak && currentPeak < peakMemory_.compare_exchange_weak(currentPeak, newPotentialPeak)) {
        // Continue trying to update peak memory
    }
    
    CLLM_DEBUG("Allocated %zu bytes, total used: %zu", bytes, prevUsed + bytes);
}

void MemoryMonitor::deallocate(size_t bytes) {
    size_t prevUsed = usedMemory_.fetch_sub(bytes);
    CLLM_DEBUG("Deallocated %zu bytes, total used: %zu", bytes, prevUsed - bytes);
}

size_t MemoryMonitor::getUsed() const {
    return usedMemory_.load();
}

size_t MemoryMonitor::getPeak() const {
    return peakMemory_.load();
}

void MemoryMonitor::setLimitCallback(MemoryLimitCallback callback) {
    limitCallback_ = callback;
}

void MemoryMonitor::resetPeak() {
    peakMemory_.store(0);
    CLLM_DEBUG("Peak memory reset");
}

void MemoryMonitor::resetAll() {
    usedMemory_.store(0);
    peakMemory_.store(0);
    CLLM_DEBUG("All memory statistics reset");
}

}  // namespace cllm