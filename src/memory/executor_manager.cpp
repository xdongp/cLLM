/**
 * @file executor_manager.cpp
 * @brief 模型执行器内存管理器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/memory/executor_manager.h"
#include <stdexcept>
#include <cstring>

namespace cllm {

ModelExecutorMemoryManager::ModelExecutorMemoryManager(size_t maxMemoryMb)
    : maxMemoryBytes_(maxMemoryMb * 1024 * 1024), tempMemoryUsed_(0), weightsMemoryUsed_(0) {
}

ModelExecutorMemoryManager::~ModelExecutorMemoryManager() {
    clearAll();
}

void* ModelExecutorMemoryManager::allocateTempBuffer(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (tempMemoryUsed_ + size > maxMemoryBytes_) {
        throw std::runtime_error("Temp memory limit exceeded");
    }
    
    void* ptr = new char[size];
    std::memset(ptr, 0, size);
    
    BufferInfo info;
    info.ptr = ptr;
    info.size = size;
    info.isTemp = true;
    
    buffers_.push_back(info);
    tempMemoryUsed_ += size;
    
    return ptr;
}

void ModelExecutorMemoryManager::deallocateTempBuffer(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
        if (it->ptr == ptr && it->isTemp) {
            tempMemoryUsed_ -= it->size;
            delete[] static_cast<char*>(it->ptr);
            buffers_.erase(it);
            return;
        }
    }
}

void* ModelExecutorMemoryManager::allocateWeightsCache(size_t size) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (weightsMemoryUsed_ + size > maxMemoryBytes_) {
        throw std::runtime_error("Weights memory limit exceeded");
    }
    
    void* ptr = new char[size];
    std::memset(ptr, 0, size);
    
    BufferInfo info;
    info.ptr = ptr;
    info.size = size;
    info.isTemp = false;
    
    buffers_.push_back(info);
    weightsMemoryUsed_ += size;
    
    return ptr;
}

void ModelExecutorMemoryManager::deallocateWeightsCache(void* ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto it = buffers_.begin(); it != buffers_.end(); ++it) {
        if (it->ptr == ptr && !it->isTemp) {
            weightsMemoryUsed_ -= it->size;
            delete[] static_cast<char*>(it->ptr);
            buffers_.erase(it);
            return;
        }
    }
}

size_t ModelExecutorMemoryManager::getTempMemoryUsed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tempMemoryUsed_;
}

size_t ModelExecutorMemoryManager::getWeightsMemoryUsed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return weightsMemoryUsed_;
}

size_t ModelExecutorMemoryManager::getTotalMemoryUsed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return tempMemoryUsed_ + weightsMemoryUsed_;
}

void ModelExecutorMemoryManager::clearAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (const auto& info : buffers_) {
        delete[] static_cast<char*>(info.ptr);
    }
    
    buffers_.clear();
    tempMemoryUsed_ = 0;
    weightsMemoryUsed_ = 0;
}

}  // namespace cllm
