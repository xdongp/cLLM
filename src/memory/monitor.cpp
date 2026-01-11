/**
 * @file monitor.cpp
 * @brief 全局内存监控器实现
 * @author cLLM Team
 * @date 2024-01-01
 */

#include "cllm/memory/monitor.h"
#include <stdexcept>
#include <iostream>

namespace cllm {

MemoryMonitor::MemoryMonitor() {
    usedMemory_ = 0;
    peakMemory_ = 0;
    memoryLimit_ = 0;
}

MemoryMonitor& MemoryMonitor::instance() {
    static MemoryMonitor instance;
    return instance;
}

void MemoryMonitor::setLimit(size_t limitBytes) {
    memoryLimit_.store(limitBytes);
}

size_t MemoryMonitor::getLimit() const {
    return memoryLimit_.load();
}

void MemoryMonitor::allocate(size_t bytes) {
    size_t limit = memoryLimit_.load();
    if (limit > 0) {
        size_t used = usedMemory_.load();
        if (used + bytes > limit) {
            if (limitCallback_) {
                limitCallback_(used, limit);
            }
            throw std::runtime_error("Memory limit exceeded");
        }
    }
    
    usedMemory_.fetch_add(bytes);
    
    size_t current = usedMemory_.load();
    size_t peak = peakMemory_.load();
    while (current > peak) {
        if (peakMemory_.compare_exchange_weak(peak, current)) {
            break;
        }
    }
}

void MemoryMonitor::deallocate(size_t bytes) {
    usedMemory_.fetch_sub(bytes);
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
}

void MemoryMonitor::resetAll() {
    usedMemory_.store(0);
    peakMemory_.store(0);
}

}  // namespace cllm
