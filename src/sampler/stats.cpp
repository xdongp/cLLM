/**
 * @file stats.cpp
 * @brief 采样器统计信息类实现
 * @author cLLM Team
 * @date 2026-01-09
 */

#include "cllm/sampler/stats.h"
#include <sstream>
#include <iomanip>

namespace cllm {

SamplerStats::SamplerStats()
    : totalSamples_(0)
    , greedySamples_(0)
    , topKSamples_(0)
    , topPSamples_(0)
    , temperatureSamples_(0) {
}

SamplerStats::SamplerStats(const SamplerStats& other) 
    : totalSamples_(0)
    , greedySamples_(0)
    , topKSamples_(0)
    , topPSamples_(0)
    , temperatureSamples_(0) {
    std::lock_guard<std::mutex> lock(other.mutex_);
    totalSamples_ = other.totalSamples_;
    greedySamples_ = other.greedySamples_;
    topKSamples_ = other.topKSamples_;
    topPSamples_ = other.topPSamples_;
    temperatureSamples_ = other.temperatureSamples_;
}

SamplerStats& SamplerStats::operator=(const SamplerStats& other) {
    if (this != &other) {
        std::lock_guard<std::mutex> lock1(mutex_);
        std::lock_guard<std::mutex> lock2(other.mutex_);
        totalSamples_ = other.totalSamples_;
        greedySamples_ = other.greedySamples_;
        topKSamples_ = other.topKSamples_;
        topPSamples_ = other.topPSamples_;
        temperatureSamples_ = other.temperatureSamples_;
    }
    return *this;
}

SamplerStats::~SamplerStats() {
}

void SamplerStats::incrementTotalSamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++totalSamples_;
}

void SamplerStats::incrementGreedySamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++greedySamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTopKSamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++topKSamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTopPSamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++topPSamples_;
    ++totalSamples_;
}

void SamplerStats::incrementTemperatureSamples() {
    std::lock_guard<std::mutex> lock(mutex_);
    ++temperatureSamples_;
    ++totalSamples_;
}

long long SamplerStats::getTotalSamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return totalSamples_;
}

long long SamplerStats::getGreedySamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return greedySamples_;
}

long long SamplerStats::getTopKSamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return topKSamples_;
}

long long SamplerStats::getTopPSamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return topPSamples_;
}

long long SamplerStats::getTemperatureSamples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return temperatureSamples_;
}

float SamplerStats::getGreedyPercentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculatePercentage(greedySamples_);
}

float SamplerStats::getTopKPercentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculatePercentage(topKSamples_);
}

float SamplerStats::getTopPPercentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculatePercentage(topPSamples_);
}

float SamplerStats::getTemperaturePercentage() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return calculatePercentage(temperatureSamples_);
}

void SamplerStats::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    totalSamples_ = 0;
    greedySamples_ = 0;
    topKSamples_ = 0;
    topPSamples_ = 0;
    temperatureSamples_ = 0;
}

std::string SamplerStats::toString() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    oss << "SamplerStats {" << std::endl;
    oss << "  Total Samples: " << totalSamples_ << std::endl;
    oss << "  Greedy: " << greedySamples_ 
        << " (" << calculatePercentage(greedySamples_) << "%)" << std::endl;
    oss << "  Top-K: " << topKSamples_ 
        << " (" << calculatePercentage(topKSamples_) << "%)" << std::endl;
    oss << "  Top-P: " << topPSamples_ 
        << " (" << calculatePercentage(topPSamples_) << "%)" << std::endl;
    oss << "  Temperature: " << temperatureSamples_ 
        << " (" << calculatePercentage(temperatureSamples_) << "%)" << std::endl;
    oss << "}";
    
    return oss.str();
}

float SamplerStats::calculatePercentage(long long count) const {
    if (totalSamples_ == 0) {
        return 0.0f;
    }
    return (static_cast<float>(count) / static_cast<float>(totalSamples_)) * 100.0f;
}

}
